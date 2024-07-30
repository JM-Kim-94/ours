from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.cluster import estimate_bandwidth

import math

import random
from collections import deque


class Cbuffer:  # group_c_buffers
    def __init__(self, capacity):
        self.c_buffer = deque(maxlen=capacity)

    def add_c(self, c):  # ([10]), ([10])  -->  ([1, 10]), ([1, 10])
        dims = len(c.size())
        c = c.detach()  # .cpu().numpy()
        if dims == 1:
            # self.c_buffer.append(c.unsqueeze(0))
            self.c_buffer.append(c.unsqueeze(0))
        elif dims == 2:
            for i in range(len(c)):
                # self.c_buffer.append(c[i].unsqueeze(0))
                self.c_buffer.append(c[i].unsqueeze(0))

    def sample_c(self, num_samples):
        c_batches = random.sample(self.c_buffer, num_samples)
        # c_batches = np.array(c_batches)
        c_batches_tensor = torch.cat(c_batches).to(ptu.device)
        return c_batches_tensor

    def len(self):
        return len(self.c_buffer)


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            target_enc_tau=0.001,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.c_buffer = Cbuffer(capacity=self.c_buffer_size)

        self.ce_loss = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        print("psi trainable parameters :", sum(p.numel() for p in self.agent.psi.parameters() if p.requires_grad))
        if self.use_W:
            self.agent.psi.register_parameter('W', nn.Parameter(torch.rand(latent_dim, latent_dim)))
            self.agent.psi_target.register_parameter('W', nn.Parameter(torch.rand(latent_dim, latent_dim)))
            self.agent.psi_target.load_state_dict(self.agent.psi.state_dict())
            print("psi trainable parameters after parameter W addition :",
                  sum(p.numel() for p in self.agent.psi.parameters() if p.requires_grad))

        self.psi_optim = optimizer_class(
            self.agent.psi.parameters(),
            lr=context_lr,
        )
        # self.psi2_optim = optimizer_class(
        #     self.agent.psi2.parameters(),
        #     lr=context_lr,
        # )
        self.k_decoder_optim = optimizer_class(
            self.agent.k_decoder.parameters(),
            lr=context_lr,
        )
        self.c_decoder_optim = optimizer_class(
            self.agent.c_decoder.parameters(),
            lr=context_lr,
        )
        self.disc_optim = optimizer_class(
            self.agent.disc.parameters(),
            lr=0.0001
        )

        # self.z_autoencoder_optim = optimizer_class(
        #     self.agent.z_autoencoder.parameters(),
        #     lr=0.0003
        # )

        self.c_distribution_vae_optim = optimizer_class(
            self.agent.c_distribution_vae.parameters(),
            lr=0.001
        )

        self.sim_fn = torch.nn.CosineSimilarity(dim=-1)
        # self.decoder_optimizer = optimizer_class(
        #     self.agent.decoder.parameters(),
        #     lr=0.0005,
        # )
        self.target_enc_tau = target_enc_tau

        random_obs = torch.randn(16, 20).to('cuda')
        random_obs = [o.repeat(256, 1).unsqueeze(0) for o in random_obs]
        self.random_obs = torch.cat(random_obs, dim=0)  # [16, 256, 5])

        random_actions = torch.randn(16, 6).to('cuda')
        random_actions = [a.repeat(256, 1).unsqueeze(0) for a in random_actions]
        self.random_actions = torch.cat(random_actions, dim=0)  # [16, 256, 5])

        self.random_obs_flat, self.random_actions_flat = self.random_obs.view(-1, 20), self.random_actions.view(-1, 6)

    def target_enc_soft_update(self):
        for param_target, param in zip(self.agent.psi_target.parameters(), self.agent.psi.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.target_enc_tau) + param.data * self.target_enc_tau)
            # param_target * 0.999 + param_online * 0.001

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices, batchsize):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense

        # batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=batchsize)) for idx in
                   indices]

        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_total_task_transition(self, indices_len, batch_size):
        # prior_enc_buffer_size = np.zeros(indices_len)
        # for task_idx in range(indices_len):
        #     prior_enc_buffer_size[task_idx] = self.total_enc_replay_buffer.task_buffers[task_idx].size()
        # print(prior_enc_buffer_size)
        # [10503.     0.     0.     0.]
        batches = [ptu.np_to_pytorch_batch(
            self.total_enc_replay_buffer.random_batch(0, batch_size=batch_size)) for idx in range(indices_len)]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices, which_buffer="default", b_size="default", return_unpacked=False):  # which_buffer : "prior" or "online"
        ''' sample batch of context from a list of tasks from the replay buffer '''
        if which_buffer=="default":
            which_buffer = self.offpol_ctxt_sampling_buffer
        if b_size == "default":
            b_size = self.embedding_batch_size

        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        # if self.use_context_buffer:

        if which_buffer == "rl":
            batches = [ptu.np_to_pytorch_batch(
                self.replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]

        elif which_buffer == "prior":
            batches = [ptu.np_to_pytorch_batch(
                self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]


        elif which_buffer == "online":
            t = [self.online_enc_replay_buffer.task_buffers[idx].size() > 0 for idx in indices]
            if math.prod(t):  # 샘플링된 모든 인덱스의 온라인 컨텍스트 버퍼에 샘플이 차 있으면:
                batches = [ptu.np_to_pytorch_batch(
                    self.online_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]
            else:
                batches = [ptu.np_to_pytorch_batch(
                    self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]


        elif which_buffer == "both":
            t = [self.online_enc_replay_buffer.task_buffers[idx].size() > 0 for idx in indices]  # 그 태스크 인덱스에 대한 온라인 버퍼가 차 있는지 확인하는 리스트
            # print("t :", t, math.prod(t))  # [True, False, False, True, ... ]
            if math.prod(t):  # 샘플링된 모든 인덱스의 온라인 컨텍스트 버퍼에 샘플이 차있으면:
                b_size_online = int(b_size / 5)  # --> 128개 컨텍스트 배치에서 1/5는 온라인 버퍼에서 샘플링
                b_size_prior = b_size - b_size_online  # --> 128개 배치에서 나머지는 프라이어 버퍼에서
                batches_online = [
                    ptu.np_to_pytorch_batch(self.online_enc_replay_buffer.random_batch(idx, batch_size=b_size_online)) for idx in indices]
                batches_prior = [
                    ptu.np_to_pytorch_batch(self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size_prior)) for idx in indices]
                # batches = batches1 + batches2
                # print("t list :", t, [self.online_enc_replay_buffer.task_buffers[idx].size() for idx in indices], len(batches), batches[0]["observations"].shape)  # ([64, 20])
                batches, temp_dict = [], {}
                for j in range(len(indices)):  # 16개 태스크
                    temp_dict = {}
                    perm = np.random.permutation(b_size)
                    temp_dict["observations"] = \
                        torch.cat([batches_online[j]["observations"], batches_prior[j]["observations"]])[
                            perm]  # ([64, 20]) , ([64, 20]) --> ([128, 20])
                    temp_dict["actions"] = torch.cat([batches_online[j]["actions"], batches_prior[j]["actions"]])[perm]
                    temp_dict["rewards"] = torch.cat([batches_online[j]["rewards"], batches_prior[j]["rewards"]])[perm]
                    temp_dict["terminals"] = torch.cat([batches_online[j]["terminals"], batches_prior[j]["terminals"]])[perm]
                    temp_dict["next_observations"] = \
                        torch.cat([batches_online[j]["next_observations"], batches_prior[j]["next_observations"]])[perm]
                    temp_dict["sparse_rewards"] = \
                        torch.cat([batches_online[j]["sparse_rewards"], batches_prior[j]["sparse_rewards"]])[perm]
                    batches.append(temp_dict)

                    # print(temp_dict["observations"].shape)  # ([128, 20])

            else:
                batches = [ptu.np_to_pytorch_batch(
                    self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]
                # print(len(batches))

        else:
            batches = None

        # else:
        #     batches = [ptu.np_to_pytorch_batch(
        #         self.replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size)) for idx in indices]

        unpacked = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(unpacked[:-1], dim=2)
        else:
            context = torch.cat(unpacked[:-2], dim=2)

        if return_unpacked:
            return context, unpacked
        else:
            return context

    """ 디코더 사용했을때 학습했던 함수들 """

    def _do_psi1_recon_train(self):
        indices = np.random.choice(self.train_tasks, self.meta_batch)
        transition_batch = self.sample_sac(indices, 256)  # ([16, 256, 20]), ...
        context_batch = self.sample_context(indices)  # [16,100,27]  20 + 6 + 1
        obs, actions, rewards, next_obs, terms = transition_batch
        # rewards = rewards * self.reward_scale
        # self.agent.clear_z(num_tasks=len(indices))

        z_psi1, z_psi1_means, z_psi1_vars = self.agent.infer_posterior(context_batch, which_enc='psi1')
        z_psi1 = [c.repeat(256, 1).unsqueeze(0) for c in z_psi1]
        z_psi1 = torch.cat(z_psi1, dim=0)  # ([16, 256, 5])

        """ (1) 리컨스트럭션 """

        state_embed = self.agent.decoder.state_embed(obs)
        obs_pred, action_pred = self.agent.decoder.dec1(z_psi1, state_embed)
        reward_pred, next_obs_pred, _, _, terms_pred = self.agent.decoder.dec2(z_psi1, state_embed, actions)

        obs_recon_loss = F.mse_loss(obs, obs_pred)
        action_recon_loss = F.mse_loss(actions, action_pred)
        reward_recon_loss = F.mse_loss(rewards, reward_pred)
        next_obs_recon_loss = F.mse_loss(next_obs, next_obs_pred)
        terms_recon = F.mse_loss(terms, terms_pred)
        total_recon_loss = obs_recon_loss + action_recon_loss + reward_recon_loss + next_obs_recon_loss + terms_recon

        """ (2) z_psi1 KL """
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                        zip(torch.unbind(z_psi1_means), torch.unbind(z_psi1_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))

        psi1_total_loss = self.kl_lambda * kl_div_sum + 10 * total_recon_loss

        self.psi1_optim.zero_grad()
        self.decoder_optim.zero_grad()
        psi1_total_loss.backward()
        self.psi1_optim.step()
        self.decoder_optim.step()

        return psi1_total_loss.item(), total_recon_loss.item(), kl_div_sum.item(), \
            obs_recon_loss.item(), action_recon_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item()

    def _do_wgan_disc(self):
        indices = np.random.choice(self.train_tasks, self.meta_batch)
        task_1_indices = indices[:8]
        task_2_indices = indices[8:]
        tran_bsize = 128

        t1_tran_bat = self.sample_sac(task_1_indices, tran_bsize)
        t2_tran_bat = self.sample_sac(task_2_indices, tran_bsize)
        t1_ctxt_bat = self.sample_context(task_1_indices)
        t2_ctxt_bat = self.sample_context(task_2_indices)

        o1, a1, r1, next_o1, _ = t1_tran_bat
        o2, a2, r2, next_o2, _ = t2_tran_bat

        with torch.no_grad():
            c1, _, _ = self.agent.infer_posterior(t1_ctxt_bat, which_enc="psi1")
            c1 = [c.repeat(tran_bsize, 1).unsqueeze(0) for c in c1]
            c1 = torch.cat(c1, dim=0)  # [8, 256, 10])

            c2, _, _ = self.agent.infer_posterior(t2_ctxt_bat, which_enc="psi1")
            c2 = [c.repeat(tran_bsize, 1).unsqueeze(0) for c in c2]
            c2 = torch.cat(c2, dim=0)  # [8, 256, 10])

            alpha = np.random.uniform(0, 1, 8)
            c_alpha = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
                c_alpha.append(interpolation.unsqueeze(0))
            c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 256, 10])

            s1 = self.agent.decoder.state_embed(o1)
            s2 = self.agent.decoder.state_embed(o2)
            o_1alpha, a_1alpha = self.agent.decoder.dec1(c_alpha, s1)  # ([8, 256, 20])
            o_2alpha, a_2alpha = self.agent.decoder.dec1(c_alpha, s2)  # ([8, 256, 20])
            r_1alpha, next_o_1alpha, _, _, _ = self.agent.decoder.dec2(c_alpha, s1, a_1alpha)
            r_2alpha, next_o_2alpha, _, _, _ = self.agent.decoder.dec2(c_alpha, s2, a_2alpha)

            alpha_GP = np.random.uniform(0, 1, 8)
            c_alpha_GP = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                interpolation = alpha_GP[i] * c1[i] + (1 - alpha_GP[i]) * c2[i]  # ([256, 10])
                c_alpha_GP.append(interpolation.unsqueeze(0))
            c_alpha_GP = torch.cat(c_alpha_GP, dim=0)  # ([8, 256, 10])

            o_1alpha_GP, a_1alpha_GP = self.agent.decoder.dec1(c_alpha_GP, s1)  # ([8, 256, 20])
            o_2alpha_GP, a_2alpha_GP = self.agent.decoder.dec1(c_alpha_GP, s2)  # ([8, 256, 20])
            r_1alpha_GP, next_o_1alpha_GP, _, _, _ = self.agent.decoder.dec2(c_alpha_GP, s1, a_1alpha_GP)
            r_2alpha_GP, next_o_2alpha_GP, _, _, _ = self.agent.decoder.dec2(c_alpha_GP, s2, a_2alpha_GP)

        d_real_sample1 = torch.cat([o1, a1, r1, next_o1], dim=-1).view(-1, 47)
        d_real_sample2 = torch.cat([o2, a2, r2, next_o2], dim=-1).view(-1, 47)
        d_fake_sample1 = torch.cat([o_1alpha, a_1alpha, r_1alpha, next_o_1alpha], dim=-1).view(-1, 47)
        d_fake_sample2 = torch.cat([o_2alpha, a_2alpha, r_2alpha, next_o_2alpha], dim=-1).view(-1, 47)

        d_real_score1 = self.agent.disc(d_real_sample1).mean()
        d_real_score2 = self.agent.disc(d_real_sample2).mean()
        d_fake_score1 = self.agent.disc(d_fake_sample1).mean()
        d_fake_score2 = self.agent.disc(d_fake_sample2).mean()

        # gradient penalty
        d_fake_sample1_GP = torch.cat([o_1alpha, a_1alpha, r_1alpha, next_o_1alpha], dim=-1).view(-1, 47)
        d_fake_sample2_GP = torch.cat([o_2alpha, a_2alpha, r_2alpha, next_o_2alpha], dim=-1).view(-1, 47)
        d_fake_sample_GP = torch.cat([d_fake_sample1_GP, d_fake_sample2_GP], dim=0)
        d_fake_sample_GP.requires_grad = True
        d_fake_score_GP = self.agent.disc(d_fake_sample_GP)
        gradients = torch.autograd.grad(outputs=d_fake_score_GP, inputs=d_fake_sample_GP,
                                        grad_outputs=torch.ones(d_fake_score_GP.size()).to(ptu.device),
                                        create_graph=True, retain_graph=True)[0]  # ([256, 57])
        gradients_norm = gradients.norm(2, 1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        gradients_norm = gradients_norm.mean()

        d_total_loss = - d_real_score1 - d_real_score2 + d_fake_score1 + d_fake_score2 + self.wgan_lambda * gradient_penalty

        self.disc_optim.zero_grad()
        d_total_loss.backward()
        self.disc_optim.step()

        return d_total_loss.item(), (d_real_score1 + d_real_score2).item(), (d_fake_score1 + d_fake_score2).item(), \
            gradient_penalty.item(), gradients_norm.item()

    def _do_wgan_gen(self):
        indices = np.random.choice(self.train_tasks, self.meta_batch)
        task_1_indices = indices[:8]
        task_2_indices = indices[8:]
        tran_bsize = 128

        t1_tran_bat = self.sample_sac(task_1_indices, tran_bsize)
        t2_tran_bat = self.sample_sac(task_2_indices, tran_bsize)
        t1_ctxt_bat = self.sample_context(task_1_indices)
        t2_ctxt_bat = self.sample_context(task_2_indices)

        o1, a1, r1, next_o1, _ = t1_tran_bat
        o2, a2, r2, next_o2, _ = t2_tran_bat

        with torch.no_grad():
            c1, _, _ = self.agent.infer_posterior(t1_ctxt_bat, which_enc="psi1")
            c1 = [c.repeat(tran_bsize, 1).unsqueeze(0) for c in c1]
            c1 = torch.cat(c1, dim=0)  # [8, 256, 10])

            c2, _, _ = self.agent.infer_posterior(t2_ctxt_bat, which_enc="psi1")
            c2 = [c.repeat(tran_bsize, 1).unsqueeze(0) for c in c2]
            c2 = torch.cat(c2, dim=0)  # [8, 256, 10])

            alpha = np.random.uniform(0, 1, 8)
            c_alpha = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
                c_alpha.append(interpolation.unsqueeze(0))
            c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 256, 10])

        s1 = self.agent.decoder.state_embed(o1)
        s2 = self.agent.decoder.state_embed(o2)
        o_1alpha, a_1alpha = self.agent.decoder.dec1(c_alpha, s1)  # ([8, 256, 20])
        o_2alpha, a_2alpha = self.agent.decoder.dec1(c_alpha, s2)  # ([8, 256, 20])
        r_1alpha, next_o_1alpha, _, _, _ = self.agent.decoder.dec2(c_alpha, s1, a_1alpha)
        r_2alpha, next_o_2alpha, _, _, _ = self.agent.decoder.dec2(c_alpha, s2, a_2alpha)

        g_real_sample1 = torch.cat([o1, a1, r1, next_o1], dim=-1).view(-1, 47)
        g_real_sample2 = torch.cat([o2, a2, r2, next_o2], dim=-1).view(-1, 47)
        g_fake_sample1 = torch.cat([o_1alpha, a_1alpha, r_1alpha, next_o_1alpha], dim=-1).view(-1, 47)
        g_fake_sample2 = torch.cat([o_2alpha, a_2alpha, r_2alpha, next_o_2alpha], dim=-1).view(-1, 47)

        g_real_score1 = self.agent.disc(g_real_sample1).mean().detach()
        g_real_score2 = self.agent.disc(g_real_sample2).mean().detach()
        g_fake_score1 = self.agent.disc(g_fake_sample1).mean()
        g_fake_score2 = self.agent.disc(g_fake_sample2).mean()

        g_total_loss = - g_fake_score1 - g_fake_score2

        self.decoder_optim.zero_grad()
        g_total_loss.backward()
        self.decoder_optim.step()

        w_distance = g_real_score1 + g_real_score2 - g_fake_score1 - g_fake_score2

        return g_total_loss.item(), w_distance.item(), (g_real_score1 + g_real_score2).item(), (
                g_fake_score1 + g_fake_score2).item()

    #

    def _do_wgan_disc2(self, indices):
        task_1_indices = indices[:8]  # [indices[0]]
        task_2_indices = indices[8:]  # [indices[1]]
        tran_bsize = 128

        t1_tran_bat = self.sample_sac(task_1_indices, tran_bsize)
        t2_tran_bat = self.sample_sac(task_2_indices, tran_bsize)
        t1_ctxt_bat = self.sample_context(task_1_indices)
        t2_ctxt_bat = self.sample_context(task_2_indices)

        o1, a1, r1, next_o1, _ = t1_tran_bat
        o2, a2, r2, next_o2, _ = t2_tran_bat

        with torch.no_grad():
            c1 = self.agent.get_context_embedding(t1_ctxt_bat, use_target=False)
            # c1_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c1], dim=0)

            c2 = self.agent.get_context_embedding(t2_ctxt_bat, use_target=False)
            # c2_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c2], dim=0)

            # alpha = np.random.uniform(0, 1, 8)
            # c_alpha = []  # ([C1] --------(1-a)------------------------ [Ca] ----(a)----- [C2])
            # for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
            #     interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
            #     c_alpha.append(interpolation.unsqueeze(0))
            # c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 256, 10])
            c_alpha = []
            for i in range(8):
                temp = []
                a = np.random.dirichlet(np.ones(16)) * np.random.uniform(0.8, 1.5)
                for j in range(16):
                    temp.append(a[j] * torch.cat([c1, c2])[j])
                temp = sum(temp).unsqueeze(0)
                c_alpha.append(temp)
            c_alpha = torch.cat(c_alpha)  # ([8, 10])
            c_alpha = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c_alpha], dim=0)

            # a_1alpha = self.agent.policy(torch.cat([o1.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1), reparameterize=True, return_log_prob=True)[0]
            # a_2alpha = self.agent.policy(torch.cat([o2.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1), reparameterize=True, return_log_prob=True)[0]
            a_1alpha = self.agent.decoder.dec1(o1.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim))
            a_2alpha = self.agent.decoder.dec1(o2.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim))
            r_1alpha, next_o_1alpha, _, _ = self.agent.decoder(o1.view(-1, self.o_dim), a_1alpha, c_alpha.view(-1, self.l_dim))
            r_2alpha, next_o_2alpha, _, _ = self.agent.decoder(o2.view(-1, self.o_dim), a_2alpha, c_alpha.view(-1, self.l_dim))

            # alpha_GP = np.random.uniform(0, 1, 8)
            # c_alpha_GP = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            # for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
            #     interpolation = alpha_GP[i] * c1[i] + (1 - alpha_GP[i]) * c2[i]  # ([256, 10])
            #     c_alpha_GP.append(interpolation.unsqueeze(0))
            # c_alpha_GP = torch.cat(c_alpha_GP, dim=0)  # ([8, 256, 10])
            c_alpha_GP = []
            for i in range(8):
                temp = []
                a = np.random.dirichlet(np.ones(16)) * np.random.uniform(0.8, 1.5)
                for j in range(16):
                    temp.append(a[j] * torch.cat([c1, c2])[j])
                temp = sum(temp).unsqueeze(0)
                c_alpha_GP.append(temp)
            c_alpha_GP = torch.cat(c_alpha_GP)  # ([8, 10])
            c_alpha_GP = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c_alpha_GP], dim=0)

            # a_1alpha_GP = self.agent.policy(torch.cat([o1.view(-1, self.o_dim), c_alpha_GP.view(-1, self.l_dim)], dim=1), reparameterize=True, return_log_prob=True)[0]
            # a_2alpha_GP = self.agent.policy(torch.cat([o2.view(-1, self.o_dim), c_alpha_GP.view(-1, self.l_dim)], dim=1),  reparameterize=True, return_log_prob=True)[0]
            a_1alpha_GP = self.agent.decoder.dec1(o1.view(-1, self.o_dim), c_alpha_GP.view(-1, self.l_dim))
            a_2alpha_GP = self.agent.decoder.dec1(o2.view(-1, self.o_dim), c_alpha_GP.view(-1, self.l_dim))
            r_1alpha_GP, next_o_1alpha_GP, _, _ = self.agent.decoder(o1.view(-1, self.o_dim), a_1alpha_GP, c_alpha_GP.view(-1, self.l_dim))
            r_2alpha_GP, next_o_2alpha_GP, _, _ = self.agent.decoder(o2.view(-1, self.o_dim), a_2alpha_GP, c_alpha_GP.view(-1, self.l_dim))

        d_real_sample1 = torch.cat([o1.view(-1, self.o_dim), a1.view(-1, self.a_dim), r1.view(-1, 1), next_o1.view(-1, self.o_dim)], dim=-1)
        d_real_sample2 = torch.cat([o2.view(-1, self.o_dim), a2.view(-1, self.a_dim), r2.view(-1, 1), next_o2.view(-1, self.o_dim)], dim=-1)
        d_fake_sample1 = torch.cat([o1.view(-1, self.o_dim), a_1alpha, r_1alpha, next_o_1alpha], dim=-1)
        d_fake_sample2 = torch.cat([o2.view(-1, self.o_dim), a_2alpha, r_2alpha, next_o_2alpha], dim=-1)

        d_real_score1 = self.agent.disc(d_real_sample1).mean()
        d_real_score2 = self.agent.disc(d_real_sample2).mean()
        d_fake_score1 = self.agent.disc(d_fake_sample1).mean()
        d_fake_score2 = self.agent.disc(d_fake_sample2).mean()

        # gradient penalty
        d_fake_sample1_GP = torch.cat([o1.view(-1, self.o_dim), a_1alpha_GP, r_1alpha_GP, next_o_1alpha_GP], dim=-1)
        d_fake_sample2_GP = torch.cat([o2.view(-1, self.o_dim), a_2alpha_GP, r_2alpha_GP, next_o_2alpha_GP], dim=-1)
        d_fake_sample_GP = torch.cat([d_fake_sample1_GP, d_fake_sample2_GP], dim=0)
        d_fake_sample_GP.requires_grad = True
        d_fake_score_GP = self.agent.disc(d_fake_sample_GP)
        gradients = torch.autograd.grad(outputs=d_fake_score_GP, inputs=d_fake_sample_GP,
                                        grad_outputs=torch.ones(d_fake_score_GP.size()).to(ptu.device),
                                        create_graph=True, retain_graph=True)[0]  # ([256, 57])
        gradients_norm = gradients.norm(2, 1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        gradients_norm = gradients_norm.mean()

        d_total_loss = - d_real_score1 - d_real_score2 + d_fake_score1 + d_fake_score2 + self.wgan_lambda * gradient_penalty
        # d_total_loss = 10 * d_total_loss

        self.disc_optim.zero_grad()
        d_total_loss.backward()
        self.disc_optim.step()

        return d_total_loss.item(), (d_real_score1 + d_real_score2).item(), (d_fake_score1 + d_fake_score2).item(), \
            gradient_penalty.item(), gradients_norm.item()

    def _do_wgan_gen2(self, indices):
        task_1_indices = indices[:8]  # [indices[0]]
        task_2_indices = indices[8:]  # [indices[1]]
        tran_bsize = 128

        t1_tran_bat = self.sample_sac(task_1_indices, tran_bsize)
        t2_tran_bat = self.sample_sac(task_2_indices, tran_bsize)
        t1_ctxt_bat = self.sample_context(task_1_indices)
        t2_ctxt_bat = self.sample_context(task_2_indices)

        o1, a1, r1, next_o1, _ = t1_tran_bat
        o2, a2, r2, next_o2, _ = t2_tran_bat

        with torch.no_grad():
            c1 = self.agent.get_context_embedding(t1_ctxt_bat, use_target=False)
            # c1 = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c1], dim=0)

            c2 = self.agent.get_context_embedding(t2_ctxt_bat, use_target=False)
            # c2 = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c2], dim=0)

            # alpha = np.random.uniform(0, 1, 8)
            # c_alpha = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            # for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
            #     interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
            #     c_alpha.append(interpolation.unsqueeze(0))
            # c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 256, 10])
            c_alpha = []
            for i in range(8):
                temp = []
                a = np.random.dirichlet(np.ones(16)) * np.random.uniform(0.8, 1.5)
                for j in range(16):
                    temp.append(a[j] * torch.cat([c1, c2])[j])
                temp = sum(temp).unsqueeze(0)
                c_alpha.append(temp)
            c_alpha = torch.cat(c_alpha)  # ([8, 10])
            c_alpha = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c_alpha], dim=0)

        # a_1alpha = self.agent.policy(torch.cat([o1.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1), reparameterize=True, return_log_prob=True)[0]
        # a_2alpha = self.agent.policy(torch.cat([o2.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1), reparameterize=True, return_log_prob=True)[0]
        a_1alpha = self.agent.decoder.dec1(o1.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim))
        a_2alpha = self.agent.decoder.dec1(o2.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim))
        r_1alpha, next_o_1alpha, _, _ = self.agent.decoder(o1.view(-1, self.o_dim), a_1alpha.detach(), c_alpha.view(-1, self.l_dim))
        r_2alpha, next_o_2alpha, _, _ = self.agent.decoder(o2.view(-1, self.o_dim), a_2alpha.detach(), c_alpha.view(-1, self.l_dim))

        g_real_sample1 = torch.cat([o1.view(-1, self.o_dim), a1.view(-1, self.a_dim), r1.view(-1, 1), next_o1.view(-1, self.o_dim)], dim=-1)
        g_real_sample2 = torch.cat([o2.view(-1, self.o_dim), a2.view(-1, self.a_dim), r2.view(-1, 1), next_o2.view(-1, self.o_dim)], dim=-1)
        g_fake_sample1 = torch.cat([o1.view(-1, self.o_dim), a_1alpha, r_1alpha, next_o_1alpha], dim=-1)
        g_fake_sample2 = torch.cat([o2.view(-1, self.o_dim), a_2alpha, r_2alpha, next_o_2alpha], dim=-1)

        g_real_score1 = self.agent.disc(g_real_sample1).mean().detach()
        g_real_score2 = self.agent.disc(g_real_sample2).mean().detach()
        g_fake_score1 = self.agent.disc(g_fake_sample1).mean()
        g_fake_score2 = self.agent.disc(g_fake_sample2).mean()

        g_total_loss = - g_fake_score1 - g_fake_score2
        # g_total_loss = 10 * g_total_loss

        self.decoder_optim.zero_grad()
        # self.policy_optimizer.zero_grad()
        g_total_loss.backward()
        self.decoder_optim.step()
        # self.policy_optimizer.step()

        w_distance = g_real_score1 + g_real_score2 - g_fake_score1 - g_fake_score2

        return g_total_loss.item(), w_distance.item(), (g_real_score1 + g_real_score2).item(), (
                g_fake_score1 + g_fake_score2).item()

    """"""





    def _do_gan_disc(self, indices_):

        indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
        indices_diric_inter_tasks = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)  # 3개
        tran_bsize = 128

        o, a, r, next_o, _ = self.sample_sac(indices, tran_bsize)  # 5개
        ctxt_bat = self.sample_context(indices_diric_inter_tasks, which_buffer=self.offpol_ctxt_sampling_buffer)

        with torch.no_grad():
            task_c = self.agent.get_context_embedding(ctxt_bat, use_target=False)  # ([3, 10])
            task_c_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in task_c], dim=0)  # ([3, 128, 10])
            task_c_repeat = task_c_repeat.view(-1, self.l_dim)

            task_c_alpha = []
            for i in range(self.num_fake_tasks):  # 5번
                alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                task_c_alpha.append(alpha @ task_c)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
            task_c_alpha = torch.cat(task_c_alpha)  # ([5, 10])
            task_c_alpha_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c_alpha], dim=0)  # ([5, 212856, 10])
            task_c_alpha_repeat = task_c_alpha_repeat.view(-1, self.l_dim)  # ([5x128, 10])
            
            r_alpha, n_o_alpha, _, _ = self.agent.c_decoder(o.view(-1, self.o_dim), a.view(-1, self.a_dim), task_c_alpha_repeat)

        d_real_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r.view(-1, 1), next_o.view(-1, self.o_dim)], dim=-1)
        if self.use_decoder_next_state:
            d_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, n_o_alpha], dim=-1)
        else:
            d_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, next_o.view(-1, self.o_dim)], dim=-1)

        d_real_score = self.agent.disc(d_real_sample)  # ([5x128, 1])
        d_fake_score = self.agent.disc(d_fake_sample)  # ([5x128, 1])
        
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(d_real_score)
        fake_labels = torch.zeros_like(d_fake_score)

        d_loss_real = criterion(d_real_score, real_labels)
        d_loss_fake = criterion(d_fake_score, fake_labels)
        d_total_loss = d_loss_real + d_loss_fake

        # d_total_loss = - d_real_score + d_fake_score + self.wgan_lambda * gradient_penalty

        self.disc_optim.zero_grad()
        d_total_loss.backward()
        self.disc_optim.step()

        return d_total_loss.item(), d_real_score.mean().item(), d_fake_score.mean().item(), 0, 0

    def _do_gan_gen(self, indices_):

        indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
        indices_diric_inter_tasks = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)  # 3개
        tran_bsize = 128

        o, a, r, next_o, _ = self.sample_sac(indices, tran_bsize)  # 5개
        ctxt_bat = self.sample_context(indices_diric_inter_tasks, which_buffer=self.offpol_ctxt_sampling_buffer)

        with torch.no_grad():
            task_c = self.agent.get_context_embedding(ctxt_bat, use_target=False)  # ([3, 10])
            task_c_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in task_c], dim=0)  # ([3, 128, 10])
            task_c_repeat = task_c_repeat.view(-1, self.l_dim)

            task_c_alpha = []
            for i in range(self.num_fake_tasks):  # 5번
                alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                task_c_alpha.append(alpha @ task_c)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
            task_c_alpha = torch.cat(task_c_alpha)  # ([5, 10])
            task_c_alpha_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c_alpha], dim=0)  # ([5, 212856, 10])
            task_c_alpha_repeat = task_c_alpha_repeat.view(-1, self.l_dim)  # ([5x128, 10])
            
        r_alpha, n_o_alpha, _, _ = self.agent.c_decoder(o.view(-1, self.o_dim), a.view(-1, self.a_dim), task_c_alpha_repeat)

        # g_real_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r.view(-1, 1), next_o.view(-1, self.o_dim)], dim=-1)
        if self.use_decoder_next_state:
            g_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, n_o_alpha], dim=-1)
        else:
            g_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, next_o.view(-1, self.o_dim)], dim=-1)

        # g_real_score = self.agent.disc(d_real_sample)  # ([5x128, 1])
        g_fake_score = self.agent.disc(g_fake_sample)  # ([5x128, 1])
        
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(g_fake_score)

        g_loss_fake = criterion(g_fake_score, real_labels)

        self.c_decoder_optim.zero_grad()
        # self.policy_optimizer.zero_grad()
        g_loss_fake.backward()
        self.c_decoder_optim.step()
        # self.policy_optimizer.step()

        # w_distance = g_real_score - g_fake_score

        return g_loss_fake.item(), 0, 0, g_fake_score.mean().item()

    """   """



    def _do_wgan_disc3(self, indices_):

        indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
        indices_diric_inter_tasks = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)  # 3개
        tran_bsize = 128

        o, a, r, next_o, _ = self.sample_sac(indices, tran_bsize)
        ctxt_bat = self.sample_context(indices_diric_inter_tasks, which_buffer=self.offpol_ctxt_sampling_buffer)

        with torch.no_grad():
            task_c = self.agent.get_context_embedding(ctxt_bat, use_target=False)
            # task_z, _ = self.agent.z_autoencoder(task_c)
            task_c_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c], dim=0)  # ([16, 256, 10])
            task_c_repeat = task_c_repeat.view(-1, self.l_dim)

            task_c_alpha = []
            for i in range(self.num_fake_tasks):  # 16번
                if self.use_full_interpolation:
                    alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                else:
                    alpha = np.random.uniform(0.1, 0.9)
                    alpha = np.array([alpha, 1 - alpha]).reshape(1, 2)
                    alpha = torch.from_numpy(alpha).float().to(ptu.device)
                task_c_alpha.append(alpha @ task_c)  # ([1, 16]) @ ([16, 10]) --> ([1, 10])
            task_c_alpha = torch.cat(task_c_alpha)  # ([16, 10])
            task_c_alpha_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c_alpha], dim=0)  # ([16, 256, 10])
            task_c_alpha_repeat = task_c_alpha_repeat.view(-1, self.l_dim)

            r_alpha, n_o_alpha, _, _ = self.agent.c_decoder(o.view(-1, self.o_dim), a.view(-1, self.a_dim), task_c_alpha_repeat)

            task_c_alpha_GP = []
            for i in range(self.num_fake_tasks):  # 16번
                if self.use_full_interpolation:
                    alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                else:
                    alpha = np.random.uniform(0.1, 0.9)
                    alpha = np.array([alpha, 1 - alpha]).reshape(1,2)
                    alpha = torch.from_numpy(alpha).float().to(ptu.device)
                task_c_alpha_GP.append(alpha @ task_c)  # ([1, 16]) @ ([16, 10]) --> ([1, 10])
            task_c_alpha_GP = torch.cat(task_c_alpha_GP)  # ([16, 10])
            task_c_alpha_repeat_GP = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c_alpha_GP], dim=0)  # ([16, 256, 10])
            task_c_alpha_repeat_GP = task_c_alpha_repeat_GP.view(-1, self.l_dim)

            r_alpha_GP, n_o_alpha_GP, _, _ = self.agent.c_decoder(o.view(-1, self.o_dim), a.view(-1, self.a_dim), task_c_alpha_repeat_GP)

        d_real_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r.view(-1, 1), next_o.view(-1, self.o_dim)], dim=-1)
        if self.use_decoder_next_state:
            d_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, n_o_alpha], dim=-1)
        else:
            d_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, next_o.view(-1, self.o_dim)], dim=-1)

        d_real_score = self.agent.disc(d_real_sample).mean()
        d_fake_score = self.agent.disc(d_fake_sample).mean()

        # gradient penalty
        if self.use_decoder_next_state:
            d_fake_sample_GP = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha_GP, n_o_alpha], dim=-1)
        else:
            d_fake_sample_GP = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha_GP, next_o.view(-1, self.o_dim)], dim=-1)
        d_fake_sample_GP.requires_grad = True
        d_fake_score_GP = self.agent.disc(d_fake_sample_GP)
        gradients = torch.autograd.grad(outputs=d_fake_score_GP, inputs=d_fake_sample_GP,
                                        grad_outputs=torch.ones(d_fake_score_GP.size()).to(ptu.device),
                                        create_graph=True, retain_graph=True)[0]  # ([256, 57])
        gradients_norm = gradients.norm(2, 1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        gradients_norm = gradients_norm.mean()

        d_total_loss = - d_real_score + d_fake_score + self.wgan_lambda * gradient_penalty

        self.disc_optim.zero_grad()
        d_total_loss.backward()
        self.disc_optim.step()

        return d_total_loss.item(), d_real_score.item(), d_fake_score.item(), \
            gradient_penalty.item(), gradients_norm.item()

    def _do_wgan_gen3(self, indices_):

        indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
        indices_diric_inter_tasks = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)  # 3개
        tran_bsize = 128

        o, a, r, next_o, _ = self.sample_sac(indices, tran_bsize)
        ctxt_bat = self.sample_context(indices_diric_inter_tasks, which_buffer=self.offpol_ctxt_sampling_buffer)

        with torch.no_grad():
            task_c = self.agent.get_context_embedding(ctxt_bat, use_target=False)
            # task_z, _ = self.agent.z_autoencoder(task_c)
            task_c_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c], dim=0)  # ([16, 256, 10])
            task_c_repeat = task_c_repeat.view(-1, self.l_dim)

            task_c_alpha = []
            for i in range(self.num_fake_tasks):  # 16번
                if self.use_full_interpolation:
                    alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                else:
                    alpha = np.random.uniform(0.1, 0.9)
                    alpha = np.array([alpha, 1 - alpha]).reshape(1,2)
                    alpha = torch.from_numpy(alpha).float().to(ptu.device)
                task_c_alpha.append(alpha @ task_c)  # ([1, 16]) @ ([16, 10]) --> ([1, 10])
            task_c_alpha = torch.cat(task_c_alpha)  # ([16, 10])
            task_c_alpha_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c_alpha], dim=0)  # ([16, 256, 10])
            task_c_alpha_repeat = task_c_alpha_repeat.view(-1, self.l_dim)

        r_alpha, n_o_alpha, _, _ = self.agent.c_decoder(o.view(-1, self.o_dim), a.view(-1, self.a_dim), task_c_alpha_repeat)


        g_real_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r.view(-1, 1), next_o.view(-1, self.o_dim)], dim=-1)
        if self.use_decoder_next_state:
            g_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, n_o_alpha], dim=-1)
        else:
            g_fake_sample = torch.cat([o.view(-1, self.o_dim), a.view(-1, self.a_dim), r_alpha, next_o.view(-1, self.o_dim)], dim=-1)

        g_real_score = self.agent.disc(g_real_sample).detach().mean()
        g_fake_score = self.agent.disc(g_fake_sample).mean()

        g_total_loss = - g_fake_score

        self.c_decoder_optim.zero_grad()
        # self.policy_optimizer.zero_grad()
        g_total_loss.backward()
        self.c_decoder_optim.step()
        # self.policy_optimizer.step()

        w_distance = g_real_score - g_fake_score

        return g_total_loss.item(), w_distance.item(), (g_real_score).item(), (g_fake_score).item()

    """"""

    def fake_sample_representation_train(self, indices):

        """ * fake 샘플로 z 사이클 """

        # (1) 0~400 전체 버퍼에서 o,a 배치 샘플링, 컨텍스크 배치 샘플링
        tran_bsize = 256
        obs, actions, _, _, _ = self.sample_total_task_transition(len(indices), batch_size=tran_bsize)  # [16, 128, 20], ...
        ctxt_bat = self.sample_context(indices)
        obs_flat, actions_flat = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim)

        # (2) c, z 인퍼런스 및 z_alpha 생성
        task_c = self.agent.get_context_embedding(ctxt_bat, use_target=False)  # ([16, 10])
        task_z, _ = self.agent.z_autoencoder(task_c)
        task_z_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in task_z], dim=0)  # ([16, 256, 10])
        task_z_repeat = task_z_repeat.view(-1, self.l_dim)

        z_alpha = []
        for i in range(len(task_z)):  # 16번
            alpha = Dirichlet(torch.ones(1, len(task_z))).sample() * self.beta - (self.beta - 1) / len(task_z)
            z_alpha.append(alpha.to(ptu.device) @ task_z)  # ([1, 16]) @ ([16, 10]) --> ([1, 10])
        z_alpha = torch.cat(z_alpha)  # ([16, 10])
        z_alpha_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in z_alpha], dim=0)  # ([16, 256, 10])
        z_alpha_repeat = z_alpha_repeat.view(-1, self.l_dim)

        # (3) fake샘플 생성 (o, a, r_alpha) 및 z_alpha_from_fake_context
        with torch.no_grad():
            a_alpha = self.agent.decoder.dec1(obs_flat, z_alpha_repeat)
            r_alpha, next_o_alpha, _, _ = self.agent.decoder(obs_flat, a_alpha, z_alpha_repeat)
            fake_context_bat = torch.cat([obs, a_alpha.view(-1, tran_bsize, self.a_dim), r_alpha.view(-1, tran_bsize, 1)], dim=-1)

        c_alpha_from_z_alpha = self.agent.z_autoencoder.dec_fc(z_alpha)
        c_alpha_from_fake_context = self.agent.get_context_embedding(fake_context_bat, use_target=False)  # ([16, 10])

        # z_alpha_from_fake_context, _ = self.agent.z_autoencoder(c_alpha_from_fake_context)

        # (4) 사이클 로스 계산
        # cycle_loss = torch.abs(z_alpha.detach() - z_alpha_from_fake_context).sum(-1).mean()
        # cycle_loss = F.mse_loss(z_alpha, z_alpha_from_fake_context)
        cycle_loss = F.mse_loss(c_alpha_from_z_alpha.detach(), c_alpha_from_fake_context)

        """ * fake 샘플로 c bisim """

        # (1) 태스크마다 같은 o,a 배치 만들기
        # obs, actions = obs[0].repeat(len(indices), 1, 1), actions[0].repeat(len(indices), 1, 1)  # ([256, 20]) --> (16, 256, 20)
        # obs, actions = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim)
        obs_perm, actions_perm = np.random.permutation(tran_bsize), np.random.permutation(tran_bsize)
        obs, actions = obs[0][obs_perm], actions[0][actions_perm]
        obs, actions = obs.repeat(len(indices), 1, 1), actions.repeat(len(indices), 1, 1)
        obs, actions = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim)

        # (2)
        # c_alpha_from_z_alpha = self.agent.z_autoencoder.dec_fc(z_alpha)

        # (3) fake sample 생성
        with torch.no_grad():
            r_alpha, _, next_o_alpha_mean, next_o_alpha_std = self.agent.decoder(obs, actions, z_alpha_repeat)
            r, _, next_o_mean, next_o_std = self.agent.decoder(obs, actions, task_z_repeat)

        r_1, n_o_mean_1, n_o_std_1 = r_alpha.view(-1, tran_bsize, 1), next_o_alpha_mean.view(-1, tran_bsize, self.o_dim), next_o_alpha_std.view(-1, tran_bsize, self.o_dim)
        r_2, n_o_mean_2, n_o_std_2 = r.view(-1, tran_bsize, 1), next_o_mean.view(-1, tran_bsize, self.o_dim), next_o_std.view(-1, tran_bsize, self.o_dim)

        # (4) bisim_dist 및 bisim_loss 계산
        transition_dist = torch.sum((n_o_mean_2 - n_o_mean_1).pow(2), dim=-1) + \
                          torch.sum((n_o_std_2 - n_o_std_1).pow(2), dim=-1)
        transition_dist = torch.sqrt(transition_dist)  # ([16, 128])
        r_dist = torch.sqrt(torch.sum((r_2 - r_1).pow(2), dim=-1))  # ([16, 128])
        sample_dist = (r_dist + 0.9 * transition_dist).mean(dim=-1)  # ([16])

        sample_dist = 5 * sample_dist

        c_dist = torch.abs(task_c.detach() - c_alpha_from_z_alpha).sum(dim=-1)

        bisim_c_loss = (c_dist - sample_dist).pow(2).mean()

        """ 학습 """
        total_loss = cycle_loss + bisim_c_loss

        self.psi_optim.zero_grad()
        # self.z_autoencoder_optim.zero_grad()
        total_loss.backward()
        self.psi_optim.step()
        # self.z_autoencoder_optim.step()

        return cycle_loss.item(), bisim_c_loss.item()

    def fake_sample_cycle_train(self, use_full_interpolation):
        # indices = np.random.choice(self.train_tasks, 8, replace=False)
        indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
        indices_diric_inter_tasks = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)  # 3개
        tran_bsize = 128

        """ (1) off fake 샘플로 c 사이클 """
        # (1) 0~400 전체 버퍼에서 o,a 배치 샘플링, 컨텍스크 배치 샘플링
        obs, actions, _, _, _ = self.sample_sac(indices, tran_bsize)
        obs_flat, actions_flat = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim)
        ctxt_bat_off = self.sample_context(indices_diric_inter_tasks, which_buffer=self.offpol_ctxt_sampling_buffer)
        ctxt_bat_on = self.sample_context(indices_diric_inter_tasks, which_buffer='online')

        # (2) c, z 인퍼런스 및 z_alpha 생성
        task_c_off = self.agent.get_context_embedding(ctxt_bat_off, use_target=False)  # ([3, 10])
        task_c_on = self.agent.get_context_embedding(ctxt_bat_on, use_target=False)  # ([3, 10])

        c_alpha_off, c_alpha_on = [], []
        for i in range(self.num_fake_tasks):  # 5번
            if use_full_interpolation:
                alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
            else:
                alpha = np.random.uniform(0.1, 0.9)
                alpha = np.array([alpha, 1 - alpha]).reshape(1,2)
                alpha = torch.from_numpy(alpha).float().to(ptu.device)

            c_alpha_off.append(alpha @ task_c_off)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
            c_alpha_on.append(alpha @ task_c_on)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
        c_alpha_off = torch.cat(c_alpha_off)  # ([5, 10])
        c_alpha_off_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c_alpha_off], dim=0)  # ([5, 128, 10])
        c_alpha_off_repeat = c_alpha_off_repeat.view(-1, self.l_dim)
        c_alpha_on = torch.cat(c_alpha_on)  # ([5, 10])
        # c_alpha_on_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in c_alpha_on], dim=0)  # ([5, 128, 10])
        # c_alpha_on_repeat = c_alpha_on_repeat.view(-1, self.l_dim)
        
        # (3) fake샘플 생성 (o, a, r_alpha) 및 z_alpha_from_fake_context
        r_alpha_off, n_obs_alpha_off, _, _ = self.agent.c_decoder(obs_flat, actions_flat, c_alpha_off_repeat)
        #print("r_alpha_off", r_alpha_off.view(-1, tran_bsize, 1).shape)
        #print("n_obs_alpha_off", n_obs_alpha_off.view(-1, tran_bsize, self.o_dim).shape)
        if self.use_decoder_next_state:
            fake_context_bat_off = torch.cat([obs, actions, r_alpha_off.view(-1, tran_bsize, 1), n_obs_alpha_off.view(-1, tran_bsize, self.o_dim)], dim=-1)
        else:
            fake_context_bat_off = torch.cat([obs, actions, r_alpha_off.view(-1, tran_bsize, 1)], dim=-1)
        #print("fake_context_bat_off", fake_context_bat_off.shape)
        c_alpha_hat_off = self.agent.get_context_embedding(fake_context_bat_off, use_target=False)  # ([5, 10])
        #print("c_alpha_hat_off", c_alpha_hat_off.shape)
        # (4) 사이클 로스 계산
        cycle_loss_off    = F.mse_loss(c_alpha_hat_off, c_alpha_off, reduction="sum") + F.mse_loss(c_alpha_hat_off, c_alpha_on, reduction="sum")



        """ (2) on fake 샘플로 c 사이클 """
        # (1) 0~400 전체 버퍼에서 o,a 배치 샘플링, 컨텍스크 배치 샘플링
        _, (obs, actions, _, _, _) = self.sample_context(indices, which_buffer='online', b_size=tran_bsize, return_unpacked=True)
        obs_flat, actions_flat = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim)

        # (3) fake샘플 생성 (o, a, r_alpha) 및 z_alpha_from_fake_context
        r_alpha_on, n_obs_alpha_on, _, _ = self.agent.c_decoder(obs_flat, actions_flat, c_alpha_off_repeat)
        if self.use_decoder_next_state:
            fake_context_bat_on = torch.cat([obs, actions, r_alpha_on.view(-1, tran_bsize, 1), n_obs_alpha_on.view(-1, tran_bsize, self.o_dim)], dim=-1)
        else:
            fake_context_bat_on = torch.cat([obs, actions, r_alpha_on.view(-1, tran_bsize, 1)], dim=-1)
        c_alpha_hat_on = self.agent.get_context_embedding(fake_context_bat_on, use_target=False)  # ([5, 10])

        # (4) 사이클 로스 계산
        cycle_loss_on    = F.mse_loss(c_alpha_hat_on, c_alpha_on, reduction="sum")

        loss = cycle_loss_off + cycle_loss_on
        loss = self.cycle_coeff * loss





        """ 학습 """
        self.psi_optim.zero_grad()
        self.c_decoder_optim.zero_grad()
        loss.backward()
        self.psi_optim.step()
        self.c_decoder_optim.step()

        return cycle_loss_off.item(), cycle_loss_on.item()
    #

    def batch_flatten(self, sample):
        total_dim = len(sample.size())
        if total_dim == 3:
            meta_b_size, b_size, dim = sample.size()
            return sample.view(-1, dim)
        else:
            return sample

    def recover_flatten(self, sample, meta_bach_size, tr_b_size):
        total_dim = len(sample.size())
        if total_dim == 2:
            dim = sample.size(1)
            return sample.view(meta_bach_size, tr_b_size, dim)
        else:
            return sample


    def compute_bisimilarity(self, n_o1_mean, n_o1_std, r1, c1, n_o2_mean, n_o2_std, r2, c2):
        transition_dist = torch.sum((n_o1_mean - n_o2_mean).pow(2), dim=-1) + \
                          torch.sum((n_o1_std - n_o2_std).pow(2), dim=-1)
        transition_dist = torch.sqrt(transition_dist)  # ([256])
        # r_dist = torch.abs(r1 - r2)  #
        r_dist = torch.sqrt(torch.sum((r1 - r2).pow(2), dim=-1))  # ([256])
        sample_dist = (r_dist + 0.9 * transition_dist).mean().detach()  # ([])
        # sample_dist = 100 * max(sample_dist, 0)

        c_dist = torch.sum((c1 - c2).pow(2) + 1e-7, dim=-1)
        c_dist = torch.sqrt(c_dist)
        # c_dist = F.smooth_l1_loss(c1, c2, reduction='none')
        # print("c_dist:", c_dist, "  sample_dist:", sample_dist)

        return sample_dist, c_dist

    def task_info(self, indices1, indices2, ratio):

        label1 = []
        for idx in indices1:
            label1.append(np.around(self.total_tasks_dict_list[idx]['goal'], 4))
        label1 = np.array(label1)

        goal_dists1 = []
        for i in range(len(label1)):
            goal_dists1.append(np.sqrt(label1[i][0] ** 2 + label1[i][1] ** 2))
        # goal_dists1 = np.array(goal_dists1)

        label2 = []
        for idx in indices2:
            label2.append(np.around(self.total_tasks_dict_list[idx]['goal'], 4))
        label2 = np.array(label2)

        goal_dists2 = []
        for i in range(len(label2)):
            goal_dists2.append(np.sqrt(label2[i][0] ** 2 + label2[i][1] ** 2))
        # goal_dists2 = np.array(goal_dists2)

        mask = torch.ones(len(indices1))
        for i, (gd1, gd2) in enumerate(zip(goal_dists1, goal_dists2)):
            if np.abs(gd1 - gd2) > 1:  # 반지름 0~1 - 반지름 2.5~3의 차이
                mask[i] = 1 / ratio

        return mask.to(ptu.device)

    def train_c_distribution_vae(self):
        # indice = np.random.choice(self.train_tasks, 256)
        # ctxt_batch = self.sample_context(indice)  # ([16, 100, 27])
        # c_batch = self.agent.get_context_embedding(ctxt_batch, use_target=False).detach()  # ([16, 5])

        # c_batch = self.c_buffer.sample_c(num_samples=256)  # ([16, 256])

        indices = np.random.choice(self.train_tasks, 256)
        on_pol_ctxt_bat = self.sample_context(indices, which_buffer="online", b_size=128)
        c_batch = self.agent.get_context_embedding(on_pol_ctxt_bat, use_target=False).detach()  # ([256, 10]) ?

        c_recon, mu, logvar = self.agent.c_distribution_vae(c_batch)

        c_recon_loss = F.mse_loss(c_recon, c_batch, reduction='sum')
        c_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = c_recon_loss + self.c_kl_lambda * c_kl_loss

        self.c_distribution_vae_optim.zero_grad()
        loss.backward()
        self.c_distribution_vae_optim.step()

        return loss.item(), c_recon_loss.item(), c_kl_loss.item()





    def _bisim_c_train4(self, tasks_indices, training_epoch):

        num_tasks = len(tasks_indices)

        tran_bsize = 256
        obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(len(tasks_indices), batch_size=tran_bsize)  # [16, 128, 20], ...
        ctxt_bat = self.sample_context(tasks_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # [16, 128, 47]
        ctxt_bat2 = self.sample_context(tasks_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # [16, 128, 47]

        if self.sa_perm:
            obs_perm, actions_perm = np.random.permutation(tran_bsize), np.random.permutation(tran_bsize)
            obs_c, actions_c = obs_c[0][obs_perm], actions_c[0][actions_perm]
        else:
            obs_c, actions_c = obs_c[0], actions_c[0]
        obs_c, actions_c = obs_c.repeat(len(tasks_indices), 1, 1), actions_c.repeat(len(tasks_indices), 1, 1)

        task_c_1 = self.agent.get_context_embedding(ctxt_bat, use_target=False)  # ([4, 20])
        task_c_1_ = self.agent.get_context_embedding(ctxt_bat2, use_target=False)  # ([4, 20])

        task_k = self.index_to_onehot(tasks_indices)  # [16, 150]

        if self.use_z_autoencoder:
            task_z_1, task_c_1_recon = self.agent.z_autoencoder(task_c_1)
            task_z_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in task_z_1])  # ([4, 128, 20])
        else:
            task_z_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in task_c_1])  # ([4, 128, 20])
            task_k_repeat = torch.cat([k.repeat(tran_bsize, 1).unsqueeze(0) for k in task_k])  # ([4, 128, 20])

        # task_z_repeat = torch.cat([z.repeat(tran_bsize, 1).unsqueeze(0) for z in task_z_1])  # ([4, 128, 20])

        task_c_on_pol, task_c_on_pol2 = [], []
        for j in range(2):
            on_pol_ctxt_bat = self.sample_context(tasks_indices, which_buffer="online", b_size=128)  # [16,128,38]
            on_pol_ctxt_bat2 = self.sample_context(tasks_indices, which_buffer="online", b_size=128)  # [16,128,38]
            task_c_on_pol.append(self.agent.get_context_embedding(on_pol_ctxt_bat, use_target=False).unsqueeze(0))  # [16,128,38] --> [1,16,10] -->
            task_c_on_pol2.append(self.agent.get_context_embedding(on_pol_ctxt_bat2, use_target=False).unsqueeze(0))
        task_c_on_pol  = torch.cat(task_c_on_pol, dim=0).mean(dim=0)  # [5,16,10] --> [16, 10]
        task_c_on_pol2 = torch.cat(task_c_on_pol2, dim=0).mean(dim=0)

        obs_c, actions_c = obs_c.view(-1, self.o_dim), actions_c.view(-1, self.a_dim)
        task_z_repeat = task_z_repeat.view(-1, self.l_dim)
        task_k_repeat = task_k_repeat.view(-1, len(self.train_tasks))

        with torch.no_grad():
            r_pred, _, next_o_pred_mean, next_o_pred_std = self.agent.k_decoder(obs_c, actions_c, task_k_repeat)
            r_pred = r_pred * self.r_dist_coeff # * self.bisim_r_coef

        # r_1, n_o_mean_1, n_o_std_1 = r_pred.view(-1, tran_bsize, 1), next_o_pred_mean.view(-1, tran_bsize, self.o_dim), next_o_pred_std.view(-1, tran_bsize, self.o_dim)
        # perm = np.random.permutation(len(r_1))
        # r_2, n_o_mean_2, n_o_std_2 = r_1[perm], n_o_mean_1[perm], n_o_std_1[perm]
        # task_c_2 = task_c_1[perm]
        perm = np.random.permutation(num_tasks)
        r_1 = r_pred.view(-1, tran_bsize, 1)
        r_2 = r_1[perm]
        if self.use_next_state_bisim:
            n_o_mean_1, n_o_std_1 = next_o_pred_mean.view(-1, tran_bsize, self.o_dim), next_o_pred_std.view(-1, tran_bsize, self.o_dim)
            n_o_mean_2, n_o_std_2 = n_o_mean_1[perm], n_o_std_1[perm]
        task_c_2 = task_c_1[perm]
        tasks_indices = np.array(tasks_indices)

        # onpol_c에 대해서도 bisim
        # task_c_on_pol_2 = task_c_on_pol[perm]



        tasks_indices1 = tasks_indices
        tasks_indices2 = tasks_indices[perm]

        # transition_dist = ((n_o_mean_2 - n_o_mean_1).pow(2) + 1e-7).sum(dim=-1) + \
        #                   ((n_o_std_2 - n_o_std_1).pow(2) + 1e-7).sum(dim=-1)
        # transition_dist = torch.sqrt(transition_dist + 1e-7)  # ([16, 128])
        # r_dist = ((r_2 - r_1).pow(2) + 1e-7).sum(dim=-1)  # ([16, 128])
        # r_dist = torch.sqrt(r_dist + 1e-7)
        # sample_dist = (r_dist + 0.9 * transition_dist).mean(dim=-1)  # ([16])
        r_dist = ((r_2 - r_1).pow(2) + 1e-7).sum(dim=-1)  # ([16, 128])
        r_dist = torch.sqrt(r_dist + 1e-7)
        if self.use_next_state_bisim:
            transition_dist = ((n_o_mean_2 - n_o_mean_1).pow(2) + 1e-7).sum(dim=-1) + \
                              ((n_o_std_2 - n_o_std_1).pow(2) + 1e-7).sum(dim=-1)
            transition_dist = torch.sqrt(transition_dist + 1e-7)  # ([16, 128])
            sample_dist = (r_dist + 0.9 * transition_dist).mean(dim=-1)
        else:
            sample_dist = r_dist.mean(dim=-1)
        sample_dist = self.sample_dist_coeff * sample_dist

        c_dist = torch.abs(task_c_2 - task_c_1).sum(dim=-1)
        # c_dist_on = torch.abs(task_c_on_pol_2 - task_c_on_pol).sum(dim=-1)

        """템퍼리처조절"""
        if self.use_decrease_mask:
            mask = self.task_info(tasks_indices1, tasks_indices2, ratio=self.decrease_rate)  # --> List
            sample_dist = sample_dist * mask
        """템퍼리처조절"""

        # bisim_c_loss = (c_dist - sample_dist).pow(2).mean() + 10 * F.mse_loss(task_c_1, task_c_1_)
        # bisim_c_loss = (c_dist - sample_dist).pow(2).mean()
        bisim_c_loss = F.mse_loss(c_dist, sample_dist, reduction="mean")   # + F.mse_loss(c_dist_on, sample_dist, reduction="mean")



        # same_task_c_loss = torch.sum((task_c_1 - task_c_1_).pow(2))  # F.mse_loss(task_c_1, task_c_1_)
        # on_pol_c_loss = torch.sum((task_c_1 - task_c_on_pol).pow(2))  # F.mse_loss(task_c_1, task_c_on_pol)

        on_pol_c_loss = F.mse_loss(task_c_1, task_c_on_pol, reduction="mean")

        # same_task_c_loss = 0.1 * F.mse_loss(task_c_1, task_c_1_) + 10 * F.mse_loss(task_c_on_pol, task_c_on_pol2)
        # same_task_c_loss = 10 * F.mse_loss(task_c_on_pol, task_c_on_pol2)
        same_task_c_loss = F.mse_loss(task_c_1, task_c_1_, reduction="mean") + \
                           F.mse_loss(task_c_on_pol, task_c_on_pol2, reduction='mean')

        # same_task_c_loss = torch.tensor([0]).to(ptu.device)
        # on_pol_c_loss = torch.tensor([0]).to(ptu.device)

        c_cycle_loss, bisim_c_alpha_loss = torch.tensor([0]).to(ptu.device), torch.tensor([0]).to(ptu.device)

        return bisim_c_loss, same_task_c_loss, on_pol_c_loss, c_cycle_loss, bisim_c_alpha_loss


    def compute_sample_dist(self, n_o_mean1, n_o_std1, r1, n_o_mean2, n_o_std2, r2):
        transition_dist = torch.sum((n_o_mean2 - n_o_mean1).pow(2), dim=-1) + \
                          torch.sum((n_o_std2 - n_o_std1).pow(2), dim=-1)
        transition_dist = torch.sqrt(transition_dist)  # ([128])
        r_dist = torch.sqrt(torch.sum((r2 - r1).pow(2), dim=-1))  # ([128])
        sample_dist = (r_dist + 0.9 * transition_dist).mean()
        return sample_dist

    def index_to_onehot(self, indices):
        # print("indices", indices)
        onehot = torch.zeros(len(indices), len(self.train_tasks))  # [16, 100]
        for i in range(len(indices)):
            onehot[i, indices[i]] = 1
        return onehot.to(ptu.device)

    def pretrain3(self, indices, train_step, training_epoch, train_c_distribution_vae=True):

        """ (0) 학습할 미니 배치 준비 """
        tran_batch_size = 256
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices, tran_batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...
        ctxt_batch = self.sample_context(indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([16, 100, 27])

        # print("obs", obs.shape)  ([16, 256, 17])
        # print("actions", actions.shape)  ([16, 256, 6])
        # print("rewards", rewards.shape)  ([16, 256, 1])
        # print("next_obs", next_obs.shape)  ([16, 256, 17])


        """ (1) Reconstruction 학습 : psi + decoder 학습됨 """
        # self.agent.clear_z(num_tasks=len(indices))
        # with torch.no_grad():

        # task_z = self.agent.get_context_embedding(ctxt_batch, use_target=False)  # ([16, 5])
        task_c = self.agent.get_context_embedding(ctxt_batch, use_target=False)  # ([16, 5])
        task_k = self.index_to_onehot(indices)  # [16, 150]
        
        task_z_repeat = [z.repeat(tran_batch_size, 1).unsqueeze(0) for z in task_c]
        task_k_repeat = [k.repeat(tran_batch_size, 1).unsqueeze(0) for k in task_k]

        on_pol_ctxt_bat = self.sample_context(indices, which_buffer="online", b_size=128)
        task_c_on_pol = self.agent.get_context_embedding(on_pol_ctxt_bat, use_target=False)  # ([4, 20])

        ###############################3
        # indices_diric_inter_tasks = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)  # 3개
        # ctxt_bat_on_4_interpolation = self.sample_context(indices_diric_inter_tasks, which_buffer='online')
        # task_c_on_4_interpolation = self.agent.get_context_embedding(ctxt_bat_on_4_interpolation, use_target=False)  # ([3, 10])
        # c_alpha_on = []
        # for i in range(self.meta_batch):  # 5번
        #     alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - ( self.beta - 1) / self.num_dirichlet_tasks
        #     c_alpha_on.append(alpha @ task_c_on_4_interpolation)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
        # c_alpha_on = torch.cat(c_alpha_on)  # ([5, 10])
        #################################

        self.c_buffer.add_c(task_c_on_pol)  # c_vae는 policy가 실제로 받는 c를 학습해야 함 --> onpol_c가 버퍼에 저장돼야함
        # self.c_buffer.add_c(c_alpha_on)
        if train_c_distribution_vae and (train_step % self.c_distri_vae_train_freq == 0):
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = self.train_c_distribution_vae()
        else:
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = None, None, None

        task_z_repeat = torch.cat(task_z_repeat, dim=0)  # [16, 256, 5])
        task_k_repeat = torch.cat(task_k_repeat, dim=0)  # [16, 256, 150])

        obs_flat, actions_flat, rewards_flat, next_obs_flat = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim), rewards.view(-1, 1), next_obs.view(-1, self.o_dim)  # ([4096, 20]), ...
        task_z_repeat = task_z_repeat.view(-1, self.l_dim)  # ([4096, 5])
        task_k_repeat = task_k_repeat.view(-1, len(self.train_tasks))  # ([4096, 5])

        """ reconstruction loss 계산 """
        rewards_pred_k, next_obs_pred_k, _, _ = self.agent.k_decoder(obs_flat, actions_flat, task_k_repeat)  # ([4096, 20])
        rewards_pred_c, next_obs_pred_c, _, _ = self.agent.c_decoder(obs_flat, actions_flat, task_z_repeat)  # ([4096, 20])

        reward_recon_loss_k = F.mse_loss(rewards_flat, rewards_pred_k, reduction="mean")
        reward_recon_loss_c = F.mse_loss(rewards_flat, rewards_pred_c, reduction="mean")
        if self.use_decoder_next_state:
            next_obs_recon_loss_k = F.mse_loss(next_obs_flat, next_obs_pred_k)
            next_obs_recon_loss_c = F.mse_loss(next_obs_flat, next_obs_pred_c)
        else:
            next_obs_recon_loss_k = torch.Tensor([0]).mean().to(ptu.device)
            next_obs_recon_loss_c = torch.Tensor([0]).mean().to(ptu.device)

        reward_recon_loss = reward_recon_loss_k + reward_recon_loss_c
        next_obs_recon_loss = next_obs_recon_loss_k + next_obs_recon_loss_c
        # recon_loss = 200 * (reward_recon_loss + next_obs_recon_loss)
        """ reconstruction loss 계산 """

        # self.psi_optim.zero_grad()
        # self.decoder_optim.zero_grad()
        # total_recon_loss.backward()
        # self.psi_optim.step()
        # self.decoder_optim.step()

        """ (2) bisimulation 학습 : 태스크 레프레젠테이션 c의 거리가 태스크 샘플들의 bisim 거리가 되도록 학습 : psi만 학습 """
        # bisim_z_loss = self._bisim_c_train3(indices)  # bisim_bias 방식
        bisim_c_loss, same_task_c_loss, on_pol_c_loss, c_cycle_loss, bisim_c_alpha_loss = self._bisim_c_train4(indices, training_epoch)  # bisim_no_bias 방식
        # bisim_c_loss, same_task_c_loss, on_pol_c_loss, c_cycle_loss, bisim_c_alpha_loss = self._bisim_c_train5(indices)  # bisim_no_bias 방식 + fakesample
        # self.psi_optim.zero_grad()
        # loss.backward()
        # self.psi_optim.step()

        total_loss = self.recon_coeff * reward_recon_loss + \
                     self.recon_coeff * next_obs_recon_loss + \
                     self.bisim_coeff * bisim_c_loss + \
                     self.same_c_coeff * same_task_c_loss + \
                     self.onpol_c_coeff * on_pol_c_loss       
                    # bisim_coeff 원래 50
                    # recon_loss원래 100 / same_c_coeff 200 / onpol_c_coeff 200

        self.psi_optim.zero_grad()
        self.k_decoder_optim.zero_grad()
        self.c_decoder_optim.zero_grad()
        total_loss.backward()
        self.psi_optim.step()
        self.k_decoder_optim.step()
        self.c_decoder_optim.step()


        return total_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item(), \
            reward_recon_loss_k.item(), reward_recon_loss_c.item(), \
            next_obs_recon_loss_k.item(), next_obs_recon_loss_c.item(), \
            bisim_c_loss.item(), same_task_c_loss.item(), on_pol_c_loss.item(), \
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss  


    def z_cluster_train(self, decrease_rate=2):
        for params in self.agent.z_autoencoder.parameters():
            params.requires_grad = True

        # c_batch = self.c_buffer.sample_c(self.c_batch_num)  # nd array
        # print("c_batch", c_batch.shape)

        # bandwidth = estimate_bandwidth(c_batch.cpu().numpy())
        # meanshift_model = MeanShift(bandwidth=bandwidth)

        # cluster_labels = meanshift_model.fit_predict(c_batch.cpu().numpy())
        # cluster_unique_labels = np.unique(cluster_labels)
        # print("cluster_unique_labels", cluster_unique_labels)

        c_batch = []
        for _ in range(20):
            ctxt = self.sample_context(self.train_tasks)  # ([150, 100, 27])
            c_batch.append(self.agent.get_context_embedding(ctxt, use_target=False).detach())  # ([150, 10])
        c_batch = torch.cat(c_batch)  # ([3000, 10])

        kmeans_model = KMeans(n_clusters=2)
        kmeans_model.fit(c_batch.cpu().numpy())
        cluster_labels = kmeans_model.labels_
        cluster_unique_labels = np.unique(cluster_labels)

        # c_batch = torch.from_numpy(c_batch).to(torch.float).to(ptu.device)  # ([500, 5])

        z, c_recon = self.agent.z_autoencoder(c_batch)

        # if len(cluster_unique_labels) == 1 or len(cluster_unique_labels) > 2:
        #     recon_loss = F.mse_loss(c_batch, c_recon)  # + F.mse_loss(c_alpha_batch, c_alpha_recon)
        #     return recon_loss.item(), None, None, None  #

        z_groups, c_groups = [], []
        for c_label in cluster_unique_labels:
            z_groups.append(z[cluster_labels == c_label])
            c_groups.append(c_batch[cluster_labels == c_label])

        """그룹간 pairwise dist 줄이기"""
        sample_length = min([len(z_groups[i]) for i in range(len(z_groups))])
        print("sample_length", sample_length)  # 그룹내의 샘플 수가 다를 수 있으므로 더 작은 샘플 수를 가진 그룹을 기준으로 함.
        c0_bat = c_groups[0][:sample_length, :]  # ([248, 5])
        c1_bat = c_groups[1][:sample_length, :]
        z0_bat = z_groups[0][:sample_length, :]
        z1_bat = z_groups[1][:sample_length, :]

        c_dist_bat = torch.sqrt((c0_bat - c1_bat).pow(2).sum(-1))
        z_dist_bat = torch.sqrt((z0_bat - z1_bat).pow(2).sum(-1))
        # n = 2
        groups_dist_loss = F.mse_loss(z_dist_bat, c_dist_bat / decrease_rate)

        """그룹내 pairwise dist 유지"""
        in_group_dist_loss = []
        for i in range(len(z_groups)):
            z_group_i, c_group_i = z_groups[i], c_groups[i]  # ([214, 5]), ([214, 5])
            perm = np.random.permutation(z_group_i.size(0))
            z_group_i_perm, c_group_i_perm = z_group_i[perm], c_group_i[perm]

            z_dist = ((z_group_i - z_group_i_perm).pow(2) + 1e-7).sum(dim=-1)
            z_dist = torch.sqrt(z_dist + 1e-7)
            c_dist = ((c_group_i - c_group_i_perm).pow(2) + 1e-7).sum(dim=-1)
            c_dist = torch.sqrt(c_dist + 1e-7)

            in_group_dist_loss.append(F.mse_loss(c_dist, z_dist))

        in_group_dist_loss = sum(in_group_dist_loss) / len(in_group_dist_loss)

        """ reconstruction """
        recon_loss = F.mse_loss(c_batch, c_recon)  # + F.mse_loss(c_alpha_batch, c_alpha_recon)

        """ 인터폴레이션 """
        n = 16
        z_alpha_batch = []
        for i in range(50):
            z_batch = z[np.random.permutation(200)][: n]  # ([10, l_dim])
            alpha = Dirichlet(torch.ones(1, n)).sample() * self.beta - (self.beta - 1) / n  # ([1, 10])
            z_alpha_batch.append(alpha.to(ptu.device) @ z_batch)  # ([1, 10]) @ ([10, l_dim]) --> ([1, l_dim])
        z_alpha_batch = torch.cat(z_alpha_batch)  # ([50, l_dim])
        c_alpha_batch = self.agent.z_autoencoder.dec_fc(z_alpha_batch)
        z_alpha_batch_hat, _ = self.agent.z_autoencoder(c_alpha_batch)
        interpolation_recon_loss = F.mse_loss(z_alpha_batch.detach(), z_alpha_batch_hat)

        loss = groups_dist_loss + in_group_dist_loss + recon_loss + interpolation_recon_loss

        self.z_autoencoder_optim.zero_grad()
        loss.backward()
        self.z_autoencoder_optim.step()
        # groups_dist_loss + in_group_dist_loss + recon_loss

        for params in self.agent.z_autoencoder.parameters():
            params.requires_grad = False

        return recon_loss.item(), groups_dist_loss.item(), in_group_dist_loss.item(), interpolation_recon_loss.item()

    def _do_training(self, indices, current_step):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)  # ([16, 128, 27])

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop, so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]  # ([16, 128, 27])

            # self._take_step(indices, context)  #
            # sac_loss = self._take_step_interpolation(indices, context, use_inter_samples=use_inter_samples, inter_update_coeff=inter_update_coeff)
            # sac_loss = self._take_step_contrastive_2(indices, context)
            # sac_loss = self._take_step_contrastive(indices, context)

            sac_loss = self._take_step_bisim(indices, current_step)  # sac만
            # sac_loss = self._take_step_bisim2(indices, current_step)  # recon+bisim+wgan 합친거

            # stop backprop
            self.agent.detach_z()

        return sac_loss

    def _min_q(self, obs, actions, task_z):
        if self.use_q_contrastive:
            _, q1 = self.qf1(obs, actions, task_z)
            _, q2 = self.qf2(obs, actions, task_z)
        else:
            q1 = self.qf1(obs, actions, task_z)
            q2 = self.qf2(obs, actions, task_z)
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step_bisim(self, indices, current_step):


        losses = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        total_recon_loss, kl_loss, kl_div, reward_recon_loss, next_obs_recon_loss, actions_recon_loss, \
            bisim_z_loss, same_task_c_loss, on_pol_c_loss, _, _ = losses

        tran_batch_size = 256
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices, tran_batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...

        ctxt_batch = self.sample_context(indices, which_buffer="online")  # ([16, 100, 27])
        task_c_ = self.agent.get_context_embedding(ctxt_batch, use_target=False).detach()

        if self.use_inter_samples:
            rl_indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
            inter_indices = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False )
            inter_tran_batch_size = self.fakesample_rl_tran_batch_size
            inter_num_tasks = len(rl_indices)

            obs_inter, actions_inter, _, next_obs_, terms_inter = self.sample_sac(rl_indices, inter_tran_batch_size)  # ([5, 256, 20]), ([5, 256, 6]), ...
            # obs_on, actions_on, _, next_obs_on, terms_on = self.sample_context(rl_indices, which_buffer='online', b_size=tran_bsize, return_unpacked=True)
            obs_inter = obs_inter.view(inter_num_tasks * inter_tran_batch_size, -1)
            actions_inter = actions_inter.view(inter_num_tasks * inter_tran_batch_size, -1)
            if not self.use_decoder_next_state:
                next_obs_inter = next_obs_.view(inter_num_tasks * inter_tran_batch_size, -1)
            terms_inter = terms_inter.view(inter_num_tasks * inter_tran_batch_size, -1)

            ctxt_batch_off = self.sample_context(inter_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([16, 100, 27])
            ctxt_batch_on = self.sample_context(inter_indices, which_buffer="online")  # ([16, 100, 27])
            task_c_off = self.agent.get_context_embedding(ctxt_batch_off, use_target=False).detach()
            task_c_on = self.agent.get_context_embedding(ctxt_batch_on, use_target=False).detach()

            c_alpha_diric_off, c_alpha_diric_on = [], []
            for i in range(inter_num_tasks):
                alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                c_alpha_diric_off.append(alpha @ task_c_off)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
                c_alpha_diric_on.append(alpha @ task_c_on)
            
            c_alpha_diric_off = torch.cat(c_alpha_diric_off)  # ([5, 10])
            c_alpha_diric_off_repeat = torch.cat([c.repeat(inter_tran_batch_size, 1).unsqueeze(0) for c in c_alpha_diric_off], dim=0)
            c_alpha_diric_off_repeat = c_alpha_diric_off_repeat.view(-1, self.l_dim)

            c_alpha_diric_on = torch.cat(c_alpha_diric_on)  # ([5, 10])
            c_alpha_diric_on_repeat = torch.cat([c.repeat(inter_tran_batch_size, 1).unsqueeze(0) for c in c_alpha_diric_on], dim=0)
            c_alpha_diric_on_repeat = c_alpha_diric_on_repeat.view(-1, self.l_dim)

            rewards_inter, n_obs_inter, _, _ = self.agent.c_decoder(obs_inter, actions_inter, c_alpha_diric_off_repeat)
            if self.use_decoder_next_state:
                next_obs_inter = n_obs_inter
            rewards_inter = rewards_inter.detach() * self.reward_scale


        # t, b, _ = obs.size()  # t:16,  b:256
        obs = obs.view(num_tasks * tran_batch_size, -1)
        actions = actions.view(num_tasks * tran_batch_size, -1)
        next_obs = next_obs.view(num_tasks * tran_batch_size, -1)
        task_c = [c.repeat(tran_batch_size, 1) for c in task_c_]
        task_c = torch.cat(task_c, dim=0)  # ([2048, 5])

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_c.detach()], dim=1)
        policy_outputs = self.agent.policy(in_, reparameterize=True, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        if self.use_inter_samples:
            in_inter = torch.cat([obs_inter, c_alpha_diric_on_repeat], dim=1)
            policy_outputs_inter = self.agent.policy(in_inter, reparameterize=True, return_log_prob=True)
            new_actions_inter, policy_mean_inter, policy_log_std_inter, log_pi_inter = policy_outputs_inter[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.use_q_contrastive:
            _, q1_pred = self.qf1(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            _, q2_pred = self.qf2(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        else:
            q1_pred = self.qf1(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            q2_pred = self.qf2(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        v_pred = self.vf(obs, task_c.detach())  # ([4096, 20]), ([4096, 5])  -->  ([4096, 1])
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_c)
        if self.use_inter_samples:
            q1_pred_inter = self.qf1(obs_inter, actions_inter, c_alpha_diric_on_repeat)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            q2_pred_inter = self.qf2(obs_inter, actions_inter, c_alpha_diric_on_repeat)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            v_pred_inter = self.vf(obs_inter, c_alpha_diric_on_repeat)
            with torch.no_grad():
                target_v_values_inter = self.target_vf(next_obs_inter, c_alpha_diric_on_repeat)


        # qf and encoder update (note encoder does not get grads from policy or vf)
        rewards_flat = rewards.view(tran_batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(tran_batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        if self.use_inter_samples:
            q_target_inter = rewards_inter + (1. - terms_inter) * self.discount * target_v_values_inter
            qf_loss_inter = torch.mean((q1_pred_inter - q_target_inter) ** 2) + torch.mean((q2_pred_inter - q_target_inter) ** 2)
            # qf_loss = self.inter_update_coeff * qf_loss_inter + qf_loss
        else:
            qf_loss_inter = torch.zeros_like(qf_loss)

        qf_loss_total = qf_loss + self.inter_update_coeff * qf_loss_inter

        # qf_loss.backward()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss_total.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()



        # min_q_new_actions = self._min_q(obs, new_actions, task_z)
        min_q_new_actions = self._min_q(obs, new_actions, task_c.detach())
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        if self.use_inter_samples:
            min_q_new_actions_inter = self._min_q(obs_inter, new_actions_inter, c_alpha_diric_on_repeat)
            v_target_inter = min_q_new_actions_inter - log_pi_inter
            vf_loss_inter = self.vf_criterion(v_pred_inter, v_target_inter.detach())
            # vf_loss = vf_loss + self.inter_update_coeff * vf_loss_inter
        else:
            vf_loss_inter = torch.zeros_like(vf_loss)

        vf_loss_total = vf_loss + self.inter_update_coeff * vf_loss_inter

        # vf_loss.backward()
        self.vf_optimizer.zero_grad()
        vf_loss_total.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        policy_loss = (log_pi - log_policy_target).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        if self.use_inter_samples:
            log_policy_target_inter = min_q_new_actions_inter
            policy_loss_inter = (log_pi_inter - log_policy_target_inter).mean()

            mean_reg_loss_inter = self.policy_mean_reg_weight * (policy_mean_inter ** 2).mean()
            std_reg_loss_inter = self.policy_std_reg_weight * (policy_log_std_inter ** 2).mean()
            pre_tanh_value_inter = policy_outputs_inter[-1]
            pre_activation_reg_loss_inter = self.policy_pre_activation_weight * (
                (pre_tanh_value_inter ** 2).sum(dim=1).mean()
            )
            policy_reg_loss_inter = mean_reg_loss_inter + std_reg_loss_inter + pre_activation_reg_loss_inter
            policy_loss_inter = policy_loss_inter + policy_reg_loss_inter
            # policy_loss = policy_loss + self.inter_update_coeff * policy_loss_inter
        else:
            policy_loss_inter = torch.zeros_like(policy_loss)

        policy_loss_total = policy_loss + self.inter_update_coeff * policy_loss_inter

        # policy_loss.backward()
        self.policy_optimizer.zero_grad()
        policy_loss_total.backward()
        self.policy_optimizer.step()




        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                # self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                # self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

        if not self.use_inter_samples:
            qf_loss_inter, vf_loss_inter, policy_loss_inter = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        return qf_loss.item(), vf_loss.item(), policy_loss.item(), \
            qf_loss_inter.item(), vf_loss_inter.item(), policy_loss_inter.item(), \
            qf_loss_total.item(), vf_loss_total.item(), policy_loss_total.item(), \
            total_recon_loss, kl_loss, kl_div, reward_recon_loss, next_obs_recon_loss, actions_recon_loss, bisim_z_loss
        # kl_loss.item(), bisim_z_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item()

    def _take_step_bisim2(self, indices, current_step):
        """ (0) 학습할 미니 배치 준비 """
        tran_batch_size = 256
        tran_batch = self.sample_sac(indices, tran_batch_size)  # ([2, 256, 113]), ...
        obs, actions, rewards, next_obs, terms = tran_batch  # ([2, 256, 113]), ...
        ctxt_batch = self.sample_context(indices, which_buffer="both")  # ([2, 128, 122])
        ctxt_batch2 = self.sample_context(indices, which_buffer="both")  # ([2, 128, 122])
        task_z = self.agent.get_context_embedding(ctxt_batch, use_target=False)  # ([2, 5])
        task_z2 = self.agent.get_context_embedding(ctxt_batch2, use_target=False)  # ([2, 5])

        same_task_loss = F.mse_loss(task_z, task_z2)

        # if current_step % 4 == 0:
        actions_recon_loss, reward_recon_loss, next_obs_recon_loss = self.compute_recon_loss(tran_batch, task_z,
                                                                                             tran_batch_size)
        bisim_c_loss = self.compute_bisim_loss(ctxt_batch, task_z)
        # d_real_score, d_fake_score, gradient_penalty, gradients_norm = self.compute_wgan_disc_loss(tran_batch, task_z, tran_batch_size)

        # if current_step % self.gen_freq == 0:
        #     g_real_score, g_fake_score, w_dist = self.compute_wgan_gen_loss(tran_batch, task_z, tran_batch_size)

        num_tasks = len(indices)

        #
        if self.use_inter_samples:
            o1, a1, r1, next_o1 = obs[:8], actions[:8], rewards[:8], next_obs[:8]  # ([8, 128, 20]), ([8, 128, 6]), ...
            o2, a2, r2, next_o2 = obs[8:], actions[8:], rewards[8:], next_obs[8:]  # ([8, 128, 20]), ...
            t1_ctxt_bat = ctxt_bat_query[:8]
            t2_ctxt_bat = ctxt_bat_query[8:]
            c1, _, _ = self.agent.infer_posterior(t1_ctxt_bat, which_enc="psi1")
            c2, _, _ = self.agent.infer_posterior(t2_ctxt_bat, which_enc="psi1")

            with torch.no_grad():
                c1 = [c.repeat(tran_batch_size, 1).unsqueeze(0) for c in c1]
                c1 = torch.cat(c1, dim=0)  # [8, 128, 5])
                c2 = [c.repeat(tran_batch_size, 1).unsqueeze(0) for c in c2]
                c2 = torch.cat(c2, dim=0)  # [8, 128, 5])

                alpha = np.random.uniform(0, 1, 8)
                c_alpha = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
                for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                    interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
                    c_alpha.append(interpolation.unsqueeze(0))
                c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 128, 5])

                s1 = self.agent.decoder.state_embed(o1)
                s2 = self.agent.decoder.state_embed(o2)
                o_1alpha, a_1alpha = self.agent.decoder.dec1(c_alpha, s1)  # ([8, 256, 20])
                o_2alpha, a_2alpha = self.agent.decoder.dec1(c_alpha, s2)  # ([8, 256, 20])
                r_1alpha, next_o_1alpha, _, _, _ = self.agent.decoder.dec2(c_alpha, s1, a_1alpha)
                r_2alpha, next_o_2alpha, _, _, _ = self.agent.decoder.dec2(c_alpha, s2, a_2alpha)
                o_inter = torch.cat([o_1alpha, o_2alpha], dim=0)
                a_inter = torch.cat([a_1alpha, a_2alpha], dim=0)
                r_inter = torch.cat([r_1alpha, r_2alpha], dim=0)
                o_next_inter = torch.cat([next_o_1alpha, next_o_2alpha], dim=0)
                terms_inter = terms

            inter_sample_batch = torch.cat([o_inter, a_inter, r_inter], dim=-1)  # ([16, 128, 27])
            z_inter = self.agent.infer_posterior(inter_sample_batch, which_enc='psi2')  # ([16, 5])
            t_, b_, _ = o_inter.size()
            z_inter = [z.repeat(b_, 1) for z in z_inter]
            z_inter = torch.cat(z_inter, dim=0)  # ([2048, 5])

            kl_div_sum_inter = self.agent.compute_kl_div()
            kl_loss_inter = 0.02 * kl_div_sum_inter
            kl_loss = kl_loss + kl_loss_inter

        task_z = task_z.detach()
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)  # ([2048, 5])
        if self.use_inter_samples:
            o_inter = o_inter.view(t * b, -1)  # ([2048, 20])
            a_inter = a_inter.view(t * b, -1)  # ([2048, 6])
            o_next_inter = o_next_inter.view(t * b, -1)
            z_inter = z_inter.view(-1, 5)  # ([2048, 5])

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.agent.policy(in_, reparameterize=True, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        if self.use_inter_samples:
            in_inter_ = torch.cat([o_inter, z_inter.detach()], dim=1)
            policy_outputs_inter = self.agent.policy(in_inter_, reparameterize=True, return_log_prob=True)
            new_actions_inter, policy_mean_inter, policy_log_std_inter, log_pi_inter = policy_outputs_inter[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.use_q_contrastive:
            _, q1_pred = self.qf1(obs, actions, task_z)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            _, q2_pred = self.qf2(obs, actions, task_z)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        else:
            q1_pred = self.qf1(obs, actions, task_z)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            q2_pred = self.qf2(obs, actions, task_z)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        v_pred = self.vf(obs, task_z.detach())  # ([4096, 20]), ([4096, 5])  -->  ([4096, 1])
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)
        if self.use_inter_samples:
            q1_pred_inter = self.qf1(o_inter, a_inter, z_inter)
            q2_pred_inter = self.qf2(o_inter, a_inter, z_inter)
            v_pred_inter = self.vf(o_inter, z_inter.detach())
            with torch.no_grad():
                target_v_values_inter = self.target_vf(o_next_inter, z_inter)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        rewards_flat = rewards.view(tran_batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(tran_batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        if self.use_inter_samples:
            rewards_flat_inter = rewards.view(tran_batch_size * num_tasks, -1)
            rewards_flat_inter = rewards_flat_inter * self.reward_scale
            terms_flat_inter = terms_inter.view(tran_batch_size * num_tasks, -1)
            q_target_inter = rewards_flat_inter + (1. - terms_flat_inter) * self.discount * target_v_values_inter
            qf_loss_inter = torch.mean((q1_pred_inter - q_target_inter) ** 2) + torch.mean(
                (q2_pred_inter - q_target_inter) ** 2)

            qf_loss_norm = float(torch.sqrt(qf_loss ** 2).detach().cpu().numpy())
            qf_loss_inter_norm = float(torch.sqrt(qf_loss_inter ** 2).detach().cpu().numpy())
            qf_loss_inter = qf_loss_inter * qf_loss_norm / qf_loss_inter_norm

            qf_loss = qf_loss + self.inter_update_coeff * qf_loss_inter

        # self.psi_optim.zero_grad()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # compute min Q on the new actions
        # min_q_new_actions = self._min_q(obs, new_actions, task_z)
        min_q_new_actions = self._min_q(obs, new_actions, task_z.detach())
        if self.use_inter_samples:
            min_q_new_actions_inter = self._min_q(o_inter, new_actions_inter, z_inter.detach())

        # vf update fd sd
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        if self.use_inter_samples:
            v_target_inter = min_q_new_actions_inter - log_pi_inter
            vf_loss_inter = self.vf_criterion(v_pred_inter, v_target_inter.detach())

            vf_loss_norm = float(torch.sqrt(vf_loss ** 2).detach().cpu().numpy())
            vf_loss_inter_norm = float(torch.sqrt(vf_loss_inter ** 2).detach().cpu().numpy())
            vf_loss_inter = vf_loss_inter * vf_loss_norm / vf_loss_inter_norm

            vf_loss = vf_loss + self.inter_update_coeff * vf_loss_inter

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        policy_loss = (log_pi - log_policy_target).mean()
        if self.use_inter_samples:
            log_policy_target_inter = min_q_new_actions_inter
            policy_loss_inter = (log_pi_inter - log_policy_target_inter).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss
        if self.use_inter_samples:
            mean_reg_loss_inter = self.policy_mean_reg_weight * (policy_mean_inter ** 2).mean()
            std_reg_loss_inter = self.policy_std_reg_weight * (policy_log_std_inter ** 2).mean()
            pre_tanh_value_inter = policy_outputs_inter[-1]
            pre_activation_reg_loss_inter = self.policy_pre_activation_weight * (
                (pre_tanh_value_inter ** 2).sum(dim=1).mean()
            )
            policy_reg_loss_inter = mean_reg_loss_inter + std_reg_loss_inter + pre_activation_reg_loss_inter
            policy_loss_inter = policy_loss_inter + policy_reg_loss_inter

            policy_loss_norm = float(torch.sqrt(policy_loss ** 2).detach().cpu().numpy())
            policy_loss_inter_norm = float(torch.sqrt(policy_loss_inter ** 2).detach().cpu().numpy())
            policy_loss_inter = policy_loss_inter * policy_loss_norm / policy_loss_inter_norm

            policy_loss = policy_loss_inter + self.inter_update_coeff * policy_loss_inter

        # self.disc_optim.zero_grad()
        self.decoder_optim.zero_grad()
        self.psi_optim.zero_grad()
        same_task_loss.backward(retain_graph=True)  # --> psi
        # if current_step % 4 == 0:
        recon_loss = actions_recon_loss + reward_recon_loss + next_obs_recon_loss
        recon_loss.backward(retain_graph=True)  # --> decoder + psi
        bisim_c_loss.backward()  # --> psi

        # disc_loss = - d_real_score + d_fake_score + self.wgan_lambda * gradient_penalty
        # disc_loss.backward()  # --> disc

        # if current_step % self.gen_freq == 0:
        #     gen_loss = -g_fake_score
        #     gen_loss.backward(retain_graph=True)  # --> decoder + policy

        # self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # self.policy_optimizer.step()

        self.psi_optim.step()
        # self.disc_optim.step()
        self.decoder_optim.step()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                # self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                # self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

        if not self.use_inter_samples:
            qf_loss_inter, vf_loss_inter, policy_loss_inter = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        if current_step % self.gen_freq != 0:
            gen_loss, g_real_score, g_fake_score, w_dist = None, None, None, None
        else:
            # gen_loss, g_real_score, g_fake_score, w_dist = gen_loss.item(), g_real_score.item(), g_fake_score.item(), w_dist
            gen_loss, g_real_score, g_fake_score, w_dist = None, None, None, None

        if current_step % 4 != 0:
            recon_loss, actions_recon_loss, reward_recon_loss, next_obs_recon_loss, bisim_c_loss = None, None, None, None, None
            disc_loss, d_real_score, d_fake_score, gradient_penalty, gradients_norm = None, None, None, None, None
        else:
            # recon_loss, actions_recon_loss, reward_recon_loss, next_obs_recon_loss, bisim_c_loss = recon_loss.item(), actions_recon_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item(), bisim_c_loss.item()
            # disc_loss, d_real_score, d_fake_score, gradient_penalty, gradients_norm = disc_loss.item(), d_real_score.item(), d_fake_score.item(), gradient_penalty.item(), gradients_norm.item()
            recon_loss, actions_recon_loss, reward_recon_loss, next_obs_recon_loss, bisim_c_loss = None, None, None, None, None
            disc_loss, d_real_score, d_fake_score, gradient_penalty, gradients_norm = None, None, None, None, None

        return qf_loss.item(), vf_loss.item(), policy_loss.item(), \
            qf_loss_inter.item(), vf_loss_inter.item(), policy_loss_inter.item(), \
            recon_loss, 0, 0, actions_recon_loss, reward_recon_loss, next_obs_recon_loss, bisim_c_loss, \
            disc_loss, d_real_score, d_fake_score, gradient_penalty, gradients_norm, \
            gen_loss, g_real_score, g_fake_score, w_dist
        # kl_loss.item(), bisim_z_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item()

    def compute_recon_loss(self, tran_batch, task_z, tran_batch_size):
        obs, actions, rewards, next_obs, terms = tran_batch
        task_z_repeat = torch.cat([z.repeat(tran_batch_size, 1).unsqueeze(0) for z in task_z], dim=0)

        obs_flat, actions_flat, rewards_flat, next_obs_flat = obs.view(-1, self.o_dim), actions.view(-1,
                                                                                                     self.a_dim), rewards.view(
            -1, 1), next_obs.view(-1, self.o_dim)  # ([4096, 20]), ...
        task_z_repeat_flat = task_z_repeat.view(-1, self.l_dim)  # ([4096, 5])

        actions_pred = self.agent.decoder.dec1(obs_flat, task_z_repeat_flat)  # ([4096, 20])
        rewards_pred, next_obs_pred, _, _ = self.agent.decoder(obs_flat, actions_flat,
                                                               task_z_repeat_flat)  # ([4096, 20])

        actions_recon_loss = F.mse_loss(actions_flat, actions_pred)
        reward_recon_loss = F.mse_loss(rewards_flat, rewards_pred)
        next_obs_recon_loss = F.mse_loss(next_obs_flat, next_obs_pred)
        # recon_loss = reward_recon_loss + next_obs_recon_loss + actions_recon_loss

        return actions_recon_loss, reward_recon_loss, next_obs_recon_loss

    def compute_bisim_loss(self, ctxt_batch, task_c):
        obs_c = ctxt_batch[:, :, :self.o_dim]
        tran_bsize = ctxt_batch.size(1)  # 128

        task_c_repeat = torch.cat([c.repeat(tran_bsize, 1).unsqueeze(0) for c in task_c])  # ([4, 128, 20])

        obs_c, task_c_repeat = obs_c.view(-1, self.o_dim), task_c_repeat.view(-1, self.l_dim)

        perm = np.random.permutation(obs_c.size(0))
        obs_c = obs_c[perm]

        with torch.no_grad():
            actions_c = self.agent.policy(torch.cat([obs_c, task_c_repeat.detach()], dim=1), reparameterize=True,
                                          return_log_prob=True)[0]
            # actions_c = self.agent.decoder.dec1(obs_c, task_c_repeat)
            r_pred, _, next_o_pred_mean, next_o_pred_std = self.agent.decoder(obs_c, actions_c, task_c_repeat)
            r_pred = r_pred * self.reward_scale

        r_pred, next_o_pred_mean, next_o_pred_std = r_pred.view(-1, tran_bsize, 1), next_o_pred_mean.view(-1,
                                                                                                          tran_bsize,
                                                                                                          self.o_dim), next_o_pred_std.view(
            -1, tran_bsize, self.o_dim)

        bisim_c_loss = []
        for i in range(int(len(ctxt_batch) / 2)):  #
            task_1, task_2 = i, i + int(len(ctxt_batch) / 2)  # 0, 2 / 1, 3
            r_pred1, next_o_pred_mean1, next_o_pred_std1 = r_pred[task_1], next_o_pred_mean[task_1], next_o_pred_std[
                task_1]
            r_pred2, next_o_pred_mean2, next_o_pred_std2 = r_pred[task_2], next_o_pred_mean[task_2], next_o_pred_std[
                task_2]
            task_c1, task_c2 = task_c[task_1], task_c[task_2]

            transition_dist = torch.sum((next_o_pred_mean2 - next_o_pred_mean1).pow(2), dim=-1) + \
                              torch.sum((next_o_pred_std2 - next_o_pred_std1).pow(2), dim=-1)
            transition_dist = torch.sqrt(transition_dist)  # ([128])
            r_dist = torch.sqrt(torch.sum((r_pred2 - r_pred1).pow(2), dim=-1))  # ([128])
            sample_dist = (r_dist + 0.9 * transition_dist).mean()

            c_dist = torch.sum((task_c2 - task_c1).pow(2) + 1e-7, dim=-1)
            c_dist = torch.sqrt(c_dist)

            bisim_c_loss.append((c_dist - sample_dist).pow(2))

        bisim_c_loss = sum(bisim_c_loss) / len(bisim_c_loss)

        return bisim_c_loss

    def compute_wgan_disc_loss(self, tran_batch, task_z, tran_batch_size):
        obs, actions, rewards, next_obs, terms = tran_batch
        # o1, a1, r1, next_o1 = obs[0].unsqueeze(0), actions[0].unsqueeze(0), rewards[0].unsqueeze(0), next_obs[0].unsqueeze(0)
        # o2, a2, r2, next_o2 = obs[1].unsqueeze(0), actions[1].unsqueeze(0), rewards[1].unsqueeze(0), next_obs[1].unsqueeze(0)
        dl = int(self.meta_batch / 2)
        o1, a1, r1, next_o1 = obs[:dl], actions[:dl], rewards[:dl], next_obs[:dl]
        o2, a2, r2, next_o2 = obs[dl:], actions[dl:], rewards[dl:], next_obs[dl:]

        with torch.no_grad():
            # c1 = task_z[0].unsqueeze(0)
            c1 = task_z[:dl]
            c1 = torch.cat([c.repeat(tran_batch_size, 1).unsqueeze(0) for c in c1], dim=0)

            # c2 = task_z[1].unsqueeze(0)
            c2 = task_z[dl:]
            c2 = torch.cat([c.repeat(tran_batch_size, 1).unsqueeze(0) for c in c2], dim=0)

            alpha = np.random.uniform(0, 1, 8)
            c_alpha = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
                c_alpha.append(interpolation.unsqueeze(0))
            c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 256, 10])

            a_1alpha = self.agent.policy(torch.cat([o1.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1),
                                         reparameterize=True, return_log_prob=True)[0]
            a_2alpha = self.agent.policy(torch.cat([o2.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1),
                                         reparameterize=True, return_log_prob=True)[0]
            r_1alpha, next_o_1alpha, _, _ = self.agent.decoder(o1.view(-1, self.o_dim), a_1alpha,
                                                               c_alpha.view(-1, self.l_dim))
            r_2alpha, next_o_2alpha, _, _ = self.agent.decoder(o2.view(-1, self.o_dim), a_2alpha,
                                                               c_alpha.view(-1, self.l_dim))

            alpha_GP = np.random.uniform(0, 1, 8)
            c_alpha_GP = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                interpolation = alpha_GP[i] * c1[i] + (1 - alpha_GP[i]) * c2[i]  # ([256, 10])
                c_alpha_GP.append(interpolation.unsqueeze(0))
            c_alpha_GP = torch.cat(c_alpha_GP, dim=0)  # ([8, 256, 10])

            a_1alpha_GP = \
                self.agent.policy(torch.cat([o1.view(-1, self.o_dim), c_alpha_GP.view(-1, self.l_dim)], dim=1),
                                  reparameterize=True, return_log_prob=True)[0]
            a_2alpha_GP = \
                self.agent.policy(torch.cat([o2.view(-1, self.o_dim), c_alpha_GP.view(-1, self.l_dim)], dim=1),
                                  reparameterize=True, return_log_prob=True)[0]
            r_1alpha_GP, next_o_1alpha_GP, _, _ = self.agent.decoder(o1.view(-1, self.o_dim), a_1alpha_GP,
                                                                     c_alpha_GP.view(-1, self.l_dim))
            r_2alpha_GP, next_o_2alpha_GP, _, _ = self.agent.decoder(o2.view(-1, self.o_dim), a_2alpha_GP,
                                                                     c_alpha_GP.view(-1, self.l_dim))

        d_real_sample1 = torch.cat(
            [o1.view(-1, self.o_dim), a1.view(-1, self.a_dim), r1.view(-1, 1), next_o1.view(-1, self.o_dim)], dim=-1)
        d_real_sample2 = torch.cat(
            [o2.view(-1, self.o_dim), a2.view(-1, self.a_dim), r2.view(-1, 1), next_o2.view(-1, self.o_dim)], dim=-1)
        d_fake_sample1 = torch.cat([o1.view(-1, self.o_dim), a_1alpha, r_1alpha, next_o_1alpha], dim=-1)
        d_fake_sample2 = torch.cat([o2.view(-1, self.o_dim), a_2alpha, r_2alpha, next_o_2alpha], dim=-1)

        d_real_score1 = self.agent.disc(d_real_sample1).mean()
        d_real_score2 = self.agent.disc(d_real_sample2).mean()
        d_fake_score1 = self.agent.disc(d_fake_sample1).mean()
        d_fake_score2 = self.agent.disc(d_fake_sample2).mean()

        # gradient penalty
        d_fake_sample1_GP = torch.cat([o1.view(-1, self.o_dim), a_1alpha_GP, r_1alpha_GP, next_o_1alpha_GP], dim=-1)
        d_fake_sample2_GP = torch.cat([o2.view(-1, self.o_dim), a_2alpha_GP, r_2alpha_GP, next_o_2alpha_GP], dim=-1)
        d_fake_sample_GP = torch.cat([d_fake_sample1_GP, d_fake_sample2_GP], dim=0)
        d_fake_sample_GP.requires_grad = True
        d_fake_score_GP = self.agent.disc(d_fake_sample_GP)
        gradients = torch.autograd.grad(outputs=d_fake_score_GP, inputs=d_fake_sample_GP,
                                        grad_outputs=torch.ones(d_fake_score_GP.size()).to(ptu.device),
                                        create_graph=True, retain_graph=True)[0]  # ([256, 57])
        gradients_norm = gradients.norm(2, 1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        gradients_norm = gradients_norm.mean()

        # d_total_loss = - d_real_score1 - d_real_score2 + d_fake_score1 + d_fake_score2 + self.wgan_lambda * gradient_penalty
        return d_real_score1 + d_real_score2, d_fake_score1 + d_fake_score2, gradient_penalty, gradients_norm

    def compute_wgan_gen_loss(self, tran_batch, task_z, tran_batch_size):
        obs, actions, rewards, next_obs, terms = tran_batch
        # o1, a1, r1, next_o1 = obs[0].unsqueeze(0), actions[0].unsqueeze(0), rewards[0].unsqueeze(0), next_obs[0].unsqueeze(0)
        # o2, a2, r2, next_o2 = obs[1].unsqueeze(0), actions[1].unsqueeze(0), rewards[1].unsqueeze(0), next_obs[1].unsqueeze(0)
        dl = int(self.meta_batch / 2)
        o1, a1, r1, next_o1 = obs[:dl], actions[:dl], rewards[:dl], next_obs[:dl]
        o2, a2, r2, next_o2 = obs[dl:], actions[dl:], rewards[dl:], next_obs[dl:]

        with torch.no_grad():
            # c1 = task_z[0].unsqueeze(0)
            c1 = task_z[:dl]
            c1 = torch.cat([c.repeat(tran_batch_size, 1).unsqueeze(0) for c in c1], dim=0)

            # c2 = task_z[1].unsqueeze(0)
            c2 = task_z[dl:]
            c2 = torch.cat([c.repeat(tran_batch_size, 1).unsqueeze(0) for c in c2], dim=0)

            alpha = np.random.uniform(0, 1, 8)
            c_alpha = []  # ([C1] --------(1-a)--------- [Ca] ----(a)----- [C2])
            for i in range(int(self.meta_batch / 2)):  # 16 / 2 == 8
                interpolation = alpha[i] * c1[i] + (1 - alpha[i]) * c2[i]  # ([256, 10])
                c_alpha.append(interpolation.unsqueeze(0))
            c_alpha = torch.cat(c_alpha, dim=0)  # ([8, 256, 10])

        a_1alpha = self.agent.policy(torch.cat([o1.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1),
                                     reparameterize=True, return_log_prob=True)[0]
        a_2alpha = self.agent.policy(torch.cat([o2.view(-1, self.o_dim), c_alpha.view(-1, self.l_dim)], dim=1),
                                     reparameterize=True, return_log_prob=True)[0]
        r_1alpha, next_o_1alpha, _, _ = self.agent.decoder(o1.view(-1, self.o_dim), a_1alpha,
                                                           c_alpha.view(-1, self.l_dim))
        r_2alpha, next_o_2alpha, _, _ = self.agent.decoder(o2.view(-1, self.o_dim), a_2alpha,
                                                           c_alpha.view(-1, self.l_dim))

        g_real_sample1 = torch.cat(
            [o1.view(-1, self.o_dim), a1.view(-1, self.a_dim), r1.view(-1, 1), next_o1.view(-1, self.o_dim)], dim=-1)
        g_real_sample2 = torch.cat(
            [o2.view(-1, self.o_dim), a2.view(-1, self.a_dim), r2.view(-1, 1), next_o2.view(-1, self.o_dim)], dim=-1)
        g_fake_sample1 = torch.cat([o1.view(-1, self.o_dim), a_1alpha, r_1alpha, next_o_1alpha], dim=-1)
        g_fake_sample2 = torch.cat([o2.view(-1, self.o_dim), a_2alpha, r_2alpha, next_o_2alpha], dim=-1)

        g_real_score1 = self.agent.disc(g_real_sample1).mean().detach()
        g_real_score2 = self.agent.disc(g_real_sample2).mean().detach()
        g_fake_score1 = self.agent.disc(g_fake_sample1).mean()
        g_fake_score2 = self.agent.disc(g_fake_sample2).mean()

        # g_total_loss = - g_fake_score1 - g_fake_score2
        w_dist = (g_real_score1 + g_real_score2 - g_fake_score1 - g_fake_score2).item()

        return g_real_score1 + g_real_score2, g_fake_score1 + g_fake_score2, w_dist

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            psi=self.agent.psi.state_dict(),
            # decoder=self.agent.decoder.state_dict(),
        )
        return snapshot