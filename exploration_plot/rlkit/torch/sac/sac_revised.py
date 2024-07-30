"""코드 간소화 - 2024.03.22
- 학습 로스는 그대로 유지하되,
- 배치 뽑는것 최소화
-
"""



from collections import OrderedDict
import numpy as np

import time
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

from torch.distributions import Normal

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

    def size(self):
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
            # optimizer_class=optim.Adam,
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
        # self.task_c_on_buffer  = TaskCBuffer(num_tasks=len(self.train_tasks), capacity=self.num_meta_train_steps * 10)
        # self.task_c_off_buffer = TaskCBuffer(num_tasks=len(self.train_tasks), capacity=self.num_meta_train_steps * 10)
        self.c_off_buffer_history_que = deque(maxlen=10)
        self.c_on_buffer_history_que  = deque(maxlen=10)

        self.ce_loss = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()


        if self.optimizer == "adam":
            optimizer_class = optim.Adam
        elif self.optimizer == "adagrad":
            optimizer_class = optim.Adagrad
        elif self.optimizer == "rmsprop":
            optimizer_class = optim.RMSprop
        elif self.optimizer == "sgd":
            optimizer_class = optim.SGD
        else:
            optimizer_class = None

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.alpha_optimizer = optimizer_class(
            self.agent.log_alpha_net.parameters(),
            lr=policy_lr,
            weight_decay=self.alpha_net_weight_decay,
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

        print("policy_lr", policy_lr)
        print("qf_lr", qf_lr)
        print("vf_lr", vf_lr)
        print("context_lr", context_lr)

        print("psi trainable parameters :", sum(p.numel() for p in self.agent.psi.parameters() if p.requires_grad))

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
            lr=context_lr
        )

        # self.z_autoencoder_optim = optimizer_class(
        #     self.agent.z_autoencoder.parameters(),
        #     lr=0.0003
        # )

        self.c_distribution_vae_optim = optimizer_class(
            self.agent.c_distribution_vae.parameters(),
            lr=0.0002
        )

        self.sim_fn = torch.nn.CosineSimilarity(dim=-1)

        self.target_enc_tau = target_enc_tau

        self.soft_target_update(self.agent.c_decoder, self.agent.c_decoder_target, 1.0)

    # def target_enc_soft_update(self):
    #     for param_target, param in zip(self.agent.psi_target.parameters(), self.agent.psi.parameters()):
    #         param_target.data.copy_(param_target.data * (1.0 - self.target_enc_tau) + param.data * self.target_enc_tau)
    #         # param_target * 0.999 + param_online * 0.001
    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

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

    def sample_sac(self, indices, batchsize, randomly_sample=False):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense

        # batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=batchsize)) for idx in indices]

        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]

        if randomly_sample:
            t = len(unpacked[0])
            random_ind = np.random.choice(range(t * batchsize), batchsize, replace=False)
            unpacked = [x.view(t * batchsize, -1)[random_ind] for x in unpacked]

        return unpacked

    def sample_total_task_transition(self, indices_len, batch_size):
        # prior_enc_buffer_size = np.zeros(indices_len)
        # for task_idx in range(indices_len):
        #     prior_enc_buffer_size[task_idx] = self.total_enc_replay_buffer.task_buffers[task_idx].size()
        # print(prior_enc_buffer_size)
        # [10503.     0.     0.     0.]
        batches = [ptu.np_to_pytorch_batch(
            self.bisim_target_sample_buffer.random_batch(0, batch_size=batch_size)) for idx in range(indices_len)]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices, which_buffer="default", b_size="default", return_unpacked=False):  # which_buffer : "prior" or "online"
        ''' sample batch of context from a list of tasks from the replay buffer '''
        if which_buffer == "default":
            which_buffer = self.offpol_ctxt_sampling_buffer
        if b_size == "default":
            b_size = self.embedding_batch_size

        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        # if self.use_context_buffer:

        if which_buffer == "rl":
            batches = [ptu.np_to_pytorch_batch( self.replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]

        elif which_buffer == "prior":
            batches = [ptu.np_to_pytorch_batch(
                self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]


        elif which_buffer == "online":
            if not self.use_episodic_online_buffer:
                t = [self.online_enc_replay_buffer.task_buffers[idx].size() > 0 for idx in indices]
                if math.prod(t):  # 샘플링된 모든 인덱스의 온라인 컨텍스트 버퍼에 샘플이 차 있으면:
                    batches = [ptu.np_to_pytorch_batch(
                        self.online_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]
                else:
                    batches = [ptu.np_to_pytorch_batch(
                        self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size)) for idx in indices]
            else:
                ctxt_batch = self.episodic_online_buffer.sample_context(task_indices=indices, batch_size=self.embedding_batch_size)  # ([1, 128, 36])
                # print("ctxt_batch in sample_context using episodic_online_buffer :", ctxt_batch.shape)  # ([4, 128, 36])
                if return_unpacked:
                    o = ctxt_batch[:, :, :self.o_dim]
                    a = ctxt_batch[:, :, self.o_dim:self.o_dim+self.a_dim]
                    r = ctxt_batch[:, :, self.o_dim+self.a_dim:self.o_dim+self.a_dim+self.r_dim]
                    if self.agent.use_next_obs_in_context:
                        no = ctxt_batch[:, :, self.o_dim+self.a_dim+self.r_dim:self.o_dim+self.a_dim+self.r_dim+self.o_dim]
                    else:
                        no = None
                    return ctxt_batch, [o, a, r, no, None]
                else:
                    return ctxt_batch

        elif which_buffer == "both":
            t = [self.online_enc_replay_buffer.task_buffers[idx].size() > 0 for idx in indices]  # 그 태스크 인덱스에 대한 온라인 버퍼가 차 있는지 확인하는 리스트
            # print("t :", t, math.prod(t))  # [True, False, False, True, ... ]
            if math.prod(t):  # 샘플링된 모든 인덱스의 온라인 컨텍스트 버퍼에 샘플이 차있으면:
                b_size_online = int(b_size / 5)  # --> 128개 컨텍스트 배치에서 1/5는 온라인 버퍼에서 샘플링
                b_size_prior = b_size - b_size_online  # --> 128개 배치에서 나머지는 프라이어 버퍼에서
                batches_online = [
                    ptu.np_to_pytorch_batch(self.online_enc_replay_buffer.random_batch(idx, batch_size=b_size_online))
                    for idx in indices]
                batches_prior = [
                    ptu.np_to_pytorch_batch(self.prior_enc_replay_buffer.random_batch(idx, batch_size=b_size_prior)) for
                    idx in indices]
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
                    temp_dict["terminals"] = torch.cat([batches_online[j]["terminals"], batches_prior[j]["terminals"]])[
                        perm]
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

    def get_repeat(self, latent, tran_batch_size, dim):  # [16, 10]
        latent_repeat = torch.cat([z.repeat(tran_batch_size, 1).unsqueeze(0) for z in latent], dim=0)
        latent_repeat = latent_repeat.view(-1, dim)
        return latent_repeat  # [16, 256, 10]

    def pick_specific_tasks(self, target_indices, tran_batch):
        if type(tran_batch) == list:
            o, a, r, n_o, t = tran_batch
            o_, a_, r_, n_o_, t_ = o[target_indices], a[target_indices], r[target_indices], n_o[target_indices], t[target_indices]
            return [o_, a_, r_, n_o_, t_]
        elif type(tran_batch) == torch.Tensor:
            return tran_batch[target_indices]
    #######################


    def pretrain(self, train_step, pretrain=False):

        task_indices = np.random.choice(self.train_tasks, self.meta_batch, replace=True)

        off_tran_batch1 = self.sample_sac(task_indices, self.batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...
        off_tran_batch2 = self.sample_sac(task_indices, self.batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...

        task_k = self.index_to_onehot(task_indices)  # ([16, 150]) #asdfgasdfg

        off_ctxt_batch = self.sample_context(task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
        on_ctxt_batch = self.sample_context(task_indices, which_buffer="online")
        off_ctxt_batch2 = self.sample_context(task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
        on_ctxt_batch2 = self.sample_context(task_indices, which_buffer="online")

        c_1_off = self.agent.get_context_embedding(off_ctxt_batch, which_enc="psi")  # [([4, 10]), ([4, 10]), ...]
        c_1_on  = self.agent.get_context_embedding(on_ctxt_batch, which_enc="psi")
        c_2_off = self.agent.get_context_embedding(off_ctxt_batch2, which_enc="psi")  # [([4, 10]), ([4, 10]), ...]
        c_2_on  = self.agent.get_context_embedding(on_ctxt_batch2, which_enc="psi")

        self.agent.task_c_off_buffer.add_c(task_indices, c_1_off)
        self.agent.task_c_off_buffer.add_c(task_indices, c_2_off)
        self.agent.task_c_on_buffer.add_c(task_indices, c_1_on)
        self.agent.task_c_on_buffer.add_c(task_indices, c_2_on)

        if self.agent.task_c_off_buffer.check_buffers_size(threshold=512):
            c_off_mean = self.agent.task_c_off_buffer.sample_c(tasks_indices=task_indices,
                                                               all_element_sampling=self.c_off_all_element_sampling,
                                                               num_samples=512)
            c_off_mean = c_off_mean.mean(dim=1)
        else:
            c_offs = []
            for _ in range(10):
                temp = self.sample_context(task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
                c_offs.append(self.agent.get_context_embedding(temp, which_enc="psi_target"))  # [([4, 10]), ([4, 10]), ...]
            c_off_mean = torch.stack(c_offs, dim=0).mean(dim=0)  # ([10, 4, 10])  -->  ([4, 10])

        task_k_repeat  = self.get_repeat(task_k,  self.batch_size, len(self.train_tasks))
        c_1_off_repeat = self.get_repeat(c_1_off, self.batch_size, self.l_dim)  # ([4096, 10])
        c_1_on_repeat  = self.get_repeat(c_1_on,  self.batch_size, self.l_dim)  # ([4096, 10])





        self.c_buffer.add_c(c_1_on)
        if (train_step % self.c_distri_vae_train_freq == 0) and self.use_c_vae and self.agent.task_c_on_buffer.check_buffers_size(threshold=256):
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = self.train_c_distribution_vae()
        else:
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = 0, 0, 0

        obs1, actions1, rewards1, next_obs1, terms1 = off_tran_batch1
        obs_flat1, actions_flat1, rewards_flat1, next_obs_flat1, terms_flat1 = obs1.view(-1, self.o_dim), actions1.view(-1, self.a_dim), rewards1.view(-1, 1), next_obs1.view(-1, self.o_dim), terms1.view(-1, 1)


        rewards_flat_pred1_k, next_obs_flat_pred1_sample_k, next_obs_flat_pred1_mean_k, next_obs_flat_pred1_std_k = self.agent.k_decoder(obs_flat1, actions_flat1, task_k_repeat)  # ([4096, 20])
        rewards_flat_pred1_c, next_obs_flat_pred1_sample_c, next_obs_flat_pred1_mean_c, next_obs_flat_pred1_std_c = self.agent.c_decoder(obs_flat1, actions_flat1, c_1_off_repeat)
        """ 1. 기본 representation loss 계산 """

        # (1) recon loss 계산
        reward_recon_loss_k, next_obs_recon_loss_k = self.compute_recon_loss_probablistic(rewards_flat1, next_obs_flat1, rewards_flat_pred1_k, next_obs_flat_pred1_sample_k, next_obs_flat_pred1_mean_k, next_obs_flat_pred1_std_k)
        reward_recon_loss_c, next_obs_recon_loss_c = self.compute_recon_loss_probablistic(rewards_flat1, next_obs_flat1, rewards_flat_pred1_c, next_obs_flat_pred1_sample_c, next_obs_flat_pred1_mean_c, next_obs_flat_pred1_std_c)
        # reward_recon_loss_c, next_obs_recon_loss_c = torch.zeros_like(reward_recon_loss_k), torch.zeros_like(reward_recon_loss_k)
        reward_recon_loss = reward_recon_loss_k + reward_recon_loss_c
        next_obs_recon_loss = next_obs_recon_loss_k + next_obs_recon_loss_c

        # (2) bisim, same task loss, onpol c loss 계산
        bisim_c_loss, bisim_c_loss_alpha = self.compute_bisim_alpha_loss(c_1_off, task_k, c_1_off, mse_reduction="sum")
        # same_task_c_loss = F.mse_loss(c_1_off, c_2_off, reduction="sum")  # + F.mse_loss(c_1_on, c_2_on, reduction="sum")  # if self.use_next_state_bisim else torch.zeros_like(bisim_c_loss)  #
        on_off_loss    = F.mse_loss(c_off_mean.detach(), c_1_on, reduction="sum")
        # off_off_loss   = F.mse_loss(c_off_mean.detach(), c_1_off, reduction="sum")






        """ representation loss 업데이트 """
        total_loss = self.recon_coeff * (reward_recon_loss + next_obs_recon_loss ) +\
                     self.bisim_coeff * (bisim_c_loss + bisim_c_loss_alpha) # +\
                     # self.onpol_c_coeff * on_off_loss
                     # self.offpol_c_coeff * off_off_loss

        self.k_decoder_optim.zero_grad()
        self.c_decoder_optim.zero_grad()
        self.psi_optim.zero_grad()
        total_loss.backward()
        self.k_decoder_optim.step()
        self.c_decoder_optim.step()
        self.psi_optim.step()


        # self.target_enc_soft_update()
        # self.soft_target_update(self.agent.psi, self.agent.psi_target, tau=0.005)
        if self.use_target_c_dec:
            self.soft_target_update(self.agent.c_decoder, self.agent.c_decoder_target, tau=0.005)


        return total_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item(), \
            reward_recon_loss_k.item(), reward_recon_loss_c.item(), \
            next_obs_recon_loss_k.item(), next_obs_recon_loss_c.item(), \
            bisim_c_loss.item(), on_off_loss.item(), \
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss




    #################3
    def meta_train(self, train_step, ep, pretrain=False):

        coff_buffer_fix = False
        bisim_coeff = self.bisim_coeff
        offpol_c_coeff = 0.0
        if ep > self.coff_buffer_fix_ep:
            coff_buffer_fix = True
            bisim_coeff = 0.0
            offpol_c_coeff = self.offpol_c_coeff



        # print("self.close_method", self.close_method)

        task_indices = np.random.choice(self.train_tasks, self.meta_batch, replace=True)

        off_tran_batch1 = self.sample_sac(task_indices, self.batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...
        off_tran_batch2 = self.sample_sac(task_indices, self.batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...

        task_k = self.index_to_onehot(task_indices)  # ([16, 150]) #asdfgasdfg

        off_ctxt_batch = self.sample_context(task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
        on_ctxt_batch = self.sample_context(task_indices, which_buffer="online")
        off_ctxt_batch2 = self.sample_context(task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
        on_ctxt_batch2 = self.sample_context(task_indices, which_buffer="online")

        c_1_off = self.agent.get_context_embedding(off_ctxt_batch, which_enc="psi")  # [([4, 10]), ([4, 10]), ...]
        c_1_on  = self.agent.get_context_embedding(on_ctxt_batch, which_enc="psi")
        c_2_off = self.agent.get_context_embedding(off_ctxt_batch2, which_enc="psi")  # [([4, 10]), ([4, 10]), ...]
        c_2_on  = self.agent.get_context_embedding(on_ctxt_batch2, which_enc="psi")

        self.agent.task_c_off_buffer.add_c(task_indices, c_1_off, coff_buffer_fix)
        self.agent.task_c_off_buffer.add_c(task_indices, c_2_off, coff_buffer_fix)
        self.agent.task_c_on_buffer.add_c(task_indices, c_1_on)
        self.agent.task_c_on_buffer.add_c(task_indices, c_2_on)

        if self.agent.task_c_off_buffer.check_buffers_size(threshold=512):
            c_off_mean = self.agent.task_c_off_buffer.sample_c(tasks_indices=task_indices,
                                                               all_element_sampling=self.c_off_all_element_sampling,
                                                               num_samples=512)
            c_off_mean = c_off_mean.mean(dim=1)
        else:
            c_offs = []
            for _ in range(10):
                temp = self.sample_context(task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
                c_offs.append(self.agent.get_context_embedding(temp, which_enc="psi_target"))  # [([4, 10]), ([4, 10]), ...]
            c_off_mean = torch.stack(c_offs, dim=0).mean(dim=0)  # ([10, 4, 10])  -->  ([4, 10])

        task_k_repeat  = self.get_repeat(task_k,  self.batch_size, len(self.train_tasks))
        c_1_off_repeat = self.get_repeat(c_1_off, self.batch_size, self.l_dim)  # ([4096, 10])
        c_1_on_repeat  = self.get_repeat(c_1_on,  self.batch_size, self.l_dim)  # ([4096, 10])



        """"""
        c_off_alpha, c_on_alpha = [], []
        for i in range(self.num_fake_tasks):
            mixing_task_indices = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=True)
            mixing_off_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="rl")  # ([16, 100, 27])
            mixing_on_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="online")  # ([16, 100, 27])
            c_off_mixing = self.agent.get_context_embedding(mixing_off_ctxt_batch, which_enc="psi")
            c_on_mixing = self.agent.get_context_embedding(mixing_on_ctxt_batch, which_enc="psi")
            alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
            c_off_alpha.append(alpha @ c_off_mixing)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
            c_on_alpha.append(alpha @ c_on_mixing)
        c_off_alpha = torch.cat(c_off_alpha)  # ([5, 10])
        c_on_alpha = torch.cat(c_on_alpha)  # ([5, 10])
        """"""

        self.c_buffer.add_c(c_1_on)
        if (train_step % self.c_distri_vae_train_freq == 0) and self.use_c_vae and self.agent.task_c_on_buffer.check_buffers_size(threshold=256):
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = self.train_c_distribution_vae()
        else:
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = 0, 0, 0

        obs1, actions1, rewards1, next_obs1, terms1 = off_tran_batch1
        obs_flat1, actions_flat1, rewards_flat1, next_obs_flat1, terms_flat1 = obs1.view(-1, self.o_dim), actions1.view(-1, self.a_dim), rewards1.view(-1, 1), next_obs1.view(-1, self.o_dim), terms1.view(-1, 1)



        rewards_flat_pred1_c, next_obs_flat_pred1_sample_c, next_obs_flat_pred1_mean_c, next_obs_flat_pred1_std_c = self.agent.c_decoder(obs_flat1, actions_flat1, c_1_off_repeat)
        reward_recon_loss_c, next_obs_recon_loss_c = self.compute_recon_loss_probablistic(rewards_flat1, next_obs_flat1, rewards_flat_pred1_c, next_obs_flat_pred1_sample_c, next_obs_flat_pred1_mean_c, next_obs_flat_pred1_std_c)
        # reward_recon_loss_c, next_obs_recon_loss_c = torch.zeros_like(reward_recon_loss_k), torch.zeros_like(reward_recon_loss_k)

        if not coff_buffer_fix:
            rewards_flat_pred1_k, next_obs_flat_pred1_sample_k, next_obs_flat_pred1_mean_k, next_obs_flat_pred1_std_k = self.agent.k_decoder(obs_flat1, actions_flat1, task_k_repeat)  # ([4096, 20])
            reward_recon_loss_k, next_obs_recon_loss_k = self.compute_recon_loss_probablistic(rewards_flat1, next_obs_flat1, rewards_flat_pred1_k, next_obs_flat_pred1_sample_k, next_obs_flat_pred1_mean_k, next_obs_flat_pred1_std_k)
        else:
            reward_recon_loss_k, next_obs_recon_loss_k = torch.zeros_like(reward_recon_loss_c), torch.zeros_like(reward_recon_loss_c)


        reward_recon_loss = reward_recon_loss_k + reward_recon_loss_c
        next_obs_recon_loss = next_obs_recon_loss_k + next_obs_recon_loss_c

        # (2) bisim, same task loss, onpol c loss 계산
        bisim_c_loss, bisim_c_loss_alpha = self.compute_bisim_alpha_loss(c_1_off, task_k, c_off_alpha, mse_reduction="sum")
        same_task_c_loss = F.mse_loss(c_1_off, c_2_off, reduction="sum")  # + F.mse_loss(c_1_on, c_2_on, reduction="sum")  # if self.use_next_state_bisim else torch.zeros_like(bisim_c_loss)  #
        on_off_loss = F.mse_loss(c_off_mean.detach(), c_1_on, reduction="sum")
        off_off_loss = F.mse_loss(c_off_mean.detach(), c_1_off, reduction="sum")




        """ 2. fake sample 활용한 representation loss 계산 """
        if self.use_fakesample_representation:
            # 방법 1
            # # 랜덤으로 15개 뽑아서 섞음 --> 5개로 만듦
            # mixing_task_indices = np.random.choice(self.train_tasks, self.num_fake_tasks * self.num_dirichlet_tasks, replace=False)
            # alphas = [Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks for _ in range(self.num_fake_tasks)]
            # alphas = torch.cat(alphas, dim=0).view(self.num_fake_tasks, -1, 1)  # [5, 3, 1]
            # # c_on_mixing  = c_on_total_tasks[mixing_task_indices].view(self.num_fake_tasks, self.num_dirichlet_tasks, -1)
            # # c_off_mixing = c_off_total_tasks[mixing_task_indices].view(self.num_fake_tasks, self.num_dirichlet_tasks, -1)
            #
            # mixing_off_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
            # mixing_on_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="online")
            # c_off_mixing = self.agent.get_context_embedding(mixing_off_ctxt_batch, which_enc="psi").view(self.num_fake_tasks, self.num_dirichlet_tasks, -1)  # [([4, 10]), ([4, 10]), ...]
            # c_on_mixing = self.agent.get_context_embedding(mixing_on_ctxt_batch, which_enc="psi").view(self.num_fake_tasks, self.num_dirichlet_tasks, -1)
            #
            # c_on_alpha = (alphas * c_on_mixing).sum(dim=1)    # [5, 3, 1] * [5, 3, 10] -> [5, 3, 10] --> [5, 10]
            # c_off_alpha = (alphas * c_off_mixing).sum(dim=1)  # [5, 3, 1] * [5, 3, 10] -> [5, 3, 10] --> [5, 10]

            # 방법 2


            c_off_alpha_repeat = self.get_repeat(c_off_alpha, self.batch_size, self.l_dim)
            c_off_alpha_repeat_half = self.get_repeat(c_off_alpha, self.embedding_batch_size, self.l_dim)

            # real 태스크 트랜지션 샘플 (wgan할때 사용)
            real_task_indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=True)
            off_tran_batch1_real = self.sample_sac(real_task_indices, self.batch_size)  # --> 디스크리미네이터 학습에 사용
            off_tran_batch2_real = self.sample_sac(real_task_indices, self.batch_size)  # --> 제너레이터 학습에 사용
            _, on_ctxt_batch_tran1 = self.sample_context(real_task_indices, which_buffer="online", return_unpacked=True)

            # c_off_real = c_on_total_tasks[real_task_indices]
            real_off_ctxt_batch = self.sample_context(real_task_indices, which_buffer=self.offpol_ctxt_sampling_buffer)  # ([4, 128, 27])
            c_off_real = self.agent.get_context_embedding(real_off_ctxt_batch, which_enc="psi")  #

            c_off_real_repeat = self.get_repeat(c_off_real, self.batch_size, self.l_dim)

            obs_real1, actions_real1, rewards_real1, next_obs_real1, _ = off_tran_batch1_real  # 태스크 5개
            obs_real2, actions_real2, rewards_real2, next_obs_real2, _ = off_tran_batch2_real  # 태스크 5개, ([1280, 29]), ([1280, 8])
            obs_real_on, actions_real_on, _, _, _ = on_ctxt_batch_tran1  # ([5, 128, 29])
            obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat = obs_real1.view(self.num_fake_tasks * self.batch_size, -1), actions_real1.view(self.num_fake_tasks * self.batch_size, -1), rewards_real1.view(self.num_fake_tasks * self.batch_size, -1), next_obs_real1.view(self.num_fake_tasks * self.batch_size, -1)
            obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat = obs_real2.view(self.num_fake_tasks * self.batch_size, -1), actions_real2.view(self.num_fake_tasks * self.batch_size, -1), rewards_real2.view(self.num_fake_tasks * self.batch_size, -1), next_obs_real2.view(self.num_fake_tasks * self.batch_size, -1)
            obs_real_on_flat, actions_real_on_flat = obs_real_on.view(self.num_fake_tasks * self.embedding_batch_size, -1), actions_real_on.view(self.num_fake_tasks * self.embedding_batch_size, -1)

            """ method 1"""
            # rewards_fake1_flat, next_obs_fake1_flat, _, _ = self.agent.c_decoder(obs_real1_flat, actions_real1_flat, c_off_alpha_repeat)  # ([1280, 1])
            # if train_step % self.gen_freq == 0:
            #     rewards_fake2_flat, next_obs_fake2_flat, _, _ = self.agent.c_decoder(obs_real2_flat, actions_real2_flat, c_off_alpha_repeat)
            # rewards_fake_on_flat, next_obs_fake_on_flat, _, _ = self.agent.c_decoder(obs_real_on_flat, actions_real_on_flat, c_off_alpha_repeat_half)
            # # (1) discriminator loss 계산
            # real_transition = [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat]
            # fake_transition = [rewards_fake1_flat, next_obs_fake1_flat, next_obs_real1_flat, c_off_alpha_repeat]
            # d_real_score, d_fake_score, gradient_penalty, gradients_norm = self.compute_disc_loss(real_transition, fake_transition)
            # d_total_loss = - d_real_score + d_fake_score + self.wgan_lambda * gradient_penalty
            # self.disc_optim.zero_grad()
            # d_total_loss.backward()
            # self.disc_optim.step()

            # # (2) generator loss 계산
            # if train_step % self.gen_freq == 0:
            #     real_transition = [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_off_real_repeat]
            #     fake_transition = [rewards_fake2_flat, next_obs_fake2_flat, next_obs_real2_flat, c_off_alpha_repeat]
            #     g_real_score, g_fake_score, w_distance = self.compute_generator_loss(real_transition, fake_transition)
            #     g_total_loss = -g_fake_score
            # else:
            #     g_total_loss, g_real_score, g_fake_score, w_distance = [torch.zeros_like(bisim_c_loss) for _ in range(4)]

            # # (3) fake sample cycle loss 계산
            # tr_batch_off = obs_real1, actions_real1, rewards_fake1_flat, next_obs_fake1_flat
            # tr_batch_on  = obs_real_on, actions_real_on, rewards_fake_on_flat, next_obs_fake_on_flat
            # cycle_loss_off, cycle_loss_on = self.compute_cycle_loss(c_off_alpha, c_on_alpha, tr_batch_off, tr_batch_on)

            """ method 2"""  # real = bat1  //  fake = bat2+c_alpha
            # rewards_fake1_flat, next_obs_fake1_flat, _, _ = self.agent.c_decoder(obs_real1_flat, actions_real1_flat, c_off_alpha_repeat)  # ([1280, 1])
            # if train_step % self.gen_freq == 0:
            rewards_fake2_flat, next_obs_fake2_flat, _, _ = self.agent.c_decoder(obs_real2_flat, actions_real2_flat, c_off_alpha_repeat)
            rewards_fake_on_flat, next_obs_fake_on_flat, _, _ = self.agent.c_decoder(obs_real_on_flat, actions_real_on_flat, c_off_alpha_repeat_half)

            # (1) discriminator loss 계산
            real_transition = [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat]
            fake_transition = [obs_real2_flat, actions_real2_flat, rewards_fake2_flat, next_obs_fake2_flat, next_obs_real2_flat, c_off_alpha_repeat]
            d_real_score, d_fake_score, gradient_penalty, gradients_norm = self.compute_disc_loss(real_transition, fake_transition)
            d_total_loss = - d_real_score + d_fake_score + self.wgan_lambda * gradient_penalty
            self.disc_optim.zero_grad()
            d_total_loss.backward()
            self.disc_optim.step()

            # (2) generator loss 계산
            if train_step % self.gen_freq == 0:
                # real_transition = [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_off_real_repeat]
                # fake_transition = [rewards_fake2_flat, next_obs_fake2_flat, next_obs_real2_flat, c_off_alpha_repeat]
                g_real_score, g_fake_score, w_distance = self.compute_generator_loss(real_transition, fake_transition)
                g_total_loss = -g_fake_score
            else:
                g_total_loss, g_real_score, g_fake_score, w_distance = [torch.zeros_like(bisim_c_loss) for _ in range(4)]

            # (3) fake sample cycle loss 계산
            tr_batch_off = obs_real2, actions_real2, rewards_fake2_flat, next_obs_fake2_flat
            tr_batch_on = obs_real_on, actions_real_on, rewards_fake_on_flat, next_obs_fake_on_flat
            cycle_loss_off, cycle_loss_on = self.compute_cycle_loss(c_off_alpha, c_on_alpha, tr_batch_off, tr_batch_on)

        else:
            d_total_loss, d_real_score, d_fake_score, gradient_penalty, gradients_norm = [torch.zeros_like(bisim_c_loss) for _ in range(5)]
            g_total_loss, g_real_score, g_fake_score, w_distance = [torch.zeros_like(bisim_c_loss) for _ in range(4)]
            cycle_loss_off, cycle_loss_on = [torch.zeros_like(bisim_c_loss) for _ in range(2)]


        """ representation loss 업데이트 """
        total_loss = self.recon_coeff * (reward_recon_loss + next_obs_recon_loss ) +\
                     bisim_coeff * (bisim_c_loss + bisim_c_loss_alpha) + \
                     self.onpol_c_coeff * on_off_loss + \
                     offpol_c_coeff * off_off_loss + \
                     self.same_c_coeff * same_task_c_loss + \
                     self.gen_coeff * g_total_loss + \
                     self.cycle_coeff * (cycle_loss_off + cycle_loss_on)
        # total_loss = self.recon_coeff * (reward_recon_loss + next_obs_recon_loss) + \
        #              self.bisim_coeff * bisim_c_loss  + \
        #              self.cycle_coeff * (cycle_loss_off + cycle_loss_on) + \
        #              self.onpol_c_coeff * on_pol_c_loss
        #              #self.same_c_coeff * same_task_c_loss + \
        #              #self.gen_coeff * g_total_loss + \

        self.k_decoder_optim.zero_grad()
        self.c_decoder_optim.zero_grad()
        self.psi_optim.zero_grad()
        total_loss.backward()
        self.k_decoder_optim.step()
        self.c_decoder_optim.step()
        self.psi_optim.step()


        # self.target_enc_soft_update()
        # self.soft_target_update(self.agent.psi, self.agent.psi_target, tau=0.005)
        if self.use_target_c_dec:
            self.soft_target_update(self.agent.c_decoder, self.agent.c_decoder_target, tau=0.005)


        return total_loss.item(), reward_recon_loss.item(), next_obs_recon_loss.item(), \
            reward_recon_loss_k.item(), reward_recon_loss_c.item(), \
            next_obs_recon_loss_k.item(), next_obs_recon_loss_c.item(), \
            bisim_c_loss.item(), on_off_loss.item(), off_off_loss.item(), same_task_c_loss.item(), \
            c_vae_loss, c_vae_recon_loss, c_vae_kl_loss, \
            d_total_loss.item(), d_real_score.item(), d_fake_score.item(), gradient_penalty.item(), gradients_norm.item(), \
            g_total_loss.item(), g_real_score.item(), g_fake_score.item(), w_distance, \
            cycle_loss_off.item(), cycle_loss_on.item()






    def compute_recon_loss(self, rewards, next_obs, rewards_pred, next_obs_pred):
        reward_recon_loss = F.mse_loss(rewards, rewards_pred)
        if self.use_decoder_next_state:
            next_obs_recon_loss = F.mse_loss(next_obs, next_obs_pred)
        else:
            next_obs_recon_loss = torch.zeros_like(reward_recon_loss)
        return reward_recon_loss, next_obs_recon_loss


    def compute_recon_loss_probablistic(self, rewards, next_obs, rewards_pred, next_obs_pred_sample, next_obs_pred_mean, next_obs_pred_std):
        reward_recon_loss = F.mse_loss(rewards, rewards_pred)
        if self.use_decoder_next_state:
            # next_obs_pred_dist = Normal(next_obs_pred_mean, next_obs_pred_std)
            # next_obs_logprob = next_obs_pred_dist.log_prob(next_obs)
            # # next_obs_logprob = next_obs_logprob.sum(dim=-1)
            # next_obs_recon_loss = -next_obs_logprob.mean()
            next_obs_recon_loss = F.mse_loss(next_obs, next_obs_pred_sample)
        else:
            next_obs_recon_loss = torch.zeros_like(reward_recon_loss)
        return reward_recon_loss, next_obs_recon_loss


    def compute_bisim_loss(self, task_z_1, task_k_repeat, mse_reduction="mean"):  # task_z_1 : [16, 10]  /// task_k_repeat :
        # obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(len(tasks_indices), batch_size=tran_bsize)  # [16, 128, 20], ...

        # if self.bisim_transition_sample == "online":
        #     obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(self.meta_batch, batch_size=self.batch_size)
        #     obs_c, actions_c = obs_c[0], actions_c[0]
        # elif self.bisim_transition_sample == "rl":
        #     obs_c, actions_c, _, _, _ = self.sample_sac(self.train_tasks, batchsize=self.batch_size, randomly_sample=True)  # ([256, 27]), ([256, 8])

        obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(self.meta_batch, batch_size=self.batch_size)
        obs_c, actions_c = obs_c[0], actions_c[0]

        obs_c, actions_c = obs_c.repeat(self.meta_batch, 1, 1).view(-1, self.o_dim), actions_c.repeat(self.meta_batch, 1, 1).view(-1, self.a_dim)

        with torch.no_grad():
            if self.use_target_c_dec:
                task_z_1_repeat = self.get_repeat(task_z_1, self.batch_size, self.l_dim)
                r_pred, _, next_o_pred_mean, next_o_pred_std = self.agent.c_decoder_target(obs_c, actions_c, task_z_1_repeat)
            else:
                r_pred, _, next_o_pred_mean, next_o_pred_std = self.agent.k_decoder(obs_c, actions_c, task_k_repeat)

        perm = np.random.permutation(self.meta_batch)
        r_1 = r_pred.view(-1, self.batch_size, 1)
        r_2 = r_1[perm]
        if self.use_next_state_bisim:
            n_o_mean_1, n_o_std_1 = next_o_pred_mean.view(-1, self.batch_size, self.o_dim), next_o_pred_std.view(-1, self.batch_size, self.o_dim)
            n_o_mean_2, n_o_std_2 = n_o_mean_1[perm], n_o_std_1[perm]
        task_z_2 = task_z_1[perm]

        r_dist = ((r_2 - r_1).pow(2) + 1e-7).sum(dim=-1)  # ([16, 128])
        r_dist = torch.sqrt(r_dist + 1e-7) * self.r_dist_coeff
        if self.use_next_state_bisim:
            transition_dist = ((n_o_mean_2 - n_o_mean_1).pow(2) + 1e-7).sum(dim=-1) + \
                              ((n_o_std_2 - n_o_std_1).pow(2) + 1e-7).sum(dim=-1)
            transition_dist = torch.sqrt(transition_dist + 1e-7)  # ([16, 128])
            transition_dist = transition_dist * self.tr_dist_coeff
            sample_dist = (r_dist + 0.9 * transition_dist).mean(dim=-1)
        else:
            sample_dist = r_dist.mean(dim=-1)

        z_dist = torch.abs(task_z_2 - task_z_1).sum(dim=-1)
        # z_dist = ((task_z_2 - task_z_1).pow(2) + 1e-7).sum(dim=-1)
        # z_dist = torch.sqrt(z_dist + 1e-7)

        bisim_loss = F.mse_loss(z_dist, sample_dist, reduction=mse_reduction)

        return bisim_loss


    def compute_bisim_alpha_loss(self, task_z_1, task_k, task_z_alpha, mse_reduction="mean"):  # task_z_1 : [16, 10]  /// task_k_repeat :
        # obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(len(tasks_indices), batch_size=tran_bsize)  # [16, 128, 20], ...

        # if self.bisim_transition_sample == "online":
        #     obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(self.meta_batch, batch_size=self.batch_size)
        #     obs_c, actions_c = obs_c[0], actions_c[0]
        # elif self.bisim_transition_sample == "rl":
        #     obs_c, actions_c, _, _, _ = self.sample_sac(self.train_tasks, batchsize=self.batch_size, randomly_sample=True)  # ([256, 27]), ([256, 8])

        task_k_repeat = self.get_repeat(task_k, self.bisim_sample_batchsize, len(self.train_tasks))

        obs_c_, actions_c_, _, _, _ = self.sample_total_task_transition(self.meta_batch, batch_size=self.bisim_sample_batchsize)
        obs_c_, actions_c_ = obs_c_[0], actions_c_[0]

        obs_c, actions_c = obs_c_.repeat(self.meta_batch, 1, 1).view(-1, self.o_dim), actions_c_.repeat(self.meta_batch, 1, 1).view(-1, self.a_dim)
        obs_c_alpha, actions_c_alpha = obs_c_.repeat(self.num_fake_tasks, 1, 1).view(-1, self.o_dim), actions_c_.repeat(self.num_fake_tasks, 1, 1).view(-1, self.a_dim)

        with torch.no_grad():
            if self.use_target_c_dec:
                task_z_1_repeat = self.get_repeat(task_z_1, self.batch_size, self.l_dim)
                task_z_alpha_repeat = self.get_repeat(task_z_alpha, self.batch_size, self.l_dim)
                r_pred, _, next_o_pred_mean, next_o_pred_std = self.agent.c_decoder_target(obs_c, actions_c, task_z_1_repeat)
                r_pred_alpha, _, next_o_pred_mean_alpha, next_o_pred_std_alpha = self.agent.c_decoder_target(obs_c_alpha, actions_c_alpha, task_z_alpha_repeat)
            else:
                r_pred, _, next_o_pred_mean, next_o_pred_std = self.agent.k_decoder(obs_c, actions_c, task_k_repeat)

        perm = np.random.permutation(self.meta_batch)
        r_1 = r_pred.view(-1, self.bisim_sample_batchsize, 1)
        r_2 = r_1[perm]
        if self.use_next_state_bisim:
            n_o_mean_1, n_o_std_1 = next_o_pred_mean.view(-1, self.bisim_sample_batchsize, self.o_dim), next_o_pred_std.view(-1, self.bisim_sample_batchsize, self.o_dim)
            n_o_mean_2, n_o_std_2 = n_o_mean_1[perm], n_o_std_1[perm]

        task_z_2 = task_z_1[perm]

        r_dist = ((r_2 - r_1).pow(2) + 1e-7).sum(dim=-1)  # ([16, 128])
        r_dist = torch.sqrt(r_dist + 1e-7) * self.r_dist_coeff
        if self.use_next_state_bisim:
            transition_dist = ((n_o_mean_2 - n_o_mean_1).pow(2) + 1e-7).sum(dim=-1) + \
                              ((n_o_std_2 - n_o_std_1).pow(2) + 1e-7).sum(dim=-1)
            transition_dist = torch.sqrt(transition_dist + 1e-7)  # ([16, 128])
            transition_dist = transition_dist * self.tr_dist_coeff
            sample_dist = (r_dist + 0.9 * transition_dist).mean(dim=-1)
        else:
            sample_dist = r_dist.mean(dim=-1)

        z_dist = torch.abs(task_z_2 - task_z_1).sum(dim=-1)
        # z_dist = ((task_z_2 - task_z_1).pow(2) + 1e-7).sum(dim=-1)
        # z_dist = torch.sqrt(z_dist + 1e-7)

        bisim_loss = F.mse_loss(z_dist, sample_dist, reduction=mse_reduction)


        # if self.use_target_c_dec:
        #     r_alpha = r_pred_alpha.view(-1, self.batch_size, 1)
        #     r_real = r_1[:self.num_fake_tasks]
        #     if self.use_next_state_bisim:
        #         n_o_mean_alpha, n_o_std_alpha = next_o_pred_mean.view(-1, self.batch_size, self.o_dim), next_o_pred_std.view(-1, self.batch_size, self.o_dim)
        #         n_o_mean_real, n_o_std_real = n_o_mean_1[:self.num_fake_tasks], n_o_std_1[:self.num_fake_tasks]
        #     task_z_real = task_z_1[:self.num_fake_tasks]
        #
        #     r_dist_alpha = ((r_alpha - r_real).pow(2) + 1e-7).sum(dim=-1)  # ([16, 128])
        #     r_dist_alpha = torch.sqrt(r_dist_alpha + 1e-7) * self.r_dist_coeff
        #     if self.use_next_state_bisim:
        #         transition_dist_alpha = ((n_o_mean_alpha - n_o_mean_real).pow(2) + 1e-7).sum(dim=-1) + \
        #                                 ((n_o_std_alpha - n_o_std_real).pow(2) + 1e-7).sum(dim=-1)
        #         transition_dist_alpha = torch.sqrt(transition_dist_alpha + 1e-7)  # ([16, 128])
        #         transition_dist_alpha = transition_dist_alpha * self.tr_dist_coeff
        #         sample_dist_alpha = (r_dist_alpha + 0.9 * transition_dist_alpha).mean(dim=-1)
        #     else:
        #         sample_dist_alpha = r_dist_alpha.mean(dim=-1)
        #
        #     z_dist_alpha = torch.abs(task_z_alpha - task_z_real.detach()).sum(dim=-1)
        #
        #     bisim_loss_alpha = F.mse_loss(z_dist_alpha, sample_dist_alpha, reduction=mse_reduction)
        #
        # else:
        #     bisim_loss_alpha = torch.zeros_like(bisim_loss)
        bisim_loss_alpha = torch.zeros_like(bisim_loss)


        return bisim_loss, bisim_loss_alpha



    def compute_disc_loss_old(self, obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat, rewards_alpha_flat, next_obs_fake_flat):

        d_real_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat], dim=-1)
        if self.use_decoder_next_state:
            d_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_fake_flat], dim=-1)
        else:
            d_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_real_flat], dim=-1)

        d_real_score = self.agent.disc(d_real_sample).mean()
        d_fake_score = self.agent.disc(d_fake_sample.detach()).mean()

        # Gragient Penalty
        with torch.no_grad():
            diric_interpol_indices_GP = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=True)
            off_ctxt_batch_for_interpol_GP = self.sample_context(diric_interpol_indices_GP, which_buffer=self.offpol_ctxt_sampling_buffer)
            c_off_for_interpol_GP = self.agent.get_context_embedding(off_ctxt_batch_for_interpol_GP)
            c_off_alpha_GP = []
            for i in range(self.num_fake_tasks):  # 5번
                alpha_GP = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                c_off_alpha_GP.append(alpha_GP @ c_off_for_interpol_GP)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
            c_off_alpha_GP = torch.cat(c_off_alpha_GP)  # ([5, 10])
            c_off_alpha_GP_repeat = self.get_repeat(c_off_alpha_GP, self.batch_size, self.l_dim)  # ([4096, 10])
            r_alpha_GP, n_o_alpha_GP, _, _ = self.agent.c_decoder(obs_real_flat, actions_real_flat, c_off_alpha_GP_repeat)

        if self.use_decoder_next_state:
            d_fake_sample_GP = torch.cat([obs_real_flat, actions_real_flat, r_alpha_GP, n_o_alpha_GP], dim=-1)
        else:
            d_fake_sample_GP = torch.cat([obs_real_flat, actions_real_flat, r_alpha_GP, next_obs_real_flat], dim=-1)
        d_fake_sample_GP.requires_grad = True
        d_fake_score_GP = self.agent.disc(d_fake_sample_GP)
        gradients = torch.autograd.grad(outputs=d_fake_score_GP, inputs=d_fake_sample_GP,
                                        grad_outputs=torch.ones(d_fake_score_GP.size()).to(ptu.device),
                                        create_graph=True, retain_graph=True)[0]  # ([256, 57])
        gradients_norm = gradients.norm(2, 1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        gradients_norm = gradients_norm.mean()

        return d_real_score, d_fake_score, gradient_penalty, gradients_norm

    def compute_generator_loss_old(self, obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat, rewards_alpha_flat, next_obs_fake_flat):

        g_real_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat], dim=-1)
        if self.use_decoder_next_state:
            g_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_fake_flat], dim=-1)
        else:
            g_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_real_flat], dim=-1)

        g_real_score = self.agent.disc(g_real_sample).mean().detach()
        g_fake_score = self.agent.disc(g_fake_sample).mean()
        w_distance = (g_real_score - g_fake_score).item()

        return g_real_score, g_fake_score, w_distance

    def compute_disc_loss(self, real_transition, fake_transition):
        # """method1"""
        # obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat = real_transition
        # rewards_fake1_flat, next_obs_fake1_flat, next_obs_real1_flat, c_off_alpha_repeat = fake_transition

        # """method2"""
        obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat = real_transition
        obs_real2_flat, actions_real2_flat, rewards_fake2_flat, next_obs_fake2_flat, next_obs_real2_flat, c_off_alpha_repeat = fake_transition

        c_off_real_repeat, c_off_alpha_repeat = c_off_real_repeat.detach().clone(), c_off_alpha_repeat.detach().clone()
        # --> 새로운걸로 클론 안해주면 밑에서 required_grad=True가 되어 c에도 gradient 계산됨. --> psi, c_decoder 학습에서 두번 미분된다는 에러 뜸

        """method1"""
        # real = [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat] if self.use_latent_in_disc else [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat]
        # fake = [obs_real1_flat, actions_real1_flat, rewards_fake1_flat]
        # if self.use_decoder_next_state:
        #     fake = fake + [next_obs_fake1_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_fake1_flat]
        # else:
        #     fake = fake + [next_obs_real1_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_real1_flat]

        """method2"""
        real = [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat] if self.use_latent_in_disc else [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat]
        fake = [obs_real2_flat, actions_real2_flat, rewards_fake2_flat]
        if self.use_decoder_next_state:
            fake = fake + [next_obs_fake2_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_fake2_flat]
        else:
            fake = fake + [next_obs_real2_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_real2_flat]

        d_real_sample = torch.cat(real, dim=-1)
        d_fake_sample = torch.cat(fake, dim=-1)

        d_real_score = self.agent.disc(d_real_sample).mean()
        d_fake_score = self.agent.disc(d_fake_sample.detach()).mean()

        if self.gan_type == "wgan":
            eps = torch.rand(d_real_sample.size(0)).view(-1, 1).repeat(1, d_real_sample.size(1)).to(ptu.device)
            GP_inter_sample = eps * d_real_sample.data + (1 - eps) * d_fake_sample.data
            GP_inter_sample.requires_grad = True
            GP_score = self.agent.disc(GP_inter_sample)
            gradients = torch.autograd.grad(outputs=GP_score, inputs=GP_inter_sample,
                                            grad_outputs=torch.ones(GP_score.size()).to(ptu.device),
                                            create_graph=True, retain_graph=True)[0]  # ([256, 57])
            gradients_norm = gradients.norm(2, 1)
            gradient_penalty = ((gradients_norm - 1) ** 2).unsqueeze(1).mean()
            gradients_norm = gradients_norm.mean()  # 사이즈확인
        else:
            gradient_penalty, gradients_norm = torch.zeros_like(d_real_score), torch.zeros_like(d_real_score.mean())

        return d_real_score, d_fake_score, gradient_penalty, gradients_norm

    def compute_generator_loss(self, real_transition, fake_transition):
        # """method1"""
        #obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat = real_transition
        #rewards_fake1_flat, next_obs_fake1_flat, next_obs_real1_flat, c_off_alpha_repeat = fake_transition

        # """method2"""
        obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat = real_transition
        obs_real2_flat, actions_real2_flat, rewards_fake2_flat, next_obs_fake2_flat, next_obs_real2_flat, c_off_alpha_repeat = fake_transition

        c_off_real_repeat, c_off_alpha_repeat = c_off_real_repeat.detach().clone(), c_off_alpha_repeat.detach().clone()  # 여기서도 해줘야되냐?

        """method1"""
        # real = [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat] if self.use_latent_in_disc else [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat]
        # fake = [obs_real1_flat, actions_real1_flat, rewards_fake1_flat]
        # if self.use_decoder_next_state:
        #     fake = fake + [next_obs_fake1_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_fake1_flat]
        # else:
        #     fake = fake + [next_obs_real1_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_real1_flat]

        """method2"""
        real = [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat, c_off_real_repeat] if self.use_latent_in_disc else [obs_real1_flat, actions_real1_flat, rewards_real1_flat, next_obs_real1_flat]
        fake = [obs_real2_flat, actions_real2_flat, rewards_fake2_flat]
        if self.use_decoder_next_state:
            fake = fake + [next_obs_fake2_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_fake2_flat]
        else:
            fake = fake + [next_obs_real2_flat, c_off_alpha_repeat] if self.use_latent_in_disc else fake + [next_obs_real2_flat]

        g_real_sample = torch.cat(real, dim=-1)
        g_fake_sample = torch.cat(fake, dim=-1)

        g_real_score = self.agent.disc(g_real_sample).mean().detach()
        g_fake_score = self.agent.disc(g_fake_sample).mean()
        w_distance = (g_real_score - g_fake_score.detach()).item()

        return g_real_score, g_fake_score, w_distance




    def compute_cycle_loss(self, c_off_alpha, c_on_alpha, tr_batch_off, tr_batch_on):
        obs_real2, actions_real2, rewards_fake2_flat, next_obs_fake2_flat = tr_batch_off
        obs_real_on, actions_real_on, rewards_fake_on_flat, next_obs_fake_on_flat = tr_batch_on

        # (2) c_off_cycle : off_pol_transition로 fake sample 만들어서 psi로 내린다음 c_off_alpha와 같아지도록
        rewards_fake2 = rewards_fake2_flat.view(-1, self.batch_size, 1)
        ind = np.random.choice(range(self.batch_size), size=self.embedding_batch_size, replace=False)
        obs_real2_half, actions_real2_half, rewards_fake2_half = obs_real2[:, ind, :], actions_real2[:, ind, :], rewards_fake2[:, ind, :]
        if self.use_decoder_next_state:
            next_obs_fake2 = next_obs_fake2_flat.view(-1, self.batch_size, self.o_dim)
            next_obs_fake2_half = next_obs_fake2[:, ind, :]

        # print("obs_real2_half", obs_real2_half.shape)  # ([5, 128, 29])
        # print("actions_real2_half", actions_real2_half.shape)  # ([5, 128, 8])
        # print("rewards_fake2_half", rewards_fake2_half.shape)  # ([5, 128, 1])


        if self.use_decoder_next_state:
            fake_context_bat_off = torch.cat( [obs_real2_half, actions_real2_half, rewards_fake2_half, next_obs_fake2_half], dim=-1)
        else:
            fake_context_bat_off = torch.cat([obs_real2_half, actions_real2_half, rewards_fake2_half], dim=-1)
        c_off_alpha_hat = self.agent.get_context_embedding(fake_context_bat_off, which_enc="psi")  # ([5, 10])

        # if self.use_cycle_detach:
        #     cycle_loss_off = F.mse_loss(c_off_alpha_hat, c_off_alpha.detach(), reduction="sum") + \
        #                      F.mse_loss(c_off_alpha_hat, c_on_alpha.detach(), reduction="sum")
        # else:
        #     cycle_loss_off = F.mse_loss(c_off_alpha_hat, c_off_alpha, reduction="sum") + \
        #                      F.mse_loss(c_off_alpha_hat, c_on_alpha, reduction="sum")

        # 사이클의 타겟을 c_off로 고정
        if self.use_cycle_detach:
            cycle_loss_off = F.mse_loss(c_off_alpha_hat, c_off_alpha.detach(), reduction="sum") + \
                             F.mse_loss(c_off_alpha_hat, c_off_alpha.detach(), reduction="sum")
        else:
            cycle_loss_off = F.mse_loss(c_off_alpha_hat, c_off_alpha, reduction="sum") + \
                             F.mse_loss(c_off_alpha_hat, c_off_alpha, reduction="sum")


        # (2) c_on_cycle : on_pol_transition로 fake sample 만들어서 psi로 내린다음 c_on_alpha와 같아지도록
        rewards_fake_on = rewards_fake_on_flat.view(-1, self.embedding_batch_size, 1)
        if self.use_decoder_next_state:
            next_obs_fake_on = next_obs_fake_on_flat.view(-1, self.embedding_batch_size, self.o_dim)

        if self.use_decoder_next_state:
            fake_context_bat_on = torch.cat( [obs_real_on, actions_real_on, rewards_fake_on, next_obs_fake_on], dim=-1)
        else:
            fake_context_bat_on = torch.cat([obs_real_on, actions_real_on, rewards_fake_on], dim=-1)
        c_on_alpha_hat = self.agent.get_context_embedding(fake_context_bat_on, which_enc="psi")  # ([5, 10])

        # if self.use_cycle_detach:
        #     cycle_loss_on = F.mse_loss(c_on_alpha_hat, c_on_alpha.detach(), reduction="sum")
        # else:
        #     cycle_loss_on = F.mse_loss(c_on_alpha_hat, c_on_alpha, reduction="sum")

        # 사이클 타겟을 c_off로 고정
        if self.use_cycle_detach:
            cycle_loss_on = F.mse_loss(c_on_alpha_hat, c_off_alpha.detach(), reduction="sum")
        else:
            cycle_loss_on = F.mse_loss(c_on_alpha_hat, c_off_alpha, reduction="sum")


        return cycle_loss_off, cycle_loss_on


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

        c_batch = self.c_buffer.sample_c(num_samples=256)  #
        # c_batch = self.agent.task_c_on_buffer.sample_c(tasks_indices=self.train_tasks, num_samples=256, randomly_sample=True)
        # c_batch = c_batch.view(-1, self.l_dim)

        # indices = np.random.choice(self.train_tasks, 256)
        # on_pol_ctxt_bat = self.sample_context(indices, which_buffer="online", b_size=128)
        # c_batch = self.agent.get_context_embedding(on_pol_ctxt_bat, use_target=False).detach()  # ([256, 10]) ?

        c_recon, mu, logvar = self.agent.c_distribution_vae(c_batch)

        c_recon_loss = F.mse_loss(c_recon, c_batch, reduction='sum')
        c_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = c_recon_loss + self.c_kl_lambda * c_kl_loss

        self.c_distribution_vae_optim.zero_grad()
        loss.backward()
        self.c_distribution_vae_optim.step()

        return loss.item(), c_recon_loss.item(), c_kl_loss.item()


    def index_to_onehot(self, indices):
        # print("indices", indices)
        onehot = torch.zeros(len(indices), len(self.train_tasks))  # [16, 100]
        for i in range(len(indices)):
            onehot[i, indices[i]] = 1
        return onehot.to(ptu.device)


    def _do_training(self, indices, current_step):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # zero out context and hidden encoder state
        # self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop, so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):

            if self.close_method == "new":
                sac_loss = self._take_step_bisim(indices, current_step)
            elif self.close_method == "old":
                sac_loss = self._take_step_bisim_closetask_old_method(indices, current_step)
            else:
                assert "choose close method"

            self.agent.detach_z()

        return sac_loss

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z)
        q2 = self.qf2(obs, actions, task_z)
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)


    def _take_step_bisim_closetask_old_method(self, indices, current_step):

        tran_batch_size = 256
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices, tran_batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...

        if self.use_c_off_rl:
            ctxt_batch = self.sample_context(indices, which_buffer="rl")  # ([16, 100, 27])
        else:
            ctxt_batch = self.sample_context(indices, which_buffer="online")  # ([16, 100, 27])
        task_c_ = self.agent.get_context_embedding(ctxt_batch, which_enc='psi').detach()

        if self.use_inter_samples:

            c_off_alpha, c_on_alpha = [], []
            closest_task_indices = []
            if self.use_closest_task and self.closest_task_method == 'total':
                c_total = self.agent.get_context_embedding(self.sample_context(self.train_tasks, which_buffer="rl"), which_enc="psi").detach()
            for i in range(self.num_fake_tasks):
                mixing_task_indices = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)
                mixing_off_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="rl")  # ([16, 100, 27])
                mixing_on_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="online")  # ([16, 100, 27])
                c_off_mixing = self.agent.get_context_embedding(mixing_off_ctxt_batch, which_enc="psi").detach()
                c_on_mixing = self.agent.get_context_embedding(mixing_on_ctxt_batch, which_enc="psi").detach()
                alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                c_off_alpha_ = alpha @ c_off_mixing
                c_on_alpha_ = alpha @ c_on_mixing
                c_off_alpha.append(c_off_alpha_)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
                c_on_alpha.append(c_on_alpha_)

                """가장 가까운 태스크 선택 : (방법1 - 섞은것들중에서 가까운태스크) // (방법2 - 전체트레인중에서 가까운태스크) """  # / output : 태스크인덱스 outof 전체 트레인 태스크인덱스
                if self.use_closest_task:  # default=False
                    if self.closest_task_method == 'mix':
                        temp_distance = ((c_off_mixing - c_off_alpha_) ** 2).sum(-1)  # [3,10] - [1,10] --(sum-1)--> [3]
                        temp_distance = torch.sqrt(temp_distance + 1e-6)
                        closest_task_idx = int(torch.argmin(temp_distance).detach().cpu().numpy())
                        closest_task_idx = mixing_task_indices[closest_task_idx]
                        closest_task_indices.append(closest_task_idx)
                    elif self.closest_task_method == 'total':
                        temp_distance = ((c_total - c_off_alpha_) ** 2).sum(-1)  # [100,10] - [1,10] --(sum-1)--> [100]
                        temp_distance = torch.sqrt(temp_distance + 1e-6)
                        closest_task_idx = int(torch.argmin(temp_distance).detach().cpu().numpy())
                        closest_task_idx = self.train_tasks[closest_task_idx]
                        closest_task_indices.append(closest_task_idx)



            c_off_alpha = torch.cat(c_off_alpha)  # ([5, 10])
            c_on_alpha = torch.cat(c_on_alpha)  # ([5, 10])

            c_off_alpha_repeat = self.get_repeat(c_off_alpha, self.batch_size, self.l_dim)
            c_on_alpha_repeat = self.get_repeat(c_on_alpha, self.batch_size, self.l_dim)

            if self.use_closest_task:
                off_tran_batch_real = self.sample_sac(closest_task_indices, self.batch_size)
            else:
                real_task_indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=False)  # 5개
                off_tran_batch_real = self.sample_sac(real_task_indices, self.batch_size)  # --> 디스크리미네이터 학습에 사용

            obs_real, actions_real, rewards_real, next_obs_real, terms_real = off_tran_batch_real  # 태스크 5개
            obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat, terms_real_flat = obs_real.view(self.num_fake_tasks * self.batch_size, -1), actions_real.view(self.num_fake_tasks * self.batch_size, -1), rewards_real.view(self.num_fake_tasks * self.batch_size, -1), next_obs_real.view(self.num_fake_tasks * self.batch_size, -1), terms_real.view(self.num_fake_tasks * self.batch_size, -1)

            with torch.no_grad():
                rewards_fake_flat, next_obs_fake_flat, _, _ = self.agent.c_decoder(obs_real_flat, actions_real_flat, c_off_alpha_repeat)

            if self.use_decoder_next_state:
                next_obs_inter = next_obs_fake_flat
            else:
                next_obs_inter = next_obs_real_flat * 1.0
            rewards_inter = rewards_fake_flat * self.reward_scale
            rewards_real_flat = rewards_real_flat * self.reward_scale

            if self.use_c_off_rl:
                c_alpha = c_off_alpha_repeat
            else:
                c_alpha = c_on_alpha_repeat

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
            in_inter = torch.cat([obs_real_flat, c_alpha], dim=1)
            policy_outputs_inter = self.agent.policy(in_inter, reparameterize=True, return_log_prob=True)
            new_actions_inter, policy_mean_inter, policy_log_std_inter, log_pi_inter = policy_outputs_inter[:4]
        else:
            log_pi_inter = torch.zeros_like(log_pi)

        ################################################################################
        if self.use_auto_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy * self.target_entropy_coeff).detach()).mean()
            if self.use_inter_samples:
                alpha_loss_inter = -(self.log_alpha_inter * (log_pi_inter + self.target_entropy * self.target_entropy_coeff).detach()).mean()
            else:
                alpha_loss_inter = torch.zeros_like(alpha_loss)
            alpha_loss_total = alpha_loss + alpha_loss_inter
            self.alpha_optimizer.zero_grad()
            alpha_loss_total.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
            alpha_inter = self.log_alpha_inter.exp()
        else:
            alpha_loss, alpha_loss_inter = torch.tensor([0.0]).mean(), torch.tensor([0.0]).mean()
            alpha = torch.tensor([self.entropy_coeff]).mean().to(ptu.device)
            alpha_inter = torch.tensor([self.entropy_coeff]).mean().to(ptu.device)
        ################################################################################

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        q2_pred = self.qf2(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        v_pred = self.vf(obs, task_c.detach())  # ([4096, 20]), ([4096, 5])  -->  ([4096, 1])
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_c)
        if self.use_inter_samples:
            q1_pred_inter = self.qf1(obs_real_flat, actions_real_flat, c_alpha)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            q2_pred_inter = self.qf2(obs_real_flat, actions_real_flat, c_alpha)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            v_pred_inter = self.vf(obs_real_flat, c_alpha)
            with torch.no_grad():
                if self.use_next_obs_beta:  # bets=0 -> use full real sample // beta=1 -> use full fake sample
                    weighted_next_obs = self.next_obs_beta * next_obs_inter + (1 - self.next_obs_beta) * next_obs_real_flat
                    target_v_values_inter = self.target_vf(weighted_next_obs, c_alpha)
                else:
                    target_v_values_inter = self.target_vf(next_obs_inter, c_alpha)

                # if not self.use_entropy_for_q_target:
                #     target_v_values_inter = target_v_values_inter + alpha_inter * log_pi_inter


        # qf and encoder update (note encoder does not get grads from policy or vf)
        rewards_flat = rewards.view(tran_batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(tran_batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        if self.use_inter_samples:
            if self.use_rewards_beta:  # bets=0 -> use full real sample // beta=1 -> use full fake sample
                rewards_inter = self.rewards_beta * rewards_inter + (1 - self.rewards_beta) * rewards_real_flat
            q_target_inter = rewards_inter + (1. - terms_real_flat) * self.discount * target_v_values_inter
            qf_loss_inter = torch.mean((q1_pred_inter - q_target_inter) ** 2) + torch.mean((q2_pred_inter - q_target_inter) ** 2)
            # qf_loss = self.inter_update_coeff * qf_loss_inter + qf_loss
        else:
            qf_loss_inter = torch.zeros_like(qf_loss)
            q1_pred_inter, q2_pred_inter, v_pred_inter, q_target_inter = [torch.zeros_like(q1_pred) for _ in range(4)]

        if self.use_inter_samples:
            if self.use_next_obs_Q_reg:
                # (1) fake sample Q
                # (1-1) fake sample 넥스트 액션 : a'_alpha ~ pi(a'_alpha | o'_alpha, c_alpha)
                next_pi_alpha_outputs = self.agent.policy(torch.cat([next_obs_inter, c_alpha], dim=1), reparameterize=True, return_log_prob=True)
                next_action_alpha, next_action_alpha_mean, next_action_alpha_log_std, next_action_alpha_log_pi = next_pi_alpha_outputs[:4]
                # (1-2) fake sample 넥스트 Q : Q(o'_alpha, a'_alpha)
                next_q1_alpha = self.qf1(next_obs_inter, next_action_alpha.detach(), c_alpha)
                next_q2_alpha = self.qf2(next_obs_inter, next_action_alpha.detach(), c_alpha)

                # (2) real sample Q
                # (2-1) real sample 넥스트 액션 : a' ~ pi(a' | o', c)
                # next_pi_real_outputs = self.agent.policy(torch.cat([next_obs_real_flat, real_c_on_repeat], dim=1), reparameterize=True, return_log_prob=True)
                # next_action_real, next_action_real_mean, next_action_real_log_std, next_action_real_log_pi = next_pi_real_outputs[:4]
                # (2-2) real sample 넥스트 Q : Q(o', a')  obs_real_flat, actions_real_flat
                next_q1_real = self.qf1(obs_real_flat, actions_real_flat, c_alpha)
                next_q2_real = self.qf2(obs_real_flat, actions_real_flat, c_alpha)

                # (3) Q regularize loss
                q_reg_loss = (next_q1_alpha + next_q2_alpha - next_q1_real - next_q2_real).mean()
            else:
                q_reg_loss = torch.zeros_like(qf_loss)
        else:
            q_reg_loss = torch.zeros_like(qf_loss)


        qf_loss_total = qf_loss \
                        + self.inter_update_coeff * qf_loss_inter \
                        + self.q_reg_coeff * q_reg_loss

        # qf_loss.backward()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss_total.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # min_q_new_actions = self._min_q(obs, new_actions, task_z)
        min_q_new_actions = self._min_q(obs, new_actions, task_c.detach())
        v_target = min_q_new_actions - alpha * log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        if self.use_inter_samples:
            min_q_new_actions_inter = self._min_q(obs_real_flat, new_actions_inter, c_alpha)
            v_target_inter = min_q_new_actions_inter - alpha_inter * log_pi_inter
            vf_loss_inter = self.vf_criterion(v_pred_inter, v_target_inter.detach())
            # vf_loss = vf_loss + self.inter_update_coeff * vf_loss_inter
        else:
            vf_loss_inter = torch.zeros_like(vf_loss)
            v_target_inter = torch.zeros_like(vf_loss)

        vf_loss_total = vf_loss + self.inter_update_coeff * vf_loss_inter

        # vf_loss.backward()
        self.vf_optimizer.zero_grad()
        vf_loss_total.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        policy_loss = (alpha * log_pi - log_policy_target).mean()

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
            policy_loss_inter = (alpha_inter * log_pi_inter - log_policy_target_inter).mean()

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

        return qf_loss.item(),       vf_loss.item(),       policy_loss.item(), \
               qf_loss_inter.item(), vf_loss_inter.item(), policy_loss_inter.item(), \
               qf_loss_total.item(), vf_loss_total.item(), policy_loss_total.item(), \
               q_reg_loss.item(), \
               q1_pred.mean().item(), q2_pred.mean().item(), v_pred.mean().item(), q_target.mean().item(), \
               q1_pred_inter.mean().item(), q2_pred_inter.mean().item(), v_pred_inter.mean().item(), q_target_inter.mean().item(), \
               v_target.mean().item(), v_target_inter.mean().item(), log_pi.mean().item(), log_pi_inter.mean().item(), \
               alpha.mean().item(), alpha_inter.mean().item(), alpha_loss.item(), alpha_loss_inter.item()






    def _take_step_bisim(self, indices, current_step):

        tran_batch_size = 256
        num_tasks = len(indices)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices, tran_batch_size)  # ([16, 256, 20]), ([16, 256, 6]), ...

        if self.use_c_off_rl:
            ctxt_batch = self.sample_context(indices, which_buffer="rl")  # ([16, 100, 27])
        else:
            ctxt_batch = self.sample_context(indices, which_buffer="online")  # ([16, 100, 27])
        task_c_ = self.agent.get_context_embedding(ctxt_batch, which_enc='psi').detach()

        if self.use_inter_samples:

            c_off_alpha, c_on_alpha = [], []
            # closest_task_indices = []
            # if self.use_sample_reg and self.closest_task_method == 'total':
            #     c_total = self.agent.get_context_embedding(self.sample_context(self.train_tasks, which_buffer="rl"), which_enc="psi").detach()
            for i in range(self.num_fake_tasks):
                mixing_task_indices = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=True)
                mixing_off_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="rl")  # ([16, 100, 27])
                mixing_on_ctxt_batch = self.sample_context(mixing_task_indices, which_buffer="online")  # ([16, 100, 27])
                c_off_mixing = self.agent.get_context_embedding(mixing_off_ctxt_batch, which_enc="psi").detach()
                c_on_mixing = self.agent.get_context_embedding(mixing_on_ctxt_batch, which_enc="psi").detach()
                alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
                c_off_alpha_ = alpha @ c_off_mixing
                c_on_alpha_ = alpha @ c_on_mixing
                c_off_alpha.append(c_off_alpha_)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
                c_on_alpha.append(c_on_alpha_)


            c_off_alpha = torch.cat(c_off_alpha)  # ([5, 10])
            c_on_alpha = torch.cat(c_on_alpha)  # ([5, 10])

            c_off_alpha_repeat = self.get_repeat(c_off_alpha, self.batch_size, self.l_dim)
            c_on_alpha_repeat = self.get_repeat(c_on_alpha, self.batch_size, self.l_dim)



            # else:
            real_task_indices = np.random.choice(self.train_tasks, self.num_fake_tasks, replace=True)  # 5개
            off_tran_batch_real = self.sample_sac(real_task_indices, self.batch_size)  # --> 디스크리미네이터 학습에 사용

            obs_real, actions_real, rewards_real, next_obs_real, terms_real = off_tran_batch_real  # 태스크 5개
            obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat, terms_real_flat = obs_real.view(self.num_fake_tasks * self.batch_size, -1), actions_real.view(self.num_fake_tasks * self.batch_size, -1), rewards_real.view(self.num_fake_tasks * self.batch_size, -1), next_obs_real.view(self.num_fake_tasks * self.batch_size, -1), terms_real.view(self.num_fake_tasks * self.batch_size, -1)

            with torch.no_grad():
                rewards_fake_flat, next_obs_fake_flat, _, _ = self.agent.c_decoder(obs_real_flat, actions_real_flat, c_off_alpha_repeat)


            if self.use_sample_reg:
                if self.sample_reg_method == 'closest':
                    c_total = self.agent.get_context_embedding(self.sample_context(self.train_tasks, which_buffer="rl"), which_enc="psi").detach()
                    closest_task_indices = []
                    for i in range(self.num_fake_tasks):
                        temp_distance = ((c_total - c_off_alpha[i].unsqueeze(0)) ** 2).sum(-1)  # [100,10] - [1,10] --(sum-1)--> [100]
                        temp_distance = torch.sqrt(temp_distance + 1e-6)
                        closest_task_idx = int(torch.argmin(temp_distance).detach().cpu().numpy())
                        closest_task_idx = self.train_tasks[closest_task_idx]
                        closest_task_indices.append(closest_task_idx)
                    # closest_ctxt_batch = self.sample_context(closest_task_indices, which_buffer="online")  # ([16, 100, 27])
                    # closest_c = self.agent.get_context_embedding(closest_ctxt_batch, which_enc='psi').detach()
                    closest_c = c_total[closest_task_indices]
                    closest_c_repeat = self.get_repeat(closest_c, self.batch_size, self.l_dim)
                    with torch.no_grad():
                        rewards_real_reg_flat, next_obs_real_reg_flat, _, _ = self.agent.c_decoder(obs_real_flat, actions_real_flat, closest_c_repeat)

                elif self.sample_reg_method == 'random':
                    rewards_real_reg_flat, next_obs_real_reg_flat = rewards_real_flat * 1.0, next_obs_real_flat * 1.0
                else:
                    rewards_real_reg_flat, next_obs_real_reg_flat = None, None

            # if self.use_decoder_next_state:
            #     next_obs_inter = next_obs_fake_flat.detach()
            # else:
            #     next_obs_inter = next_obs_real_flat * 1.0
            rewards_inter = rewards_fake_flat.detach() * self.reward_scale
            # rewards_real_flat = rewards_real_flat.detach() * self.reward_scale

            if self.use_c_off_rl:
                c_alpha = c_off_alpha_repeat
            else:
                c_alpha = c_on_alpha_repeat

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
            in_inter = torch.cat([obs_real_flat, c_alpha], dim=1)
            policy_outputs_inter = self.agent.policy(in_inter, reparameterize=True, return_log_prob=True)
            new_actions_inter, policy_mean_inter, policy_log_std_inter, log_pi_inter = policy_outputs_inter[:4]
        else:
            log_pi_inter = torch.zeros_like(log_pi)




        # print("log_pi", log_pi.shape)  # ([2048, 1])
        # print("log_pi_inter", log_pi_inter.shape)  # ([1024, 1])
        ################################################################################
        if self.use_auto_entropy:
            log_alpha = self.agent.log_alpha_net(task_c)
            alpha_loss = -(log_alpha * (log_pi + self.target_entropy * self.target_entropy_coeff).detach()).mean()
            if self.use_inter_samples:
                log_alpha_inter = self.agent.log_alpha_net(c_alpha)
                alpha_loss_inter = -(log_alpha_inter * (log_pi_inter + self.target_entropy * self.target_entropy_coeff).detach()).mean()
            else:
                alpha_loss_inter = torch.zeros_like(alpha_loss)
            alpha_loss_total = alpha_loss + alpha_loss_inter * self.inter_update_coeff
            self.alpha_optimizer.zero_grad()
            alpha_loss_total.backward()
            self.alpha_optimizer.step()

            alpha = self.agent.log_alpha_net(task_c).exp().detach()
            if self.use_inter_samples:
                alpha_inter = self.agent.log_alpha_net(c_alpha).exp().detach()
        else:
            alpha_loss, alpha_loss_inter = torch.tensor([0.0]).mean(), torch.tensor([0.0]).mean()
            alpha = torch.tensor([self.entropy_coeff]).mean().to(ptu.device)
            alpha_inter = torch.tensor([self.entropy_coeff]).mean().to(ptu.device)
        ################################################################################

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        q2_pred = self.qf2(obs, actions, task_c)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
        v_pred = self.vf(obs, task_c.detach())  # ([4096, 20]), ([4096, 5])  -->  ([4096, 1])
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_c)
        if self.use_inter_samples:
            q1_pred_inter = self.qf1(obs_real_flat, actions_real_flat, c_alpha)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            q2_pred_inter = self.qf2(obs_real_flat, actions_real_flat, c_alpha)  # ([4096, 20]), ([4096, 6]), ([4096, 5])  -->  ([4096, 1])
            v_pred_inter = self.vf(obs_real_flat, c_alpha)
            with torch.no_grad():
                if self.use_decoder_next_state:
                    if self.use_sample_reg:
                        weighted_next_obs = self.next_obs_beta * next_obs_fake_flat + (1 - self.next_obs_beta) * next_obs_real_reg_flat
                        target_v_values_inter = self.target_vf(weighted_next_obs, c_alpha)
                    else:
                        target_v_values_inter = self.target_vf(next_obs_fake_flat, c_alpha)
                else:  # 넥스트옵스 안쓰는 리워드 환경
                    target_v_values_inter = self.target_vf(next_obs_real_flat, c_alpha)

                # if not self.use_entropy_for_q_target:
                #     target_v_values_inter = target_v_values_inter + alpha_inter * log_pi_inter


        # qf and encoder update (note encoder does not get grads from policy or vf)
        rewards_flat = rewards.view(tran_batch_size * num_tasks, -1)
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(tran_batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        if self.use_inter_samples:
            if self.use_sample_reg: # bets=0 -> use full real sample // beta=1 -> use full fake sample
                rewards_inter = self.rewards_beta * rewards_inter + (1 - self.rewards_beta) * rewards_real_reg_flat
            q_target_inter = rewards_inter + (1. - terms_real_flat) * self.discount * target_v_values_inter
            qf_loss_inter = torch.mean((q1_pred_inter - q_target_inter) ** 2) + torch.mean((q2_pred_inter - q_target_inter) ** 2)
            # qf_loss = self.inter_update_coeff * qf_loss_inter + qf_loss
        else:
            qf_loss_inter = torch.zeros_like(qf_loss)
            q1_pred_inter, q2_pred_inter, v_pred_inter, q_target_inter = [torch.zeros_like(q1_pred) for _ in range(4)]

        # if self.use_inter_samples:
        #     if self.use_next_obs_Q_reg:
        #         # (1) fake sample Q
        #         # (1-1) fake sample 넥스트 액션 : a'_alpha ~ pi(a'_alpha | o'_alpha, c_alpha)
        #         next_pi_alpha_outputs = self.agent.policy(torch.cat([next_obs_inter, c_alpha], dim=1), reparameterize=True, return_log_prob=True)
        #         next_action_alpha, next_action_alpha_mean, next_action_alpha_log_std, next_action_alpha_log_pi = next_pi_alpha_outputs[:4]
        #         # (1-2) fake sample 넥스트 Q : Q(o'_alpha, a'_alpha)
        #         next_q1_alpha = self.qf1(next_obs_inter, next_action_alpha.detach(), c_alpha)
        #         next_q2_alpha = self.qf2(next_obs_inter, next_action_alpha.detach(), c_alpha)
        #
        #         # (2) real sample Q
        #         # (2-1) real sample 넥스트 액션 : a' ~ pi(a' | o', c)
        #         # next_pi_real_outputs = self.agent.policy(torch.cat([next_obs_real_flat, real_c_on_repeat], dim=1), reparameterize=True, return_log_prob=True)
        #         # next_action_real, next_action_real_mean, next_action_real_log_std, next_action_real_log_pi = next_pi_real_outputs[:4]
        #         # (2-2) real sample 넥스트 Q : Q(o', a')  obs_real_flat, actions_real_flat
        #         next_q1_real = self.qf1(obs_real_flat, actions_real_flat, c_alpha)
        #         next_q2_real = self.qf2(obs_real_flat, actions_real_flat, c_alpha)
        #
        #         # (3) Q regularize loss
        #         q_reg_loss = (next_q1_alpha + next_q2_alpha - next_q1_real - next_q2_real).mean()
        #     else:
        #         q_reg_loss = torch.zeros_like(qf_loss)
        # else:
        #     q_reg_loss = torch.zeros_like(qf_loss)
        q_reg_loss = torch.zeros_like(qf_loss)

        qf_loss_total = qf_loss \
                        + self.inter_update_coeff * qf_loss_inter \
                        + self.q_reg_coeff * q_reg_loss

        # qf_loss.backward()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss_total.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # min_q_new_actions = self._min_q(obs, new_actions, task_z)
        min_q_new_actions = self._min_q(obs, new_actions, task_c.detach())
        v_target = min_q_new_actions - alpha * log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        if self.use_inter_samples:
            min_q_new_actions_inter = self._min_q(obs_real_flat, new_actions_inter, c_alpha)
            v_target_inter = min_q_new_actions_inter - alpha_inter * log_pi_inter
            vf_loss_inter = self.vf_criterion(v_pred_inter, v_target_inter.detach())
            # vf_loss = vf_loss + self.inter_update_coeff * vf_loss_inter
        else:
            vf_loss_inter = torch.zeros_like(vf_loss)
            v_target_inter = torch.zeros_like(vf_loss)

        vf_loss_total = vf_loss + self.inter_update_coeff * vf_loss_inter

        # vf_loss.backward()
        self.vf_optimizer.zero_grad()
        vf_loss_total.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        policy_loss = (alpha * log_pi - log_policy_target).mean()

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
            policy_loss_inter = (alpha_inter * log_pi_inter - log_policy_target_inter).mean()

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

        return qf_loss.item(),       vf_loss.item(),       policy_loss.item(), \
               qf_loss_inter.item(), vf_loss_inter.item(), policy_loss_inter.item(), \
               qf_loss_total.item(), vf_loss_total.item(), policy_loss_total.item(), \
               q_reg_loss.item(), \
               q1_pred.mean().item(), q2_pred.mean().item(), v_pred.mean().item(), q_target.mean().item(), \
               q1_pred_inter.mean().item(), q2_pred_inter.mean().item(), v_pred_inter.mean().item(), q_target_inter.mean().item(), \
               v_target.mean().item(), v_target_inter.mean().item(), log_pi.mean().item(), log_pi_inter.mean().item(), \
               alpha.mean().item(), alpha_inter.mean().item(), alpha_loss.item(), alpha_loss_inter.item()



    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            psi=self.agent.psi.state_dict(),
            c_decoder=self.agent.c_decoder.state_dict(),
            k_decoder=self.agent.k_decoder.state_dict(),
            discriminator=self.agent.disc.state_dict(),
            # psi_target=self.agent.psi_target.state_dict(),
            # decoder=self.agent.decoder.state_dict(),
        )
        return snapshot