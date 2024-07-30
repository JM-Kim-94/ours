"""코드 간소화 - 2024.04.04
- alpha 비율
-
"""



from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm_revised_alpha import MetaRLAlgorithm

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

        self.psi_optim = optimizer_class(
            self.agent.psi.parameters(),
            lr=context_lr,
        )
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

        self.c_distribution_vae_optim = optimizer_class(
            self.agent.c_distribution_vae.parameters(),
            lr=0.001
        )




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
        if which_buffer == "default":
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
            t = [self.online_enc_replay_buffer.task_buffers[idx].size() > 0 for idx in
                 indices]  # 그 태스크 인덱스에 대한 온라인 버퍼가 차 있는지 확인하는 리스트
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


    ########################################################################################################
    def meta_train(self, train_step, mode):

        """ *** 0. 배치 준비 *** """

        # 트레인 태스크 리스트에서 16개(meta_batch) 선택
        indices = np.random.choice(self.train_tasks, self.meta_batch, replace=False)

        # 기본 트랜지션 배치
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices, self.batch_size)
        obs2, actions2, rewards2, next_obs2, terms2 = self.sample_sac(indices, self.batch_size)  # --> 디스크리미네이터 리얼 샘플로 들어감
        obs_flat, actions_flat, rewards_flat, next_obs_flat, terms_flat = obs.view(-1, self.o_dim), actions.view(-1, self.a_dim), rewards.view(-1, 1), next_obs.view(-1, self.o_dim), terms.view(-1, 1)
        transition_batch_flat = [obs_flat, actions_flat, rewards_flat, next_obs_flat, terms_flat]

        # 레이턴트 베리어블
        ctxt_bat_on1,  ctxt_tr_bat_on1  = self.sample_context(indices, which_buffer="online", return_unpacked=True)
        ctxt_bat_off1, ctxt_tr_bat_off1 = self.sample_context(indices, which_buffer=self.offpol_ctxt_sampling_buffer, return_unpacked=True)
        c_on1  = self.agent.get_context_embedding(ctxt_bat_on1)  # [16,10]
        c_off1 = self.agent.get_context_embedding(ctxt_bat_off1)  # 기본
        if mode == 'meta_train':
            ctxt_bat_on2  = self.sample_context(indices, which_buffer="online")
            ctxt_bat_off2 = self.sample_context(indices, which_buffer=self.offpol_ctxt_sampling_buffer)
            c_on2  = self.agent.get_context_embedding(ctxt_bat_on2)
            c_off2 = self.agent.get_context_embedding(ctxt_bat_off2)

        # 컨텍스트 트랜지션 배치 (사이클에 사용), on, off ctxt bat
        obs_on_c, actions_on_c, rewards_on_c, next_obs_on_c, _     = ctxt_tr_bat_on1
        obs_off_c, actions_off_c, rewards_off_c, next_obs_off_c, _ = ctxt_tr_bat_off1


        # 태스크 인덱스
        task_k = self.index_to_onehot(indices)  # [16, 150]

        task_k_repeat = self.get_repeat(task_k, self.batch_size, len(self.train_tasks))
        # c_on1_repeat = self.get_repeat(c_on1, self.batch_size, self.l_dim)  # ([4096, 10])
        c_off1_repeat = self.get_repeat(c_off1, self.batch_size, self.l_dim)  # ([4096, 10])

        # 디리클레 인터폴레이션
        alphas = [Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks for _ in range(self.num_fake_tasks)]
        alphas = torch.cat(alphas, dim=0).view(self.num_fake_tasks, -1, 1)  # [5, 3, 1]

        alphas_prime = []
        for one_alpha in torch.unbind(alphas):
            negative_flag, denomi = False, 1
            for i in range(len(one_alpha)):
                if one_alpha[i] < 0:
                    negative_flag = True
                    denomi = denomi + torch.abs(one_alpha[i])
            if negative_flag:
                one_alpha_temp = torch.where(one_alpha < 0, 0, one_alpha / denomi)
            else:
                one_alpha_temp = one_alpha
            alphas_prime.append(one_alpha_temp)
        alphas_prime = torch.cat(alphas_prime).view(self.num_fake_tasks, -1, 1)


        # ------------- 16개 중 15개 선택 --------------

        # 0~15 16개 중에 랜덤으로 15개 (재사용)
        indices_alpha = np.random.choice(range(len(indices)), self.num_fake_tasks * self.num_dirichlet_tasks, replace=True)

        # 기본 배치 (15개 태스크) # 이거에 대한 c는 아래 c_on1_real, c_off1_real, c_off1_real (세개는 같음) / c_alpha_diric_off_repeat
        obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat, terms_real_flat = obs[indices_alpha].view(-1, self.o_dim), actions[indices_alpha].view(-1, self.a_dim), rewards[indices_alpha].view(-1, 1), next_obs[indices_alpha].view(-1, self.o_dim), terms[indices_alpha].view(-1, 1)
        obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, terms_real2_flat = obs2[indices_alpha].view(-1, self.o_dim), actions2[indices_alpha].view(-1, self.a_dim), rewards2[indices_alpha].view(-1, 1), next_obs2[indices_alpha].view(-1, self.o_dim), terms2[indices_alpha].view(-1, 1)
        # obs_real2_flat <- 디스크리미네이터 리얼 샘플로 들어감

        # 컨텍스트 tr 배치 --> 사이클에 사용 on, off
        obs_on_c_real_flat, actions_on_c_real_flat, rewards_on_c_real_flat, next_obs_on_c_real_flat = obs_on_c[indices_alpha].view(-1, self.o_dim), actions_on_c[indices_alpha].view(-1, self.a_dim), rewards_on_c[indices_alpha].view(-1, 1), next_obs_on_c[indices_alpha].view(-1, self.o_dim)
        obs_off_c_real_flat, actions_off_c_real_flat, rewards_off_c_real_flat, next_obs_off_c_real_flat = obs_off_c[indices_alpha].view(-1, self.o_dim), actions_off_c[indices_alpha].view(-1, self.a_dim), rewards_off_c[indices_alpha].view(-1, 1), next_obs_off_c[indices_alpha].view(-1, self.o_dim)
        # ctxt_on_tr_bat_real_flat = [obs_on_c_real_flat, actions_on_c_real_flat, rewards_on_c_real_flat, next_obs_on_c_real_flat]
        # ctxt_off_tr_bat_real_flat = [obs_off_c_real_flat, actions_off_c_real_flat, rewards_off_c_real_flat, next_obs_off_c_real_flat]

        # [16, 10] --> [15, 10] --> [5, 3, 10]
        c_on1_real = c_on1[indices_alpha].view(self.num_fake_tasks, self.num_dirichlet_tasks, self.l_dim)
        c_off1_real = c_off1[indices_alpha].view(self.num_fake_tasks, self.num_dirichlet_tasks, self.l_dim)

        c_alpha_diric_on = (alphas * c_on1_real).sum(dim=1)  # [5, 3, 1] * [5, 3, 10] -> [5, 3, 10] --> [5, 10]
        # [5, 10] -> [5, 1, 10] -> [5, 3*256, 10] -> [5, 3, 256, 10] --> [5*3*256, 10]
        c_alpha_diric_on_repeat = c_alpha_diric_on.unsqueeze(1).repeat(1, self.num_dirichlet_tasks * self.batch_size, 1).view(self.num_fake_tasks, self.num_dirichlet_tasks, self.batch_size, -1)
        c_alpha_diric_on_repeat = c_alpha_diric_on_repeat.view(-1, self.l_dim)

        c_alpha_diric_off = (alphas * c_off1_real).sum(dim=1)  # [5,10]
        c_alpha_diric_off_repeat = c_alpha_diric_off.unsqueeze(1).repeat(1, self.num_dirichlet_tasks * self.batch_size, 1).view(self.num_fake_tasks, self.num_dirichlet_tasks, self.batch_size, -1)
        c_alpha_diric_off_repeat = c_alpha_diric_off_repeat.view(-1, self.l_dim)  # [5*3*256, 10]
        c_alpha_diric_off_half_repeat = c_alpha_diric_off.unsqueeze(1).repeat(1, self.num_dirichlet_tasks * self.embedding_batch_size, 1).view(self.num_fake_tasks, self.num_dirichlet_tasks, self.embedding_batch_size, -1)
        c_alpha_diric_off_half_repeat = c_alpha_diric_off_half_repeat.view(-1, self.l_dim)  # [5*3*128, 10]

        # 사이클에 사용  # [5, 10] -> [5, 1, 10] -> [5, 3, 10] -> [5*3, 10]
        c_alpha_diric_on  = c_alpha_diric_on.unsqueeze(1).repeat(1, self.num_dirichlet_tasks, 1).view(self.num_fake_tasks * self.num_dirichlet_tasks, -1)
        c_alpha_diric_off = c_alpha_diric_off.unsqueeze(1).repeat(1, self.num_dirichlet_tasks, 1).view(self.num_fake_tasks * self.num_dirichlet_tasks, -1)


        c_off1_real = c_off1[indices_alpha]  # 15개 --> discriminator 실제 샘플에 사용
        c_off1_real_repeat = self.get_repeat(c_off1_real, self.batch_size, self.l_dim)  # ([4096, 10])

        # ------------- 16개 중 15개 선택 --------------


        # ----------------------------------------------------
        # ------------- 디코더 활용 fake 데이터 생성 --------------
        # (트레이닝 c로 학습하는 것들 - recon, bisim, same c, onpol c - 각 함수 내에서 처리)
        # (c_alpha 사용하는 것들 - disc, gen, cycle, fake rl - 여기서 디코더로 만들어 사용)

        rewards_fake_flat, next_obs_fake_flat, _, _ = self.agent.c_decoder(obs_real_flat, actions_real_flat, c_alpha_diric_off_repeat)  # ([1280, 1])
        fake_transition = [obs_real_flat, actions_real_flat, rewards_fake_flat, next_obs_fake_flat, next_obs_real_flat, c_alpha_diric_off_repeat]
        real_transition = [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_off1_real_repeat]
        if mode == 'meta_train':
            # 사이클 할때 필요
            rewards_off_c_fake_flat, next_obs_off_c_fake_flat, _, _ = self.agent.c_decoder(obs_off_c_real_flat, actions_off_c_real_flat, c_alpha_diric_off_half_repeat)  # ([1280, 1])
            rewards_on_c_fake_flat, next_obs_on_c_fake_flat, _, _   = self.agent.c_decoder(obs_on_c_real_flat, actions_on_c_real_flat, c_alpha_diric_off_half_repeat)  # ([1280, 1])
            tr_batch_off = obs_off_c_real_flat, actions_off_c_real_flat, rewards_off_c_fake_flat, next_obs_off_c_fake_flat
            tr_batch_on = obs_on_c_real_flat, actions_on_c_real_flat, rewards_on_c_fake_flat, next_obs_on_c_fake_flat
        # ------------- 디코더 활용 fake 데이터 생성 --------------
        # ----------------------------------------------------



        if mode == 'meta_train':

            self.c_buffer.add_c(c_on1)
            if train_step % self.c_distri_vae_train_freq == 0:
                c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = self.train_c_distribution_vae()
            else:
                c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = 0.0, 0.0, 0.0

            """ (1) 기본 representation loss 계산 """
            r_recon_loss_k, r_recon_loss_c, n_o_recon_loss_k, n_o_recon_loss_c = self.compute_recon_loss(transition_batch_flat, c_off1_repeat, task_k_repeat)
            r_recon_loss   = r_recon_loss_k   + r_recon_loss_c
            n_o_recon_loss = n_o_recon_loss_k + n_o_recon_loss_c

            bisim_c_loss     = self.compute_bisim_loss(c_off1, task_k_repeat)
            same_task_c_loss = F.mse_loss(c_off1, c_off2) + F.mse_loss(c_on1, c_on2)
            on_pol_c_loss    = F.mse_loss(c_off1, c_on1)


            """ (2) fake sample 활용한 representation loss 계산 """
            #                                                                                     obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat
            d_real_score, d_fake_score, gradient_penalty, gradients_norm = self.compute_disc_loss(real_transition, fake_transition)
            d_total_loss = (- d_real_score + d_fake_score).view(self.num_fake_tasks, self.num_dirichlet_tasks, self.batch_size) * alphas_prime  # [5, 3, 256] * [5, 3, 1]
            d_total_loss = d_total_loss.sum(dim=1).mean() + self.wgan_lambda * gradient_penalty.mean()
            self.disc_optim.zero_grad()
            d_total_loss.backward()
            self.disc_optim.step()

            if train_step % self.gen_freq == 0:  #
                g_real_score, g_fake_score, w_distance = self.compute_generator_loss(real_transition, fake_transition)
                g_total_loss = - g_fake_score.view(self.num_fake_tasks, self.num_dirichlet_tasks, self.batch_size) * alphas_prime
                g_total_loss = g_total_loss.sum(dim=1).mean()
                w_distance = w_distance.mean()
            else:
                g_total_loss, g_real_score, g_fake_score, w_distance = [torch.zeros_like(d_total_loss) for _ in range(4)]

            # # (3) fake sample cycle loss 계산
            cycle_loss_off, cycle_loss_on = self.compute_cycle_loss(c_alpha_diric_off, c_alpha_diric_on, tr_batch_off, tr_batch_on)
            cycle_total_loss = (cycle_loss_off + cycle_loss_on).view(self.num_fake_tasks, self.num_dirichlet_tasks, -1) * alphas_prime
            cycle_total_loss = cycle_total_loss.sum(dim=1).mean()



            """ (3) representation loss 업데이트 """
            total_loss = self.recon_coeff * (r_recon_loss + n_o_recon_loss) + \
                         self.bisim_coeff * bisim_c_loss + \
                         self.same_c_coeff * same_task_c_loss + \
                         self.onpol_c_coeff * on_pol_c_loss + \
                         self.gen_coeff * g_total_loss + \
                         self.cycle_coeff * cycle_total_loss

            self.psi_optim.zero_grad()
            self.k_decoder_optim.zero_grad()
            self.c_decoder_optim.zero_grad()
            total_loss.backward()
            self.psi_optim.step()
            self.k_decoder_optim.step()
            self.c_decoder_optim.step()


            return total_loss.item(), r_recon_loss.item(), n_o_recon_loss.item(), \
                r_recon_loss_k.item(), r_recon_loss_c.item(), \
                n_o_recon_loss_k.item(), n_o_recon_loss_c.item(), \
                bisim_c_loss.item(), same_task_c_loss.item(), on_pol_c_loss.item(), \
                c_vae_loss, c_vae_recon_loss, c_vae_kl_loss, \
                d_total_loss.item(), d_real_score.mean().item(), d_fake_score.mean().item(), gradient_penalty.mean().item(), gradients_norm.mean().item(), \
                g_total_loss.item(), g_real_score.mean().item(), g_fake_score.mean().item(), w_distance.item(), \
                cycle_loss_off.mean().item(), cycle_loss_on.mean().item()


        elif mode == "rl":
            c_off1_repeat = c_off1_repeat.detach()
            c_alpha_diric_on_repeat = c_alpha_diric_on_repeat.detach()

            # real task 액션 뽑기
            pi_input_train = torch.cat([obs_flat, c_off1_repeat], dim=1)
            pi_outputs_train = self.agent.policy(pi_input_train, reparameterize=True, return_log_prob=True)
            new_actions_train, pi_mean_train, pi_log_std_train, log_pi_train = pi_outputs_train[:4]

            if self.use_inter_samples:
                # alpha task 액션 뽑기
                pi_input_alpha = torch.cat([obs_real_flat, c_alpha_diric_on_repeat], dim=1)
                pi_outputs_alpha = self.agent.policy(pi_input_alpha, reparameterize=True, return_log_prob=True)
                new_actions_alpha, pi_mean_alpha, pi_log_std_alpha, log_pi_alpha = pi_outputs_alpha[:4]

            # (1) Q train
            q1_pred_train = self.qf1(obs_flat, actions_flat, c_off1_repeat)
            q2_pred_train = self.qf2(obs_flat, actions_flat, c_off1_repeat)
            v_pred_train = self.vf(obs_flat, c_off1_repeat)
            with torch.no_grad():
                target_v_values_train = self.target_vf(next_obs_flat, c_off1_repeat)
            q_target_train = rewards_flat * self.reward_scale + (1. - terms_flat) * self.discount * target_v_values_train
            qf_loss = torch.mean((q1_pred_train - q_target_train) ** 2) + torch.mean((q2_pred_train - q_target_train) ** 2)

            if self.use_inter_samples:
                q1_pred_alpha = self.qf1(obs_real_flat, actions_real_flat, c_alpha_diric_on_repeat)
                q2_pred_alpha = self.qf2(obs_real_flat, actions_real_flat, c_alpha_diric_on_repeat)
                v_pred_alpha = self.vf(obs_real_flat, c_alpha_diric_on_repeat)

                # 타겟값 및 페널티 계산
                if self.use_decoder_next_state:
                    target_v_values_alpha = self.target_vf(next_obs_fake_flat, c_alpha_diric_on_repeat)
                else:
                    target_v_values_alpha = self.target_vf(next_obs_real_flat, c_alpha_diric_on_repeat)
                if self.use_penalty:
                    real_score, fake_score, w_distance = self.compute_generator_loss(real_transition, fake_transition)
                    penalty = (w_distance - w_distance.min()) / (w_distance.max() - w_distance.min())  # normalized w_distance
                else:
                    penalty = torch.zeros_like(rewards_fake_flat)

                q_target_alpha = + rewards_fake_flat * self.reward_scale \
                                 - penalty * self.bisim_penalty_coeff \
                                 + (1. - terms_real_flat) * self.discount * target_v_values_alpha.detach()

                qf_loss_inter = (q1_pred_alpha - q_target_alpha) ** 2 + (q2_pred_alpha - q_target_alpha) ** 2

                qf_loss_inter = qf_loss_inter.view(self.num_fake_tasks, self.num_dirichlet_tasks, -1) * alphas_prime
                qf_loss_inter = qf_loss_inter.sum(dim=1).mean()
                # qf_loss_inter = qf_loss_inter.mean()
            else:
                qf_loss_inter = torch.zeros_like(qf_loss)

            q_total_loss = qf_loss + qf_loss_inter * self.inter_update_coeff

            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
            q_total_loss.backward()
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()

            min_q_new_actions_train = self._min_q(obs_flat, new_actions_train, c_off1_repeat)
            v_target_train = min_q_new_actions_train - log_pi_train
            vf_loss = torch.mean((v_pred_train - v_target_train.detach()).pow(2))
            if self.use_inter_samples:
                min_q_new_actions_alpha = self._min_q(obs_real_flat, new_actions_alpha, c_alpha_diric_on_repeat)
                v_target_alpha = min_q_new_actions_alpha - log_pi_alpha
                vf_loss_inter = (v_pred_alpha - v_target_alpha.detach()).pow(2)

                vf_loss_inter = vf_loss_inter.view(self.num_fake_tasks, self.num_dirichlet_tasks, -1) * alphas_prime
                vf_loss_inter = vf_loss_inter.sum(dim=1).mean()
                # vf_loss_inter = vf_loss_inter.mean()
            else:
                vf_loss_inter = torch.zeros_like(vf_loss)

            v_total_loss = vf_loss + self.inter_update_coeff * vf_loss_inter

            self.vf_optimizer.zero_grad()
            v_total_loss.backward()
            self.vf_optimizer.step()
            self._update_target_network()

            log_policy_target_train = min_q_new_actions_train
            pi_loss = (log_pi_train - log_policy_target_train).mean()
            # new_actions_train, pi_mean_train, pi_log_std_train, log_pi_train = pi_outputs_train[:4]
            mean_reg_loss = self.policy_mean_reg_weight * (pi_mean_train ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (pi_log_std_train ** 2).mean()
            pre_tanh_value = pi_outputs_train[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value ** 2).sum(dim=1).mean()
            )
            pi_reg_loss_train = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            pi_loss = pi_loss + pi_reg_loss_train

            if self.use_inter_samples:
                log_policy_target_alpha = min_q_new_actions_alpha
                pi_loss_inter = log_pi_alpha - log_policy_target_alpha  # mean()추가함
                # pi_loss_inter = pi_loss_inter.mean()

                # alpha mean()
                # mean_reg_loss_alpha = self.policy_mean_reg_weight * (pi_mean_alpha ** 2)  # .mean()
                # std_reg_loss_alpha = self.policy_std_reg_weight * (pi_log_std_alpha ** 2)  # .mean()
                # pre_tanh_value_alpha = pi_outputs_alpha[-1]
                # pre_activation_reg_loss_alpha = self.policy_pre_activation_weight * (
                #     (pre_tanh_value_alpha ** 2).sum(dim=1)  # .mean()
                # )
                # pi_reg_loss_alpha = mean_reg_loss_alpha.mean(dim=1) + std_reg_loss_alpha.mean(dim=1) + pre_activation_reg_loss_alpha
                # pi_reg_loss_alpha = pi_reg_loss_alpha.unsqueeze(1)

                # 그냥 mean()
                # mean_reg_loss_alpha = self.policy_mean_reg_weight * (pi_mean_alpha ** 2).mean()
                # std_reg_loss_alpha = self.policy_std_reg_weight * (pi_log_std_alpha ** 2).mean()
                # pre_tanh_value_alpha = pi_outputs_alpha[-1]
                # pre_activation_reg_loss_alpha = self.policy_pre_activation_weight * (
                #     (pre_tanh_value_alpha ** 2).sum(dim=1).mean()
                # )
                # pi_reg_loss_alpha = mean_reg_loss_alpha + std_reg_loss_alpha + pre_activation_reg_loss_alpha
                pi_loss_inter = pi_loss_inter.view(self.num_fake_tasks, self.num_dirichlet_tasks, -1) * alphas_prime
                pi_loss_inter = pi_loss_inter.sum(dim=1).mean()
            else:
                pi_loss_inter = torch.zeros_like(pi_loss)

            pi_total_loss = pi_loss + self.inter_update_coeff * pi_loss_inter
            self.policy_optimizer.zero_grad()
            pi_total_loss.backward()
            self.policy_optimizer.step()

            return qf_loss.item(), vf_loss.item(), pi_loss.item(), \
                   qf_loss_inter.item(), vf_loss_inter.item(), pi_loss_inter.item(), \
                   q_total_loss.item(), v_total_loss.item(), pi_total_loss.item(), \
                   rewards_real_flat.mean().item(), rewards_fake_flat.mean().item(), penalty.mean().item()

    ########################################################################################################


    def compute_recon_loss(self, transition_batch_flat, task_z_repeat, task_k_repeat):

        obs, actions, rewards, next_obs, terms = transition_batch_flat
        rewards_pred_k, next_obs_pred_k, _, _ = self.agent.k_decoder(obs, actions, task_k_repeat)  # ([4096, 20])
        rewards_pred_c, next_obs_pred_c, _, _ = self.agent.c_decoder(obs, actions, task_z_repeat)  # ([4096, 20])

        reward_recon_loss_k = F.mse_loss(rewards, rewards_pred_k)
        reward_recon_loss_c = F.mse_loss(rewards, rewards_pred_c)
        if self.use_decoder_next_state:
            next_obs_recon_loss_k = F.mse_loss(next_obs, next_obs_pred_k)
            next_obs_recon_loss_c = F.mse_loss(next_obs, next_obs_pred_c)
        else:
            next_obs_recon_loss_k = torch.Tensor([0]).mean().to(ptu.device)
            next_obs_recon_loss_c = torch.Tensor([0]).mean().to(ptu.device)

        return reward_recon_loss_k, reward_recon_loss_c, next_obs_recon_loss_k, next_obs_recon_loss_c


    def compute_bisim_loss(self, task_z_1, task_k_repeat):  # task_z_1 : [16, 10]  /// task_k_repeat :
        # obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(len(tasks_indices), batch_size=tran_bsize)  # [16, 128, 20], ...

        obs_c, actions_c, rewards_c, next_obs_c, terms_c = self.sample_total_task_transition(self.meta_batch, batch_size=self.batch_size)
        obs_c, actions_c = obs_c[0], actions_c[0]
        obs_c, actions_c = obs_c.repeat(self.meta_batch, 1, 1).view(-1, self.o_dim), actions_c.repeat(self.meta_batch, 1, 1).view(-1, self.a_dim)

        with torch.no_grad():
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

        bisim_loss = F.mse_loss(z_dist, sample_dist, reduction="mean")

        return bisim_loss

    def compute_disc_loss(self, real_transition, fake_transition):
        obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_real_flat = real_transition
        obs_real_flat, actions_real_flat, rewards_fake_flat, next_obs_fake_flat, next_obs_real_flat, c_alpha_flat = fake_transition

        c_real_flat, c_alpha_flat = c_real_flat.detach().clone(), c_alpha_flat.detach().clone()
        # --> 새로운걸로 클론 안해주면 밑에서 required_grad=True가 되어 c에도 gradient 계산됨. --> psi, c_decoder 학습에서 두번 미분된다는 에러 뜸

        real = [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_real_flat] if self.use_latent_in_disc else [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat]
        fake = [obs_real_flat, actions_real_flat, rewards_fake_flat]
        if self.use_decoder_next_state:
            fake = fake + [next_obs_fake_flat, c_alpha_flat] if self.use_latent_in_disc else fake + [next_obs_fake_flat]
        else:
            fake = fake + [next_obs_real_flat, c_alpha_flat] if self.use_latent_in_disc else fake + [next_obs_real_flat]


        d_real_sample = torch.cat(real, dim=-1)
        d_fake_sample = torch.cat(fake, dim=-1)

        # d_real_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat], dim=-1)
        # if self.use_decoder_next_state:
        #     d_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_fake_flat], dim=-1)
        # else:
        #     d_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_real_flat], dim=-1)

        d_real_score = self.agent.disc(d_real_sample)
        d_fake_score = self.agent.disc(d_fake_sample.detach())

        # Gradient Penalty
        # with torch.no_grad():
        #     diric_interpol_indices_GP = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)
        #     off_ctxt_batch_for_interpol_GP = self.sample_context(diric_interpol_indices_GP, which_buffer=self.offpol_ctxt_sampling_buffer)
        #     c_off_for_interpol_GP = self.agent.get_context_embedding(off_ctxt_batch_for_interpol_GP)
        #     c_off_alpha_GP = []
        #     for i in range(self.num_fake_tasks):  # 5번
        #         alpha_GP = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
        #         c_off_alpha_GP.append(alpha_GP @ c_off_for_interpol_GP)  # ([1, 3]) @ ([3, 10]) --> ([1, 10])
        #     c_off_alpha_GP = torch.cat(c_off_alpha_GP)  # ([5, 10])
        #     c_off_alpha_GP_repeat = self.get_repeat(c_off_alpha_GP, self.batch_size, self.l_dim)  # ([4096, 10])
        #     r_alpha_GP, n_o_alpha_GP, _, _ = self.agent.c_decoder(obs_real_flat, actions_real_flat, c_off_alpha_GP_repeat)
        # if self.use_decoder_next_state:
        #     d_fake_sample_GP = torch.cat([obs_real_flat, actions_real_flat, r_alpha_GP, n_o_alpha_GP], dim=-1)
        # else:
        #     d_fake_sample_GP = torch.cat([obs_real_flat, actions_real_flat, r_alpha_GP, next_obs_real_flat], dim=-1)
        # d_fake_sample_GP.requires_grad = True
        # d_fake_score_GP = self.agent.disc(d_fake_sample_GP)
        # gradients = torch.autograd.grad(outputs=d_fake_score_GP, inputs=d_fake_sample_GP,
        #                                 grad_outputs=torch.ones(d_fake_score_GP.size()).to(ptu.device),
        #                                 create_graph=True, retain_graph=True)[0]  # ([256, 57])
        # gradients_norm = gradients.norm(2, 1)
        # gradient_penalty = ((gradients_norm - 1) ** 2)
        # gradients_norm = gradients_norm.mean()
        if self.gan_type == "wgan":
            eps = torch.rand(d_real_sample.size(0)).view(-1, 1).repeat(1, d_real_sample.size(1)).to(ptu.device)
            GP_inter_sample = eps * d_real_sample.data + (1 - eps) * d_fake_sample.data
            GP_inter_sample.requires_grad = True
            GP_score = self.agent.disc(GP_inter_sample)
            gradients = torch.autograd.grad(outputs=GP_score, inputs=GP_inter_sample,
                                            grad_outputs=torch.ones(GP_score.size()).to(ptu.device),
                                            create_graph=True, retain_graph=True)[0]  # ([256, 57])
            gradients_norm = gradients.norm(2, 1)
            gradient_penalty = ((gradients_norm - 1) ** 2).unsqueeze(1)  # .mean()
            gradients_norm = gradients_norm.mean()  # 사이즈확인
        else:
            gradient_penalty, gradients_norm = torch.zeros_like(d_real_score), torch.zeros_like(d_real_score.mean())

        return d_real_score, d_fake_score, gradient_penalty, gradients_norm

    def compute_generator_loss(self, real_transition, fake_transition):

        obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_real_flat = real_transition
        obs_real_flat, actions_real_flat, rewards_fake_flat, next_obs_fake_flat, next_obs_real_flat, c_alpha_flat = fake_transition

        real = [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat, c_real_flat] if self.use_latent_in_disc else [obs_real2_flat, actions_real2_flat, rewards_real2_flat, next_obs_real2_flat]
        fake = [obs_real_flat, actions_real_flat, rewards_fake_flat]
        if self.use_decoder_next_state:
            fake = fake + [next_obs_fake_flat, c_alpha_flat] if self.use_latent_in_disc else fake + [next_obs_fake_flat]
        else:
            fake = fake + [next_obs_real_flat, c_alpha_flat] if self.use_latent_in_disc else fake + [next_obs_real_flat]

        g_real_sample = torch.cat(real, dim=-1)
        g_fake_sample = torch.cat(fake, dim=-1)

        # g_real_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_real_flat, next_obs_real_flat], dim=-1)
        # if self.use_decoder_next_state:
        #     g_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_fake_flat], dim=-1)
        # else:
        #     g_fake_sample = torch.cat([obs_real_flat, actions_real_flat, rewards_alpha_flat, next_obs_real_flat], dim=-1)

        g_real_score = self.agent.disc(g_real_sample).detach()
        g_fake_score = self.agent.disc(g_fake_sample)
        w_distance = (g_real_score - g_fake_score).detach()

        return g_real_score, g_fake_score, w_distance



    def compute_cycle_loss(self, c_off_alpha, c_on_alpha, tr_batch_off, tr_batch_on):
        # obs_off_c_real_flat, actions_off_c_real_flat, rewards_off_c_fake_flat, next_obs_off_c_fake_flat = tr_batch_off
        # obs_on_c_real_flat, actions_on_c_real_flat, rewards_on_c_fake_flat, next_obs_on_c_fake_flat = tr_batch_on
        tr_batch_off = [sample.view(self.num_fake_tasks * self.num_dirichlet_tasks, self.embedding_batch_size, -1) if sample is not None else sample for sample in tr_batch_off]
        tr_batch_on = [sample.view(self.num_fake_tasks * self.num_dirichlet_tasks, self.embedding_batch_size, -1) if sample is not None else sample for sample in tr_batch_on]

        if self.use_decoder_next_state:
            fake_context_bat_off = torch.cat( tr_batch_off, dim=-1)
        else:
            fake_context_bat_off = torch.cat(tr_batch_off[:-1], dim=-1)
        c_off_alpha_hat = self.agent.get_context_embedding(fake_context_bat_off, use_target=False)  # ([5, 10])

        # cycle_loss_off = (F.mse_loss(c_off_alpha_hat, c_off_alpha, reduction="sum")
        #                 + F.mse_loss(c_off_alpha_hat, c_on_alpha, reduction="sum"))
        cycle_loss_off = (c_off_alpha_hat - c_off_alpha).pow(2) + \
                         (c_off_alpha_hat - c_on_alpha).pow(2)


        if self.use_decoder_next_state:
            fake_context_bat_on = torch.cat( tr_batch_on, dim=-1)
        else:
            fake_context_bat_on = torch.cat(tr_batch_on[:-1], dim=-1)
        c_on_alpha_hat = self.agent.get_context_embedding(fake_context_bat_on, use_target=False)  # ([5, 10])

        # cycle_loss_on = F.mse_loss(c_on_alpha_hat, c_on_alpha, reduction="sum")
        cycle_loss_on = (c_on_alpha_hat - c_on_alpha).pow(2)

        return cycle_loss_off, cycle_loss_on








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


    def index_to_onehot(self, indices):
        # print("indices", indices)
        onehot = torch.zeros(len(indices), len(self.train_tasks))  # [16, 100]
        for i in range(len(indices)):
            onehot[i, indices[i]] = 1
        return onehot.to(ptu.device)


    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z)
        q2 = self.qf2(obs, actions, task_z)
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)



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