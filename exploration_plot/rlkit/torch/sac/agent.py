

import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu

import random
from collections import deque
from torch.distributions.dirichlet import Dirichlet


class TaskCBuffer:
    def __init__(self, num_tasks, capacity):
        self.buffers = [deque(maxlen=capacity) for _ in range(num_tasks)]
        # self.total_buffer = deque(maxlen=capacity * num_tasks)
        self.num_tasks = num_tasks

    def add_c(self, tasks_indices, c, c_buffer_fix=False):
        if not c_buffer_fix:
            c = c.detach()  # [4, 10]
            for i in range(len(tasks_indices)):
                self.buffers[tasks_indices[i]].append(c[i].unsqueeze(0))  # [1, 10]으로 저장
                # self.total_buffer.append(c[i].unsqueeze(0))

    def sample_c(self, tasks_indices, num_samples, all_element_sampling=False, randomly_sample=False):
        if not randomly_sample:
            if all_element_sampling:
                c_batches = [torch.cat(random.sample(self.buffers[task_idx], len(self.buffers[task_idx]))) for task_idx in tasks_indices]
                c_batches_tensor = torch.stack(c_batches).to(ptu.device)  # --> [4, 32, 10]
            else:
                c_batches = [torch.cat(random.sample(self.buffers[task_idx], num_samples)) for task_idx in tasks_indices]
                c_batches_tensor = torch.stack(c_batches).to(ptu.device)  # --> [4, 32, 10]

        else:  # 태스크와 관계없는 c 샘플
            # c_batches = torch.cat(random.sample(self.total_buffer, num_samples))
            # print("c_batches", c_batches.shape)
            num_c_per_buffer =  1 + (num_samples // self.num_tasks)  # 예1) num_samples=5000, num_tasks=100  --> num_c_per_buffer = 50+1 = 51
            #                                                        # 예2) num_samples=1,    num_tasks=100  --> num_c_per_buffer = 0+1  = 1
            c_batches = [torch.cat(random.sample(self.buffers[task_idx], num_c_per_buffer)) for task_idx in tasks_indices]
            c_batches_tensor = torch.stack(c_batches).to(ptu.device)
            c_batches_tensor = c_batches_tensor.view(-1, c_batches_tensor.size(-1))  # 예1) ([100, 51, 10]) --> ([5100, 10])
            #                                                                        # 예2) ([100, 1, 10]) --> ([100, 10])
            rand_ind = np.random.choice(range(len(c_batches_tensor)), num_samples, replace=False)
            c_batches_tensor = c_batches_tensor[rand_ind]
        return c_batches_tensor

    def check_buffers_size(self, threshold):
        """check all tasks latent variable buffers length whether over than threshold"""
        check = [len(buffer) > threshold for buffer in self.buffers ]
        return np.prod(check)



def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 psi,
                 psi_aux_vae_dec,
                 k_decoder,
                 c_decoder,
                 c_decoder_target,
                 discriminator,
                 policy,
                 log_alpha_net,
                 c_distribution_vae,
                 train_tasks,
                 **kwargs
                 ):
        super().__init__()
        self.latent_dim = latent_dim

        self.psi = psi
        self.psi_target = self.psi.copy()
        self.psi_aux_vae_dec = psi_aux_vae_dec
        self.k_decoder = k_decoder
        self.c_decoder = c_decoder
        self.c_decoder_target = c_decoder_target
        self.disc = discriminator
        self.policy = policy
        self.log_alpha_net = log_alpha_net
        self.train_tasks = train_tasks


        print("ptu.device", ptu.device)

        self.c_distribution_vae = c_distribution_vae.to(ptu.device)

        self.use_c_dist_clear = kwargs['use_c_dist_clear']
        self.use_c_vae = kwargs["use_c_vae"]

        self.use_index_rl = kwargs['use_index_rl']

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        self.num_dirichlet_tasks = kwargs['num_dirichlet_tasks']
        self.beta = kwargs['beta']
        self.dirichlet_sample_freq_for_exploration = kwargs['dirichlet_sample_freq_for_exploration']
        self.wide_exp = kwargs['wide_exp']


        self.num_meta_train_steps = kwargs['num_meta_train_steps']

        self.task_c_on_buffer  = TaskCBuffer(num_tasks=len(train_tasks), capacity=kwargs["c_off_buffer_length"])  # self.num_meta_train_steps * 10 # 태스크 당 5000개 x 150개?
        self.task_c_off_buffer = TaskCBuffer(num_tasks=len(train_tasks), capacity=kwargs["c_off_buffer_length"])

        self.c_buffer_check = False

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.register_buffer('k', torch.zeros(1, len(train_tasks)))

        logvar_min = -10  # --> log(var) = -10 --> var = exp(-10)
        logvar_max = 2   # --> log(var) = 2   --> var = exp(2)
        self.var_min = torch.tensor(logvar_min).to(ptu.device).exp()
        self.var_max = torch.tensor(logvar_max).to(ptu.device).exp()


        self.c_vae_curr_c = []
        self.c_vae_distance_margin = 0.1

        # self.clear_z()
        self.clear_z(random_task_ctxt_batch=None)

    def index_to_onehot(self, indices):
        # print("indices", indices)
        onehot = torch.zeros(len(indices), len(self.train_tasks))  # [16, 100]
        for i in range(len(indices)):
            onehot[i, indices[i]] = 1
        return onehot.to(ptu.device)


    def clear_z(self, random_task_ctxt_batch, context_clear=True, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior

        if self.use_index_rl:
            self.k = self.index_to_onehot()
            if context_clear:
                self.context = None

        else:

            if self.use_c_vae:
                c_latent = torch.randn(num_tasks, self.latent_dim).to(ptu.device)
                self.z = self.c_distribution_vae.decode(c_latent)

                self.z_means, logvar = self.c_distribution_vae.encode(self.z)
                self.z_vars = torch.exp(logvar)

                if context_clear:
                    self.context = None
                # reset any hidden state in the encoder network (relevant for RNN)
                self.psi.reset(num_tasks)
                self.psi_target.reset(num_tasks)

            else:
                # if random_task_ctxt_batch is None:
                #     mu = ptu.zeros(num_tasks, self.latent_dim)
                #     if self.use_ib:
                #         var = ptu.ones(num_tasks, self.latent_dim)
                #     else:
                #         var = ptu.zeros(num_tasks, self.latent_dim)
                #     self.z_means = mu
                #     self.z_vars = var
                #
                #     self.sample_z()
                #     # reset the context collected so far
                #     if context_clear:
                #         self.context = None
                #     # reset any hidden state in the encoder network (relevant for RNN)
                #     self.psi.reset(num_tasks)
                #     self.psi_target.reset(num_tasks)
                # else:
                #     with torch.no_grad():
                #         task_z = self.get_context_embedding(random_task_ctxt_batch, which_enc="psi")
                #         self.z = task_z
                #         self.z_means = task_z  # ([1, 10])
                #         self.z_vars = self.z_vars[0].unsqueeze(0)  # ([1, 10])
                #         if context_clear:
                #             self.context = None
                self.sample_dirichlet_c()  # self.z ~ diriclet(c세개~c_buffer) if len(buffer)>3 else self.z ~ N(0,1)
                if context_clear:
                    self.context = None



    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        # if self.recurrent:
        #     self.psi.hidden = self.psi.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        
        if self.use_next_obs_in_context:
            # print("o", o.shape)
            # print("a", a.shape)
            # print("r", r.shape)
            # print("no", no.shape)
            # o torch.Size([1, 1, 17])
            # a torch.Size([1, 1, 6])
            # r torch.Size([1, 1, 1, 1])
            # no torch.Size([1, 1, 17])
            if r.dim() == 4:
                r = r.squeeze(0)
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, which_enc="psi"):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        if which_enc == "psi":
            params = self.psi(context)
            params = params.view(context.size(0), -1, self.psi.output_size)
        elif which_enc == "psi_target":
            params = self.psi_target(context).detach()
            params = params.view(context.size(0), -1, self.psi_target.output_size)

        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])

            # 스토캐스티시티 바운드 추가
            # self.z_vars = torch.clamp(self.z_vars, self.var_min, self.var_max)
        else:
            self.z_means = torch.mean(params, dim=1)

        self.sample_z()

        # return self.z


    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_context_embedding(self, context, which_enc="psi", return_z_distribution=False):
        self.infer_posterior(context, which_enc=which_enc)
        # self.sample_z()
        task_z = self.z
        if return_z_distribution:
            return task_z, self.z_means, self.z_vars
        else:
            return task_z

    def get_action_original(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)


    #######################################################################################################
    #######################################################################################################
    def get_action(self, obs, deterministic=False):
        """ sample action from the policy, conditioned on the task embedding """
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def sample_dirichlet_c(self):
        # if self.c_buffer.size() > self.num_dirichlet_tasks:
        #     random_c = self.c_buffer.sample_c(num_samples=self.num_dirichlet_tasks)  # [3, 10]
        #     alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * self.beta - (self.beta - 1) / self.num_dirichlet_tasks
        #     self.dirichlet_c = alpha @ random_c
        # else:
        #     self.dirichlet_c = self.z

        if self.c_buffer_check:  # 10개 있는지.
            # print("dirichlet sampling sample_dirichlet_c func in agent ")

            beta = self.beta + 5 if self.wide_exp else self.beta

            random_c = self.task_c_on_buffer.sample_c(self.train_tasks, self.num_dirichlet_tasks, randomly_sample=True)  # [3, 10]
            alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample().to(ptu.device) * beta - (beta - 1) / self.num_dirichlet_tasks
            self.z = alpha @ random_c
            # self.z = self.task_c_on_buffer.sample_c(self.train_tasks, 1, randomly_sample=True)  # [1, 10]
        else:
            if self.task_c_on_buffer.check_buffers_size(threshold=10):
                self.c_buffer_check = True
            # print("normal sampling sample_dirichlet_c func in agent ")
            mu = ptu.zeros(1, self.latent_dim)
            if self.use_ib:
                var = ptu.ones(1, self.latent_dim)
            else:
                var = ptu.zeros(1, self.latent_dim)
            self.z_means = mu
            self.z_vars = var
            if self.use_ib:
                posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
                z = [d.rsample() for d in posteriors]
                self.z = torch.stack(z)
            else:
                self.z = self.z_means
    #######################################################################################################
    #######################################################################################################

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, which_enc="psi"):
        ''' given context, get statistics under the current policy of a set of observations '''

        self.infer_posterior(context, which_enc=which_enc)
        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig






    def infer_posterior_aux(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''

        # with torch.no_grad():
        params = self.psi_aux(context)
        params = params.view(context.size(0), -1, self.psi_aux.output_size)

        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        z_means = torch.stack([p[0] for p in z_params])
        z_vars = torch.stack([p[1] for p in z_params])

        # smaple_z 함수 포함됨.
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        z = [d.rsample() for d in posteriors]
        task_z = torch.stack(z)

        return task_z, z_means, z_vars

    def compute_kl_div_psi_aux(self, means, vars):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(means), torch.unbind(vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum




    @property
    def networks(self):
        return [self.psi,
                self.psi_target,
                self.psi_aux_vae_dec,
                self.k_decoder,
                self.c_decoder,
                self.c_decoder_target,
                self.disc,
                self.policy,
                self.log_alpha_net,
                self.c_distribution_vae]



