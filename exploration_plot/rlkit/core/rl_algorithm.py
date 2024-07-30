import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np
import os
import shutil

import torchvision.io as vision_io
import torch
from torch.distributions.dirichlet import Dirichlet

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu

# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from rlkit.core.latent_vectors import LatentVectors, EnvType

# pip install git+https://github.com/jorvis/Multicore-TSNE

import pickle

import matplotlib.pyplot as plt

# from torch.utils.tensorboard import SummaryWriter

import wandb
import datetime

import random
from collections import deque


class EpisodicOnlineReplayBuffer:
    def __init__(self, num_task, max_episode_per_buffer=10):
        self.num_task = num_task
        self.max_episode_per_buffer = max_episode_per_buffer
        self.buffers = [deque(maxlen=max_episode_per_buffer) for i in range(num_task)]

    def add_context(self, task_index, context):  # context = ([1, 446, 36])
        self.buffers[task_index].append(context)

    def sample_context(self, task_indices, batch_size):  # output : ([16, 128, 36])
        batches = []
        for idx in task_indices:
            random_epi_context_in_buffer = random.sample(self.buffers[idx], 1)[0]  # ([1, 446, 36]) --> 이제 여기서 128개 뽑으면 되는거임.
            rand_tran_indices = np.random.choice(range(random_epi_context_in_buffer.size(1)), batch_size)
            batches.append(random_epi_context_in_buffer[:, rand_tran_indices, :])  # ([1, 128, 36])
        batches = torch.cat(batches)
        return batches


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            indistribution_tasks,
            tsne_tasks,
            train_tsne_tasks,
            test_tsne_tasks,
            total_tasks_dict_list,

            use_state_noise=True,
            use_new_batch_4_fake=False,

            ood="inter",
            env_name_suffix="",

            recon_coeff=200,
            onpol_c_coeff=200,
            same_c_coeff=200,
            cycle_coeff=10,
            gen_coeff=1,

            bisim_penalty_coeff=1,

            which_sac_file='',

            use_full_interpolation=True,

            bisim_coeff=50,

            num_dirichlet_tasks=4,
            num_fake_tasks=5,

            r_dist_coeff=1,
            tr_dist_coeff=1,
            policy_kl_reg_coeff=0,
            pretrain_tsne_freq=5000,

            use_c_dist_clear=True,

            sa_perm=False,

            gan_type=None,
            use_gan=False,

            make_prior_to_rl=False,
            fakesample_cycle=False,

            use_decrease_mask=False,
            use_fakesample_representation=False,

            offpol_ctxt_sampling_buffer="rl",

            c_distri_vae_train_freq=50,

            fakesample_rl_tran_batch_size=128,

            sample_dist_coeff=1,
            use_decoder_next_state=False,
            use_next_state_bisim=False,

            c_kl_lambda=0.1,

            decrease_rate=2,

            bisim_r_coef=1,
            bisim_dist_coef=1,

            c_buffer_size=2000,
            c_batch_num=500,

            beta=2,

            use_z_autoencoder=False,

            z_dist_compute_method="euclidian",
            env_name="cheetah-vel",

            algorithm="ours",

            same_task_loss_pow=1,

            clear_enc_buffer=1,
            prior_enc_buffer_size=50000,
            online_enc_buffer_size=10000,

            pretrain_steps=50000,

            wgan_lambda=10,
            gen_freq=5,

            use_context_buffer=1,

            use_W=1,

            use_q_contrastive=0,
            q_embed_size=20,

            enc_q_recover_train=1,

            use_z_contrastive=1,

            use_new_batch_for_fake=True,

            target_enc_tau=0.005,

            use_c_vae=True,

            num_tsne_evals=10,
            tsne_plot_freq=5,
            tsne_perplexity=[50, 50],

            num_meta_train_steps=200,

            use_inter_samples=0,
            inter_update_coeff=0.01,

            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,

            use_latent_in_disc=True,
            use_penalty=False,
            psi_aux_vae_beta=1,

            dirichlet_sample_freq_for_exploration=10,
            bisim_transition_sample='rl',
            use_fake_value_bound=False,
            use_episodic_online_buffer=True,
            c_off_batch_size=32,
            c_off_all_element_sampling=False,
            use_c_off_rl=False,
            use_index_rl=False,
            use_first_samples_for_bisim_samples=True,
            use_next_obs_Q_reg=False,
            q_reg_coeff=1,
            use_auto_entropy=False,
            use_target_c_dec=False,
            use_next_obs_beta=False,
            next_obs_beta=1,
            use_kl_penalty=False,
            pi_kl_reg_coeff=0,
            task_mix_freq=1,
            use_rewards_beta=False,
            rewards_beta=1,
            use_sample_reg=False,
            sample_reg_method="",
            target_entropy_coeff=1,
            use_cycle_detach=False,
            seed=1234,
            wide_exp=False,
            optimizer='adam',
            alpha_net_weight_decay=0.0,
            entropy_coeff=1.0,
            close_method="",
            use_closest_task=False,
            closest_task_method="total",
            offpol_c_coeff=1,
            bisim_sample_batchsize=256,
            coff_buffer_fix_ep="",
            c_off_buffer_length=5000,
            plot_beta_area=True,
            compute_overestimation_bias=False,
            plot_wide_tsne=False,
            expert=False,
            wandb_project="",
            wandb_group="",
            eval_generator=False,
            pretrained_model_path="",

            log_dir='',
            launch_file_name='',
            dims={},
            exp_name="",
            config={}
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.pretrained_model_path = pretrained_model_path
        self.eval_generator = eval_generator
        self.wandb_group = wandb_group
        self.wandb_project = wandb_project
        self.expert = expert
        self.plot_wide_tsne = plot_wide_tsne
        self.train_tsne_tasks = train_tsne_tasks
        self.test_tsne_tasks = test_tsne_tasks
        self.compute_overestimation_bias = compute_overestimation_bias
        self.plot_beta_area = plot_beta_area
        self.c_off_buffer_length = c_off_buffer_length
        self.coff_buffer_fix_ep = coff_buffer_fix_ep
        self.bisim_sample_batchsize = bisim_sample_batchsize
        self.offpol_c_coeff = offpol_c_coeff
        self.use_closest_task = use_closest_task
        self.closest_task_method = closest_task_method
        self.close_method = close_method
        self.entropy_coeff = entropy_coeff
        self.alpha_net_weight_decay = alpha_net_weight_decay
        self.optimizer = optimizer
        self.seed = seed
        self.wide_exp = wide_exp
        self.use_cycle_detach = use_cycle_detach
        self.target_entropy_coeff = target_entropy_coeff
        self.sample_reg_method = sample_reg_method
        self.use_sample_reg = use_sample_reg
        self.rewards_beta = rewards_beta
        self.use_rewards_beta = use_rewards_beta
        self.task_mix_freq = task_mix_freq
        self.use_kl_penalty = use_kl_penalty
        self.pi_kl_reg_coeff = pi_kl_reg_coeff
        self.next_obs_beta = next_obs_beta
        self.use_next_obs_beta = use_next_obs_beta
        self.config = config
        self.use_target_c_dec = use_target_c_dec
        self.q_reg_coeff = q_reg_coeff
        self.use_next_obs_Q_reg = use_next_obs_Q_reg
        self.use_first_samples_for_bisim_samples = use_first_samples_for_bisim_samples
        self.use_index_rl = use_index_rl
        self.use_c_off_rl = use_c_off_rl
        self.c_off_all_element_sampling = c_off_all_element_sampling
        self.c_off_batch_size = c_off_batch_size
        self.use_episodic_online_buffer = use_episodic_online_buffer
        self.use_fake_value_bound = use_fake_value_bound
        self.bisim_transition_sample = bisim_transition_sample
        self.use_fakesample_representation = use_fakesample_representation
        self.dirichlet_sample_freq_for_exploration = dirichlet_sample_freq_for_exploration
        self.bisim_penalty_coeff = bisim_penalty_coeff
        self.psi_aux_vae_beta = psi_aux_vae_beta
        self.use_penalty = use_penalty
        self.use_latent_in_disc = use_latent_in_disc
        self.env = env
        self.agent = agent
        self.exploration_agent = agent  # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.indistribution_tasks = indistribution_tasks
        self.tsne_tasks = tsne_tasks
        self.total_tasks_dict_list = total_tasks_dict_list
        # print("total_tasks_dict_list : ", total_tasks_dict_list)  # --> len() = 16
        # print("total_tasks_dict[0,1,2,4]", self.get_task_info([0, 1, 2, 4]))

        self.onpol_c_coeff = onpol_c_coeff
        self.same_c_coeff = same_c_coeff
        self.cycle_coeff = cycle_coeff

        curr_time = datetime.datetime.now()
        month = "0" + str(curr_time.month) if len(str(curr_time.month)) < 2 else str(curr_time.month)  # 04
        day = "0" + str(curr_time.day) if len(str(curr_time.day)) < 2 else str(curr_time.day)  # 16
        self.date = month + day

        self.use_new_batch_for_fake = use_new_batch_for_fake

        self.which_sac_file = which_sac_file

        self.bisim_coeff = bisim_coeff
        self.gen_coeff = gen_coeff

        self.recon_coeff = recon_coeff
        # self.s_recon_coeff = s_recon_coeff

        print("self.train_tasks", self.train_tasks)
        print("self.eval_tasks", self.eval_tasks)
        print("self.indistribution_tasks", self.indistribution_tasks)
        print("self.tsne_tasks", self.tsne_tasks)

        self.use_c_vae = use_c_vae

        self.use_new_batch_4_fake = use_new_batch_4_fake

        self.ood = ood
        self.env_name_suffix = env_name_suffix

        self.use_full_interpolation = use_full_interpolation

        self.use_c_dist_clear = use_c_dist_clear

        self.r_dist_coeff = r_dist_coeff
        self.tr_dist_coeff = tr_dist_coeff
        self.policy_kl_reg_coeff = policy_kl_reg_coeff
        self.pretrain_tsne_freq = pretrain_tsne_freq

        self.sa_perm = sa_perm

        self.num_fake_tasks = num_fake_tasks
        self.gan_type = gan_type
        self.use_gan = use_gan

        self.num_dirichlet_tasks = num_dirichlet_tasks

        self.fakesample_cycle = fakesample_cycle

        self.use_decrease_mask = use_decrease_mask

        self.make_prior_to_rl = make_prior_to_rl

        self.offpol_ctxt_sampling_buffer = offpol_ctxt_sampling_buffer

        self.c_distri_vae_train_freq = c_distri_vae_train_freq

        self.fakesample_rl_tran_batch_size = fakesample_rl_tran_batch_size

        self.sample_dist_coeff = sample_dist_coeff

        self.use_decoder_next_state = use_decoder_next_state
        self.use_next_state_bisim = use_next_state_bisim

        self.c_kl_lambda = c_kl_lambda

        self.bisim_r_coef = bisim_r_coef
        self.bisim_dist_coef = bisim_dist_coef

        self.exp_name = exp_name

        self.beta = beta
        self.c_buffer_size = c_buffer_size
        self.c_batch_num = c_batch_num

        self.decrease_rate = decrease_rate

        self.use_z_autoencoder = use_z_autoencoder

        self.same_task_loss_pow = same_task_loss_pow

        self.z_dist_compute_method = z_dist_compute_method

        self.env_name = env_name

        self.clear_enc_buffer = clear_enc_buffer
        self.prior_enc_buffer_size = prior_enc_buffer_size
        self.online_enc_buffer_size = online_enc_buffer_size

        self.pretrain_steps = pretrain_steps

        self.wgan_lambda = wgan_lambda
        self.gen_freq = gen_freq

        self.use_context_buffer = use_context_buffer

        self.use_W = use_W

        self.use_q_contrastive = use_q_contrastive
        self.q_embed_size = q_embed_size
        self.enc_q_recover_train = enc_q_recover_train

        self.use_z_contrastive = use_z_contrastive

        self.target_enc_tau = target_enc_tau

        self.num_tsne_evals = num_tsne_evals
        self.tsne_plot_freq = tsne_plot_freq
        self.tsne_perplexity = tsne_perplexity

        self.num_meta_train_steps = num_meta_train_steps

        self.use_inter_samples = use_inter_samples
        self.inter_update_coeff = inter_update_coeff

        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.launch_file_name = launch_file_name

        self.log_dir = log_dir
        # shutil.copy("launch_experiment.py", os.path.join(self.log_dir, "save_files", "launch_experiment.py"))
        shutil.copytree("configs", os.path.join(self.log_dir, "save_files", "configs"))
        shutil.copytree("rand_param_envs", os.path.join(self.log_dir, "save_files", "rand_param_envs"))
        shutil.copytree("rlkit", os.path.join(self.log_dir, "save_files", "rlkit"))
        shutil.copyfile(launch_file_name, os.path.join(self.log_dir, "save_files", launch_file_name))

        print("dims", dims)
        self.o_dim = dims["obs_dim"]
        self.a_dim = dims["action_dim"]
        self.r_dim = dims["reward_dim"]
        self.l_dim = dims["latent_dim"]

        # self.writer = SummaryWriter(log_dir=self.log_dir)
        self.algorithm = algorithm

        self.use_auto_entropy = use_auto_entropy
        self.alpha_net_weight_decay = alpha_net_weight_decay
        self.target_entropy = -dims["action_dim"]
        # self.log_alpha = torch.zeros(1).to(ptu.device).requires_grad_(True)
        # if use_auto_entropy:
        #     self.log_alpha_inter = torch.zeros(1).to(ptu.device).requires_grad_(True)
        # alpha_param = [self.log_alpha, self.log_alpha_inter] if use_auto_entropy else [self.log_alpha]
        # self.alpha_optimizer = torch.optim.Adam(alpha_param, lr=0.0003)

        self.wide_tsne_log_dir = f"logs/{self.env_name}/{datetime.datetime.now()}/tsne"
        os.makedirs(self.wide_tsne_log_dir)


        # 실행 이름 설정
        # wandb.run.name = self.exp_name
        # wandb.run.save()


        wandb.login(key="7316f79887c82500a01a529518f2af73d5520255")
        entity = "jmkim22" if wandb_project == "mo_metaRL" else "mlic_academic"
        wandb.init(
            # set the wandb project where this run will be logged
            entity=entity,
            project=self.wandb_project,
            group=self.wandb_group if len(self.wandb_group) > 0 else self.env_name,  # "onoff-off",  # "pearl-antgoal",#self.env_name,
            # name=self.algorithm + '/' + self.exp_name  # "baseline"#self.exp_name,
            name=self.date + self.env_name + self.env_name_suffix + '/' + self.exp_name,  # "baseline"#self.exp_name,
            config=config
        )

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )

        #########################################################################33
        #########################################################################33
        self.prior_enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,  # 50,000개
            env,
            self.train_tasks,
        )

        self.online_enc_replay_buffer = MultiTaskReplayBuffer(
            # self.online_enc_buffer_size,  # 10,000개
            # self.num_steps_prior + self.num_extra_rl_steps_posterior,
            self.num_steps_prior,
            env,
            self.train_tasks,
        )
        # 프라이어 인코더 버퍼는 처음 5만개 데이터만 저장 후 고정
        # 온라인 데이터는 1만개 데이터가 저장 & pi 학습 되면서 저장되는 샘플 계속 변함
        #
        # --> 타임스텝에 대해 안정화 + robustness
        self.bisim_target_sample_buffer_size = 100000 * len(self.train_tasks)  # ex) ant-dir-4 : buffer_size = 100,000 x 4 = 400,000
        self.bisim_target_sample_buffer = MultiTaskReplayBuffer(  # ex) ant-goal-inter : buffer_size = 100,000 x 150 = 15,000,000
            self.bisim_target_sample_buffer_size + 1000,
            env,
            self.train_tasks,
        )
        #########################################################################33
        #########################################################################33

        ###################################################################################
        if self.use_episodic_online_buffer:
            self.episodic_online_buffer = EpisodicOnlineReplayBuffer(num_task=len(self.train_tasks), max_episode_per_buffer=10)
        ###################################################################################

        self._n_env_steps_total = 0
        self._init_n_env_steps = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

        self.current_pretrain_steps = 0

    def get_task_info(self, indices):
        label = []
        # print("self.env_name", self.env_name)

        if self.env_name in ["cheetah-vel-inter", "cheetah-vel-extra"]:
            for idx in indices:
                label.append(round(self.total_tasks_dict_list[idx]['velocity'], 4))
            indices, label = np.array(indices), np.array(label)

            sorted_indices = label.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        elif self.env_name in ["ant-goal-inter", "ant-goal-extra", "ant-goal-extra-hard"]:
            for idx in indices:
                # temp.append(round(self.total_tasks_dict_list[idx]['goal'], 4))
                label.append(np.around(self.total_tasks_dict_list[idx]['goal'], 4))
            indices, label = np.array(indices), np.array(label)

            goal_dists = []
            for i in range(len(label)):
                goal_dists.append(np.sqrt(label[i][0] ** 2 + label[i][1] ** 2))
            goal_dists = np.array(goal_dists)
            sorted_indices = goal_dists.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        elif self.env_name in ["ant-dir-4개", "ant-dir-2개"]:
            for idx in indices:
                label.append(np.around(self.total_tasks_dict_list[idx]['goal'], 4))
            indices, label = np.array(indices), np.array(label)

            sorted_indices = label.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        # elif self.env_name in ["cheetah-mass-inter", "cheetah-mass-extra",
        #                         "hopper-rand-params-inter", "hopper-rand-params-extra",
        #                         "walker-rand-params-inter", "walker-rand-params-extra",
        #                         "walker-mass-inter", "hopper-mass-inter", "cheetah-mass-inter",
        #                         "hopper-mass-inter-2", "walker-mass-inter-2"]:
        elif "mass" in self.env_name or "params" in self.env_name:

            for idx in indices:
                label.append(np.around(self.total_tasks_dict_list[idx], 4))
            indices, label = np.array(indices), np.array(label)

            sorted_indices = label.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        return indices, label

    def make_exploration_policy(self, policy):
        return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        # self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)

            if it_ == 0:
                print('collecting initial pool of data for train and eval... ')
                # temp for evaluating
                accum_context_during_prior_steps = True if self.use_episodic_online_buffer else False
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps,
                                      1,
                                      np.inf,
                                      accum_context=accum_context_during_prior_steps,
                                      add_to_online_enc_buffer=True)  # add_to_prior_enc_buffer=True, add_to_online_enc_buffer=True
                    if self.use_episodic_online_buffer:
                        self.episodic_online_buffer.add_context(task_index=idx, context=self.agent.context)  # ([1, 497, 36])
                        # batches = self.episodic_online_buffer.sample_context(task_indices=[idx], batch_size=128)  # ([1, 128, 36])

            self._init_n_env_steps = self._n_env_steps_total
            # self._n_env_steps_total = self._n_env_steps_total - self._init_n_env_steps


            if it_ == 0:

                print('pre training... ')

                for pretrain_step in range(self.pretrain_steps):
                    self.current_pretrain_steps = self.current_pretrain_steps + 1

                    pretrain_loss = self.pretrain(pretrain_step)
                    total_loss, reward_recon_loss, next_obs_recon_loss, reward_recon_loss_k, reward_recon_loss_c, next_obs_recon_loss_k, next_obs_recon_loss_c, bisim_c_loss, on_off_loss, c_vae_loss, c_vae_recon_loss, c_vae_kl_loss = pretrain_loss
                    # self.meta_train(pretrain_step)

                    if pretrain_step % 1000 == 0:
                        print("pretrain_step", pretrain_step)

                    pretrain_loss_dict = {
                        "pretrain/total_loss": total_loss,
                        "pretrain/reward_recon_loss": reward_recon_loss,
                        "pretrain/next_obs_recon_loss": next_obs_recon_loss,
                        "pretrain/reward_recon_loss_k": reward_recon_loss_k,
                        "pretrain/reward_recon_loss_c": reward_recon_loss_c,
                        "pretrain/next_obs_recon_loss_k": next_obs_recon_loss_k,
                        "pretrain/next_obs_recon_loss_c": next_obs_recon_loss_c,
                        "pretrain/bisim_c_loss": bisim_c_loss,
                        "pretrain/on_pol_c_loss": on_off_loss,
                        "pretrain/c_vae_loss": c_vae_loss,
                        "pretrain/c_vae_recon_loss": c_vae_recon_loss,
                        "pretrain/c_vae_kl_loss": c_vae_kl_loss,
                    }

                    wandb.log(pretrain_loss_dict, step=self._n_env_steps_total -self.pretrain_steps + self.current_pretrain_steps)


                print('done for pretraining')




            print("Bisim_target_samples_buffer length=", self.bisim_target_sample_buffer.task_buffers[0].size())
            """ 5에폭 마다 c_buffer tsne """
            if it_ % 10 == 0:

                # if self.agent.task_c_off_buffer.check_buffers_size(threshold=5000) and self.agent.task_c_on_buffer.check_buffers_size(threshold=5000):
                plt.figure(figsize=(12, 3))
                ############################## c_on_off 버퍼 tsne #####################################
                plt.subplot(131)

                c_offs = self.agent.task_c_off_buffer.sample_c(tasks_indices=self.train_tasks, num_samples=5000, randomly_sample=True)  # min([len(self.agent.task_c_off_buffer.buffers[i]) for i in range(self.agent.task_c_off_buffer.num_tasks)])
                c_ons = self.agent.task_c_on_buffer.sample_c(tasks_indices=self.train_tasks, num_samples=5000, randomly_sample=True)  # min([len(self.agent.task_c_on_buffer.buffers[i]) for i in range(self.agent.task_c_on_buffer.num_tasks)])

                c_ons_list = [c_on.detach().cpu().numpy() for c_on in c_ons.view(-1, self.l_dim)]
                c_offs_list = [c_off.detach().cpu().numpy() for c_off in c_offs.view(-1, self.l_dim)]
                c_on_off_total = c_ons_list + c_offs_list

                c_on_off_tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[0], n_jobs=4)
                c_on_off_result = c_on_off_tsne_model.fit_transform(np.array(c_on_off_total))

                c_on_off_indices_list = [range(len(c_ons_list)), range(len(c_ons_list), len(c_ons_list) + len(c_offs_list))]

                c_on_off_color = ['b', 'r']  # 파랑=c_on_buffer, r=c_off_buffer
                c_on_off_label = ['c_on_buff', 'c_off_buff']  # 파랑=c_on_buffer, r=c_off_buffer

                for i in range(2):
                    plt.scatter(c_on_off_result[:, 0][c_on_off_indices_list[i]],
                                c_on_off_result[:, 1][c_on_off_indices_list[i]],
                                c=c_on_off_color[i], s=1,
                                label=c_on_off_label[i])
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title("Current epoch's c_on, c_off buffer tsne")
                ############################## c_on_off 버퍼 tsne #####################################

                ######################### c_off_buffer history #################################
                plt.subplot(132)
                self.c_off_buffer_history_que.append({"ep": "EP" + str(it_), "c": c_offs.view(-1, self.l_dim)})

                c_offs_history_list = []  # [[c_off.detach().cpu().numpy() for c_off in c_offs.view(-1, self.l_dim)] for c_offs in self.c_off_buffer_history_que]  # [4, 200, 10] 짜리가 10개
                for data in self.c_off_buffer_history_que:
                    for c_off in data['c'].view(-1, self.l_dim):  # c_offs.view(-1, self.l_dim):
                        c_offs_history_list.append(c_off.detach().cpu().numpy())

                c_offs_history_tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[0], n_jobs=4)
                c_offs_history_tsne_result = c_offs_history_tsne_model.fit_transform(np.array(c_offs_history_list))

                length = c_offs.view(-1, self.l_dim).size(0)  # 4x500 = 2000개
                c_offs_history_indices_list = [range(i * length, (i + 1) * length) for i in range(len(self.c_off_buffer_history_que))]

                c_offs_history_color = ['red', 'orange', 'gold', 'lime', 'darkgreen', 'deepskyblue', 'blue', 'navy', 'indigo', 'purple', 'magenta']  # 파랑=c_on_buffer, r=c_off_buffer

                for i in range(len(self.c_off_buffer_history_que)):  # 지금은 10개
                    plt.scatter(c_offs_history_tsne_result[:, 0][c_offs_history_indices_list[i]],
                                c_offs_history_tsne_result[:, 1][c_offs_history_indices_list[i]],
                                c=c_offs_history_color[i], s=1, alpha=1 - 0.05 * i,
                                label=self.c_off_buffer_history_que[i]['ep'])
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title("c_off data changes as epochs tsne")
                ######################### c_off_buffer history #################################

                ######################### c_on_buffer history #################################
                plt.subplot(133)
                self.c_on_buffer_history_que.append({"ep": "EP" + str(it_), "c": c_ons.view(-1, self.l_dim)})

                c_ons_history_list = []
                for data in self.c_on_buffer_history_que:
                    for c_on in data['c'].view(-1, self.l_dim):  # c_offs.view(-1, self.l_dim):
                        c_ons_history_list.append(c_on.detach().cpu().numpy())

                c_ons_history_tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[0], n_jobs=4)
                c_ons_history_tsne_result = c_ons_history_tsne_model.fit_transform(np.array(c_ons_history_list))

                length = c_ons.view(-1, self.l_dim).size(0)  # 4x500 = 2000개
                c_ons_history_indices_list = [range(i * length, (i + 1) * length) for i in range(len(self.c_on_buffer_history_que))]

                c_ons_history_color = ['red', 'orange', 'gold', 'lime', 'darkgreen', 'deepskyblue', 'blue', 'navy', 'indigo', 'purple', 'magenta']  # 파랑=c_on_buffer, r=c_off_buffer

                for i in range(len(self.c_on_buffer_history_que)):  # 지금은 10개
                    plt.scatter(c_ons_history_tsne_result[:, 0][c_ons_history_indices_list[i]],
                                c_ons_history_tsne_result[:, 1][c_ons_history_indices_list[i]],
                                c=c_ons_history_color[i], s=1, alpha=1 - 0.05 * i,
                                label=self.c_on_buffer_history_que[i]['ep'])
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title("c_on data changes as epochs tsne")
                ######################### c_on_buffer history #################################

                c_buffers_tsne_save_path = os.path.join(self.log_dir, "c_buffers_tsne_" + str(it_) + '.png')
                plt.savefig(c_buffers_tsne_save_path, bbox_inches='tight')


                c_off_buffer_history_que_log_dir = os.path.join(self.log_dir, "c_off_buffer_history.pkl")
                c_on_buffer_history_que_log_dir = os.path.join(self.log_dir, "c_on_buffer_history.pkl")
                with open(c_off_buffer_history_que_log_dir, 'wb') as file:
                    pickle.dump(self.c_off_buffer_history_que, file)
                with open(c_on_buffer_history_que_log_dir, 'wb') as file:
                    pickle.dump(self.c_on_buffer_history_que, file)


                wandb.log({
                    "C_Buffers_tsne_every_epoch/c_buffers_EP" + str(it_): [wandb.Image(c_buffers_tsne_save_path)]
                })


            if it_ == 0:
                self.bisim_target_sample_buffer.task_buffers[0].clear()

            # Sample data from train tasks.
            accum_context_during_prior_steps = True if self.use_episodic_online_buffer else False
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                # print("idx", idx, len(self.enc_replay_buffer.task_buffers[idx]))
                if self.clear_enc_buffer:
                    self.prior_enc_replay_buffer.task_buffers[idx].clear()
                    self.online_enc_replay_buffer.task_buffers[idx].clear()
                # print("idx", idx, len(self.enc_replay_buffer.task_buffers[idx]), "\n")

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:  # 2024.04.18 add_to_rl_buffer 추가 add_to_rl_buffer=False
                    self.collect_data(self.num_steps_prior, 1, np.inf, accum_context=accum_context_during_prior_steps)  # add_to_prior_enc_buffer=True, add_to_online_enc_buffer=True
                if self.use_episodic_online_buffer:
                    self.episodic_online_buffer.add_context(task_index=idx, context=self.agent.context)  # ([1, 497, 36])
                    # batches = self.episodic_online_buffer.sample_context(task_indices=[idx], batch_size=128)  # ([1, 128, 36])

                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                      add_to_prior_enc_buffer=self.make_prior_to_rl, add_to_online_enc_buffer=False,
                                      add_to_bisim_target_sample_buffer=False)
            self.agent.c_vae_curr_c = []

            sac_loss_list = [[] for i in range(26)]
            meta_train_losses = [[] for i in range(25)]

            t1 = time.time()
            ########################################################################################
            print("메타 트레인 시작")
            for train_step in range(self.num_meta_train_steps):  # 50

                meta_train_loss = self.meta_train(train_step, ep=it_)
                for idx, loss in enumerate(meta_train_loss):
                    if not (loss == 0.0):
                        meta_train_losses[idx].append(loss)

            """"""

            ########################################################################################
            t2 = time.time()
            print("representation_time={} (s)".format(t2 - t1))

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)

                sac_losses = self._do_training(indices, train_step)
                for idx, loss in enumerate(sac_losses):
                    if loss is not None:
                        sac_loss_list[idx].append(loss)

                self._n_train_steps_total += 1
            t3 = time.time()
            print("rl_time={} (s)".format(t3 - t2))

            gt.stamp('train')

            self.training_mode(False)



            loss_dict = dict(

                qf_loss=np.mean(sac_loss_list[0]),
                vf_loss=np.mean(sac_loss_list[1]),
                policy_loss=np.mean(sac_loss_list[2]),
                qf_loss_inter=np.mean(sac_loss_list[3]),
                vf_loss_inter=np.mean(sac_loss_list[4]),
                policy_loss_inter=np.mean(sac_loss_list[5]),
                qf_loss_total=np.mean(sac_loss_list[6]),
                vf_loss_total=np.mean(sac_loss_list[7]),
                policy_loss_total=np.mean(sac_loss_list[8]),
                q_reg_loss=np.mean(sac_loss_list[9]),

                q1_pred=np.mean(sac_loss_list[10]),
                q2_pred=np.mean(sac_loss_list[11]),
                v_pred=np.mean(sac_loss_list[12]),
                q_target=np.mean(sac_loss_list[13]),
                q1_pred_inter=np.mean(sac_loss_list[14]),
                q2_pred_inter=np.mean(sac_loss_list[15]),
                v_pred_inter=np.mean(sac_loss_list[16]),
                q_target_inter=np.mean(sac_loss_list[17]),
                v_target=np.mean(sac_loss_list[18]),
                v_target_inter=np.mean(sac_loss_list[19]),
                log_pi=np.mean(sac_loss_list[20]),
                log_pi_inter=np.mean(sac_loss_list[21]),
                alpha=np.mean(sac_loss_list[22]),
                alpha_inter=np.mean(sac_loss_list[23]),
                alpha_loss=np.mean(sac_loss_list[24]),
                alpha_loss_inter=np.mean(sac_loss_list[25]),

                total_loss=np.mean(meta_train_losses[0]),
                reward_recon_loss=np.mean(meta_train_losses[1]),
                next_obs_recon_loss=np.mean(meta_train_losses[2]),
                reward_recon_loss_k=np.mean(meta_train_losses[3]),
                reward_recon_loss_c=np.mean(meta_train_losses[4]),
                next_obs_recon_loss_k=np.mean(meta_train_losses[5]),
                next_obs_recon_loss_c=np.mean(meta_train_losses[6]),
                # bisim_c_loss.item(), on_off_loss.item(), off_off_loss.item(),
                bisim_c_loss=np.mean(meta_train_losses[7]),
                on_off_loss=np.mean(meta_train_losses[8]),
                off_off_loss=np.mean(meta_train_losses[9]),
                same_c_loss=np.mean(meta_train_losses[10]),
                c_vae_loss=np.mean(meta_train_losses[11]),
                c_vae_recon_loss=np.mean(meta_train_losses[12]),
                c_vae_kl_loss=np.mean(meta_train_losses[13]),

                d_total_loss=np.mean(meta_train_losses[14]),
                d_real_score=np.mean(meta_train_losses[15]),
                d_fake_score=np.mean(meta_train_losses[16]),
                gradient_penalty=np.mean(meta_train_losses[17]),
                gradients_norm=np.mean(meta_train_losses[18]),
                g_total_loss=np.mean(meta_train_losses[19]),
                g_real_score=np.mean(meta_train_losses[20]),
                g_fake_score=np.mean(meta_train_losses[21]),
                w_distance=np.mean(meta_train_losses[22]),
                cycle_loss_off=np.mean(meta_train_losses[23]),
                cycle_loss_on=np.mean(meta_train_losses[24]),

            )

            # eval
            self._try_to_eval(it_, loss_dict)
            gt.stamp('eval')

            self._end_epoch()

    # def pretrain(self):
    #     """
    #     Do anything before the main training phase.
    #     """
    #     pass

    def collect_data(self,
                     num_samples,
                     resample_z_rate,
                     update_posterior_rate,
                     accum_context=False,
                     add_to_rl_buffer=True,
                     add_to_prior_enc_buffer=True,
                     add_to_online_enc_buffer=True,
                     add_to_bisim_target_sample_buffer=True,
                     epoch=0):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        if epoch == 0:
            random_task_ctxt_batch = None
        else:
            random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
            random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer=self.offpol_ctxt_sampling_buffer)
        self.agent.c_vae_curr_c = []
        self.agent.clear_z(random_task_ctxt_batch)

        num_transitions = 0
        if self.use_c_vae:
            sample_dirichlet_c_for_exploration_ = False
        else:
            sample_dirichlet_c_for_exploration_ = True
        while num_transitions < num_samples:
            # paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
            #                                                max_trajs=update_posterior_rate,
            #                                                accum_context=False,
            #                                                resample=resample_z_rate)
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=accum_context,
                                                           resample=resample_z_rate,
                                                           sample_dirichlet_c_for_exploration=sample_dirichlet_c_for_exploration_)  # self.max_path_length

            num_transitions += n_samples

            if add_to_rl_buffer:
                self.replay_buffer.add_paths(self.task_idx, paths)

            # print("self.prior_enc_replay_buffer.task_buffers[self.task_idx].size()", self.task_idx, self.prior_enc_replay_buffer.task_buffers[self.task_idx].size())
            # print("self.online_enc_replay_buffer.task_buffers[self.task_idx].size()", self.task_idx, self.online_enc_replay_buffer.task_buffers[self.task_idx].size())

            # if add_to_prior_enc_buffer:
            if add_to_prior_enc_buffer and self.prior_enc_replay_buffer.task_buffers[self.task_idx].size() < self.prior_enc_buffer_size:
                self.prior_enc_replay_buffer.add_paths(self.task_idx, paths)
                # self.bisim_target_sample_buffer.add_paths(0, paths)  # 0~400스텝(프라이어로 뽑은 샘플)이 하나의 태스크 버퍼에(인덱스는 항상0) 저장됨

            if add_to_online_enc_buffer:  # and self.prior_enc_replay_buffer.task_buffers[self.task_idx].size() >= self.prior_enc_buffer_size:
                self.online_enc_replay_buffer.add_paths(self.task_idx, paths)

            # if add_to_bisim_target_sample_buffer:
            #     self.bisim_target_sample_buffer.add_paths(0, paths)
            # if add_to_bisim_target_sample_buffer and self.bisim_target_sample_buffer.task_buffers[0].size() < self.bisim_target_sample_buffer_size:
            #     self.bisim_target_sample_buffer.add_paths(0, paths)
            if self.use_first_samples_for_bisim_samples:
                if self.bisim_target_sample_buffer.task_buffers[0].size() < self.bisim_target_sample_buffer_size:
                    self.bisim_target_sample_buffer.add_paths(0, paths)
            else:
                self.bisim_target_sample_buffer.add_paths(0, paths)

            # if epoch == 0:
            #     random_task_ctxt_batch = None
            # else:
            #     random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
            #     random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer=self.offpol_ctxt_sampling_buffer)
            # self.agent.clear_z(random_task_ctxt_batch)

            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx, which_buffer="online")
                # context = self.agent.context
                self.agent.infer_posterior(context, which_enc='psi')
                sample_dirichlet_c_for_exploration_ = False
        # print("num_transitions", num_transitions)

        self._n_env_steps_total = self._n_env_steps_total + num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch, loss_dict):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch, loss_dict)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation, )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run, return_z=False):
        self.task_idx = idx
        self.env.reset_task(idx)
        # print("task_idx :", idx)

        if epoch == 0:
            random_task_ctxt_batch = None
        else:
            random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
            random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer='online')
        self.agent.c_vae_curr_c = []
        self.agent.clear_z(random_task_ctxt_batch)

        # self.agent.clear_z()
        paths = []
        task_z = []
        num_transitions = 0
        num_trajs = 0
        deterministic_ = False
        if self.use_c_vae:
            sample_dirichlet_c_for_exploration_ = False
        else:
            sample_dirichlet_c_for_exploration_ = True
        while num_transitions < self.num_steps_per_eval:  # 1200
            # path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
            #                                         max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
            #                                         accum_context=True)
            path, num = self.sampler.obtain_samples(deterministic=deterministic_,
                                                    max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                    accum_context=True,
                                                    sample_dirichlet_c_for_exploration=sample_dirichlet_c_for_exploration_)

            paths += path
            # print("paths:", len(paths))
            # print("num_transitions", num_transitions)

            num_transitions += num
            num_trajs += 1

            if num_trajs >= self.num_exp_traj_eval:  # 4
                # print("Inference!")
                deterministic_ = True
                sample_dirichlet_c_for_exploration_ = False
                # print("self.agent.context:", self.agent.context.shape)
                # c = []
                # for i in range(10):
                #     context = self.agent.context
                #     context = context[:, np.random.permutation(len(context[0])), :]
                #     context = context[:, :128, :]  # [1,128,36]
                #     # print("context", context.shape)
                #     # self.agent.infer_posterior(context)
                #     # self.agent.infer_posterior(self.agent.context)
                #     c.append(self.agent.get_context_embedding(context).detach())  # [1,10]
                # c = torch.cat(c).mean(dim=0).unsqueeze(0)
                # self.agent.z = c
                # # self.agent.z_means, logvar = self.agent.c_distribution_vae.encode(c)
                # # self.agent.z_vars = torch.exp(logvar)
                c = self.agent.get_context_embedding(self.agent.context, which_enc='psi').detach()
                task_z.append(c)

            else:
                if epoch == 0:
                    random_task_ctxt_batch = None
                else:
                    random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
                    random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer='online')
                self.agent.clear_z(random_task_ctxt_batch, context_clear=False)

        # print("paths[-1][observations].shape", paths[-1]["observations"].shape)

        self.agent.c_vae_curr_c = []

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        # return paths
        if return_z:
            return paths, task_z[0]  # task_z[-2]  # task_z[0]  # task_z[1]
        else:
            return paths

    def _do_eval(self, indices, epoch, eval=False):
        final_returns = []
        online_returns = []
        r_diff_list, n_o_diff_list = [], []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths, z = self.collect_paths(idx, epoch, r, return_z=True)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])

                # print("z.shape", z.shape)  # ([1, 10])
                # print("paths[-1]", len(paths[-1]))  # 9  {"observations":ndarray, "actions":ndarray, "rewards":ndarray, "next_observations":ndarray, "terminals":ndarray, "agent_infos":[], "env_infos":[]}
                # print("paths[-1][observations].shape", paths[-1]["observations"].shape)  # (200, 20)
                if self.compute_overestimation_bias:
                    initial_state = torch.Tensor(paths[-1]["observations"][0]).to(ptu.device).unsqueeze(0)
                    initial_action = torch.Tensor(paths[-1]["actions"][0]).to(ptu.device).unsqueeze(0)

                    q_estimation = self.qf1(initial_state, initial_action, z).item()
                    log_pi_init = self.agent.policy.get_logprob(initial_state, initial_action, z).item()

                    discounted_reward_sum = 0.0
                    for s, a, r in zip(paths[-1]["observations"][::-1], paths[-1]["actions"][::-1], paths[-1]["rewards"][::-1]):
                        s = torch.Tensor(s).to(ptu.device).unsqueeze(0)
                        a = torch.Tensor(a).to(ptu.device).unsqueeze(0)
                        log_pi_ = self.agent.policy.get_logprob(s, a, z).item()
                        discounted_reward_sum = self.reward_scale * r - self.entropy_coeff * log_pi_ + self.discount * discounted_reward_sum
                    discounted_reward_sum = discounted_reward_sum + self.entropy_coeff * log_pi_init

                    q_overestimation = q_estimation - discounted_reward_sum

                else:
                    q_overestimation = 0.0



                if self.eval_generator:
                    obs, actions, rewards, next_obs = paths[-1]["observations"], paths[-1]["actions"], paths[-1]["rewards"], paths[-1]["next_observations"]
                    obs, actions, rewards, next_obs = torch.Tensor(obs).to(ptu.device), torch.Tensor(actions).to(ptu.device), torch.Tensor(rewards).to(ptu.device), torch.Tensor(next_obs).to(ptu.device)
                    path_length = len(obs)
                    z_repeat = self.get_repeat(z, path_length, self.l_dim)  # ([1, 10]) --> ([200, 10])
                    rewards_pred, next_obs_pred, _, _ = self.agent.c_decoder(obs, actions, z_repeat)

                    rewards_difference = ((rewards - rewards_pred) ** 2).mean().item()
                    if self.use_decoder_next_state:
                        next_obs_difference = ((next_obs - next_obs_pred) ** 2).mean().item()
                    else:
                        next_obs_difference = 0.0
                else:
                    rewards_difference, next_obs_difference = 0.0, 0.0

            r_diff_list.append(rewards_difference)
            n_o_diff_list.append(next_obs_difference)


            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns, q_overestimation, np.mean(r_diff_list), np.mean(n_o_diff_list)

    def get_distance_matrix(self, z_lst):
        length = len(z_lst)
        m = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                diff = z_lst[i] - z_lst[j]
                euclidian = np.linalg.norm(diff, ord=2)
                m[i, j] = euclidian
        return m

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################3
    """"""
    """"""


    def _do_tsne_plot(self, indices, epoch):
        trials = 30
        latent_samples = []
        indices_list = {task: [] for task in indices}
        i = 0

        for trial in range(trials):
            for task in indices:
                # _, latent_sample = self.collect_paths_exp(task, epoch, trial, wideeval=True, return_z=True)
                _, latent_sample = self.collect_paths(task, epoch, trial, return_z=True)
                latent_samples.append(latent_sample.squeeze().numpy(force=True))
                indices_list[task].append(i)
                i += 1

        latent_vectors = LatentVectors(np.asarray(latent_samples), indices_list, epoch, EnvType.parse(self.env_name + self.env_name_suffix))
        latent_vectors.save(self.wide_tsne_log_dir)
        latent_vectors.save_plot(self.wide_tsne_log_dir, wandb)




    def _do_tsne_eval_add_inter_plot(self, indices, epoch):
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime',
                  'yellow', 'magenta', 'coral', 'skyblue', 'indigo', 'k']


        plt.figure(figsize=(20, 8))

        #
        """ (1) 인코더 버퍼 길이 출력 """
        num_task = len(self.train_tasks)
        prior_enc_buffer_size = np.zeros(num_task)
        online_enc_buffer_size = np.zeros(num_task)
        for task_idx in range(num_task):
            prior_enc_buffer_size[task_idx] = self.prior_enc_replay_buffer.task_buffers[task_idx].size()
            online_enc_buffer_size[task_idx] = self.online_enc_replay_buffer.task_buffers[task_idx].size()
        # prior_enc_buffer_size = prior_enc_buffer_size.reshape((10, 10))
        # online_enc_buffer_size = online_enc_buffer_size.reshape((10, 10))
        print("prior_enc_buffer_size", prior_enc_buffer_size)
        print("prior_enc_buffer_size[0][0]", self.prior_enc_replay_buffer.task_buffers[0]._observations[0])
        print("online_enc_buffer_size", online_enc_buffer_size)
        print("online_enc_buffer_size[0][0]", self.online_enc_replay_buffer.task_buffers[0]._observations[0])

        """ (1) 트레인 태스크 tsne """
        if self.env_name in ["ant-dir-4개", "humanoid-dir"]:
            num_test_tasks = 4
        elif self.env_name in ["ant-dir-2개"]:
            num_test_tasks = 2
        else:
            num_test_tasks = 10
        trainset_test_indices = np.random.choice(self.train_tasks, num_test_tasks, replace=False)
        # print("trainset_test_indices", trainset_test_indices)
        # trainset_test_indices.sort()
        # print("trainset_test_indices_sort", trainset_test_indices)
        trainset_test_indices, labels = self.get_task_info(trainset_test_indices)

        z_trainset_list = [[] for _ in range(num_test_tasks)]
        for i in range(30):
            ctxt_bat = self.sample_context(trainset_test_indices)  # ([10, 128, 27])
            task_z = self.agent.get_context_embedding(ctxt_bat, which_enc="psi").detach()
            task_z = [z.cpu().numpy() for z in task_z]
            for j in range(num_test_tasks):
                z_trainset_list[j].append(task_z[j])
                # zz_trainset_list[j].append(task_zz[j])

        mean_z = [sum(z) / len(z) for z in z_trainset_list]

        z_trainset_list_flatten, zz_trainset_list_flatten = [], []
        for i in range(len(z_trainset_list)):
            for j in range(len(z_trainset_list[i])):
                z_trainset_list_flatten.append(z_trainset_list[i][j])
        print("len(z_trainset_list_flatten)", len(z_trainset_list_flatten))  # 300개

        z_trainset_indices_list = [range(i * 30, (i + 1) * 30) for i in range(len(z_trainset_list))]
        print("z_trainset_indices_list", z_trainset_indices_list)

        z_trainset_tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[1], n_jobs=4)
        z_trainset_result = z_trainset_tsne_model.fit_transform(np.array(z_trainset_list_flatten))
        length = len(z_trainset_list_flatten)

        plt.subplot(241)
        for i in range(len(z_trainset_list)):
            plt.scatter(z_trainset_result[:, 0][z_trainset_indices_list[i]],
                        z_trainset_result[:, 1][z_trainset_indices_list[i]],
                        c=colors[i], s=5, label=str(labels[i]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.subplot(242)
        distance_matrix_c = self.get_distance_matrix(mean_z)
        for i in range(len(distance_matrix_c)):
            text = 'c_' + str(i + 1)
            plt.text(i, -0.7, text, rotation=90)
            plt.text(-1.5, i, text)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.imshow(distance_matrix_c)
        plt.colorbar()

        # tsne_save_path = os.path.join(self.log_dir, "tSNE_trainset_" + str(epoch) + '.png')
        # plt.savefig(tsne_save_path, bbox_inches='tight')

        """ (0) 전체 태스크에 대한 c """
        # plt.subplot(243)
        # # plt.figure(figsize=(8, 8))
        # # c_batch = self.c_buffer.sample_c(num_samples=self.c_buffer_size).cpu().numpy()  # c_batch = self.c_buffer.sample_c(num_samples=2000).cpu().numpy()  # ([2000, 10])
        # c_batch = self.agent.task_c_on_buffer.sample_c(tasks_indices=self.train_tasks, num_samples=256)
        # c_batch = c_batch.view(-1, self.l_dim).cpu().numpy()
        #
        # c_total_tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[1])
        # c_total_result = c_total_tsne_model.fit_transform(c_batch)
        #
        # plt.scatter(c_total_result[:, 0],
        #             c_total_result[:, 1],
        #             c='b', s=2)

        ####################

        c_lst, z_lst = [], []
        for _ in range(20):
            ctxt_batch = self.sample_context(self.train_tasks)  # ([200, 100, 27])
            c = self.agent.get_context_embedding(ctxt_batch, which_enc="psi").detach().cpu()
            c_lst.append(c)  # ([200, 10]))
        c_lst = torch.cat(c_lst)  # ([4000, 10]))
        len_ = len(c_lst)

        c_latent = torch.randn(len_, self.latent_dim).to(ptu.device)  # ([4000, 10)]
        c_fake = self.agent.c_distribution_vae.decode(c_latent).detach().cpu()

        plt.subplot(247)  # 10 dim의 각 0, 1 번째 실제 데이터 출력
        plt.scatter(c_lst.numpy()[:, 0], c_lst.numpy()[:, 1], s=2, c='b')
        plt.scatter(c_fake.numpy()[:, 0], c_fake.numpy()[:, 1], s=2, alpha=0.5, c='r')

        plt.subplot(248)
        """추가할 부분"""
        result_indices = [range(len_), range(len_, len_ * 2)]
        c_lst_c_fake = torch.cat([c_lst, c_fake])
        c_lst_c_fake_tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[1], n_jobs=4)
        c_lst_c_fake_result = c_lst_c_fake_tsne_model.fit_transform(c_lst_c_fake.numpy())
        color = ['b', 'r']  # b=psi, r=c_distri
        plt.scatter(c_lst_c_fake_result[result_indices[0]][:, 0], [c_lst_c_fake_result[result_indices[0]][:, 1]], s=2, c='b')
        plt.scatter(c_lst_c_fake_result[result_indices[1]][:, 0], [c_lst_c_fake_result[result_indices[1]][:, 1]], s=2, alpha=0.5, c='r')

        # plt.subplot(223)
        # # plt.scatter(c_distri_tsne_result[:, 0], [c_distri_tsne_result[:, 1]], s=2)
        # for i in range(2):
        #     plt.scatter(c_distri_tsne_result[result_indices[i]][:, 0], [c_distri_tsne_result[result_indices[i]][:, 1]], s=2, c=color[i])
        ####################

        train_indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        z_list_train = []
        z_train_total_list = [[] for _ in range(len(train_indices))]  # [ [], [], ..., [] ]  11개 태스크 + 5개 인터폴레이션 태스크
        z_train_total_list_off = [[] for _ in range(len(train_indices))]
        z_train_total_list_on = [[] for _ in range(len(train_indices))]
        for label, task_idx in enumerate(train_indices):  #
            # print("eval idx ", task_idx, "  self.num_evals:", self.num_evals)

            one_task_z_train_ = []
            for run in range(self.num_tsne_evals):  # 30번
                paths, z = self.collect_paths(task_idx, epoch, run, return_z=True)
                z = z.view(-1, ).detach().cpu().numpy()  # z (10,)

                ctxt_bat = self.sample_context(task_idx, which_buffer=self.offpol_ctxt_sampling_buffer, b_size=128)
                z_off = self.agent.get_context_embedding(ctxt_bat, which_enc="psi")
                z_off = z_off.view(-1, ).detach().cpu().numpy()  # z (10,)

                ctxt_bat = self.sample_context(task_idx, which_buffer="online", b_size=128)
                z_on = self.agent.get_context_embedding(ctxt_bat, which_enc="psi")
                z_on = z_on.view(-1, ).detach().cpu().numpy()  # z (10,)

                # z_on : 같은 태스크 인덱스의 onpol 버퍼로 뽑은 것도 같이 비교해보자 --> z_on이 z_off처럼 한점으로 잘 모이면 z가 이상한거임

                z_train_total_list[label].append(z)
                z_train_total_list_off[label].append(z_off)
                z_train_total_list_on[label].append(z_on)

                one_task_z_train_.append(z)

            z_list_train.append(sum(one_task_z_train_) / len(one_task_z_train_))

        distance_matrix_train = self.get_distance_matrix(z_list_train)
        print("distance_matrix:\n", distance_matrix_train)  # self.result_path

        z_total_list_flatten_train, var_train_lst = [], []
        for i in range(len(z_train_total_list)):
            # z_train_total_list[i] --> 분산계산 [(10,), (10,), (10,), ..., (10,)]
            distance_matrix = self.get_distance_matrix(z_train_total_list[i])
            var_train_lst.append(np.var(distance_matrix))
            for j in range(len(z_train_total_list[i])):
                z_total_list_flatten_train.append(z_train_total_list[i][j])
        print("len(z_total_list_flatten_train)", len(z_total_list_flatten_train))  # 480개

        z_total_list_flatten_train_off, var_train_off_lst = [], []
        for i in range(len(z_train_total_list_off)):
            # z_train_total_list_off[i] --> 분산계산
            distance_matrix = self.get_distance_matrix(z_train_total_list_off[i])
            var_train_off_lst.append(np.var(distance_matrix))
            for j in range(len(z_train_total_list_off[i])):
                z_total_list_flatten_train_off.append(z_train_total_list_off[i][j])

        z_total_list_flatten_train_on, var_train_on_lst = [], []
        for i in range(len(z_train_total_list_on)):
            # z_train_total_list_off[i] --> 분산계산
            distance_matrix = self.get_distance_matrix(z_train_total_list_on[i])
            var_train_on_lst.append(np.var(distance_matrix))
            for j in range(len(z_train_total_list_on[i])):
                z_total_list_flatten_train_on.append(z_train_total_list_on[i][j])

        # indices_list = [range(i * 30, (i + 1) * 30) if i < 16
        #                 else range(i * 30, i * 30 + len(z_total_list[i]))
        #                 for i in range(len(z_total_list))]
        # indices_list_train = [range(i * 30, (i + 1) * 30) for i in range(len(z_train_total_list))]
        # print("indices_list", indices_list_train)

        z_total = z_total_list_flatten_train + z_total_list_flatten_train_off + z_total_list_flatten_train_on
        # z_total_list_flatten_train : eval c (c_on_eval) "b"
        # z_total_list_flatten_train_off : off_buffer c (c_off) "r"
        # z_total_list_flatten_train_on : on_buffer c (c_on) "g"
        tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[0], n_jobs=4)
        result = tsne_model.fit_transform(np.array(z_total))

        length = len(z_total_list_flatten_train)
        indices_list_train = [range(length), range(length, 2 * length), range(2 * length, 3 * length)]

        color = ['b', 'r', 'g']  # 파랑=onpol_eval, r= ,초록=offpol
        plot244_labels = ["c_on infered from eval_context", "c_off infered from rl_buffer", "c_on infered from online_buffer"]

        plt.subplot(244)
        for i in range(3):
            plt.scatter(result[:, 0][indices_list_train[i]],
                        result[:, 1][indices_list_train[i]],
                        c=color[i], s=5,
                        label=plot244_labels[i])
        plt.text(1, 1, str([np.round(a_, 3) for a_ in var_train_lst]))
        plt.text(1, 0.5, str([np.round(a_, 3) for a_ in var_train_off_lst]))

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #

        """ (2) 테스트 태스트 tsne """
        z_list = []
        # num_interpolated_tasks = 5 + 1  # 5 <-- c4_hat ~ c8_hat // 1 <-- 랜덤 인터폴레이션
        if self.env_name in ["cheetah-vel", "cheetah-vel-extra"]:
            num_interpolated_tasks = 0  # 5 <-- c4_hat ~ c8_hat // 1 <-- 랜덤 인터폴레이션
            num_diric_tasks = 1
        elif self.env_name in ["ant-goal"]:
            if self.ood == "inter":
                num_interpolated_tasks = 4  # 5 <-- c4_hat ~ c8_hat // 1 <-- 랜덤 인터폴레이션
            elif self.ood == "extra":
                num_interpolated_tasks = 0  # 5 <-- c4_hat ~ c8_hat // 1 <-- 랜덤 인터폴레이션
            num_diric_tasks = 1
            print("num_interpolated_tasks", num_interpolated_tasks)
        else:
            num_interpolated_tasks = 0
            num_diric_tasks = 1

        z_total_list = [[] for _ in range(len(indices) + num_interpolated_tasks + num_diric_tasks)]  # [ [], [], ..., [] ]  11개 태스크 + 5개 인터폴레이션 태스크
        print("len z_total_list", len(z_total_list))  # 17  0~11:tsne_eval, 12,13,14,15:intertask, 16:total_c
        print("len indices", len(indices))  # 12
        for label, task_idx in enumerate(indices):  # label==0~11,
            # print("eval idx ", task_idx, "  self.num_evals:", self.num_evals)
            print("label", label, ",  task_idx", task_idx)

            one_task_z_ = []
            for run in range(self.num_tsne_evals):  # 30번
                paths, z = self.collect_paths(task_idx, epoch, run, return_z=True)
                z = z.view(-1, ).detach().cpu().numpy()  # z (10,)

                z_total_list[label].append(z)

                one_task_z_.append(z)  # 디스턴스 매트릭스 계산시
            z_list.append(sum(one_task_z_) / len(one_task_z_))  # 디스턴스 매트릭스 계산시 필요

        if self.ood == "inter":
            alpha_ = 1.25 / 2.25
        for idx in range(num_interpolated_tasks):  # 4개  0~3
            print("interpolated_tasks idx", idx)

            for run in range(self.num_tsne_evals):  # 30번

                if idx == 0:  # c4=0.75 --> 0.9c3 + 0.1c9 --> 0.9*z_total_list[2] + 0.1*z_total_list[8]
                    interpolated_c_temp = (1 - alpha_) * z_total_list[0][run] + alpha_ * z_total_list[8][run]
                    # interpolated_c_temp = 0.9 * z_total_list[2][run] + 0.1 * z_total_list[8][run]
                    z_total_list[len(indices) + idx].append(interpolated_c_temp)  # z_total_list[12]

                elif idx == 1:  # c5=1.25 --> 0.75c3 + 0.25c11 --> 0.75*z_total_list[2] + 0.25*z_total_list[10]
                    # interpolated_c_temp = 0.75 * z_total_list[2][run] + 0.25 * z_total_list[10][run]
                    interpolated_c_temp = (1 - alpha_) * z_total_list[1][run] + alpha_ * z_total_list[9][run]
                    z_total_list[len(indices) + idx].append(interpolated_c_temp)  # z_total_list[13]

                elif idx == 2:  # c6=1.75 --> 0.5c2 + 0.5c10 --> 0.5*z_total_list[1] + 0.5*z_total_list[9]
                    # interpolated_c_temp = 0.5 * z_total_list[1][run] + 0.5 * z_total_list[9][run]
                    interpolated_c_temp = (1 - alpha_) * z_total_list[2][run] + alpha_ * z_total_list[10][run]
                    z_total_list[len(indices) + idx].append(interpolated_c_temp)  # z_total_list[14]

                elif idx == 3:  # c7=2.25 --> 0.3c3 + 0.7c9 --> 0.3*z_total_list[2] + 0.7*z_total_list[8]
                    # interpolated_c_temp = 0.3 * z_total_list[2][run] + 0.7 * z_total_list[8][run]
                    interpolated_c_temp = (1 - alpha_) * z_total_list[3][run] + alpha_ * z_total_list[11][run]
                    z_total_list[len(indices) + idx].append(interpolated_c_temp)  # z_total_list[15]

            print("idx:", len(z_total_list[len(indices) + idx]))

        for idx in range(num_diric_tasks):  # 1개
            for run in range(self.num_tsne_evals):  # 30번
                for _ in range(100):
                    # onpol_c로 전체 태스크에 대한 인터폴레이션 공간 그려보기
                    # 트레이닝에서 했던것처럼 트레이닝 태스크의 c를 16개씩 섞어서 테스트 c 공간에 같이 그려보기
                    train_indices = np.random.choice(self.train_tasks, self.num_dirichlet_tasks, replace=False)
                    ctxt_batch = self.sample_context(train_indices, which_buffer="online")
                    task_c = self.agent.get_context_embedding(ctxt_batch, which_enc="psi").detach().cpu()  # ([16, 10])
                    alpha = Dirichlet(torch.ones(1, self.num_dirichlet_tasks)).sample() * self.beta - (self.beta - 1) / self.num_dirichlet_tasks  # ([1, 16])
                    c_alpha = alpha @ task_c  # ([1, 10])
                    c_alpha = c_alpha.squeeze().cpu().numpy()
                    z_total_list[len(indices) + num_interpolated_tasks + idx].append(c_alpha)

        for i in range(len(z_total_list)):
            print("z_total_list i", i, len(z_total_list[i]))

        distance_matrix = self.get_distance_matrix(z_list)
        print("distance_matrix:\n", distance_matrix)  # self.result_path

        z_total_list_flatten = []
        for i in range(len(z_total_list)):  # 17
            for j in range(len(z_total_list[i])):
                z_total_list_flatten.append(z_total_list[i][j])
        print("len(z_total_list_flatten)", len(z_total_list_flatten))  # 330개

        # indices_list = [range(i * 30, (i + 1) * 30) if i < 15
        #                 else range(i * 30, i * 30 + len(z_total_list[i]))
        #                 for i in range(len(z_total_list))]
        indices_list = [range(i * self.num_tsne_evals, (i + 1) * self.num_tsne_evals) for i in range(len(z_total_list) - 1)]
        indices_list = indices_list + [range(i * self.num_tsne_evals, i * self.num_tsne_evals + len(z_total_list[-1]))]
        print("indices_list", indices_list)

        tsne_model = TSNE(n_components=2, random_state=0, perplexity=self.tsne_perplexity[0], n_jobs=4)
        result = tsne_model.fit_transform(np.array(z_total_list_flatten))
        print("len result", len(result))

        if self.env_name in ["cheetah-vel-inter"]:

            [0.1, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.1, 3.25],

            tsne_tasks_lst = ["0.1(c0)", "0.25(c1)", "0.75(c2)", "1.25(c3)",
                              "1.75(c4)", "2.25(c5)", "2.75(c6)", "3.1(c7)", "3.25(c8)", "c_inter"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c$"]

            # tsne_tasks_lst = ["0.1(c0)", "0.25(c1)", "0.5(c2)", "0.75(c3)", "1.25(c4)",
            #                   "1.75(c5)", "2.25(c6)", "2.75(c7)", "3.0(c8)", "3.25(c9)", "3.5(c10)",
            #                   '0.75_hat=0.9c3+0.1c9', '1.25_hat=0.75c3+0.25c11',
            #                   '1.75_hat=0.5c2+0.5c10', '2.25_hat=0.3c3+0.7c9', '2.75_hat=0.1c3+0.9c9']
            # tsne_tasks_lst = ["0.25(c1)", "0.5(c2)", "0.75(c3)", "1.2(c4)", "1.75(c5)",
            #                   "2.25(c6)", "2.75(c7)", "3.4(c8)", "3.75(c9)", "4.0(c10)", "4.25(c11)",
            #                   '0.75_hat=0.9c3+0.1c9', '1.25_hat=0.75c3+0.25c11',
            #                   '1.75_hat=0.5c2+0.5c10', '2.25_hat=0.3c3+0.7c9', '2.75_hat=0.1c3+0.9c9']

            # marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c_{9}$", "$c_{10}$",
            #                "$\hat{c_4}$", "$\hat{c_5}$", "$\hat{c_6}$", "$\hat{c_7}$", "$\hat{c_8}$", ]

        elif self.env_name in ["cheetah-vel-extra"]:
            # [0.1, 0.25, 1, 1.5, 2, 2.5, 3.1, 3.25, 3.4]
            tsne_tasks_lst = ["0.1(c0)", "0.25(c1)", "1.0(c2)", "1.5(c3)", "2.0(c4)",
                              "2.5(c5)", "3.1(c6)", "3.25(c7)", "3.4(c8)", "c_inter"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c$"]

        elif self.env_name in ["ant-goal-inter", "ant-goal-extra", "ant-goal-extra-hard", "humanoid-goal"]:

            if self.ood == "inter":
                tsne_tasks_lst = ["[0.5,  0]c0", "[0, 0.5 ]c1", "[-0.5,  0]c2", "[0, -0.5 ]c3",
                                  "[1.75, 0]c4", "[0, 1.75]c5", "[-1.75, 0]c6", "[0, -1.75]c7",
                                  "[2.75, 0]c8", "[0, 2.75]c9", "[-2.75, 0]c10", '[0, -2.75]c11',
                                  "[1.75, 0]c4_hat", "[0, 1.75]c5_hat", "[-1.75, 0]c6_hat", "[0, -1.75]c7_hat",
                                  "c_inter"]  # 1개

                marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c_{9}$",
                               "$c_{10}$", "$c_{11}$",
                               "$\hat{c_4}$", "$\hat{c_5}$", "$\hat{c_6}$", "$\hat{c_7}$", "$c$"]

            elif self.ood == "extra":
                tsne_tasks_lst = ["[0.5,  0]c0", "[0, 0.5 ]c1", "[-0.5,  0]c2", "[0, -0.5 ]c3",
                                  "[1.75, 0]c4", "[0, 1.75]c5", "[-1.75, 0]c6", "[0, -1.75]c7",
                                  "[3.0, 0]c8", "[0, 3.0]c9", "[-3.0, 0]c10", '[0, -3.0]c11',
                                  "c_inter"]  # 1개

                marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c_{9}$",
                               "$c_{10}$", "$c_{11}$", "$c$"]

            elif self.ood == "extra-hard":
                tsne_tasks_lst = ["[0.5,  0]c0", "[0, 0.5 ]c1", "[-0.5,  0]c2", "[0, -0.5 ]c3",
                                  "[1.75, 0]c4", "[0, 1.75]c5", "[-1.75, 0]c6", "[0, -1.75]c7",
                                  "[3.0, 0]c8", "[0, 3.0]c9", "[-3.0, 0]c10", '[0, -3.0]c11',
                                  "c_inter"]  # 1개

                marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c_{9}$",
                               "$c_{10}$", "$c_{11}$", "$c$"]

        elif self.env_name in ["ant-dir-4개"]:

            tsne_tasks_lst = ["0pi c0", "1/4pi c1", "2/4pic2", "3/4pi c3",
                              "4/4pi c4", "5/4pi c5", "6/4pi c6", "7/4pi c7", "c_inter"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c$"]

        elif self.env_name in ["ant-dir-2개"]:

            tsne_tasks_lst = ["0pi c0", "1/4pi c1", "2/4pic2", "3/4pi c3", "7/4pi c4", "c_inter"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c$"]


        elif "mass" in self.env_name or "params" in self.env_name:

            tsne_tasks_lst = ["c" + str(i) for i in range(len(self.tsne_tasks))] + ["c_inter"]
            marker_list = ["$c_{" + str(i) + "}$" for i in range(len(self.tsne_tasks))] + ["$c$"]
            print("tsne_tasks_lst", tsne_tasks_lst)
            print("marker_list", marker_list)

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'lime',
                  'yellow', 'magenta', 'coral', 'skyblue', 'indigo', 'k']

        # plt.figure(figsize=(20, 5))

        """ 추가할것 : Distance Matrix plot 한 plot에 나오도록 추가   """

        plt.subplot(245)
        total_diric_idx = len(z_total_list)  # 49
        print("total_diric_idx", total_diric_idx)
        for i in range(len(z_total_list)):  # i ==  0~16
            print("i", i)
            if i < total_diric_idx - 1:  # 0~47
                plt.scatter(result[:, 0][indices_list[i]],
                            result[:, 1][indices_list[i]],
                            s=100, marker=marker_list[i],  # c=colors[i], # 's',
                            # alpha=0.02, edgecolor='k',
                            label=str(tsne_tasks_lst[i]))
            else:  # 48 (idx48=49번째=total_c)
                if self.plot_beta_area:
                    plt.scatter(result[:, 0][indices_list[i]],
                                result[:, 1][indices_list[i]],
                                s=10, marker=marker_list[i], c='k',  # 's',
                                alpha=0.1,  # edgecolor='k',
                                label=str(tsne_tasks_lst[i]))
            # elif i == 16:
            #     plt.scatter(result[:, 0][indices_list[i]],
            #                 result[:, 1][indices_list[i]],
            #                 c=colors[i], s=1, marker='*',  # 'x',
            #                 # alpha=0.02, edgecolor='k',
            #                 label="random_interpolation")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Task latent variable(c) T-SNE plot")

        plt.subplot(246)
        for i in range(len(distance_matrix)):
            text = 'c_' + str(i + 1)
            plt.text(i, -0.7, text, rotation=90)
            plt.text(-1.5, i, text)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.imshow(distance_matrix)
        plt.colorbar()

        tsne_save_path = os.path.join(self.log_dir, "tSNE_" + str(epoch) + '.png')
        plt.savefig(tsne_save_path, bbox_inches='tight')

        wandb.log({
            "Eval_tsne/tSNE_EP" + str(epoch): [wandb.Image(tsne_save_path)]
        })

        z_total_list_logdir = os.path.join(self.log_dir, "z_total_list.pkl")
        with open(z_total_list_logdir, 'wb') as file:
            pickle.dump(z_total_list[:-1], file)


        # tsne_saved_img = vision_io.read_image(tsne_save_path)
        # self.writer.add_image("plt", tsne_saved_img)

        plt.clf()
        #

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################3

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################3

    def evaluate(self, epoch, loss_dict):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                         max_samples=self.max_path_length * 20,
                                                         accum_context=False,
                                                         resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            num_episode = self.num_steps_per_eval // self.max_path_length  # 3
            deterministic_ = False
            for i in range(num_episode):  # 600 // 200 = 3 , i=0,1,2
                if i == num_episode - 1:  # 0/2 , 1/2, 2/2
                    deterministic_ = True
                context = self.sample_context(idx)
                self.agent.infer_posterior(context, which_enc='psi')
                # p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                #                                    max_samples=self.max_path_length,
                #                                    accum_context=False,
                #                                    max_trajs=1,
                #                                    resample=np.inf)
                p, _ = self.sampler.obtain_samples(deterministic=deterministic_,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf,
                                                   sample_dirichlet_c_for_exploration=False)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)

        ### eval train tasks with on-policy data to match eval of test tasks
        print("indices _do_eval", indices)
        train_final_returns, train_online_returns, _, _, _ = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns, q_overestimation, r_diff, n_o_diff = self._do_eval(self.eval_tasks, epoch, eval=True)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        ### test tasks
        if len(self.indistribution_tasks) > 0:
            eval_util.dprint('evaluating on {} indistribution tasks'.format(len(self.indistribution_tasks)))
            indistribution_final_returns, indistribution_online_returns, _, _, _ = self._do_eval(self.indistribution_tasks, epoch)
            eval_util.dprint('indistribution online returns')
            eval_util.dprint(indistribution_online_returns)
        else:
            indistribution_final_returns, indistribution_online_returns = [0.0], [0.0]

        num_task = len(self.train_tasks)
        rl_buffer_size = np.zeros(num_task)
        prior_enc_buffer_size = np.zeros(num_task)
        online_enc_buffer_size = np.zeros(num_task)
        for task_idx in range(num_task):
            rl_buffer_size[task_idx] = self.replay_buffer.task_buffers[task_idx].size()
            prior_enc_buffer_size[task_idx] = self.prior_enc_replay_buffer.task_buffers[task_idx].size()
            online_enc_buffer_size[task_idx] = self.online_enc_replay_buffer.task_buffers[task_idx].size()
        # prior_enc_buffer_size = prior_enc_buffer_size.reshape((10, 10))
        # online_enc_buffer_size = online_enc_buffer_size.reshape((10, 10))
        print("rl_buffer_size", rl_buffer_size)
        print("prior_enc_buffer_size", prior_enc_buffer_size)
        print("online_enc_buffer_size", online_enc_buffer_size)

        # tsne_plot
        if epoch % self.tsne_plot_freq == 0:

            if self.plot_wide_tsne:
                print("TSNE plot 시작")
                self._do_tsne_eval_add_inter_plot(self.tsne_tasks, epoch)

                print("Wide TSNE plot 시작")
                self._do_tsne_plot(self.train_tsne_tasks + self.test_tsne_tasks, epoch)


            print("TSNE plot 끝")



        c_on_buffer_log_dir = os.path.join(self.log_dir, "c_on_buffer.pkl")
        c_off_buffer_log_dir = os.path.join(self.log_dir, "c_off_buffer.pkl")
        with open(c_on_buffer_log_dir, 'wb') as file:
            pickle.dump(self.agent.task_c_on_buffer.buffers, file)
        with open(c_off_buffer_log_dir, 'wb') as file:
            pickle.dump(self.agent.task_c_off_buffer.buffers, file)



        print("train_final_returns", train_final_returns)
        print("test_final_returns", test_final_returns)
        # print("train_final_returns_AVG", sum(train_final_returns) / len(train_final_returns))
        # print("test_final_returns_AVG", sum(test_final_returns) / len(test_final_returns))

        train_returns = np.mean(train_returns)

        train_avg_return = np.mean(train_final_returns)
        test_avg_return = np.mean(test_final_returns)
        indistribution_avg_return = np.mean(indistribution_final_returns)

        train_avg_online_return = np.mean(train_online_returns)
        test_avg_online_return = np.mean(test_online_returns)
        indistribution_avg_online_return = np.mean(indistribution_online_returns)

        tsne_save_path = os.path.join(self.log_dir, "tSNE_" + str(epoch) + '.png')
        env_step = self._n_env_steps_total  # - self._init_n_env_steps
        wandb_log_dict = {
            "epoch": epoch,
            "env_step": env_step,
            "rl_train_step": self._n_train_steps_total,

            "SAC/qf_loss": loss_dict["qf_loss"],
            "SAC/vf_loss": loss_dict["vf_loss"],
            "SAC/policy_loss": loss_dict["policy_loss"],
            "SAC/qf_loss_inter": loss_dict["qf_loss_inter"],
            "SAC/vf_loss_inter": loss_dict["vf_loss_inter"],
            "SAC/policy_loss_inter": loss_dict["policy_loss_inter"],
            "SAC/qf_loss_total": loss_dict["qf_loss_total"],
            "SAC/vf_loss_total": loss_dict["vf_loss_total"],
            "SAC/policy_loss_total": loss_dict["policy_loss_total"],
            "SAC/q_reg_loss": loss_dict["q_reg_loss"],
            "SAC/q1_pred": loss_dict["q1_pred"],
            "SAC/q2_pred": loss_dict["q2_pred"],
            "SAC/v_pred": loss_dict["v_pred"],
            "SAC/q_target": loss_dict["q_target"],
            "SAC/q1_pred_inter": loss_dict["q1_pred_inter"],
            "SAC/q2_pred_inter": loss_dict["q2_pred_inter"],
            "SAC/v_pred_inter": loss_dict["v_pred_inter"],
            "SAC/q_target_inter": loss_dict["q_target_inter"],
            "SAC/v_target": loss_dict["v_target"],
            "SAC/v_target_inter": loss_dict["v_target_inter"],
            "SAC/log_pi": loss_dict["log_pi"],
            "SAC/log_pi_inter": loss_dict["log_pi_inter"],
            "SAC/alpha": loss_dict["alpha"],
            "SAC/alpha_inter": loss_dict["alpha_inter"],
            "SAC/alpha_loss": loss_dict["alpha_loss"],
            "SAC/alpha_loss_inter": loss_dict["alpha_loss_inter"],

            "model_psi_train/total_loss": loss_dict["total_loss"],
            "model_psi_train/reward_recon_loss": loss_dict["reward_recon_loss"],
            "model_psi_train/next_obs_recon_loss": loss_dict["next_obs_recon_loss"],
            "model_psi_train/reward_recon_loss_k": loss_dict["reward_recon_loss_k"],
            "model_psi_train/reward_recon_loss_c": loss_dict["reward_recon_loss_c"],
            "model_psi_train/next_obs_recon_loss_k": loss_dict["next_obs_recon_loss_k"],
            "model_psi_train/next_obs_recon_loss_c": loss_dict["next_obs_recon_loss_c"],
            "model_psi_train/bisim_c_loss": loss_dict["bisim_c_loss"],
            "model_psi_train/off_off_loss": loss_dict["off_off_loss"],
            "model_psi_train/same_c_loss": loss_dict["same_c_loss"],
            "model_psi_train/on_pol_c_loss": loss_dict["on_off_loss"],
            "model_psi_train/c_vae_loss": loss_dict["c_vae_loss"],
            "model_psi_train/c_vae_recon_loss": loss_dict["c_vae_recon_loss"],
            "model_psi_train/c_vae_kl_loss": loss_dict["c_vae_kl_loss"],

            "wgan/d_total_loss": loss_dict["d_total_loss"],
            "wgan/d_real_score": loss_dict["d_real_score"],
            "wgan/d_fake_score": loss_dict["d_fake_score"],
            "wgan/gradient_penalty": loss_dict["gradient_penalty"],
            "wgan/gradients_norm": loss_dict["gradients_norm"],
            "wgan/g_total_loss": loss_dict["g_total_loss"],
            "wgan/g_real_score": loss_dict["g_real_score"],
            "wgan/g_fake_score": loss_dict["g_fake_score"],
            "wgan/w_distance": loss_dict["w_distance"],

            "fakesample_cycle/cycle_loss_on": loss_dict["cycle_loss_on"],
            "fakesample_cycle/cycle_loss_off": loss_dict["cycle_loss_off"],

            "Eval/train_returns": train_returns,
            "Eval/train_avg_online_return": train_avg_online_return,
            "Eval/test_avg_online_return": test_avg_online_return,
            "Eval/indistribution_avg_online_return": indistribution_avg_online_return,
            "Eval/indistribution_avg_return": indistribution_avg_return,
            "Eval/train_avg_return": train_avg_return,
            "Eval/test_avg_return": test_avg_return,
            "Eval/Q_overstimation_bias": q_overestimation,
            "Eval/Rewards_difference": r_diff,
            "Eval/NextObs_difference": n_o_diff,



        }

        # wandb.log(wandb_log_dict, step=env_step)
        wandb.log(wandb_log_dict, step=env_step -self.pretrain_steps + self.current_pretrain_steps)
        # wandb.log(wandb_log_dict, step=epoch+1)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        # logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        # logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self, indices):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    @abc.abstractmethod
    def pretrain(self, step, pretrain):
        pass
    #
    # @abc.abstractmethod
    # def _do_wgan_gen3(self):
    #     pass
