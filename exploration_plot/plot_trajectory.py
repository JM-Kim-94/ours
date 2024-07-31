"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import random
import pickle
import time

import argparse

import warnings
warnings.filterwarnings("ignore")

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, TransitionDecoder4PEARL, MyFlattenMlp
from rlkit.torch.networks import WGanCritic, Decoder, Discriminator, Autoencoder, SimpleVAE, SNWGanCritic, ReverseDynamics, AlphaNet
from rlkit.torch.networks import PsiAuxVaeDec
# from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

import matplotlib.pyplot as plt

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout

###########################
# from rlkit.torch.sac.sac_old import PEARLSoftActorCritic
# from rlkit.torch.sac.sac import PEARLSoftActorCritic
# from rlkit.torch.sac.sac_constrain import PEARLSoftActorCritic
################################

def experiment(variant):

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(  ENVS[  variant['env_name']  ]   (**variant['env_params'])  )

    which_sac_file = variant['algo_params']['which_sac_file']
    if which_sac_file == "sac":
        from rlkit.torch.sac.sac import PEARLSoftActorCritic
    elif which_sac_file == "sac_revised":
        from rlkit.torch.sac.sac_revised import PEARLSoftActorCritic
    elif which_sac_file == "sac_revised_alpha":
        from rlkit.torch.sac.sac_revised_alpha import PEARLSoftActorCritic
    elif which_sac_file == "sac_revised_dir2cyclebatch5":
        from rlkit.torch.sac.sac_revised_dir2cyclebatch5 import PEARLSoftActorCritic



    

    """ 시드설정 """
    seed = variant['seed']
    print("SEED :", seed)

    # env.reset(seed=seed)
    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    tasks, total_tasks_dict_list = env.get_all_task_idx()
    print("tasks", tasks)
    print("total_tasks_dict_list[0]", total_tasks_dict_list[0])
    print("total_tasks_dict_list[1]", total_tasks_dict_list[1])
    print("total_tasks_dict_list[2]", total_tasks_dict_list[2])

    obs_dim = env.get_obs_dim()
    print("obs_dim", obs_dim)
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    print("obs_dim : {},  antion_dim : {}".format(obs_dim, action_dim))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] \
                                else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder


    num_train = variant["n_train_tasks"]  # 50
    num_test = variant["n_eval_tasks"]  # 5
    num_indistribution = variant["n_indistribution_tasks"]  # 5
    num_tsne = variant["n_tsne_tasks"]  # 5
    if variant["env_params"]["expert"]:
        if "mass" in variant["env_name"]:
            num_train = num_train + 5
        elif "vel" in variant["env_name"]:
            num_train = num_train + 5
        elif "goal" in variant["env_name"]:
            num_train = num_train + 4
        elif "dir" in variant["env_name"]:
            if variant["env_params"]["ood_type"] == "extra":
                num_train = 5
            elif variant["env_params"]["ood_type"] == "inter":
                num_train = 8

    num_train_tsne = variant["n_train_tsne_tasks"]
    num_test_tsne = variant["n_test_tsne_tasks"]

    print("num_train", num_train)
    print("num_test", num_test)
    print("num_indistribution", num_indistribution)
    print("num_tsne", num_tsne)

    print("train_tasks = ", tasks[: num_train])  # 0~49
    print("eval_tasks = ", tasks[num_train:  num_train + num_test])  # 50 ~ 54
    print("indistribution_tasks = ", tasks[num_train + num_test: num_train + num_test + num_indistribution])  # 55 ~ 59
    print("tsne_tasks = ", tasks[num_train + num_test + num_indistribution: num_train + num_test + num_indistribution + num_tsne])

    print("train_tsne_tasks = ", tasks[num_train + num_test + num_indistribution + num_tsne: num_train + num_test + num_indistribution + num_tsne + num_train_tsne])
    print("test_tsne_tasks = ", tasks[num_train + num_test + num_indistribution + num_tsne + num_train_tsne: num_train + num_test + num_indistribution + num_tsne + num_train_tsne + num_test_tsne])


    c_distribution_vae = SimpleVAE(input_dim=latent_dim, latent_dim=latent_dim)

    psi = encoder_model(
        hidden_sizes=[300, 300, 300],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    psi_aux_vae_dec = PsiAuxVaeDec(latent_dim=latent_dim,
                                   obs_dim=obs_dim,
                                   action_dim=action_dim)

    use_decoder_next_state = variant['algo_params']['use_decoder_next_state']
    use_state_noise = variant['algo_params']['use_state_noise']
    print("use_state_noise", use_state_noise)
    use_target_c_dec = variant['algo_params']['use_target_c_dec']

    k_decoder = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        use_next_state=use_decoder_next_state,
        use_k=True,
        num_tasks=num_train,
        use_state_noise=use_state_noise
    )
    c_decoder = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        use_next_state=use_decoder_next_state,
        use_k=False,
        num_tasks=num_train,
        use_state_noise=use_state_noise
    )
    # if use_target_c_dec:
    c_decoder_target = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        use_next_state=use_decoder_next_state,
        use_k=False,
        num_tasks=num_train,
        use_state_noise=use_state_noise
    )
    # else:
    #     c_decoder_target = None


    if variant['algo_params']['gan_type'] == 'wgan':
        disc_model = WGanCritic
    elif variant['algo_params']['gan_type'] == 'sngan':
        disc_model = SNWGanCritic
    else:
        disc_model = None
    disc_l_dim = latent_dim if variant['algo_params']['use_latent_in_disc'] else 0

    discriminator = disc_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        latent_dim=disc_l_dim)  # 0,  # latent_dim

    print("k_decoder", k_decoder)
    print("c_decoder", c_decoder)
    print("discriminator", discriminator)

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim if not variant["algo_params"]["use_index_rl"] else obs_dim + num_train,
        latent_dim=latent_dim if not variant["algo_params"]["use_index_rl"] else num_train,
        action_dim=action_dim,
    )


    log_alpha_net = AlphaNet(latent_dim=latent_dim)

    agent = PEARLAgent(
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
        tasks[: num_train],
        **variant['algo_params']
    )

    exp_name = variant['exp_name']
    launch_file_name = os.path.basename(__file__)
    print("__file__", __file__)
    print("launch_file_name = os.path.basename(__file__)", launch_file_name)
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_name,
                                      base_log_dir=variant['util_params']['base_log_dir'])

    print("train_tsne_tasks = ", tasks[num_train + num_test + num_indistribution + num_tsne: num_train + num_test + num_indistribution + num_tsne + num_train_tsne])
    print("test_tsne_tasks = ", tasks[num_train + num_test + num_indistribution + num_tsne + num_train_tsne: num_train + num_test + num_indistribution + num_tsne + num_train_tsne + num_test_tsne])


    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)


    pretrained_path = variant["algo_params"]["pretrained_model_path"]
    if len(pretrained_path) > 0:
        agent.psi.load_state_dict(torch.load(os.path.join(pretrained_path, 'psi.pth')))
        agent.policy.load_state_dict(torch.load(os.path.join(pretrained_path, 'policy.pth')))
        agent.psi.to(ptu.device)
        agent.policy.to(ptu.device)
        print("psi and policy loaded")

        task_c_on_buffer_dir = os.path.join(pretrained_path, "c_on_buffer.pkl")
        with open(task_c_on_buffer_dir, 'rb') as f:
            agent.task_c_on_buffer.buffers = pickle.load(f)
        print("task_c_on_buffer loaded")


    """Start"""
    agent = MakeDeterministic(agent)
    eval_tasks_indices = tasks[num_train:  num_train + num_test + num_indistribution]
    num_trajs = variant['algo_params']['num_exp_traj_eval'] + 1
    print("num_trajs", num_trajs)  # 3
    all_rets = []
    trajs = []
    task_z = []
    for idx in eval_tasks_indices:
        print("task eval on {}".format(total_tasks_dict_list[idx]))
        env.reset_task(idx)
        agent.clear_z(random_task_ctxt_batch=None)
        sample_dirichlet_c_for_exploration_ = True
        paths = []
        task_traj = []
        for n in range(num_trajs):  # 3
            print("n", n)
            path = rollout(env, agent,
                            max_path_length=variant['algo_params']['max_path_length'],
                            accum_context=True,
                            sample_dirichlet_c_for_exploration=sample_dirichlet_c_for_exploration_)
            paths.append(path)
            epi_traj = np.array([info['xyz_coor'] for info in path['env_infos']])
            task_traj.append(epi_traj)
            if n >= variant['algo_params']['num_exp_traj_eval']-1:
                sample_dirichlet_c_for_exploration_ = False
                # agent.infer_posterior(agent.context)
                z = agent.get_context_embedding(agent.context, which_enc='psi').detach()
                print("INFERENCE")
                task_z.append(z)

        all_rets.append([sum(p['rewards']) for p in paths])
        trajs.append(task_traj)

    tasks = []
    for task_dict in total_tasks_dict_list[num_train:num_train+num_test+num_indistribution]:
        tasks.append(task_dict['goal'])
    task_traj = {
        'task': tasks,
        'traj': trajs,
    }
    np.save(f'{pretrained_path}/beta{variant["algo_params"]["beta"]}.npy', task_traj)


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to



def main(args):
    print("main")



    variant = default_config
    config = "configs/" + args.env_name + ".json"
    if config:

        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

        variant["algo_params"]["beta"] = args.beta

        variant["algo_params"]["pretrain_steps"] = args.pretrain_steps
        variant["algo_params"]["num_meta_train_steps"] = args.num_meta_train_steps
        variant["algo_params"]["num_train_steps_per_itr"] = args.num_train_steps_per_itr
        variant["algo_params"]["num_initial_steps"] = args.num_initial_steps

        variant["algo_params"]["reward_scale"] = args.reward_scale

        variant["algo_params"]["target_entropy_coeff"] = args.target_entropy_coeff



        # 페널티
        # variant["algo_params"]["policy_kl_reg_coeff"] = args.policy_kl_reg_coeff
        # variant["algo_params"]["state_penalty_coeff"] = args.state_penalty_coeff

        variant["algo_params"]["recon_coeff"] = args.recon_coeff
        variant["algo_params"]["same_c_coeff"] = args.same_c_coeff
        variant["algo_params"]["onpol_c_coeff"] = args.onpol_c_coeff
        variant["algo_params"]["bisim_coeff"] = args.bisim_coeff
        variant["algo_params"]["bisim_penalty_coeff"] = args.bisim_penalty_coeff
        variant["algo_params"]["cycle_coeff"] = args.cy_coef
        variant["algo_params"]["r_dist_coeff"] = args.r_dist_coeff
        variant["algo_params"]["tr_dist_coeff"] = args.tr_dist_coeff

        variant["algo_params"]["gen_coeff"] = args.gen_coeff


        variant["algo_params"]["gen_freq"] = args.gen_freq

        # variant["algo_params"]["pretrain_tsne_freq"] = args.pretrain_tsne_freq

        variant["algo_params"]["wgan_lambda"] = args.wgan_lambda

        variant["algo_params"]["c_kl_lambda"] = args.c_kl

        # if args.eval_ep == 2:
        #     variant["algo_params"]["num_steps_prior"] = 400
        #     variant["algo_params"]["num_extra_rl_steps_posterior"] = 600
        #     variant["algo_params"]["num_steps_per_eval"] = 600
        #     variant["algo_params"]["num_exp_traj_eval"] = 2
        # elif args.eval_ep == 4:
        #     variant["algo_params"]["num_steps_prior"] = 800
        #     variant["algo_params"]["num_extra_rl_steps_posterior"] = 600
        #     variant["algo_params"]["num_steps_per_eval"] = 1000
        #     variant["algo_params"]["num_exp_traj_eval"] = 4

        variant["algo_params"]["num_exp_traj_eval"] = args.num_exp_traj_eval

        variant["algo_params"]["c_distri_vae_train_freq"] = args.c_vae_freq
        variant["algo_params"]["fakesample_rl_tran_batch_size"] = args.f_bsize

        variant["algo_params"]["which_sac_file"] = args.which_sac_file

        variant["algo_params"]["gan_type"] = args.gan_type




        if 'goal' in args.env_name:
            if args.contact == "True":
                variant["env_params"]["use_cfrc"] = True
            else:
                variant["env_params"]["use_cfrc"] = False

        if args.use_state_noise == "True":
            variant["algo_params"]["use_state_noise"] = True
        elif args.use_state_noise == "False":
            variant["algo_params"]["use_state_noise"] = False

        if args.use_decoder_next_state == "True":
            variant["algo_params"]["use_decoder_next_state"] = True
        elif args.use_decoder_next_state == "False":
            variant["algo_params"]["use_decoder_next_state"] = False

        if args.use_c_dist_clear == "True":
            variant["algo_params"]["use_c_dist_clear"] = True
        elif args.use_c_dist_clear == "False":
            variant["algo_params"]["use_c_dist_clear"] = False

        if args.use_gan == "True":
            variant["algo_params"]["use_gan"] = True
        elif args.use_gan == "False":
            variant["algo_params"]["use_gan"] = False

        if args.fakesample_cycle == "True":
            variant["algo_params"]["fakesample_cycle"] = True
        elif args.fakesample_cycle == "False":
            variant["algo_params"]["fakesample_cycle"] = False


        if args.use_latent_in_disc == "True":
            variant["algo_params"]["use_latent_in_disc"] = True
        elif args.use_latent_in_disc == "False":
            variant["algo_params"]["use_latent_in_disc"] = False

        # if args.use_decrease_mask == "True":
        #     variant["algo_params"]["use_decrease_mask"] = True
        # elif args.use_decrease_mask == "False":
        #     variant["algo_params"]["use_decrease_mask"] = False

        if args.use_inter_samples == "True":
            variant["algo_params"]["use_inter_samples"] = True
        elif args.use_inter_samples == "False":
            variant["algo_params"]["use_inter_samples"] = False

        if args.use_first_samples_for_bisim_samples == "True":
            variant["algo_params"]["use_first_samples_for_bisim_samples"] = True
        elif args.use_first_samples_for_bisim_samples == "False":
            variant["algo_params"]["use_first_samples_for_bisim_samples"] = False


        if args.use_c_vae == "True":
            variant["algo_params"]["use_c_vae"] = True
        elif args.use_c_vae == "False":
            variant["algo_params"]["use_c_vae"] = False

        if args.use_new_batch_for_fake == "True":
            variant["algo_params"]["use_new_batch_for_fake"] = True
        elif args.use_new_batch_for_fake == "False":
            variant["algo_params"]["use_new_batch_for_fake"] = False

        if args.use_penalty == "True":
            variant["algo_params"]["use_penalty"] = True
        elif args.use_penalty == "False":
            variant["algo_params"]["use_penalty"] = False

        if args.use_next_state_bisim == "True":
            variant["algo_params"]["use_next_state_bisim"] = True
        elif args.use_next_state_bisim == "False":
            variant["algo_params"]["use_next_state_bisim"] = False

        if args.use_next_obs_in_context == "True":
            variant["algo_params"]["use_next_obs_in_context"] = True
        elif args.use_next_obs_in_context == "False":
            variant["algo_params"]["use_next_obs_in_context"] = False

        if args.use_fakesample_representation == "True":
            variant["algo_params"]["use_fakesample_representation"] = True
        elif args.use_fakesample_representation == "False":
            variant["algo_params"]["use_fakesample_representation"] = False

        if args.use_fake_value_bound == "True":
            variant["algo_params"]["use_fake_value_bound"] = True
        elif args.use_fake_value_bound == "False":
            variant["algo_params"]["use_fake_value_bound"] = False

        if args.use_episodic_online_buffer == "True":
            variant["algo_params"]["use_episodic_online_buffer"] = True
        elif args.use_episodic_online_buffer == "False":
            variant["algo_params"]["use_episodic_online_buffer"] = False

        if args.c_off_all_element_sampling == "True":
            variant["algo_params"]["c_off_all_element_sampling"] = True
        elif args.c_off_all_element_sampling == "False":
            variant["algo_params"]["c_off_all_element_sampling"] = False

        if args.use_c_off_rl == "True":
            variant["algo_params"]["use_c_off_rl"] = True
        elif args.use_c_off_rl == "False":
            variant["algo_params"]["use_c_off_rl"] = False

        if args.use_index_rl == "True":
            variant["algo_params"]["use_index_rl"] = True
        elif args.use_index_rl == "False":
            variant["algo_params"]["use_index_rl"] = False

        if args.use_next_obs_Q_reg == "True":
            variant["algo_params"]["use_next_obs_Q_reg"] = True
        elif args.use_next_obs_Q_reg == "False":
            variant["algo_params"]["use_next_obs_Q_reg"] = False

        if args.use_auto_entropy == "True":
            variant["algo_params"]["use_auto_entropy"] = True
        elif args.use_auto_entropy == "False":
            variant["algo_params"]["use_auto_entropy"] = False

        if args.use_target_c_dec == "True":
            variant["algo_params"]["use_target_c_dec"] = True
        elif args.use_target_c_dec == "False":
            variant["algo_params"]["use_target_c_dec"] = False

        if args.use_next_obs_beta == "True":
            variant["algo_params"]["use_next_obs_beta"] = True
        elif args.use_next_obs_beta == "False":
            variant["algo_params"]["use_next_obs_beta"] = False

        if args.use_rewards_beta == "True":
            variant["algo_params"]["use_rewards_beta"] = True
        elif args.use_rewards_beta == "False":
            variant["algo_params"]["use_rewards_beta"] = False

        if args.use_sample_reg == "True":
            variant["algo_params"]["use_sample_reg"] = True
        elif args.use_sample_reg == "False":
            variant["algo_params"]["use_sample_reg"] = False

        if args.use_cycle_detach == "True":
            variant["algo_params"]["use_cycle_detach"] = True
        elif args.use_cycle_detach == "False":
            variant["algo_params"]["use_cycle_detach"] = False

        if args.wide_exp == "True":
            variant["algo_params"]["wide_exp"] = True
        elif args.wide_exp == "False":
            variant["algo_params"]["wide_exp"] = False

        if args.use_closest_task == "True":
            variant["algo_params"]["use_closest_task"] = True
        elif args.use_closest_task == "False":
            variant["algo_params"]["use_closest_task"] = False


        variant["algo_params"]["optimizer"] = args.optimizer

        variant["algo_params"]["sample_reg_method"] = args.sample_reg_method
        variant["algo_params"]["next_obs_beta"] = args.next_obs_beta
        variant["algo_params"]["rewards_beta"] = args.rewards_beta

        variant["algo_params"]["q_reg_coeff"] = args.q_reg_coeff

        variant["algo_params"]["c_off_batch_size"] = args.c_off_batch_size
        variant["algo_params"]["bisim_sample_batchsize"] = args.bisim_sample_batchsize


        variant["algo_params"]["bisim_transition_sample"] = args.bisim_transition_sample

        variant["algo_params"]["c_buffer_size"] = args.c_buffer_size

        variant["algo_params"]["entropy_coeff"] = args.entropy_coeff

        variant["algo_params"]["dirichlet_sample_freq_for_exploration"] = args.dirichlet_sample_freq_for_exploration

        # variant["algo_params"]["intra_group_ratio"] = args.intra_group_ratio
        # variant["algo_params"]["inter_group_ratio"] = args.inter_group_ratio

        variant["algo_params"]["wandb_project"] = args.wandb_project

        variant["algo_params"]["closest_task_method"] = args.closest_task_method
        variant['algo_params']['close_method'] = args.close_method

        variant['algo_params']['offpol_c_coeff'] = args.offpol_c_coeff

        variant["algo_params"]["seed"] = args.seed

        variant["algo_params"]["alpha_net_weight_decay"] = args.alpha_net_weight_decay

        variant["algo_params"]["psi_aux_vae_beta"] = args.psi_aux_vae_beta

        variant["algo_params"]["inter_update_coeff"] = args.lossc
        variant['util_params']['gpu_id'] =  args.gpu_num


        if args.use_manual_tasks_sampling == "True":
            variant["algo_params"]["meta_batch"] = args.meta_batch
            variant["algo_params"]["num_fake_tasks"] = args.num_fake_tasks
            variant["algo_params"]["num_dirichlet_tasks"] = args.num_dirichlet_tasks



        variant["algo_params"]["coff_buffer_fix_ep"] = args.coff_buffer_fix_ep

        variant["algo_params"]["c_off_buffer_length"] = args.c_off_buffer_length


        if args.plot_beta_area == "True":
            variant["algo_params"]["plot_beta_area"] = True
        elif args.plot_beta_area == "False":
            variant["algo_params"]["plot_beta_area"] = False

        if args.compute_overestimation_bias == "True":
            variant["algo_params"]["compute_overestimation_bias"] = True
        elif args.compute_overestimation_bias == "False":
            variant["algo_params"]["compute_overestimation_bias"] = False


        if args.plot_wide_tsne == "True":
            variant["algo_params"]["plot_wide_tsne"] = True
        elif args.plot_wide_tsne == "False":
            variant["algo_params"]["plot_wide_tsne"] = False

        if args.expert == "True":
            variant["env_params"]["expert"] = True
            variant["algo_params"]["expert"] = True
        elif args.expert == "False":
            variant["env_params"]["expert"] = False
            variant["algo_params"]["expert"] = False

        variant["algo_params"]["wandb_project"] = args.wandb_project
        variant["algo_params"]["wandb_group"] = args.wandb_group

        if args.eval_generator == "True":
            variant["algo_params"]["eval_generator"] = True
        elif args.eval_generator == "False":
            variant["algo_params"]["eval_generator"] = False

        variant["algo_params"]["pretrained_model_path"] = args.pretrained_model_path


        name = (
                # variant["algo_params"]["ood"],
                args.exp_num,

                # args.c_kl,

                # args.beta,

                # args.use_inter_samples,  # str
                # args.lossc,

                # args.eval_ep,

                args.which_sac_file,
                )
        variant["exp_name"] = "exp%s" \
                              "which_sac=%s" % name
        # variant["exp_name"] = "%s" \
        #                       "/" \
        #                       "exp%s" \
        #                       "-" \
        #                       "c_vae_kl%.3f," \
        #                       "beta%.1f," \
        #                       "fakeRL%s," \
        #                       "f_coef%.3f," \
        #                       "eval_ep%d," \
        #                       "which_sac=%s" % name




        ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])

    # variant['util_params']['gpu_id'] = gpu

    experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print(parser)


    parser.add_argument('--coff_buffer_fix_ep', default=0, type=int)

    parser.add_argument('--env_name', default='ant-goal-inter', type=str)

    parser.add_argument('--optimizer', default="adam", type=str)

    parser.add_argument('--closest_task_method', default="total", type=str)


    parser.add_argument('--pretrain_steps', default=5000, type=int)
    parser.add_argument('--num_meta_train_steps', default=1000, type=int)
    parser.add_argument('--num_rl_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps_per_itr', default=1000, type=int)

    parser.add_argument('--num_initial_steps', default=2000, type=int)

    parser.add_argument('--bisim_sample_batchsize', default=256, type=int)

    parser.add_argument('--reward_scale', default=5.0, type=float)

    parser.add_argument('--use_c_vae', default="True", type=str)
    parser.add_argument('--bisim_transition_sample', default="rl", type=str)

    parser.add_argument('--use_closest_task', default="False", type=str)



    parser.add_argument('--close_method', default="", type=str)


    parser.add_argument('--c_off_all_element_sampling', default="False", type=str)

    parser.add_argument('--entropy_coeff', default=1.0, type=float)

    parser.add_argument('--use_latent_in_disc', default="True", type=str)
    parser.add_argument('--use_auto_entropy', default="False", type=str)

    parser.add_argument('--use_target_c_dec', default="False", type=str)

    parser.add_argument('--use_c_off_rl', default="False", type=str)
    parser.add_argument('--use_index_rl', default="False", type=str)

    parser.add_argument('--use_fake_value_bound', default="False", type=str)

    parser.add_argument('--use_first_samples_for_bisim_samples', default="True", type=str)
    parser.add_argument('--use_cycle_detach', default="False", type=str)


    parser.add_argument('--alpha_net_weight_decay', default=0.0, type=float)

    parser.add_argument('--target_entropy_coeff', default=1.0, type=float)

    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--contact', default="True", type=str)

    parser.add_argument('--lossc', default=1, type=float)

    parser.add_argument('--gan_type', default="wgan", type=str)

    parser.add_argument('--sample_dist_coeff', default=1, type=float)
    parser.add_argument('--r_dist_coeff', default=1, type=float)
    parser.add_argument('--tr_dist_coeff', default=1, type=float)

    parser.add_argument('--use_next_obs_beta', default="False", type=str)
    parser.add_argument('--use_rewards_beta', default="False", type=str)
    parser.add_argument('--next_obs_beta', default=1.0, type=float)
    parser.add_argument('--rewards_beta', default=1.0, type=float)

    parser.add_argument('--sample_reg_method', default="", type=str)
    parser.add_argument('--use_sample_reg', default="False", type=str)

    parser.add_argument('--wgan_lambda', default=5, type=float)

    parser.add_argument('--wide_exp', default="False", type=str)


    # parser.add_argument('--recon_coeff', default=200, type=float)
    # parser.add_argument('--onpol_c_coeff', default=100, type=float)
    # parser.add_argument('--same_c_coeff', default=200, type=float)
    # parser.add_argument('--c_coef', default=10, type=int)
    # parser.add_argument('--bisim_coeff', default=50, type=int)
    # parser.add_argument('--gen_coeff', default=1.0, type=float)
    # parser.add_argument('--bisim_penalty_coeff', default=1.0, type=float)
    parser.add_argument('--recon_coeff', default=200, type=float)
    parser.add_argument('--onpol_c_coeff', default=100, type=float)
    parser.add_argument('--same_c_coeff', default=200, type=float)
    parser.add_argument('--cy_coef', default=10, type=int)
    parser.add_argument('--bisim_coeff', default=50, type=int)
    parser.add_argument('--gen_coeff', default=1.0, type=float)
    parser.add_argument('--bisim_penalty_coeff', default=1.0, type=float)
    parser.add_argument('--q_reg_coeff', default=1.0, type=float)
    parser.add_argument('--offpol_c_coeff', default=100, type=float)


    parser.add_argument('--c_off_batch_size', default=32, type=int)

    # parser.add_argument('--use_kl_penalty', default="False", type=str)
    # parser.add_argument('--policy_kl_reg_coeff', default=0, type=float)
    # parser.add_argument('--state_penalty_coeff', default=0, type=float)

    parser.add_argument('--use_fakesample_representation', default="False", type=str)

    parser.add_argument('--use_next_state_bisim', default="False", type=str)
    parser.add_argument('--use_next_obs_in_context', default="False", type=str)

    parser.add_argument('--use_gan', default="True", type=str)
    parser.add_argument('--use_penalty', default="False", type=str)

    parser.add_argument('--gen_freq', default=5, type=int)

    parser.add_argument('--use_c_dist_clear', default="True", type=str)

    parser.add_argument('--use_episodic_online_buffer', default="True", type=str)

    parser.add_argument('--use_next_obs_Q_reg', default="False", type=str)


    parser.add_argument('--use_new_batch_for_fake', default="True", type=str)

    parser.add_argument('--c_kl', default=0.05, type=float)  # 0.05
    parser.add_argument('--c_vae_freq', default=2, type=int)
    parser.add_argument('--c_buffer_size', default=500, type=int)

    parser.add_argument('--f_bsize', default=256, type=int)

    parser.add_argument('--seed', default=1234, type=int)

    # fakesample_rl_tran_batch_size
    # c_distri_vae_train_freq


    parser.add_argument('--which_sac_file', default="sac_revised", type=str)  # sac_revised
    # parser.add_argument('--date', type=str)  # sac_revised


    # parser.add_argument('--use_contrastive_c', default="False", type=str)
    # parser.add_argument('--use_decrease_mask', default="False", type=str)
    # parser.add_argument('--decrease_rate', default=1, type=float)
    # parser.add_argument('--intra_group_ratio', default=1, type=float)
    # parser.add_argument('--inter_group_ratio', default=1, type=float)


    parser.add_argument('--psi_aux_vae_beta', default=2.0, type=float)


    parser.add_argument('--use_state_noise', default="False", type=str)
    parser.add_argument('--use_inter_samples', default="False", type=str)

    parser.add_argument('--use_decoder_next_state', default="False", type=str)

    parser.add_argument('--fakesample_cycle', default="True", type=str)


    parser.add_argument('--eval_ep', default=2, type=int)


    parser.add_argument('--dirichlet_sample_freq_for_exploration', default=20, type=int)

    # parser.add_argument('--wandb_project', default="김정모_metaRL_new3", type=str)  # mo_metaRL <-- 개인 프로젝트
    # 7316f79887c82500a01a529518f2af73d5520255

    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--exp_num', default="1", type=str)

    parser.add_argument('--use_manual_tasks_sampling', default="False", type=str)
    parser.add_argument('--meta_batch', default=0, type=int)
    parser.add_argument('--num_fake_tasks', default=0, type=int)
    parser.add_argument('--num_dirichlet_tasks', default=0, type=int)


    parser.add_argument('--c_off_buffer_length', default=5000, type=int)

    parser.add_argument('--plot_beta_area', default="True", type=str)
    parser.add_argument('--compute_overestimation_bias', default="False", type=str)

    parser.add_argument('--plot_wide_tsne', default="False", type=str)

    parser.add_argument('--expert', default="False", type=str)

    parser.add_argument('--wandb_project', default="김정모_metaRL_new4", type=str)
    parser.add_argument('--wandb_group', default="", type=str)

    parser.add_argument('--eval_generator', default="False", type=str)
    parser.add_argument('--pretrained_model_path', default="", type=str)

    parser.add_argument('--num_exp_traj_eval', default=2, type=int)








    args, rest_args = parser.parse_known_args()

    main(args)

