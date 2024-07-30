## command
    python plot_trajectory.py --env_name ant-goal-inter --seed 4905 --pretrained_model_path pretrained/antgoal-0728 --use_c_vae False --dirichlet_sample_freq_for_exploration 20 --beta 2 --which_sac_file sac_revised --use_information_bottleneck False --eval_ep 2 --gpu_num 0

- dirichlet_sample_freq_for_exploration : exploration에서 몇 스텝마다 z를 바꿔줄건지.
- beta : 인터폴레이션을 얼마나 넓게 할건지