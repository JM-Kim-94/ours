# import numpy as np

# from rlkit.envs.ant import AntEnv
# # from gym.envs.mujoco.ant import AntEnv

# class MultitaskAntEnv(AntEnv):
#     def __init__(self, task={}, num_train_tasks=2, eval_tasks_list=[], TSNE_tasks_list=[], index_sorting=0, linear_sorting=0, use_ref_task=0, **kwargs):
#         self._task = task
#         self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, TSNE_tasks_list, index_sorting, linear_sorting, use_ref_task)
#         print("all tasks : ", self.tasks)
#         self._goal = self.tasks[0]['goal']
#         super(MultitaskAntEnv, self).__init__(**kwargs)

#     """
#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]

#         forward_vel = (xposafter - xposbefore) / self.dt
#         forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
#         ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

#         observation = self._get_obs()
#         reward = forward_reward - ctrl_cost
#         done = False
#         infos = dict(reward_forward=forward_reward,
#                      reward_ctrl=-ctrl_cost, task=self._task)
#         return (observation, reward, done, infos)
#     """


#     def get_all_task_idx(self):
#         # return range(len(self.tasks))
#         return list(range(len(self.tasks))), self.tasks

#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         self._goal = self._task['goal']  # assume parameterization of task by single vector
#         self.reset()


import numpy as np

from .ant import AntEnv
# from rlkit.envs.ant import AntEnv
# from gym.envs.mujoco.ant import AntEnv

# class MultitaskAntEnv(AntEnv):
#     def __init__(self, num_train_tasks=2, eval_tasks_list=[], TSNE_tasks_list=[], index_sorting=0, linear_sorting=0, use_ref_task=0, **kwargs):
#         self._task = {}
#         self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, TSNE_tasks_list, index_sorting, linear_sorting, use_ref_task)
#         print("all tasks : ", self.tasks)
#         self._goal = self.tasks[0]['goal']
#         super(MultitaskAntEnv, self).__init__(**kwargs)

class MultitaskAntEnv(AntEnv):
    def __init__(self, num_train_tasks=2, eval_tasks_list=[], TSNE_tasks_list=[], index_sorting=0, linear_sorting=0, use_ref_task=0):
        self._task = {}
        self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, TSNE_tasks_list, index_sorting, linear_sorting, use_ref_task)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']
        # super(MultitaskAntEnv, self).__init__()
        super().__init__()

    """
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)
    """


    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()





# num_train_tasks = 100
# eval_tasks_list = [[1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75]]
# TSNE_tasks_list = [[0.5,  0], [0, 0.5 ], [-0.5,  0], [0, -0.5 ],
#                    [1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75],
#                    [2.75, 0], [0, 2.75], [-2.75, 0], [0, -2.75]] 
# index_sorting, linear_sorting, use_ref_task =0, 0, 0
# ant_mltask_test_env = MultitaskAntEnv(num_train_tasks, eval_tasks_list, TSNE_tasks_list)

# s = ant_mltask_test_env.reset()

# obs_dim = ant_mltask_test_env.observation_space
# action_dim = ant_mltask_test_env.action_space

# print("ant_mltask_test_env OBS_DIM :", obs_dim)
# print("ant_mltask_test_env ACTION_DIM :", action_dim)
# print("ant_mltask_test_env S :", s.shape)




