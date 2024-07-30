# # import numpy as np
# # from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

# # from . import register_env


# # @register_env('walker-rand-params')
# # class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
# #     def __init__(self, n_tasks=2, randomize_tasks=True):
# #         super(WalkerRandParamsWrappedEnv, self).__init__()
# #         self.tasks = self.sample_tasks(n_tasks)
# #         self.reset_task(0)

# #     def get_all_task_idx(self):
# #         return range(len(self.tasks))

# #     def reset_task(self, idx):
# #         self._task = self.tasks[idx]
# #         self._goal = idx
# #         self.set_task(self._task)
# #         self.reset()



# import numpy as np
# from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

# from . import register_env

# # "env_params": {
# #     "num_train_tasks": 100,
# #     "eval_tasks_list":                 [0.75, 1.25, 1.75, 2.25, 2.75],
# #     "indistribution_train_tasks_list": [0.1, 0.25, 3.1, 3.25],
# #     "TSNE_tasks_list": [0.1, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.1, 3.25],
# #     "ood_type": "inter"
# # },

# @register_env('walker-mass')
# class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
#     def __init__(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, ood_type='inter'):
        
#         super(WalkerRandParamsWrappedEnv, self).__init__()
#         # n_train_tasks, n_eval_tasks, n_indistribution_tasks, n_tsne_tasks, ood='inter'
#         self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, 
#                                         indistribution_train_tasks_list, TSNE_tasks_list, ood_type)
#         self.reset_task(0)
    
#     def get_obs_dim(self):
#         return int(np.prod(self._get_obs().shape))


#     def get_all_task_idx(self):
#         # return range(len(self.tasks))
#         return list(range(len(self.tasks))), self.tasks


#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         self._goal = idx
#         self.set_task(self._task)
#         self.reset()



# import numpy as np
# from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

# from . import register_env


# @register_env('walker-rand-params')
# class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
#     def __init__(self, n_tasks=2, randomize_tasks=True):
#         super(WalkerRandParamsWrappedEnv, self).__init__()
#         self.tasks = self.sample_tasks(n_tasks)
#         self.reset_task(0)

#     def get_all_task_idx(self):
#         return range(len(self.tasks))

#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         self._goal = idx
#         self.set_task(self._task)
#         self.reset()


import numpy as np
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

from . import register_env


# "env_params": {
#     "num_train_tasks": 100,
#     "eval_tasks_list":                 [0.75, 1.25, 1.75, 2.25, 2.75],
#     "indistribution_train_tasks_list": [0.1, 0.25, 3.1, 3.25],
#     "TSNE_tasks_list": [0.1, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.1, 3.25],
#     "ood_type": "inter"
# },

@register_env('walker-mass')
class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, num_train_tasks,
                 eval_tasks_list,
                 indistribution_train_tasks_list,
                 TSNE_tasks_list,
                 ood_type='inter',
                 expert=False):

        super(WalkerRandParamsWrappedEnv, self).__init__()

        self.expert = expert

        self.tasks, self.tasks_value = self.sample_tasks(num_train_tasks, eval_tasks_list,
                                                         indistribution_train_tasks_list,
                                                         TSNE_tasks_list, ood_type, expert)
        
        self.reset_task(0)

    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks_value

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()