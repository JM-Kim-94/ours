




# import numpy as np
# import random
# from gym.envs.mujoco import HumanoidEnv as HumanoidEnv

# from . import register_env

# def mass_center(model, sim):
#     mass = np.expand_dims(model.body_mass, 1)
#     xpos = sim.data.xipos
#     return (np.sum(mass * xpos, 0) / np.sum(mass))


# @register_env('humanoid-dir')
# class HumanoidDirEnv(HumanoidEnv):

#     def __init__(self, num_train_tasks=2, eval_tasks_list=[], indistribution_train_tasks_list=[], TSNE_tasks_list=[],
#                        index_sorting=0, linear_sorting=0, use_ref_task=0):
#         self._task = {}
#         self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
#                                        index_sorting, linear_sorting, use_ref_task)
        
#         # self.reset_task(0)
#         print("all tasks : ", self.tasks)
#         self._goal = self.tasks[0]['goal']
#         super(HumanoidDirEnv, self).__init__()

#     def step(self, action):
#         pos_before = np.copy(mass_center(self.model, self.sim)[:2])
#         self.do_simulation(action, self.frame_skip)
#         pos_after = mass_center(self.model, self.sim)[:2]

#         alive_bonus = 5.0
#         data = self.sim.data
#         goal_direction = (np.cos(self._goal), np.sin(self._goal))
#         lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
#         quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
#         quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
#         quad_impact_cost = min(quad_impact_cost, 10)
#         reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
#         qpos = self.sim.data.qpos
#         done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

#         # alive_bonus = 5.0
#         # data = self.sim.data
#         # goal_direction = (np.cos(self._goal), np.sin(self._goal))
#         # lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
#         # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
#         # quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
#         # quad_impact_cost = min(quad_impact_cost, 10)
#         # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
#         # qpos = self.sim.data.qpos
#         # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

#         return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
#                                                    reward_quadctrl=-quad_ctrl_cost,
#                                                    reward_alive=alive_bonus,
#                                                    reward_impact=-quad_impact_cost)

#     def _get_obs(self):
#         data = self.sim.data
#         return np.concatenate([data.qpos.flat[2:],
#                                data.qvel.flat,
#                                data.cinert.flat,
#                                data.cvel.flat,
#                                data.qfrc_actuator.flat,
#                                data.cfrc_ext.flat])

#     def get_all_task_idx(self):
#         # return range(len(self.tasks))
#         return list(range(len(self.tasks))), self.tasks

#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         self._goal = self._task['goal']  # assume parameterization of task by single vector
#         self.reset()  # 이거 원래는 없었음.

#     def sample_tasks(self, num_train_tasks, eval_tasks_list, TSNE_tasks_list, index_sorting, linear_sorting, use_ref_task):
#         # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
#         train_dir = np.random.uniform(0., 12/8 * np.pi, size=(num_train_tasks,)).tolist()  # [i * np.pi / 8 for i in range(12)]
#         eval_dir = [i * np.pi / 8 + 6 * np.pi / 4 for i in range(4)]  # np.random.uniform(12/8 * np.pi, 16/8 * np.pi, size=(30,)).tolist()
#         tsne_dir = [i * np.pi / 8 for i in range(16)]

#         directions = train_dir + eval_dir + tsne_dir

#         tasks = [{'goal': d} for d in directions]
#         return tasks











import numpy as np
import random
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv

from . import register_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


@register_env('humanoid-dir')
class HumanoidDirEnv(HumanoidEnv):

    def __init__(self, num_train_tasks=2, eval_tasks_list=[], indistribution_train_tasks_list=[], TSNE_tasks_list=[],
                       index_sorting=0, linear_sorting=0, use_ref_task=0):
        self._task = {}
        self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                                       index_sorting, linear_sorting, use_ref_task)
        
        # self.reset_task(0)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']
        super(HumanoidDirEnv, self).__init__()

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal), np.sin(self._goal))
        lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        # done = False # 추가함 

        # alive_bonus = 5.0
        # data = self.sim.data
        # goal_direction = (np.cos(self._goal), np.sin(self._goal))
        # lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        # quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        # quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        # quad_impact_cost = min(quad_impact_cost, 10)
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        # qpos = self.sim.data.qpos
        # done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()  # 이거 원래는 없었음.
    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                           index_sorting, linear_sorting, use_ref_task):

        train_goal_dir = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]  # [i * np.pi / 2 for i in range(4)]
        eval_goal_dir  = [0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi]   # [i * np.pi / 2 + np.pi / 4 for i in range(4)]
        tsne_goal_dir  = [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]  # [i * np.pi / 4 for i in range(8)]  # train_goal_dir + eval_goal_dir

        goal_dirs = train_goal_dir + eval_goal_dir + tsne_goal_dir

        tasks = [{'goal': goal_dir} for goal_dir in goal_dirs]

        return tasks












