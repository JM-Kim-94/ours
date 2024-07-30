"""MuJoCo150버전"""
import numpy as np
import random
# from rlkit.envs.ant import AntEnv
from gym.envs.mujoco import AntEnv as AntEnv

from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(AntEnv):
    def __init__(self, num_train_tasks=2, eval_tasks_list=[], indistribution_train_tasks_list=[], TSNE_tasks_list=[],
                        use_cfrc=True, ood="inter", use_ref_task=0, expert=False):  # ood = "inter" or "extra"
        self.expert = expert
        self._task = {}
        self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                                        use_cfrc, ood, use_ref_task)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']
        self.use_cfrc = use_cfrc

        super(AntGoalEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            xyz_coor=torso_xyz_before,
        )

    # def sample_tasks(self, num_tasks):
    #     a = np.random.random(num_tasks) * 2 * np.pi
    #     r = 3 * np.random.random(num_tasks) ** 0.5
    #     goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
    #     tasks = [{'goal': goal} for goal in goals]
    #     return tasks

    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                            use_cfrc, ood, use_ref_task):

        # print("num_train_tasks", num_train_tasks)
        # print("eval_tasks_list", eval_tasks_list)
        # print("indistribution_train_tasks_list", indistribution_train_tasks_list)
        # print("TSNE_tasks_list", TSNE_tasks_list)

        if ood == "inter":
            if self.expert:
                goal_train = [[1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75]]
            else:
                goal_train = []
            for i in range(num_train_tasks):
                prob = random.random()  # np.random.uniform()
                if prob < 4.0 / 15.0:
                    r = random.random() ** 0.5  # [0, 1]
                else:
                    # r = random.random() * 0.5 + 2.5  # [2.5, 3.0]
                    r = (random.random() * 2.75 + 6.25) ** 0.5
                theta = random.random() * 2 * np.pi  # [0.0, 2pi]
                goal_train.append([r * np.cos(theta), r * np.sin(theta)])


        elif ood == "extra":
            goal_train = []
            for i in range(num_train_tasks):
                # r = random.random() * 1.5 + 1.0  # [1.0, 2.5]
                r = (random.random() * 5.25 + 1.0) ** 0.5
                theta = random.random() * 2 * np.pi  # [0.0, 2pi]
                goal_train.append([r * np.cos(theta), r * np.sin(theta)])
        
        elif ood == "extra-hard":
            goal_train = []
            for i in range(num_train_tasks):
                # r = random.random() * 0.5 + 1.5  # [1.5, 2.0]
                r = (random.random() * 1.75 + 2.25) ** 0.5
                theta = random.random() * 2 * np.pi  # [0.0, 2pi]
                goal_train.append([r * np.cos(theta), r * np.sin(theta)])
        

        goal_test = eval_tasks_list

        goal_indistribution = indistribution_train_tasks_list

        goal_tsne = TSNE_tasks_list

        #
        theta_list = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * np.pi / 4
        train_r_list = np.array([0.5, 1.0, 2.5, 3.0])
        test_r_list = np.array([1.5, 2.0])
        train_tsne_tasks_list, test_tsne_tasks_list = [], []

        for r in train_r_list:
            for theta in theta_list:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                train_tsne_tasks_list.append([x, y])

        for r in test_r_list:
            for theta in theta_list:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                test_tsne_tasks_list.append([x, y])


        goals = goal_train + goal_test + goal_indistribution + goal_tsne + train_tsne_tasks_list + test_tsne_tasks_list
        goals = np.array(goals)

        tasks = [{'goal': goal} for goal in goals]

        return tasks

    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat,
    #         self.sim.data.qvel.flat,
    #         np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    #     ])
    def _get_obs(self):
        if self.use_cfrc:
            o = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            o = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        return o

        # data = self.sim.data
        # return np.concatenate([
        #         data.qpos.flat,
        #         data.qvel.flat,
        #         data.cfrc_ext.flat
        #                       ])

        # data = self.sim.data
        # return np.concatenate([data.qpos.flat[2:],
        #                        data.qvel.flat,
        #                        data.cinert.flat,
        #                        data.cvel.flat,
        #                        data.qfrc_actuator.flat,
        #                        data.cfrc_ext.flat])

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()



# """MuJoCo210버전"""
# import numpy as np
# import random
# from gym import utils
# from gym.envs.mujoco import ant_v4
# from gym.spaces import Box

# from . import register_env


# # Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
# @register_env('ant-goal')
# class AntGoalEnv(ant_v4):
#     def __init__(self, num_train_tasks=2, eval_tasks_list=[], indistribution_train_tasks_list=[], TSNE_tasks_list=[],
#                         index_sorting=0, linear_sorting=0, use_ref_task=0):
#         self._task = {}
#         self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
#                                         index_sorting, linear_sorting, use_ref_task)
#         print("all tasks : ", self.tasks)
#         self._goal = self.tasks[0]['goal']

#         super(AntGoalEnv, self).__init__(exclude_current_positions_from_observation=False)

#     def step(self, action):

#         xy_position_before = self.get_body_com("torso")[:2].copy()
#         self.do_simulation(action, self.frame_skip)
#         xy_position_after = self.get_body_com("torso")[:2].copy()
#         print("xy_position_after", xy_position_after)

#         goal_reward = -np.sum(np.abs(xy_position_after - self._goal))  # make it happy, not suicidal
#         healthy_reward = self.healthy_reward

#         # rewards = forward_reward + healthy_reward
#         rewards = goal_reward + healthy_reward

#         costs = ctrl_cost = self.control_cost(action)

#         terminated = self.terminated
#         observation = self._get_obs()

#         info = {
#             "reward_goal": goal_reward,
#             "reward_ctrl": -ctrl_cost,
#             "reward_survive": healthy_reward,
#             "x_position": xy_position_after[0],
#             "y_position": xy_position_after[1],
#             "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
#         }

#         reward = rewards - costs

#         return observation, reward, terminated, False, info


#         # 원래 리워드 = goal_reward - ctrl_cost - contact_cost + survive_reward
#         #              goal_reward - .1 * np.square(action).sum() - 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))) + 0
#         # 수정 리워드 = forward_reward + healthy_reward - costs


#         # self.do_simulation(action, self.frame_skip)
#         # xposafter = np.array(self.get_body_com("torso"))

#         # goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal))  # make it happy, not suicidal

#         # ctrl_cost = .1 * np.square(action).sum()
#         # contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
#         # survive_reward = 0.0
#         # reward = goal_reward - ctrl_cost - contact_cost + survive_reward
#         # state = self.state_vector()
#         # done = False
#         # ob = self._get_obs()
#         # return ob, reward, done, dict(
#         #     goal_forward=goal_reward,
#         #     reward_ctrl=-ctrl_cost,
#         #     reward_contact=-contact_cost,
#         #     reward_survive=survive_reward,
#         # )

#     # def sample_tasks(self, num_tasks):
#     #     a = np.random.random(num_tasks) * 2 * np.pi
#     #     r = 3 * np.random.random(num_tasks) ** 0.5
#     #     goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
#     #     tasks = [{'goal': goal} for goal in goals]
#     #     return tasks

#     def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, index_sorting, linear_sorting,
#                      use_ref_task):

#         print("num_train_tasks", num_train_tasks)
#         print("eval_tasks_list", eval_tasks_list)
#         print("indistribution_train_tasks_list", indistribution_train_tasks_list)
#         print("TSNE_tasks_list", TSNE_tasks_list)

#         # goal_train = []
#         # for i in range(num_train_tasks):
#         #     prob = np.random.uniform()
#         #     if prob >= 0.5:
#         #         r = np.random.uniform(0.0, 1.0)
#         #     else:
#         #         r = np.random.uniform(2.5, 3.0)
#         #     theta = np.random.random() * 2 * np.pi
#         #     goal_train.append([r * np.cos(theta), r * np.sin(theta)])
#         goal_train = []
#         for i in range(num_train_tasks):
#             prob = random.random()  # np.random.uniform()
#             if prob < 4.0 / 15.0:
#                 r = random.random()  # [0, 1]
#             else:
#                 r = random.random() * 0.5 + 2.5  # [2.5, 3.0]
#             theta = random.random() * 2 * np.pi  # [0.0, 2pi]
#             goal_train.append([r * np.cos(theta), r * np.sin(theta)])

#         goal_test = eval_tasks_list

#         goal_indistribution = indistribution_train_tasks_list

#         goal_tsne = TSNE_tasks_list

#         goals = goal_train + goal_test + goal_indistribution + goal_tsne
#         goals = np.array(goals)

#         tasks = [{'goal': goal} for goal in goals]
#         # print("ANT-GOAL TASKS: ", tasks)
#         return tasks

#     # def _get_obs(self):
#     #     return np.concatenate([
#     #         self.sim.data.qpos.flat,
#     #         self.sim.data.qvel.flat,
#     #         np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
#     #     ])
#     def _get_obs(self):
#         o = np.concatenate([
#             self.sim.data.qpos.flat,
#             self.sim.data.qvel.flat,
#             # self.data.qpos.flat,
#             # self.data.qvel.flat,
#             # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
#         ])
#         # print("O_before : ", o)  29개까지는 원래 obs를 출력하지만, 29부터 113까지는 0을 출력함 --> 29까지만 사용하돌고 변경
#         # print("O_after : ", o[:29])
#         return o

#         # data = self.sim.data
#         # return np.concatenate([
#         #         data.qpos.flat,
#         #         data.qvel.flat,
#         #         data.cfrc_ext.flat
#         #                       ])

#         # data = self.sim.data
#         # return np.concatenate([data.qpos.flat[2:],
#         #                        data.qvel.flat,
#         #                        data.cinert.flat,
#         #                        data.cvel.flat,
#         #                        data.qfrc_actuator.flat,
#         #                        data.cfrc_ext.flat])

#     def get_all_task_idx(self):
#         # return range(len(self.tasks))
#         return list(range(len(self.tasks))), self.tasks

#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         self._goal = self._task['goal']  # assume parameterization of task by single vector
#         self.reset()




