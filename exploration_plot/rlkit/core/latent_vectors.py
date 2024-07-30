import json
from dataclasses import dataclass
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from MulticoreTSNE import MulticoreTSNE as TSNE


@dataclass
class EnvWrapper:
    name: str
    train_tasks: list
    test_tasks: list
    train_tasks_gradient: list
    test_tasks_gradient: list


cheetah_vel_inter = EnvWrapper(
    "cheetah-vel-inter",
    # [*range(109, 120)],
    # [*range(120, 144)],
    [*range(118, 129)],
    [*range(129, 153)],
    [0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
     2.9]
)
ant_dir_4 = EnvWrapper(
    "ant-dir-4",
    [*range(16, 20)],
    [*range(20, 24)],
    [0, 0.5, 1, 1.5],
    [0.25, 0.75, 1.25, 1.75]
)
walker_mass_inter = EnvWrapper(
    "walker-mass-inter",
    [*range(118, 129)],
    [*range(129, 153)],
    [0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
     2.9]
)
ant_goal_inter = EnvWrapper(
    "ant-goal-inter",
    [*range(174, 206)],
    [*range(206, 221)],
    [0.5 for _ in range(8)] + [1.0 for _ in range(8)] + [2.5 for _ in range(8)] + [3.0 for _ in range(8)],
    [1.5 for _ in range(8)] + [2.0 for _ in range(8)]
)


class EnvType(Enum):
    CHEETAH_VEL_INTER = cheetah_vel_inter
    ANT_DIR_4 = ant_dir_4
    WALKER_MASS_INTER = walker_mass_inter
    ANT_GOAL_INTER = ant_goal_inter

    @staticmethod
    def parse(text: str):
        if "cheetah-vel-inter" in text:
            return EnvType.CHEETAH_VEL_INTER
        elif "ant-dir-4" in text:
            return EnvType.ANT_DIR_4
        elif "walker-mass-inter" in text:
            return EnvType.WALKER_MASS_INTER
        elif "ant-goal-inter" in text:
            return EnvType.ANT_GOAL_INTER


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class LatentVectors:
    vectors: np.ndarray
    indices: dict[any, list]
    epoch: int
    env: EnvType

    tsne_model = TSNE(n_components=2, random_state=0, perplexity=50, n_jobs=4)

    @staticmethod
    def make(path: str):
        with open(path, 'r') as json_file:
            json_load = json.load(json_file)
        return LatentVectors(
            vectors=np.asarray(json_load["vectors"]),
            indices=json_load["indices"],
            epoch=json_load["epoch"],
            env=EnvType.parse(json_load["env"])
        )

    def save(self, path: str):
        json_dump = json.dumps(
            {'vectors': self.vectors, 'indices': self.indices, 'epoch': self.epoch, 'env': self.env.value.name},
            cls=NumpyEncoder)
        full_path = f"{path}/{self.epoch}.json"
        with open(full_path, 'w') as json_file:
            json_file.write(json_dump)

    def save_plot(self, path: str, wandb=None):
        full_path = f"{path}/{self.epoch}.png"
        plt.figure()
        env_detail = self.env.value
        result = LatentVectors.tsne_model.fit_transform(np.array(self.vectors))
        print("result", result)
        print("self.indices", self.indices)

        custom_reds = mcolors.LinearSegmentedColormap.from_list('custom_reds',
                                                                mpl.colormaps['Reds'](np.linspace(0.3, 0.8, 256)))
        custom_blues = mcolors.LinearSegmentedColormap.from_list('custom_blues',
                                                                 mpl.colormaps['Blues'](np.linspace(0.3, 0.8, 256)))

        train_tasks = {
            "x": np.empty(0),
            "y": np.empty(0),
            "c": np.empty(0),
        }

        test_tasks = {
            "x": np.empty(0),
            "y": np.empty(0),
            "c": np.empty(0),
        }

        for i, task_num in enumerate(env_detail.train_tasks):
            train_tasks["x"] = np.concatenate((train_tasks["x"], result[:, 0][self.indices[task_num]]))
            train_tasks["y"] = np.concatenate((train_tasks["y"], result[:, 1][self.indices[task_num]]))
            train_tasks["c"] = np.concatenate(
                (train_tasks["c"], np.full(self.indices[task_num].__len__(), env_detail.train_tasks_gradient[i])))

        for i, task_num in enumerate(env_detail.test_tasks):
            test_tasks["x"] = np.concatenate((test_tasks["x"], result[:, 0][self.indices[task_num]]))
            test_tasks["y"] = np.concatenate((test_tasks["y"], result[:, 1][self.indices[task_num]]))
            test_tasks["c"] = np.concatenate(
                (test_tasks["c"], np.full(self.indices[task_num].__len__(), env_detail.test_tasks_gradient[i])))

        train_scatter = plt.scatter(train_tasks["x"],
                                    train_tasks["y"],
                                    c=train_tasks["c"],
                                    s=10,
                                    cmap=custom_blues)

        test_scatter = plt.scatter(test_tasks["x"],
                                   test_tasks["y"],
                                   c=test_tasks["c"],
                                   s=10,
                                   cmap=custom_reds)

        plt.colorbar(test_scatter, orientation='vertical')
        plt.colorbar(train_scatter, orientation='vertical')
        plt.savefig(full_path, bbox_inches='tight')

        if wandb is not None:
            wandb.log({
                "Eval_tsne/tSNE_EP" + str(self.epoch): [wandb.Image(full_path)]
            })

# path='/home/mlic/mo/baselines/LDM_yisak_tsne/logs/logs_cheetah-vel-inter-v0/varibad_4957__16:07_13:45:22/tsne'
# path='/home/mlic/mo/baselines/LDM_yisak_tsne/logs/logs_ant-dir-4개-v0/varibad_4957__16:07_14:36:09/tsne'
# path='/home/mlic/mo/baselines/LDM_yisak_tsne/logs/logs_ant-goal-inter-v0/varibad_4957__16:07_14:53:08/tsne'
# path='/home/mlic/mo/baselines/LDM_yisak_tsne/logs/logs_walker-mass-inter-v0/varibad_4957__16:07_14:51:13/tsne'
# path='/home/user/mo/baselines/pearl/logs/walker-mass-inter/2024-07-17 02:26:06.690772/tsne'
# lv = LatentVectors.make(f'{path}/0.json')
# lv.save_plot(path)

# path = '/home/mo/PycharmProjects/metarl/aaai2025/ours_codebase/20.json'
# lv = LatentVectors.make(path)
# lv.save_plot(path)

