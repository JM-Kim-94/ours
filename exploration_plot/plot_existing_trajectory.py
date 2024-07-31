import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.collections import PatchCollection

rng = np.random.default_rng(seed=0)
pretrained_model_path = 'pretrained/antgoal-0728'
num_exp_traj_eval = 2
_beta = 2.
dirichlet_sample_freq_for_exploration = 20

pretrained_path = pretrained_model_path
beta_1_traj_file = 'beta1.0.npy'
beta_2_traj_file = 'beta2.0.npy'
beta_3_traj_file = 'beta3.0.npy'

alphas = [0.4, 0.4, 1.0]
line_styles = [':', '-.', '-']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'W']
episode_labels = ["exploration episode 1", "exploration episode 2", "RL episode"]

oods = range(0, 4)
inner_indist = range(4, 8)
outer_indist = range(8, 12)


def main():
    beta1_traj = np.load(f'{pretrained_path}/{beta_1_traj_file}', allow_pickle=True).item()
    beta2_traj = np.load(f'{pretrained_path}/{beta_2_traj_file}', allow_pickle=True).item()
    beta3_traj = np.load(f'{pretrained_path}/{beta_3_traj_file}', allow_pickle=True).item()

    task_trajs = [beta1_traj, beta2_traj, beta3_traj]
    subplot_offset = [0, 3, 6]
    betas = [1., 2., 3.]

    fig, axes = plt.subplots(3, 3, figsize=(13, 14))
    for offset, task_traj, beta in zip(subplot_offset, task_trajs, betas):
        trajs = np.array(task_traj['traj'])
        tasks = np.array(task_traj['task'])

        plot_merged_traj(tasks, trajs, offset, beta)
    for ax in axes.flat:
        ax.set(xlabel='x-position', ylabel='y-position')
        ax.label_outer()
    plt.tight_layout()
    plt.savefig(pretrained_path + f'/merged.png')


def plot_task_traj(tasks, trajs):
    for i, task_traj in enumerate(trajs):
        plt.figure(figsize=(4, 4))
        plot_setup(str(tasks[i]), [tasks[i]])
        for j, epi_traj in enumerate(task_traj):
            if j < num_exp_traj_eval:
                plt.plot(epi_traj[:, 0], epi_traj[:, 1],
                         linewidth=1.0,
                         label=episode_labels[j],
                         alpha=alphas[j],
                         color='black',
                         linestyle=line_styles[j])
                plt.scatter(epi_traj[-1, 0], epi_traj[-1, 1], s=10, color="blue")
            else:
                plt.plot(epi_traj[:, 0], epi_traj[:, 1],
                         linewidth=2.0,
                         label=episode_labels[j],
                         alpha=alphas[j],
                         color='red',
                         linestyle=line_styles[j])
                plt.scatter(epi_traj[-1, 0], epi_traj[-1, 1], s=10, color="red")
        plt.legend()
        plt.tight_layout()
        beta = str(_beta)
        randcstep = str(dirichlet_sample_freq_for_exploration)
        plt.savefig(
            pretrained_path + f'/{i}' + ',Zfrq' + randcstep + "step," + "beta" + beta + '.png')
        plt.close()


def plot_merged_traj(tasks, trajs, subplot_offset, beta):

    tasks_grouped = [tasks[oods, ...], tasks[inner_indist, ...], tasks[outer_indist, ...]]
    trajs_grouped = [trajs[oods, ...], trajs[inner_indist, ...], trajs[outer_indist, ...]]
    titles = [f'OOD (β={beta})', f'Inner Indistribution (β={beta})', f'Outer Indistribution (β={beta})']
    subplot_index = np.array([1, 2, 3]) + subplot_offset

    for _tasks, _trajs, title, i in zip(tasks_grouped, trajs_grouped, titles, subplot_index):
        plt.subplot(3, 3, i)
        plot_setup(title, _tasks)

        traj_expl_1 = rng.choice(_trajs[:, 0, ...])
        traj_expl_2 = rng.choice(_trajs[:, 1, ...])
        merged_traj = _trajs[:, 2, ...]

        for j, epi_traj in enumerate([traj_expl_1, traj_expl_2, *merged_traj]):
            if j < 2:
                plt.plot(epi_traj[:, 0], epi_traj[:, 1],
                         linewidth=1.0,
                         label=episode_labels[j],
                         alpha=alphas[j],
                         color='black',
                         linestyle=line_styles[j])
                plt.scatter(epi_traj[-1, 0], epi_traj[-1, 1], s=10, color="blue")
            else:
                plt.plot(epi_traj[:, 0], epi_traj[:, 1],
                         linewidth=2.0,
                         label=episode_labels[-1],
                         alpha=alphas[-1],
                         color='red',
                         linestyle=line_styles[-1])
                plt.scatter(epi_traj[-1, 0], epi_traj[-1, 1], s=10, color="red")

        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys())

def plot_setup(title, tasks):
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title(title)

    patches = PatchCollection([
        Wedge((.0, .0), .75, 0, 360),  # Full circle
        Wedge((.0, .0), 3., 0, 360, width=3. - 2.5),  # Full ring
    ])
    patches.set_color('gray')
    patches.set_alpha(0.5)
    plt.gca().add_collection(patches)

    for task in tasks:
        plt.plot(task[0], task[1], marker='x', color='r')

    plt.tight_layout()
    plt.gca().set_aspect('equal')


if __name__ == "__main__":
    main()
