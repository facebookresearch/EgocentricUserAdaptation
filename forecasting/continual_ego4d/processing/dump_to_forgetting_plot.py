"""
From a Re-exposure forgetting experiment dump, make a plot.
"""
import os
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# To save imgs
local_csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
csv_dirname = local_csv_dirname

# Dump paths to load
user_dir_fmt = "user_{}"
dump_filename = "stream_info_dump.pth"
parent_outputdir = "/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/results/ego4d_action_recog/forgetting_eval/logs/2022-09-23_21-12-33_UIDec506d45-3018-468f-b63a-89744c5d10f9/user_logs/"

# Plot configs
plot_config = {
    "color": 'royalblue',
    "dpi": 600,
    "figsize": (6, 6),
    "xlabel": "re-exposure iterations",
    "ylabel": ""
}

# KEYS
saved_dumpkeys = [
    'train_action_past/FORG_EXPOSE_loss',
    'train_action_past/FORG_EXPOSE_top1acc',

    'train_verb_past/FORG_EXPOSE_loss',
    'train_verb_past/FORG_EXPOSE_top1acc',
    'train_verb_past/FORG_EXPOSE_top5acc',
    'train_verb_past/FORG_EXPOSE_top20acc',

    'train_noun_past/FORG_EXPOSE_loss',
    'train_noun_past/FORG_EXPOSE_top1acc',
    'train_noun_past/FORG_EXPOSE_top5acc',
    'train_noun_past/FORG_EXPOSE_top20acc'
]
CHOSEN_KEY = 'train_action_past/FORG_EXPOSE_loss'

# Adapt cfg
ylabel_map = {
    'train_action_past/FORG_EXPOSE_loss': r"$RF_{\text{action}}$",
}
plot_config['ylabel'] = ylabel_map[CHOSEN_KEY]

modes = ['action', 'verb', 'noun']

# Paths
main_outdir = ""
title = "FORG_REEXPOSURE"
parent_dirname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + title
parent_dirpath = os.path.join(main_outdir, parent_dirname)

new_dump_filename = "stream_info_dump.pkl"


def convert_pth_to_pkl():
    for user_subdir in os.scandir(parent_outputdir):
        if not user_subdir.is_dir():
            continue
        user_dump_path = os.path.join(parent_outputdir, user_subdir.name, dump_filename)
        assert os.path.isfile(user_dump_path)

        dump = torch.load(user_dump_path)

        new_user_dump_path = os.path.join(parent_outputdir, user_subdir.name, new_dump_filename)
        f = open(new_user_dump_path, 'wb')
        pickle.dump(dump, f)
        f.close()
        continue

        results = dump[CHOSEN_KEY]

        # Plot user
        fig = plot_single_user(results)

        path = os.path.join(parent_dirpath, filename.format(mode))
        fig.savefig(path, )

        plt.show()

        import pdb;

        pdb.set_trace()


def plot_single_user(action_results_over_time: dict):
    assert len(action_results_over_time["delta"]) > 0  # Don't log if state is not kept

    # Get deltas on x-axis
    deltas_x_per_action = defaultdict(list)
    for action, prev_res_over_time in action_results_over_time["prev_after_update_iter"].items():
        cur_res_over_time = action_results_over_time["current_before_update_iter"][action]
        for prev_t, new_t in zip(prev_res_over_time, cur_res_over_time):
            assert new_t >= prev_t, f"New iteration {new_t} <= prev iteration {prev_t}"
            deltas_x_per_action[action].append(new_t - prev_t)

    # Get values on y-axis
    deltas_y_per_action = action_results_over_time["delta"]

    # Plot task-agnostic
    print(f"Plotting scatter: x={deltas_x_per_action}, y={deltas_y_per_action}")
    fig = plt.figure(figsize=plot_config['figsize'],
                     dpi=plot_config['dpi'])  # So all bars are visible!
    for action, deltas_x in deltas_x_per_action.items():
        deltas_y = deltas_y_per_action[action]
        plt.scatter(deltas_x, deltas_y, c=plot_config['color'])

    plt.ylim(None, None)
    plt.xlim(0, None)

    plt.xlabel(plot_config['xlabel'])
    plt.ylabel(plot_config['ylabel'])
    plt.title(plot_config['title'])

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    main()
