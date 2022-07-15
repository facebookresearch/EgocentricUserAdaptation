""" Split in train and test sets and generate summary. """

# TODO use notebooks

import pandas as pd
import numpy as np
from collections import Counter
import os.path as osp
import json
import matplotlib.pyplot as plt
import argparse
import datetime
import os
from collections import defaultdict

parser = argparse.ArgumentParser(description="User split for ego4d LTA task.")
parser.add_argument(
    "--nb_users_thresh",
    help="Number users to keep as subset",
    default=50,
    type=int,
)
parser.add_argument(
    "--nb_users_train",
    help="Number users to use for training, should be <= nb_users_thresh."
         "The 'nb_users_thresh - nb_users_train' are used for testing.",
    default=40,
    type=int,
)

parser.add_argument(
    "--user_videotime_min_thresh",
    help="Number of videominutes users need to remain in the subset",
    default=None,
    type=int,
)
parser.add_argument(
    "--sort_by_col",
    help="Column in dataframe to sort on. (Default: total sum of user clip-video length)",
    default="sum_clip_duration_min",
    type=str,
)
parser.add_argument(
    "--p_output_dir",
    help="Parent dir to output timestamped dir including plots and json splits.",
    default="./usersplit_data",
    type=str,
)
parser.add_argument(
    "--seed",
    help="Seed numpy for deterministic splits",
    default=0,
    type=int,
)


def generate_usersplit_from_trainval(
        meta_data_file_path: str,
        train_annotation_file: str,
        val_annotation_file: str,
        user_id_col="fb_participant_id"):
    args = parser.parse_args()
    nb_users_test = args.nb_users_thresh - args.nb_users_train

    np.random.seed(args.seed)

    # check args
    assert args.nb_users_thresh is None or args.user_videotime_min_thresh is None, \
        "Can only define one thresholding method!"

    # Open meta data object
    with open(meta_data_file_path, 'r') as meta_data_file:
        meta_data_obj = json.load(meta_data_file)
    meta_df = pd.json_normalize(meta_data_obj['videos'])
    print(f"meta_data.shape={meta_df.shape}")

    # Open train and val objects
    with open(train_annotation_file, 'r') as train_file, \
            open(val_annotation_file, 'r') as val_file:
        train_clips = json.load(train_file)['clips']
        val_clips = json.load(val_file)['clips']

    train_clips_df = pd.json_normalize(train_clips)
    val_clips_df = pd.json_normalize(val_clips)
    print(f"trainshape={train_clips_df.shape}, valshape={val_clips_df.shape}")

    # Show overlapping
    print(f"Meta colnames={list(meta_df)}")
    print(f"Annotation colnames={list(train_clips_df)}")
    overlapping_colnames = [x for x in list(meta_df) if x in list(train_clips_df)]
    print(f"Overlapping colnames={overlapping_colnames}")

    # MERGE dataframes on video_uid (Right join: keep annotation entries, but add video_uid info)
    train_joined_df = pd.merge(meta_df, train_clips_df,
                               on="video_uid", validate="one_to_many", how="right")
    val_joined_df = pd.merge(meta_df, val_clips_df,
                             on="video_uid", validate="one_to_many", how="right")
    print(f"train_joined_df={train_joined_df.shape}, val_joined_df={val_joined_df.shape}")

    # CONCAT the dataframes (312 rows × 12 columns)
    trainval_joined_df = pd.concat([train_joined_df, val_joined_df], ignore_index=True, sort=False)

    # FIND USERS that satisfy video-length threshold
    # Note: video_uid relates to the entire uncut raw video,
    # these are split into ~5-min clips, denoted with clip_id for the annotations.
    trainval_user_df = summarize_clips_by_user(trainval_joined_df)

    # Sort users on sum_length
    trainval_user_df = trainval_user_df.sort_values(by=[args.sort_by_col], ascending=False)

    # Keep only highest in sorted
    if args.nb_users_thresh is not None:
        print(f"Thresholding on nb_users_thresh")
        sorted_user_ids = trainval_user_df[: args.nb_users_thresh][user_id_col].tolist()
        user_sort_values = trainval_user_df[: args.nb_users_thresh][args.sort_by_col].tolist()

    elif args.user_videotime_min_thresh is not None:
        print(f"Thresholding on user_videotime_min_thresh")
        user_subset_df = trainval_user_df.loc[trainval_user_df[args.sort_by_col] >= args.user_videotime_min_thresh]
        sorted_user_ids = user_subset_df[user_id_col].tolist()
        user_sort_values = user_subset_df[args.sort_by_col].tolist()

    else:  # No cutoff
        print(f"No thresholding")
        sorted_user_ids = trainval_user_df[user_id_col].tolist()
        user_sort_values = trainval_user_df[args.sort_by_col].tolist()

    # Get train/test splits from train/val datasets
    shuffled_idxs = np.random.permutation(np.arange(len(sorted_user_ids)))
    train_idxs, test_idxs = shuffled_idxs[:args.nb_users_train], shuffled_idxs[args.nb_users_train:]
    sorted_user_ids, user_sort_values = np.array(sorted_user_ids), np.array(user_sort_values)
    train_user_ids, test_user_ids = sorted_user_ids[train_idxs], sorted_user_ids[test_idxs]
    train_sort_values, test_sort_values = user_sort_values[train_idxs], user_sort_values[test_idxs]

    # Get colors for train+test summary plot
    bar_colors = []
    train_color, test_color = 'r', 'blue'
    for bar_idx in range(len(user_sort_values)):
        if bar_idx in train_idxs:
            bar_colors.append(train_color)
        else:
            bar_colors.append(test_color)

    # Print summary
    nb_traintest_users_subset = len(sorted_user_ids)
    nb_train_users_subset = len(train_user_ids)
    nb_test_users_subset = len(test_user_ids)

    nb_total_users = trainval_user_df.shape[0]
    print_summary(user_sort_values, nb_traintest_users_subset, nb_total_users, "Train+Test user subset")
    print_summary(train_sort_values, nb_train_users_subset, nb_traintest_users_subset, "Train user subset")
    print_summary(test_sort_values, nb_test_users_subset, nb_traintest_users_subset, "Test user subset")

    # Check outdir path
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = osp.join(args.p_output_dir, f"{now}_ego4d_LTA_usersplit")
    os.makedirs(output_dir, exist_ok=True)

    # Summary plot (pdf?)
    ylabel = 'Sum of clip video lengths (min)'
    xlabel = "Users - sorted"
    title = "Sum of clip video lengths (min) per user"

    y_axis = user_sort_values
    x_axis = [idx for idx in range(len(user_sort_values))]
    plot_tag = 'TRAIN_TEST'
    plot_barchart(x_axis, y_axis, title=f'{plot_tag} {title} (train=red,test=blue)', ylabel=ylabel, xlabel=xlabel,
                  bar_colors=bar_colors, output_file=osp.join(output_dir, f'{plot_tag}_video_freq_plot.pdf'))

    y_axis = sorted(train_sort_values, reverse=True)
    x_axis = [idx for idx in range(len(train_sort_values))]
    plot_tag = 'TRAIN'
    plot_barchart(x_axis, y_axis, title=f'{plot_tag} {title}', ylabel=ylabel, xlabel=xlabel,
                  output_file=osp.join(output_dir, f'{plot_tag}_video_freq_plot.pdf'))

    y_axis = sorted(test_sort_values, reverse=True)
    x_axis = [idx for idx in range(len(test_sort_values))]
    plot_tag = 'TEST'
    plot_barchart(x_axis, y_axis, title=f'{plot_tag} {title}', ylabel=ylabel, xlabel=xlabel,
                  output_file=osp.join(output_dir, f'{plot_tag}_video_freq_plot.pdf'))

    # OUTPUT TO JSONS
    # don't need all video meta-data (reduce filesize): For train with 40 users: 212MB -> 21MB
    json_filename = 'ego4d_LTA_{}_usersplit_{}users.json'
    json_col_names = list(train_clips_df) + [
        'fb_participant_id',
        'scenarios',
    ]

    # TRAIN JSON
    split = 'train'
    json_train_filepath = osp.join(output_dir, json_filename.format(split, nb_train_users_subset))
    save_json(trainval_joined_df, user_id_col, train_user_ids, json_col_names, json_train_filepath, split)

    # TEST JSON
    split = 'test'
    json_test_filepath = osp.join(output_dir, json_filename.format(split, nb_test_users_subset))
    save_json(trainval_joined_df, user_id_col, test_user_ids, json_col_names, json_test_filepath, split)


def save_json(trainval_joined_df, user_id_col, user_ids, json_col_names, json_filepath, split):
    """Filter json dataframe, parse to json-compatible object. Dump to json file."""
    train_df = trainval_joined_df.loc[trainval_joined_df[user_id_col].isin(user_ids)]  # Get train datatframe
    train_df = train_df[json_col_names]
    train_df = train_df.rename(columns={"scenarios": "parent_video_scenarios"})

    train_json = df_to_formatted_json(train_df, user_ids, split, user_id_col)  # Convert to json
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, ensure_ascii=False, indent=4)
        print(f"Saved JSON: {osp.abspath(json_filepath)}")

    # Test if we can read them in again
    with open(json_filepath, 'r') as f:
        json_obj = json.load(f)
    df = pd.json_normalize(json_obj['users'])
    print(f"Loaded json, head=\n{df.head(n=10)}\n")


def print_summary(user_sort_values, nb_users_subset, nb_total_users, title: str):
    print(f"\n{'*' * 20} SUMMARY: {title} {'*' * 20}")
    print(f"Retaining a total of {nb_users_subset} / {nb_total_users} users")
    print(f"MAX = {user_sort_values[0]}, MIN = {user_sort_values[-1]}")
    print(f"HEAD = {user_sort_values[:10]}...")
    print(f"TAIL = ...{user_sort_values[-10:]}")
    print(f"{'*' * 50}")


def summarize_clips_by_user(joined_df):
    """Group annotation entries by clip_uid, then group those unique clips by user."""
    clip_df = joined_df.groupby(joined_df['clip_uid'], as_index=False).agg(
        {'fb_participant_id': lambda x: np.unique(x).tolist(),
         'scenarios': list,
         'verb': list, 'noun': list, 'verb_label': list, 'noun_label': list, 'action_idx': list,
         #          'video_uid':list,'duration_sec':list, # This is the raw uncut video, don't need this info
         'clip_id': list, 'clip_parent_start_sec': lambda x: np.unique(x).tolist(),
         'clip_parent_end_sec': lambda x: np.unique(x).tolist()})

    # Check users/clip_starts and ends are only 1 unique
    assert (clip_df.fb_participant_id.apply(len) == 1).all()
    assert (clip_df.clip_parent_start_sec.apply(len) == 1).all()
    assert (clip_df.clip_parent_end_sec.apply(len) == 1).all()

    # Unpack
    for col_name in ['fb_participant_id', 'clip_parent_start_sec', 'clip_parent_end_sec']:
        clip_df[col_name] = clip_df[col_name].apply(lambda x: x[0])

    # Get actual clip lengths in seconds (~5min=300s)
    clip_df['clip_duration_sec'] = clip_df.loc[:, ('clip_parent_end_sec', 'clip_parent_start_sec')].apply(
        lambda x: x[0] - x[1], axis=1)

    # Group by fb_participant_id, which has allocated multiple 5min clips (unique clip_uid's)
    user_df = clip_df.groupby(clip_df['fb_participant_id'], as_index=False).agg(
        {
            'scenarios': list,
            'verb': list, 'noun': list, 'verb_label': list, 'noun_label': list, 'action_idx': list,
            'clip_id': list, 'clip_parent_start_sec': list, 'clip_parent_end_sec': list, 'clip_duration_sec': list}
    )

    # Sum clip lengths per user
    user_df['sum_clip_duration_sec'] = user_df['clip_duration_sec'].apply(sum)
    user_df['sum_clip_duration_min'] = user_df['sum_clip_duration_sec'].apply(lambda x: x / 60)

    # The scenarios only apply to the raw uncut video, not the 5min clips
    user_df = user_df.rename(columns={"scenarios": "possible_clip_scenarios"})

    # Check that no NaN user
    assert not (user_df['fb_participant_id'].isna().any())

    return user_df


def plot_barchart(x_axis, y_vals, title, ylabel, xlabel, y_labels=None, x_labels=None,
                  grid=False, yerror=None, xerror=None, bar_align='edge', barh=False, bar_colors=None,
                  figsize=(12, 6), log=False, interactive=False, x_minor_ticks=None, output_file=None):
    max_val = max(y_vals)
    my_cmap = plt.get_cmap("plasma")
    fig = plt.figure(figsize=figsize, dpi=600)  # So all bars are visible!
    ax = plt.subplot()

    if not barh:
        bars = plt.bar(x_axis, height=y_vals, color=my_cmap.colors, align=bar_align, yerr=yerror, width=0.9, log=log)
    else:
        bars = plt.barh(y_vals, width=x_axis, color=my_cmap.colors, align=bar_align, xerr=xerror, height=0.9, log=log)

    if bar_colors:
        for idx, bar_color in enumerate(bar_colors):
            bars[idx].set_color(bar_color)

    if x_minor_ticks is not None:
        #         ax.set_xticks(major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)
    #         ax.set_yticks(major_ticks)
    #         ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    #         ax.grid(which='both')

    #         # Or if you want different settings for the grids:
    #         ax.grid(which='minor', alpha=0.2)
    #         ax.grid(which='major', alpha=0.5)

    if x_labels:
        plt.xticks(x_axis, x_labels, rotation='vertical')
    if y_labels:
        plt.yticks(y_vals, y_labels)

    plt.ylim(None, max_val * 1.01)
    plt.xlim(None, None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid, which='both')

    if interactive:
        annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        fig.canvas.mpl_connect("motion_notify_event", hover)

    # Save
    if output_file is not None:
        fig.savefig(output_file, bbox_inches='tight')
        print(f"Saved plot: {output_file}")

    plt.show()
    plt.clf()


def df_to_formatted_json(df, user_id_list, split, user_id_col_name):
    """Convert to a json with at
    L1: users, split
    L2: per user parse the annotation entries.
    """
    result = {'users': defaultdict(list), 'split': split}

    for user_id in user_id_list:  # iterate users
        user_df = df.loc[df[user_id_col_name] == user_id]
        for _, row in user_df.iterrows():  # Iterate annotations for the user
            parsed_row = {}
            for idx, val in row.iteritems():  # Convert col values to dict style
                key = idx
                parsed_row[key] = val
            result['users'][user_id].append(parsed_row)
    return result


if __name__ == "__main__":
    # META DATA
    meta_data_file_path = "/fb-agios-acai-efs/Ego4D/ego4d_data/ego4d.json"

    # ANNOTATION DATA LTA
    annotation_file_dir = "/fb-agios-acai-efs/Ego4D/ego4d_data/v1/annotations"
    annotation_file_names = {'train': "fho_lta_train.json", 'val': 'fho_lta_val.json',
                             'test': 'fho_lta_test_unannotated.json'}
    train_annotation_file = osp.join(annotation_file_dir, annotation_file_names['train'])
    val_annotation_file = osp.join(annotation_file_dir, annotation_file_names['val'])

    generate_usersplit_from_trainval(
        meta_data_file_path,
        train_annotation_file,
        val_annotation_file,
    )