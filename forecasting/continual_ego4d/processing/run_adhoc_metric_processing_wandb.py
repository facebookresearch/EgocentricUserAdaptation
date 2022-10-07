"""
Given the metrics we stored per-user. Aggregate metrics over users into a single number.
This script specifically aggregates the online-AG over users into a single metric.

Because run-selection can be cumbersome. We enable downloading a csv from wandb, and extracting for all groups the
user-results.

You can pull results directly from wandb with their API: https://docs.wandb.ai/guides/track/public-api-guide

1) Go to the table overview. Group runs based on 'Group'. Click download button and 'export as CSV'.
Additional filters: finished_run=True (fully crashed groups excluded), TAG (specific experiment),
<some_adhoc_metric>=null (Don't ad-hoc reprocess runs).
2) This script will read the csv and extract the group names.
3) It will receive for each group, all the user-runs.
4) The run.history for AG is retrieved (the last measured one = full stream AG) and we calculate the avg over all users.
5) We update the wandb entry and upload update remotely.


--------------------------------------------
Note, the train_iters is ok to normalize as first step is included, but last one is not:

user_run.history()[[metric_name,'trainer/global_step']]
Out[6]:
    train_action_batch/AG_cumul  trainer/global_step
0                           NaN                    0
1                           NaN                    0
2                     -3.403297                    0
3                           NaN                    1
4                           NaN                    1
..                          ...                  ...
72                          NaN                   24
73                          NaN                   24
74                   -34.459484                   24
75                          NaN                   25
76                          NaN                   25
"""

import pandas as pd
import wandb
import pprint
import os
from continual_ego4d.processing.utils import get_group_names_from_csv, get_group_run_iterator

api = wandb.Api()

train_users = ['68', '265', '324', '30', '24', '421', '104', '108', '27', '29']
test_users = [
    "59", "23", "17", "37", "97", "22", "31", "10", "346", "359", "120", "19", "16", "283", "28", "20", "44", "38",
    "262", "25", "51", "278", "55", "39", "45", "33", "331", "452", "453", "21", "431", "116", "35", "105", "378", "74",
    "11", "126", "123", "436"
]
MODES = [
    'aggregate_OAG_over_user_streams',
    'aggregate_test_results_over_user_streams',
]

# Adapt settings
MODE = MODES[0]
csv_filename = 'wandb_export_2022-10-06T09_29_23.798-07_00.csv'  # TODO copy file here and past name here
NB_EXPECTED_USERS = len(test_users)

# Fixed Settings
csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
PROJECT_NAME = "matthiasdelange/ContinualUserAdaptation"

# New uploaded keys
NEW_METRIC_PREFIX = 'adhoc_users_aggregate'
NEW_METRIC_PREFIX_MEAN = 'mean'
NEW_METRIC_PREFIX_SE = 'SE'  # Unbiased standard error
USER_AGGREGATE_COUNT = f"{NEW_METRIC_PREFIX}/user_aggregate_count"  # Over how many users added, also used to check if processed


def aggregate_OAG_over_user_streams(selected_group_names):
    """ Get OAG results for action/verb/noun per user, normalize over user stream samples, average and upload to wandb. """

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:

        try:
            user_ids, total_samples_per_user, final_stream_metrics_to_avg = _collect_user_AG_results(group_name)
        except Exception as e:
            print(e)
            print(f"SKIPPING: contains error: Group ={group_name}")
            continue

        if not is_user_runs_valid(user_ids, group_name):
            continue

        for name, user_val_list in final_stream_metrics_to_avg.items():
            assert len(user_val_list) == NB_EXPECTED_USERS, f"{name} has not all user values: {user_val_list}"

        print(f"Processing users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")

        # Change names for metrics
        updated_metrics_dict = {'total_samples_per_user': total_samples_per_user, 'user_id': user_ids}
        new_metric_names = []
        for k, v in final_stream_metrics_to_avg.items():
            new_metric_name = f"{NEW_METRIC_PREFIX}/{k}"
            new_metric_names.append(new_metric_name)
            updated_metrics_dict[new_metric_name] = v

        # Make averages over users from their final result on the stream, and normalize by steps in user-stream
        updated_metrics_df = pd.DataFrame.from_dict(updated_metrics_dict)
        updated_metrics_df_norm = updated_metrics_df[new_metric_names].div(
            updated_metrics_df.total_samples_per_user, axis=0)

        final_update_metrics_dict = {f"{USER_AGGREGATE_COUNT}/OAG": len(user_ids)}
        for col in updated_metrics_df_norm.columns.tolist():
            final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_MEAN}"] = updated_metrics_df_norm[col].mean()
            final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_SE}"] = updated_metrics_df_norm[col].sem()

        # Update all group entries:
        for user_run in get_group_run_iterator(PROJECT_NAME, group_name):
            for name, new_val in final_update_metrics_dict.items():
                user_run.summary[name] = new_val

            user_run.summary.update()  # UPLOAD


def _collect_user_AG_results(group_name):
    # Get metrics over runs(users)
    final_stream_metrics_to_avg = {
        'train_action_batch/AG_cumul': [],
        'train_verb_batch/AG_cumul': [],
        'train_noun_batch/AG_cumul': [],
    }
    user_ids = []  # all processed users
    total_samples_per_user = []  # nb of steps per user-stream

    # summary = final value (excludes NaN rows)
    # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
    for user_run in get_group_run_iterator(PROJECT_NAME, group_name):  # ACTS LIKE ITERATOR, CAN ONLY CALL LIKE THIS!
        user_ids.append(user_run.config['DATA.COMPUTED_USER_ID'])
        # total_iters_per_user.append(user_run.summary['trainer/global_step']) # TODO DIVIDE BY NB OF SAMPLES, NOT ITERS
        nb_samples = user_run.history()['train_batch/future_sample_count'].dropna().reset_index(drop=True)[0]
        total_samples_per_user.append(nb_samples)

        for metric_name, val_list in final_stream_metrics_to_avg.items():
            user_metric_val = user_run.summary[metric_name]
            val_list.append(user_metric_val)

    return user_ids, total_samples_per_user, final_stream_metrics_to_avg


def aggregate_test_results_over_user_streams(selected_group_names):
    """ After testing on a model with cfg.STREAM_EVAL_ONLY=True, aggregate over user stream results. """

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:

        # Filter runs in group that have finished
        run_filter = {
            "$and": [
                {"group": group_name},
                {"summary_metrics.num_samples_stream": {"$ne": None}}
            ]
        }

        try:
            user_ids, final_stream_metric_userlists = _collect_user_test_results(
                group_name, run_filter
            )
        except Exception as e:
            print(e)
            print(f"SKIPPING: contains error: Group ={group_name}")
            continue

        if not is_user_runs_valid(user_ids, group_name):
            continue

        for name, user_val_list in final_stream_metric_userlists.items():
            assert len(user_val_list) == NB_EXPECTED_USERS, f"{name} has not all user values: {user_val_list}"

        print(f"Processing users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")

        update_test_results_wandb(final_stream_metric_userlists, group_name, run_filter)


def update_test_results_wandb(metricname_to_user_results_list: dict[str, list], group_name: str, run_filter=None):
    """ Upload test results to wandb. """

    # Change names for metrics
    updated_metrics_dict = {}
    new_metric_names = []
    for orig_metric_name, metric_val in metricname_to_user_results_list.items():
        new_metric_name = f"{NEW_METRIC_PREFIX}/{orig_metric_name}"
        new_metric_names.append(new_metric_name)
        updated_metrics_dict[new_metric_name] = metric_val

    # Avg over user streams and get SE (No normalizing as is already avg metrics per stream)
    updated_metrics_df = pd.DataFrame.from_dict(updated_metrics_dict)
    total_count_key = f"{USER_AGGREGATE_COUNT}/test"
    final_update_metrics_dict = {}
    for col in updated_metrics_df.columns.tolist():
        if total_count_key not in final_update_metrics_dict:
            final_update_metrics_dict[total_count_key] = len(updated_metrics_df[col])  # Nb of users
        final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_MEAN}"] = updated_metrics_df[col].mean()
        final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_SE}"] = updated_metrics_df[col].sem()

    print(f"New metric results: {pprint.pformat(list(final_update_metrics_dict.keys()))}")

    # Update all group entries:
    import tqdm
    for user_run in tqdm.tqdm(
            get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter),
            desc=f"Uploading group results: {final_update_metrics_dict}"
    ):
        for name, new_val in final_update_metrics_dict.items():
            user_run.summary[name] = new_val

        user_run.summary.update()  # UPLOAD


def _collect_user_test_results(group_name, run_filter):
    # Get metrics over runs(users)
    final_stream_metrics_per_user = {
        'test_action_batch/loss': [],
        'test_verb_batch/loss': [],
        'test_noun_batch/loss': [],
        'test_action_batch/top1_acc': [],
        'test_verb_batch/top1_acc': [],
        'test_verb_batch/top5_acc': [],
        'test_noun_batch/top1_acc': [],
        'test_noun_batch/top5_acc': [],
        # 'num_samples_stream': [], # Can use to re-weight
    }
    user_ids = []  # all processed users

    # summary = final value (excludes NaN rows)
    # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
    for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter)):
        user_ids.append(user_run.config['DATA.COMPUTED_USER_ID'])

        for metric_name, val_list in final_stream_metrics_per_user.items():
            user_metric_val = user_run.summary[metric_name]
            val_list.append(user_metric_val)

    return user_ids, final_stream_metrics_per_user


def is_user_runs_valid(user_ids, group_name) -> bool:
    # Check if all users:
    if len(user_ids) < NB_EXPECTED_USERS:
        print(f"SKIPPING: MISSING users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")
        return False
    elif len(user_ids) > NB_EXPECTED_USERS:
        raise Exception(f"Might need to readjust EXPECTED USERS? -> user_ids={user_ids}")

    # Check no double users
    if len(set(user_ids)) != len(user_ids):
        print(f"[SKIPPING]: Contains duplicate finished users: {user_ids}: Group ={group_name}")
        return False

    return True


if __name__ == "__main__":
    csv_path = os.path.join(csv_dirname, csv_filename)
    assert os.path.isfile(csv_path)

    # From WandB csv from overview, grouped by Group. Get all the names in the csv (these are the run group names).
    selected_group_names: list[str] = get_group_names_from_csv(selected_group_names_csv_path)
    print(f"Group names={pprint.pformat(selected_group_names)}")

    locals()[MODE](selected_group_names)  # Call function
