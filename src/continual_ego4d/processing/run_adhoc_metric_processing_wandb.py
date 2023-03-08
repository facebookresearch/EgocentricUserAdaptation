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
from typing import Union
from continual_ego4d.processing.utils import get_group_run_iterator, get_delta, get_delta_mappings
import tqdm

api = wandb.Api()

# MODES = [
#     'adhoc_metrics_from_csv_dump_to_wandb',
#     'avg_and_delta_avg_results_over_user_streams',
#     'aggregate_test_results_over_user_streams',
#     'running_avg_to_avg_and_delta',
#     'running_avg_to_avg',
#     'stream_result_list_to_user_avg_and_aggregated_avg'
# ]
# # Adapt settings
# MODE = MODES[0]
# train = True
# csv_filename = 'wandb_export_2022-10-26T15_24_20.237-07_00.csv'  # TODO copy file here and past name here
# single_group_name = None
# # single_group_name = "BatchedLabelWindowPredictor_2022-10-26_20-24-45_UIDb586b449-7300-411b-baf4-9eba4300543f"
# remote = True
#
# if train:
#     train_users = ['68', '265', '324', '30', '24', '421', '104', '108', '27', '29']
#     USER_LIST = train_users
#     NB_EXPECTED_USERS = len(train_users)
#     PRETRAIN_GROUP = "FixedNetwork_2022-10-07_21-50-23_UID80e71950-cea4-44ba-ba16-7dddfe95be26"
#
# else:
#     test_users = [
#         "59", "23", "17", "37", "97", "22", "31", "10", "346", "359", "120", "19", "16", "283", "28", "20", "44", "38",
#         "262", "25", "51", "278", "55", "39", "45", "33", "331", "452", "453", "21", "431", "116", "35", "105", "378",
#         "74", "11", "126", "123", "436"]
#     USER_LIST = test_users
#     NB_EXPECTED_USERS = len(test_users)
#     PRETRAIN_GROUP = "FixedNetwork_2022-10-07_15-23-09_UID6e87e600-a447-438f-9a31-f5cae6dc9ed4"
#
# # Fixed Settings
# if remote:
#     csv_dirname = '/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/adhoc_results'
# else:
#     csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
# PROJECT_NAME = "matthiasdelange/ContinualUserAdaptation"

# New uploaded keys
NEW_METRIC_PREFIX = 'adhoc_users_aggregate'
NEW_METRIC_PREFIX_MEAN = 'mean'
NEW_METRIC_PREFIX_SE = 'SE'  # Unbiased standard error
USER_AGGREGATE_COUNT = f"{NEW_METRIC_PREFIX}/user_aggregate_count"  # Over how many users added, also used to check if processed


def avg_user_streams(
        project_name: str,
        group_name: str,
        expected_user_ids: list[str],
        pretrain_group_name: str = None,
        metric_names: list[str] = None
):
    """
    After finishing training on all user streams, aggregate their metric results into averages.
    If the pretrain_group_name is defined, also calculate the Adaptation Gains w.r.t. the population model.
    """

    if metric_names is None:
        metric_names = [
            # Loss micro-avg
            'train_action_batch/loss_running_avg',
            'train_verb_batch/loss_running_avg',
            'train_noun_batch/loss_running_avg',

            # ACC micro-avg
            'train_action_batch/top1_acc_running_avg',
            'train_verb_batch/top1_acc_running_avg',
            'train_noun_batch/top1_acc_running_avg',
            'train_verb_batch/top5_acc_running_avg',
            'train_noun_batch/top5_acc_running_avg',

            # ACC macro-avg
            'train_action_batch/top1_acc_balanced_running_avg',
            'train_verb_batch/top1_acc_balanced_running_avg',
            'train_noun_batch/top1_acc_balanced_running_avg',
        ]

    # Filter user runs in group (U_train/U_test) that have finished
    print(f"Processing user WandB-Group ={group_name}")
    user_ids, user_results = collect_wandb_group_user_results_for_metrics(
        project_name,
        group_name,
        metric_names=metric_names,
        user_ids=expected_user_ids,
        run_filter=None,
    )
    check_user_runs_valid(user_ids, group_name, len(expected_user_ids), user_results)  # Checks

    # Get pretrain results and add delta results (for OAG)
    all_results = user_results
    if pretrain_group_name is not None:
        # Check users and nb of results
        pretrain_user_ids, pretrain_results = collect_wandb_group_user_results_for_metrics(
            project_name,
            pretrain_group_name,
            metric_names=metric_names,
            run_filter=None,
            user_ids=user_ids
        )
        check_user_runs_valid(pretrain_user_ids, pretrain_group_name, len(expected_user_ids), pretrain_results)

        # Calculate deltas
        AG_dict = _get_delta_pretrain_user_streams(
            pretrain_user_ids, user_ids, user_results, pretrain_results
        )

        # Add to final results
        all_results = {**all_results, **AG_dict}

    # Upload for group the avg over runs from dict
    upload_metric_dict_to_wandb(all_results, project_name, group_name, run_filter=None)


def _get_delta_pretrain_user_streams(
        pretrain_user_ids: list[str],
        user_ids: list[str],
        user_results: dict[str, list],
        pretrain_results: dict[str, list]
) -> dict:
    """ Given pretrain and user-metrics, return dict that includes the deltas with pretrain performance."""
    AG_dict = {}
    assert pretrain_user_ids == user_ids, \
        "Order of pretrained and user ids should be same, otherwise SE will not be correct for avg the deltas. "

    for metric_name, user_val_list in user_results.items():
        assert metric_name in pretrain_results, \
            f"KEY:{metric_name} of user not in pretrain results!"

    # Get same results but delta with pretrain results
    delta_sign_map = get_delta_mappings()

    # make sure users have same order
    print("Assuming for ACC: new result - pretrain result")
    for metric_name, user_val_list in user_results.items():
        delta_sign = delta_sign_map[metric_name]
        pretrain_val_list = pretrain_results[metric_name]
        assert len(pretrain_val_list) == len(user_val_list)

        # Add as later used to calculate /mean and /SE on
        AG_dict[f"{metric_name}/PRETRAIN_abs"] = pretrain_val_list  # Absolute pretrain performance
        AG_dict[f"{metric_name}/adhoc_AG"] = [  # Relative performance to pretraining
            get_delta(delta_sign, user_val, pretrain_val)
            for pretrain_val, user_val in zip(pretrain_val_list, user_val_list)]

    return AG_dict


def check_user_runs_valid(
        user_ids: list[str],
        group_name: str,
        nb_expected_users: int,
        results: dict[str, list]
):
    """
    Checks if the following are valid:
    1) user_ids that have finished
    2) the user results
    """
    # Check if all users:
    if len(user_ids) < nb_expected_users:
        raise Exception(f"SKIPPING: MISSING users: {len(user_ids)}/{nb_expected_users} "
                        f"-> {user_ids}: Group ={group_name}")
    elif len(user_ids) > nb_expected_users:
        raise Exception(f"Might need to readjust EXPECTED USERS? -> user_ids={user_ids}")

    # Check no double users
    assert len(set(user_ids)) != len(user_ids), \
        f"[SKIPPING]: Contains duplicate finished users: {user_ids}: Group ={group_name}"

    for metric_name, user_val_list in results.items():
        assert len(user_val_list) == nb_expected_users, \
            f"{metric_name} has not all user values: {user_val_list}"


def upload_metric_dict_to_wandb(
        metricname_to_user_results_list: dict[str, list],
        project_name: str,
        group_name: str,
        run_filter=None,
        mean=True
):
    """
    Calculates the mean and SE over all results in each list in the result dict <metric_name:str, user_results:list>.
    The mean/SE are then uploaded to WandB.
    """

    # Add adhoc-prefix to metric names
    updated_metrics_dict = {}
    for orig_metric_name, metric_val in metricname_to_user_results_list.items():
        new_metric_name = f"{NEW_METRIC_PREFIX}/{orig_metric_name}"
        updated_metrics_dict[new_metric_name] = metric_val

    final_update_metrics_dict = updated_metrics_dict

    # Average over lists and get SEM
    if mean:
        updated_metrics_df = pd.DataFrame.from_dict(updated_metrics_dict)  # Dataframe with updated metric names

        total_count_key = f"{USER_AGGREGATE_COUNT}/test"
        final_update_metrics_dict = {}
        for col in updated_metrics_df.columns.tolist():
            if total_count_key not in final_update_metrics_dict:
                final_update_metrics_dict[total_count_key] = len(updated_metrics_df[col])  # Nb of users
            final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_MEAN}"] = updated_metrics_df[col].mean()
            final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_SE}"] = updated_metrics_df[col].sem()

        print(f"Averaged metric results over dict:\n{pprint.pformat(list(final_update_metrics_dict.keys()))}")

    # Update all group entries:
    for user_run in tqdm.tqdm(
            get_group_run_iterator(project_name, group_name, run_filter=run_filter),
            desc=f"Uploading group results to WandB: {final_update_metrics_dict}"
    ):
        for name, new_val in final_update_metrics_dict.items():
            user_run.summary[name] = new_val

        user_run.summary.update()  # UPLOAD


def collect_wandb_group_user_results_for_metrics(
        project_name: str,
        group_name: str,
        run_filter: Union[dict, None],
        metric_names: list[str],
        user_ids: list = None,
        metrics_strict=True
) -> (list, dict[str, list]):
    """
    For a set of metrics, download and collect results of all users in the Project and Group in WandB.
    """
    # Get metrics over runs(users)
    final_stream_metrics_per_user = {metric_name: [] for metric_name in metric_names}

    def get_dynamic_user_results():
        """ Add user results in order of downloading from WandB."""
        # summary = final value (excludes NaN rows)
        # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
        for idx, user_run in enumerate(get_group_run_iterator(project_name, group_name, run_filter=run_filter)):
            user_ids.append(user_run.config['DATA.COMPUTED_USER_ID'])
            for metric_name, val_list in final_stream_metrics_per_user.items():

                if not metrics_strict and metric_name not in user_run.summary:
                    print(f"[NOT STRICT] Skipping metric {metric_name} as not found in user run")
                    continue

                user_metric_val = user_run.summary[metric_name]  # Only takes last value
                val_list.append(user_metric_val)

    def get_static_user_results():
        """ Fill in user results based on order given by user_ids. """
        # summary = final value (excludes NaN rows)
        # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
        for idx, user_run in enumerate(get_group_run_iterator(project_name, group_name, run_filter=run_filter)):
            user_id = user_run.config['DATA.COMPUTED_USER_ID']
            remote_users.append(user_id)
            result_idx = user_ids.index(user_id)
            for metric_name, val_list in final_stream_metrics_per_user.items():

                if not metrics_strict and metric_name not in user_run.summary:
                    print(f"[NOT STRICT] Skipping metric {metric_name} as not found in user run")
                    continue

                user_metric_val = user_run.summary[metric_name]  # Only takes last value
                val_list[result_idx] = user_metric_val

    if user_ids is None:
        user_ids = []  # all processed users
        get_dynamic_user_results()

    else:
        for k, v in final_stream_metrics_per_user.items():
            final_stream_metrics_per_user[k] = [None] * len(user_ids)  # Pre-init arrays

        remote_users = []
        get_static_user_results()  # Raises value error if user not found
        assert len(remote_users) == len(user_ids), "Not all user values have been filled in!"

    return user_ids, final_stream_metrics_per_user
