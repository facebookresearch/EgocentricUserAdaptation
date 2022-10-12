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
from continual_ego4d.processing.utils import get_group_names_from_csv, get_group_run_iterator, get_delta, \
    get_delta_mappings
import tqdm
import torch
from continual_ego4d.processing.utils import loss_CE_to_likelihood, ConditionalAverageMeterDict

api = wandb.Api()

MODES = [
    'adhoc_balanced_LL_from_csv_dump_to_wandb',
    'aggregate_avg_train_results_over_user_streams',
    'aggregate_test_results_over_user_streams',
    'aggregate_OAG_over_user_streams',  # online OAG
]
# Adapt settings
MODE = MODES[0]
train = True
csv_filename = 'wandb_export_2022-10-11T18_31_26.197-07_00.csv'  # TODO copy file here and past name here
single_group_name = None
remote = True

if train:
    train_users = ['68', '265', '324', '30', '24', '421', '104', '108', '27', '29']
    USER_LIST = train_users
    NB_EXPECTED_USERS = len(train_users)
    PRETRAIN_GROUP = "FixedNetwork_2022-10-07_21-50-23_UID80e71950-cea4-44ba-ba16-7dddfe95be26"

else:
    test_users = [
        "59", "23", "17", "37", "97", "22", "31", "10", "346", "359", "120", "19", "16", "283", "28", "20", "44", "38",
        "262", "25", "51", "278", "55", "39", "45", "33", "331", "452", "453", "21", "431", "116", "35", "105", "378",
        "74", "11", "126", "123", "436"]
    USER_LIST = test_users
    NB_EXPECTED_USERS = len(test_users)
    PRETRAIN_GROUP = "FixedNetwork_2022-10-07_15-23-09_UID6e87e600-a447-438f-9a31-f5cae6dc9ed4"

# Fixed Settings
if remote:
    csv_dirname = '/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/adhoc_results'
else:
    csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
PROJECT_NAME = "matthiasdelange/ContinualUserAdaptation"

# New uploaded keys
NEW_METRIC_PREFIX = 'adhoc_users_aggregate'
NEW_METRIC_PREFIX_MEAN = 'mean'
NEW_METRIC_PREFIX_SE = 'SE'  # Unbiased standard error
USER_AGGREGATE_COUNT = f"{NEW_METRIC_PREFIX}/user_aggregate_count"  # Over how many users added, also used to check if processed


def adhoc_balanced_LL_from_csv_dump_to_wandb(selected_group_names, new_metric_name="balanced_LL"):
    """
    Get valid csv dump dir per user.
    Upload per user-stream in group the csv-dump processed balanced-Likelihood.
    Make sure to first do for pretrained model as delta is calculated afterwards in aggregating.
    """
    d = ConditionalAverageMeterDict()

    entries = (
        ('sample_idx_to_action_loss', 'action'),
        ('sample_idx_to_verb_loss', 'verb'),
        ('sample_idx_to_noun_loss', 'noun'),
    )

    # Collect from wandb
    for group_name in selected_group_names:
        print(f"\nUpdating group: {group_name}")
        user_to_dump_path = {}

        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=None)):
            user_id: str = user_run.config['DATA.COMPUTED_USER_ID']
            user_dump_path = os.path.abspath(os.path.join(user_run.config['OUTPUT_DIR'], "user_logs", f"user_{user_id}",
                                                          "stream_info_dump.pth"))
            try:
                assert os.path.exists(user_dump_path), f"Not existing: {user_dump_path}"
            except Exception as e:
                print(f"SKIPPING USER {user_id}: {e}")
                continue
            user_to_dump_path[user_id] = user_dump_path
            user_dump_dict = torch.load(user_to_dump_path[user_id])

            # Get labels
            action_labels: list[tuple[int, int]] = user_dump_dict['sample_idx_to_action_list']

            # Iterate action/verb/noun losses per instance
            for dump_name, action_mode in entries:
                sample_idx_to_loss: list[float] = user_dump_dict[dump_name]

                sample_idx_to_loss_t = torch.tensor(sample_idx_to_loss)  # To tensor
                sample_idx_to_LL_t: torch.Tensor = loss_CE_to_likelihood(sample_idx_to_loss_t)  # First convert to likelihood

                # Get label set
                if action_mode == 'action':
                    selected_labels = action_labels
                elif action_mode == 'verb':
                    selected_labels = [x[0] for x in action_labels]
                elif action_mode == 'noun':
                    selected_labels = [x[1] for x in action_labels]
                else:
                    raise ValueError()

                # Now average based on action label
                # Get AverageMeter per conditional
                d.reset()
                d.update(sample_idx_to_LL_t.tolist(), cond_list=selected_labels)
                balanced_avg_LL = d.result()  # Get total weighed avg LL

                # Upload directly as per-user result
                full_new_metric_name = f"train_{action_mode}_batch/{new_metric_name}"
                user_run.summary[full_new_metric_name] = balanced_avg_LL

                # Also get balanced loss
                d.reset()
                d.update(sample_idx_to_loss_t.tolist(), cond_list=selected_labels)
                print(f"Balanced loss: {pprint.pformat(d.d)}")
                balanced_loss = d.result()  # Get total weighed avg LL

                # Upload directly as per-user result
                full_balanced_loss_name = f"train_{action_mode}_batch/balanced_loss"
                user_run.summary[full_balanced_loss_name] = balanced_loss

                user_run.summary.update()  # UPLOAD
                print(f"USER[{user_id}] Updated [{full_new_metric_name}] = {balanced_avg_LL}, "
                      f"{full_balanced_loss_name} = {balanced_loss}")

    print(f"Finished processing")

    print(f"Aggregating over users in separate step:")
    aggregate_avg_train_results_over_user_streams(selected_group_names)


def aggregate_online_calculated_OAG_over_user_streams(selected_group_names):
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


def aggregate_avg_train_results_over_user_streams(selected_group_names):
    """ After training a model, aggregate the avg metrics over the stream such as ACC.
    Aggregate over user stream results. """

    # GET PRETRAIN as reference
    pretrain_user_ids, pretrain_final_stream_metric_userlists = _collect_wandb_group_absolute_online_results(
        PRETRAIN_GROUP, run_filter=None, user_ids=USER_LIST)

    # Check users and nb of results
    if not is_user_runs_valid(pretrain_user_ids, PRETRAIN_GROUP):
        print(f"Pretrain not valid: exit")
        return
    for metric_name, user_val_list in pretrain_final_stream_metric_userlists.items():
        assert len(user_val_list) == NB_EXPECTED_USERS, f"{metric_name} has not all user values: {user_val_list}"

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:

        # Filter runs in group that have finished
        try:
            user_ids, final_stream_metric_userlists = _collect_wandb_group_absolute_online_results(
                group_name, user_ids=USER_LIST, run_filter=None
            )
        except Exception as e:
            print(e)
            print(f"SKIPPING: contains error: Group ={group_name}")
            continue

        # Checks
        assert pretrain_user_ids == user_ids, \
            "Order of pretrained and user ids should be same, otherwise SE will not be correct for avg the deltas. "

        if not is_user_runs_valid(user_ids, group_name):
            continue
        for metric_name, user_val_list in final_stream_metric_userlists.items():
            assert len(user_val_list) == NB_EXPECTED_USERS, f"{metric_name} has not all user values: {user_val_list}"
            assert metric_name in pretrain_final_stream_metric_userlists, f"KEY:{metric_name} of user not in pretrain results!"

        print(f"Processing users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")

        # Get same results but delta with pretrain results
        delta_sign_map = get_delta_mappings()

        # make sure users have same order
        print("Assuming for ACC: new result - pretrain result")
        AG_dict = {}
        for metric_name, user_val_list in final_stream_metric_userlists.items():
            delta_sign = delta_sign_map[metric_name]
            pretrain_val_list = pretrain_final_stream_metric_userlists[metric_name]
            assert len(pretrain_val_list) == len(user_val_list)

            # Add as later used to calculate /mean and /SE on
            AG_dict[f"{metric_name}/PRETRAIN_abs"] = pretrain_val_list
            AG_dict[f"{metric_name}/adhoc_AG"] = [
                get_delta(delta_sign, user_val, pretrain_val)
                for pretrain_val, user_val in zip(pretrain_val_list, user_val_list)]

        # Add to final results
        all_results = {**final_stream_metric_userlists, **AG_dict}

        avg_per_metric_upload_wandb(all_results, group_name, run_filter=None)


def _collect_wandb_group_absolute_online_results(group_name, run_filter, user_ids: list = None):
    # Get metrics over runs(users)
    final_stream_metrics_per_user = {
        'train_action_batch/loss_running_avg': [],
        'train_verb_batch/loss_running_avg': [],
        'train_noun_batch/loss_running_avg': [],

        'train_action_batch/top1_acc_running_avg': [],
        'train_verb_batch/top1_acc_running_avg': [],
        'train_noun_batch/top1_acc_running_avg': [],
        'train_verb_batch/top5_acc_running_avg': [],
        'train_noun_batch/top5_acc_running_avg': [],
        # 'num_samples_stream': [], # Can use to re-weight

        # TODO add LL(likelihood)
        'train_action_batch/balanced_LL': [],
        'train_verb_batch/balanced_LL': [],
        'train_noun_batch/balanced_LL': [],

        'train_action_batch/balanced_loss': [],
        'train_verb_batch/balanced_loss': [],
        'train_noun_batch/balanced_loss': [],
    }

    def get_dynamic_user_results():
        """ Add user results in order of downloading from WandB."""
        # summary = final value (excludes NaN rows)
        # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter)):
            user_ids.append(user_run.config['DATA.COMPUTED_USER_ID'])
            for metric_name, val_list in final_stream_metrics_per_user.items():
                user_metric_val = user_run.summary[metric_name]  # Only takes last value
                val_list.append(user_metric_val)

    def get_static_user_results():
        """ Fill in user results based on order given by user_ids. """
        # summary = final value (excludes NaN rows)
        # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter)):
            user_id = user_run.config['DATA.COMPUTED_USER_ID']
            remote_users.append(user_id)
            result_idx = user_ids.index(user_id)
            for metric_name, val_list in final_stream_metrics_per_user.items():
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

        avg_per_metric_upload_wandb(final_stream_metric_userlists, group_name, run_filter)


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


def avg_per_metric_upload_wandb(metricname_to_user_results_list: dict[str, list], group_name: str, run_filter=None,
                                mean=True):
    """
    Calculate mean and SE for the list in each <str, list> pair in the dict.
    The str/mean and str/SE are then uploaded to WandB.
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

    print(f"New metric results:\n{pprint.pformat(list(final_update_metrics_dict.keys()))}")

    # Update all group entries:
    for user_run in tqdm.tqdm(
            get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter),
            desc=f"Uploading group results: {final_update_metrics_dict}"
    ):
        for name, new_val in final_update_metrics_dict.items():
            user_run.summary[name] = new_val

        user_run.summary.update()  # UPLOAD


if __name__ == "__main__":
    if single_group_name is None:
        csv_path = os.path.join(csv_dirname, csv_filename)
        assert os.path.isfile(csv_path), f"Non-existing: {csv_path}"

        # From WandB csv from overview, grouped by Group. Get all the names in the csv (these are the run group names).
        selected_group_names: list[str] = get_group_names_from_csv(csv_path)
    else:
        selected_group_names = [single_group_name]
    print(f"Group names={pprint.pformat(selected_group_names)}")

    locals()[MODE](selected_group_names)  # Call function
