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
from continual_ego4d.metrics.offline_metrics import per_sample_metric_to_macro_avg, \
    loss_CE_to_class_confidence
from continual_ego4d.metrics.meters import ConditionalAverageMeterDict
from continual_ego4d.metrics.offline_metrics import get_micro_macro_avg_acc
from continual_ego4d.datasets.continual_action_recog_dataset import label_tensor_to_list

api = wandb.Api()

MODES = [
    'adhoc_metrics_from_csv_dump_to_wandb',
    'avg_and_delta_avg_results_over_user_streams',
    'aggregate_test_results_over_user_streams',
    'running_avg_to_avg_and_delta',
    'running_avg_to_avg',
    'stream_result_list_to_user_avg_and_aggregated_avg'
]
# Adapt settings
MODE = MODES[3]
train = True
csv_filename = 'wandb_export_2022-10-25T22_07_14.673-07_00.csv'  # TODO copy file here and past name here
single_group_name = None
# single_group_name = "HindsightLabelWindowPredictor_2022-10-25_16-51-38_UID4b8e69a6-b174-4573-b709-067488f12b07"
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


def stream_result_list_to_user_avg_and_aggregated_avg(selected_group_names, overwrite=True, ):
    """
    Average over per-sample stream result during training, and then avg equally-weighted over users.
    Used for gradient-analysis monitoring with previous grad-steps.
    """
    res = {}
    stream_avg_postfix = "/stream_avg"
    metric_names = [f"analyze_action_batch/LOOKBACK_STEP_{nb_steps_lookback}/{model_part}_grad_cos_sim"
                    for model_part in ["full", "slow", "fast", "head", "feat", ]
                    for nb_steps_lookback in range(1, 11)]

    def _save_to_wandb_summary(metric_name, user_val, user_run):
        if overwrite or metric_name not in user_run.summary:
            user_run.summary[metric_name] = user_val
            print(f"USER[{user_id}] Updated [{metric_name}] = {user_val}")

    # Collect from wandb
    for group_name in selected_group_names:
        print(f"\nUpdating group: {group_name}")

        # Count users and check
        group_users = []
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=None)):
            group_users.append(user_run.config['DATA.COMPUTED_USER_ID'])

        if len(group_users) != NB_EXPECTED_USERS:
            print(f"Users {group_users} (#{len(group_users)}) NOT EQUAL TO expected {NB_EXPECTED_USERS}")
            print(f"SKIPPING group: {group_name}")
            continue

        # Iterate user-runs in group
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=None)):

            # Get CSV DUMP locally
            user_id: str = user_run.config['DATA.COMPUTED_USER_ID']

            # Average over stream
            user_df = user_run.history()  # Dataframe
            for metric_name in metric_names:
                metric_stream_avg = user_df[metric_name].mean()
                metric_stream_avg_name = f"{metric_name}{stream_avg_postfix}"
                res[metric_stream_avg_name] = metric_stream_avg

                # Upload
                _save_to_wandb_summary(metric_stream_avg_name, metric_stream_avg, user_run)

    print(f"Finished processing")

    print(f"Aggregating over users in separate step: \n{pprint.pformat(metric_names)}")
    avg_and_delta_avg_results_over_user_streams(
        selected_group_names,
        metrics=metric_names,
        skip_pretrain_delta=True
    )


def adhoc_metrics_from_csv_dump_to_wandb(selected_group_names, overwrite=True, conditional_analysis=False,
                                         skip_pretrain_delta=False):
    """
    Get valid csv dump dir per user.
    Upload per user-stream in group the csv-dump processed balanced-Likelihood.
    Make sure to first do for pretrained model as delta is calculated afterwards in aggregating.
    """

    action_modes = ('action', 'verb', 'noun')
    metric_names_AG = set()
    metric_names_nonAG = set()

    def _save_to_wandb_summary(metric_name, user_val, user_run, add_to_AG=True):
        if add_to_AG:
            metric_names_AG.add(metric_name)
        else:
            metric_names_nonAG.add(metric_name)
        if overwrite or metric_name not in user_run.summary:
            user_run.summary[metric_name] = user_val
            print(f"USER[{user_id}] Updated [{metric_name}] = {user_val}")

    # Collect from wandb
    for group_name in selected_group_names:
        print(f"\nUpdating group: {group_name}")
        user_to_dump_path = {}

        # Count users and check
        group_users = []
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=None)):
            group_users.append(user_run.config['DATA.COMPUTED_USER_ID'])

        if len(group_users) != NB_EXPECTED_USERS:
            print(f"Users {group_users} (#{len(group_users)}) NOT EQUAL TO expected {NB_EXPECTED_USERS}")
            print(f"SKIPPING group: {group_name}")
            continue

        # Iterate user-runs in group
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=None)):

            # Get CSV DUMP locally
            user_id: str = user_run.config['DATA.COMPUTED_USER_ID']
            user_dump_path = os.path.abspath(os.path.join(
                user_run.config['OUTPUT_DIR'], "user_logs", f"user_{user_id}", "stream_info_dump.pth"))
            try:
                assert os.path.exists(user_dump_path), f"Not existing: {user_dump_path}"
            except Exception as e:
                print(f"SKIPPING USER {user_id}: {e}")
                continue
            user_to_dump_path[user_id] = user_dump_path
            user_dump_dict = torch.load(user_to_dump_path[user_id])

            # Iterate action/verb/noun losses per instance
            for action_mode in action_modes:
                ####################################
                # LOSS/CONFIDENCE BASED METRICS
                ####################################

                # Unbalanced LL
                user_val, metric_name = dump_to_LL(user_dump_dict, action_mode, balanced=False)
                _save_to_wandb_summary(metric_name, user_val, user_run)

                # Balanced LL
                user_val, metric_name = dump_to_LL(user_dump_dict, action_mode, balanced=True)
                _save_to_wandb_summary(metric_name, user_val, user_run)

                # Also get balanced loss
                user_val, metric_name = dump_to_loss(user_dump_dict, action_mode, balanced=True)
                _save_to_wandb_summary(metric_name, user_val, user_run)

                # Get balanced ACC
                user_val, metric_name = dump_to_ACC(user_dump_dict, action_mode, macro_avg=True)
                _save_to_wandb_summary(metric_name, user_val, user_run)

                # Get decorrelated ACC and percentage + nb of sample it is measured over
                AG_metric_dict, non_AG_metric_dict = dump_to_decorrelated_ACC(
                    user_dump_dict, action_mode, macro_avg=True, correlated=False)
                for metric_name, user_val in AG_metric_dict.items():
                    _save_to_wandb_summary(metric_name, user_val, user_run, add_to_AG=True)
                for metric_name, user_val in non_AG_metric_dict.items():
                    _save_to_wandb_summary(metric_name, user_val, user_run, add_to_AG=False)

                # Get CORRELATED ACC and percentage + nb of sample it is measured over
                AG_metric_dict, non_AG_metric_dict = dump_to_decorrelated_ACC(
                    user_dump_dict, action_mode, macro_avg=True, correlated=True)
                for metric_name, user_val in AG_metric_dict.items():
                    _save_to_wandb_summary(metric_name, user_val, user_run, add_to_AG=True)
                for metric_name, user_val in non_AG_metric_dict.items():
                    _save_to_wandb_summary(metric_name, user_val, user_run, add_to_AG=False)

                # Get conditional analysis
                if conditional_analysis:
                    metric_dict: dict[str, float] = dump_to_correct_conditional_metrics(user_dump_dict, action_mode)
                    for metric_name, user_val in metric_dict.items():
                        _save_to_wandb_summary(metric_name, user_val, user_run)

    print(f"Finished processing")
    metric_names_AG = sorted(list(metric_names_AG))
    metric_names_nonAG = sorted(list(metric_names_nonAG))

    print(f"AG: Aggregating over users in separate step: \n{pprint.pformat(metric_names_AG)}")
    avg_and_delta_avg_results_over_user_streams(selected_group_names, metrics=metric_names_AG,
                                                skip_pretrain_delta=False)

    print(f"Non-AG: Aggregating over users in separate step: \n{pprint.pformat(metric_names_nonAG)}")
    avg_and_delta_avg_results_over_user_streams(selected_group_names, metrics=metric_names_nonAG,
                                                skip_pretrain_delta=True)


def dump_to_LL(user_dump_dict, action_mode, balanced=True, return_per_sample_result=False):
    """Likelihood (LL) or confidence (C) in our paper."""
    balance_name = "balanced_" if balanced else ""
    full_new_metric_name = f"train_{action_mode}_batch/{balance_name}LL"

    # Get labels
    action_labels: list[tuple[int, int]] = list(map(tuple, user_dump_dict['sample_idx_to_action_list']))
    if action_mode == 'action':
        selected_labels = action_labels
        loss_dump_name = 'sample_idx_to_action_loss'

    elif action_mode == 'verb':
        selected_labels = [x[0] for x in action_labels]
        loss_dump_name = 'sample_idx_to_verb_loss'

    elif action_mode == 'noun':
        selected_labels = [x[1] for x in action_labels]
        loss_dump_name = 'sample_idx_to_noun_loss'

    else:
        raise ValueError()

    sample_idx_to_loss: list[float] = user_dump_dict[loss_dump_name]
    sample_idx_to_loss_t = torch.tensor(sample_idx_to_loss)  # To tensor
    sample_idx_to_LL_t: torch.Tensor = loss_CE_to_class_confidence(sample_idx_to_loss_t)  # convert to likelihood

    if return_per_sample_result:
        return sample_idx_to_LL_t

    if balanced:
        result = per_sample_metric_to_macro_avg(sample_idx_to_LL_t.tolist(), selected_labels)
    else:
        result = sample_idx_to_LL_t.mean().item()

    return result, full_new_metric_name


def dump_to_correct_conditional_metrics(user_dump_dict, action_mode):
    """ Split loss and likelihood (confidence in correct class) in 2 ways:
     1) When it is predicted correctly
     2) When it is predicted incorrectly

     Results in metrics:

'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/CORRECT_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/CORRECT_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/CORRECT_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/CORRECT_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/CORRECT_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/CORRECT_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/WRONG_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/WRONG_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/WRONG_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/WRONG_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/WRONG_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/LL/WRONG_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/CORRECT_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/CORRECT_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/CORRECT_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/CORRECT_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/CORRECT_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/CORRECT_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/WRONG_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/WRONG_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/WRONG_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/WRONG_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/WRONG_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_action_batch/loss/WRONG_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/CORRECT_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/CORRECT_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/CORRECT_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/CORRECT_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/CORRECT_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/CORRECT_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/WRONG_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/WRONG_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/WRONG_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/WRONG_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/WRONG_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/LL/WRONG_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/CORRECT_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/CORRECT_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/CORRECT_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/CORRECT_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/CORRECT_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/CORRECT_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/WRONG_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/WRONG_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/WRONG_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/WRONG_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/WRONG_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_noun_batch/loss/WRONG_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/CORRECT_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/CORRECT_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/CORRECT_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/CORRECT_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/CORRECT_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/CORRECT_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/WRONG_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/WRONG_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/WRONG_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/WRONG_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/WRONG_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/LL/WRONG_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/CORRECT_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/CORRECT_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/CORRECT_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/CORRECT_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/CORRECT_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/CORRECT_COND/sum/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/WRONG_COND/avg/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/WRONG_COND/avg/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/WRONG_COND/count/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/WRONG_COND/count/SE',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/WRONG_COND/sum/mean',
 'adhoc_users_aggregate/CONDITIONAL_ANALYSIS/train_verb_batch/loss/WRONG_COND/sum/SE',

     """
    metric_name_prefix = f"CONDITIONAL_ANALYSIS/train_{action_mode}_batch"

    # Get loss and likelihood
    sample_to_likelihood_t = dump_to_LL(user_dump_dict, action_mode, return_per_sample_result=True)
    sample_to_loss_t = dump_to_loss(user_dump_dict, action_mode, return_per_sample_result=True)
    nb_samples_stream = sample_to_loss_t.shape[0]
    print(f"Nb samples = {nb_samples_stream}")

    # Get conditional
    sample_to_correct_t = dump_to_ACC(user_dump_dict, action_mode, return_per_sample_result=True)

    # Use corrects-tensor, and label list
    likelihood_d = ConditionalAverageMeterDict()  # Use it's dict
    likelihood_d.update(sample_to_likelihood_t.tolist(), cond_list=sample_to_correct_t.tolist())

    likelihood_corrects = likelihood_d.meter_dict[1]
    likelihood_wrongs = likelihood_d.meter_dict[0]

    # Loss
    loss_d = ConditionalAverageMeterDict()  # Use it's dict
    loss_d.update(sample_to_loss_t.tolist(), cond_list=sample_to_correct_t.tolist())

    loss_corrects = loss_d.meter_dict[1]
    loss_wrongs = loss_d.meter_dict[0]

    # Avgs are over correct/wrong separately per stream! Fix by using sum and total count
    # Later is avged over users as well, which is also how we calculate our metrics
    ret_metrics = {

        # Likelihood
        f"{metric_name_prefix}/LL/CORRECT_COND/avg": likelihood_corrects.sum / nb_samples_stream,
        f"{metric_name_prefix}/LL/CORRECT_COND/sum": likelihood_corrects.sum,
        f"{metric_name_prefix}/LL/CORRECT_COND/count": likelihood_corrects.count,

        f"{metric_name_prefix}/LL/WRONG_COND/avg": likelihood_wrongs.sum / nb_samples_stream,
        f"{metric_name_prefix}/LL/WRONG_COND/sum": likelihood_wrongs.sum,
        f"{metric_name_prefix}/LL/WRONG_COND/count": likelihood_wrongs.count,

        # Loss
        f"{metric_name_prefix}/loss/CORRECT_COND/avg": loss_corrects.sum / nb_samples_stream,
        f"{metric_name_prefix}/loss/CORRECT_COND/sum": loss_corrects.sum,
        f"{metric_name_prefix}/loss/CORRECT_COND/count": loss_corrects.count,

        f"{metric_name_prefix}/loss/WRONG_COND/avg": loss_wrongs.sum / nb_samples_stream,
        f"{metric_name_prefix}/loss/WRONG_COND/sum": loss_wrongs.sum,
        f"{metric_name_prefix}/loss/WRONG_COND/count": loss_wrongs.count,

    }

    return ret_metrics


def dump_to_loss(user_dump_dict, action_mode, balanced=True, return_per_sample_result=False):
    """ Loss. """
    balance_name = "balanced_" if balanced else ""
    full_new_metric_name = f"train_{action_mode}_batch/{balance_name}loss"

    # Get labels
    action_labels: list[tuple[int, int]] = list(map(tuple, user_dump_dict['sample_idx_to_action_list']))
    if action_mode == 'action':
        selected_labels = action_labels
        loss_dump_name = 'sample_idx_to_action_loss'

    elif action_mode == 'verb':
        selected_labels = [x[0] for x in action_labels]
        loss_dump_name = 'sample_idx_to_verb_loss'

    elif action_mode == 'noun':
        selected_labels = [x[1] for x in action_labels]
        loss_dump_name = 'sample_idx_to_noun_loss'

    else:
        raise ValueError()

    sample_idx_to_loss: list[float] = user_dump_dict[loss_dump_name]
    sample_idx_to_loss_t = torch.tensor(sample_idx_to_loss)  # To tensor

    if return_per_sample_result:
        return sample_idx_to_loss_t

    # Now average based on action label
    if balanced:
        result = per_sample_metric_to_macro_avg(sample_idx_to_loss_t.tolist(), selected_labels)
    else:
        result = sample_idx_to_loss_t.mean().item()

    return result, full_new_metric_name


def dump_to_ACC(user_dump_dict, action_mode, macro_avg=True, k=1, return_per_sample_result=False):
    """ ACC. """
    assert k == 1

    balance_name = "balanced_" if macro_avg else ""
    full_new_metric_name = f"train_{action_mode}_batch/{balance_name}top1_acc"

    # Get labels
    action_labels: list[tuple[int, int]] = list(map(tuple, user_dump_dict['sample_idx_to_action_list']))
    action_labels_t: torch.Tensor = torch.tensor(action_labels)

    # Get predictions for verbs/nouns
    verb_preds: list[torch.Tensor] = user_dump_dict['sample_idx_to_verb_pred']
    noun_preds: list[torch.Tensor] = user_dump_dict['sample_idx_to_noun_pred']

    verb_preds_t = torch.stack(verb_preds)
    noun_preds_t = torch.stack(noun_preds)
    action_preds = (verb_preds_t, noun_preds_t)

    result = get_micro_macro_avg_acc(
        action_mode, action_preds, action_labels_t,
        k=k, macro_avg=macro_avg, return_per_sample_result=return_per_sample_result
    )

    if return_per_sample_result:
        return result

    return result, full_new_metric_name


def dump_to_decorrelated_ACC(user_dump_dict, action_mode, macro_avg=True, k=1, return_per_sample_result=False,
                             decorrelation_window=1, batch_size=4, correlated=False) -> dict[str, float]:
    """
    :param user_dump_dict:
    :param action_mode:
    :param macro_avg:
    :param k:
    :param return_per_sample_result:
    :param decorrelation_window: Size of the window to look back to for samples in current batch of size <batch_size>.
    :param correlated: Take correlated samples instead
    :return:
    """
    assert k == 1

    balance_name = "balanced_" if macro_avg else ""
    correlated_name = "decorrelated" if not correlated else "correlated"
    full_new_metric_name = f"train_{action_mode}_batch/{balance_name}top1_acc/{correlated_name}"

    # Get labels
    action_labels: list[tuple[int, int]] = list(map(tuple, user_dump_dict['sample_idx_to_action_list']))
    action_labels_t: torch.Tensor = torch.tensor(action_labels)

    # TODO filter on action-mode, then filter all labels/ preds based on whether they have re-occuring in the
    if action_mode == 'action':
        filter_labels_t = action_labels_t

    elif action_mode == 'verb':
        filter_labels_t = action_labels_t[:, 0]

    elif action_mode == 'noun':
        filter_labels_t = action_labels_t[:, 1]

    else:
        raise ValueError()

    # Filter
    nb_orig_samples_stream = action_labels_t.shape[0]
    keep_samples_t = torch.ones(nb_orig_samples_stream, device=filter_labels_t.device, dtype=torch.long).bool()

    for sample_idx in range(1, nb_orig_samples_stream):  # First one kept always
        window_start_idx = max(0, sample_idx - decorrelation_window)

        label_window = filter_labels_t[window_start_idx: sample_idx]
        assert len(label_window) <= decorrelation_window

        current_sample_label = filter_labels_t[sample_idx]
        if current_sample_label in label_window:
            keep_samples_t[sample_idx] = False  # Don't keep

    # Inverse if instead do correlated
    if correlated:
        keep_samples_t = ~keep_samples_t

    # Summary
    num_samples_keep = keep_samples_t.sum().item()
    perc_kept = num_samples_keep / nb_orig_samples_stream
    print(f"{correlated_name} ACC: {num_samples_keep}/{nb_orig_samples_stream} ({round(perc_kept, 3)}%) samples")
    assert num_samples_keep <= nb_orig_samples_stream

    # Get predictions for verbs/nouns
    verb_preds: list[torch.Tensor] = user_dump_dict['sample_idx_to_verb_pred']
    noun_preds: list[torch.Tensor] = user_dump_dict['sample_idx_to_noun_pred']
    verb_preds_t = torch.stack(verb_preds)
    noun_preds_t = torch.stack(noun_preds)

    # Filter
    keep_sample_idxs = torch.nonzero(keep_samples_t, as_tuple=True)
    verb_preds_t = verb_preds_t[keep_sample_idxs]
    noun_preds_t = noun_preds_t[keep_sample_idxs]
    action_labels_t = action_labels_t[keep_sample_idxs]  # Filter also labels to pass
    assert verb_preds_t.shape[0] == noun_preds_t.shape[0] == action_labels_t.shape[0] == num_samples_keep

    # Combine to actions
    action_preds = (verb_preds_t, noun_preds_t)

    result = get_micro_macro_avg_acc(
        action_mode, action_preds, action_labels_t,
        k=k, macro_avg=macro_avg, return_per_sample_result=return_per_sample_result
    )

    if return_per_sample_result:
        return result

    dict_for_AG = {
        full_new_metric_name: result,
    }
    dict_not_for_AG = {
        f"{full_new_metric_name}/percentage_kept": perc_kept,
        f"{full_new_metric_name}/num_samples_keep": num_samples_keep,
        f"{full_new_metric_name}/num_samples_total": nb_orig_samples_stream,
    }

    return dict_for_AG, dict_not_for_AG


def running_avg_to_avg_and_delta(selected_group_names, metrics=None, skip_pretrain_delta=False):
    """ Pretrain didn't have running acc yet, this is fix of metric mapping. """
    metric_to_pretrain_metric_map = {
        'train_action_batch/top1_acc_balanced_running_avg': 'train_action_batch/balanced_top1_acc',
        'train_verb_batch/top1_acc_balanced_running_avg': 'train_verb_batch/balanced_top1_acc',
        'train_noun_batch/top1_acc_balanced_running_avg': 'train_noun_batch/balanced_top1_acc',
    }

    metrics = list(metric_to_pretrain_metric_map.keys())
    avg_and_delta_avg_results_over_user_streams(
        selected_group_names,
        metrics=metrics,
        metric_to_pretrain_metric_map=metric_to_pretrain_metric_map,
    )


def running_avg_to_avg(selected_group_names):
    """ """
    metrics = ['train_action_POST_UPDATE_BATCH/loss_running_avg',
               'train_noun_POST_UPDATE_BATCH/top1_acc_balanced_running_avg']
    avg_and_delta_avg_results_over_user_streams(
        selected_group_names,
        metrics=metrics,
        skip_pretrain_delta=True
    )


def avg_and_delta_avg_results_over_user_streams(selected_group_names, metrics=None, skip_pretrain_delta=False,
                                                metric_to_pretrain_metric_map=None):
    """ After training a model, aggregate the avg metrics over the stream such as ACC.
    Aggregate over user stream results. """

    default_metrics = [
        'train_action_batch/loss_running_avg',
        'train_verb_batch/loss_running_avg',
        'train_noun_batch/loss_running_avg',

        'train_action_batch/top1_acc_running_avg',
        'train_verb_batch/top1_acc_running_avg',
        'train_noun_batch/top1_acc_running_avg',
        'train_verb_batch/top5_acc_running_avg',
        'train_noun_batch/top5_acc_running_avg',

        # Only for newer runs: Implemented at training time:
        # 'train_action_batch/top1_acc_balanced_running_avg',
        # 'train_verb_batch/top1_acc_balanced_running_avg',
        # 'train_noun_batch/top1_acc_balanced_running_avg',
    ]

    # Default metrics logged at runtime
    if metrics is None:  # Always add defaults anyway
        metrics = default_metrics
    else:
        metrics = default_metrics + metrics

    if metric_to_pretrain_metric_map is not None:
        for metric_name in metric_to_pretrain_metric_map.keys():
            assert metric_name in metrics, \
                f"Defined metric mapping to pretrain metric name, but haven't included in metrics to consider: {metric_name}"

    # GET PRETRAIN as reference
    if not skip_pretrain_delta:
        online_to_adhoc_pretrain_map = {} if metric_to_pretrain_metric_map is None else metric_to_pretrain_metric_map
        adhoc_to_online_pretrain_map = {v: k for k, v in online_to_adhoc_pretrain_map.items()}

        # Map to compatible
        pretrain_metrics = [
            online_to_adhoc_pretrain_map[m] if m in online_to_adhoc_pretrain_map else m for m in metrics
        ]
        pretrain_user_ids, pretrain_metric_dict_adhoc = get_pretrain_user_results(pretrain_metrics)  # Get adhoc metrics

        # Remap:
        pretrain_final_stream_metric_userlists = {}
        for metric_name, val in pretrain_metric_dict_adhoc.items():
            if metric_name in adhoc_to_online_pretrain_map.keys():
                remap_metric_name = adhoc_to_online_pretrain_map[metric_name]
            else:
                remap_metric_name = metric_name
            pretrain_final_stream_metric_userlists[remap_metric_name] = val

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:

        # Filter runs in group that have finished
        try:
            user_ids, final_stream_metric_userlists = _collect_wandb_group_user_results_for_metrics(
                group_name, metric_names=metrics, user_ids=USER_LIST, run_filter=None,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"SKIPPING: contains error: Group ={group_name}")
            continue

        # Checks
        if not is_user_runs_valid(user_ids, group_name):
            continue
        for metric_name, user_val_list in final_stream_metric_userlists.items():
            assert len(user_val_list) == NB_EXPECTED_USERS, f"{metric_name} has not all user values: {user_val_list}"

        print(f"Processing users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")

        all_results = final_stream_metric_userlists
        if not skip_pretrain_delta:
            AG_dict = get_pretrain_delta_with_user_results(
                pretrain_user_ids, user_ids, final_stream_metric_userlists, pretrain_final_stream_metric_userlists
            )
            all_results = {**all_results, **AG_dict}  # Add to final results

        # Upload for group the avg over runs from dict
        upload_metric_dict_to_wandb(all_results, group_name, run_filter=None)


def get_pretrain_user_results(metrics):
    pretrain_user_ids, pretrain_final_stream_metric_userlists = _collect_wandb_group_user_results_for_metrics(
        PRETRAIN_GROUP, metric_names=metrics, run_filter=None, user_ids=USER_LIST)

    # Check users and nb of results
    if not is_user_runs_valid(pretrain_user_ids, PRETRAIN_GROUP):
        print(f"Pretrain not valid: exit")
        return
    for metric_name, user_val_list in pretrain_final_stream_metric_userlists.items():
        assert len(user_val_list) == NB_EXPECTED_USERS, f"{metric_name} has not all user values: {user_val_list}"

    return pretrain_user_ids, pretrain_final_stream_metric_userlists


def get_pretrain_delta_with_user_results(pretrain_user_ids,
                                         user_ids,
                                         final_stream_metric_userlists,
                                         pretrain_final_stream_metric_userlists
                                         ) -> dict:
    """ Given pretrain and user-metrics, return dict that includes the deltas with pretrain performance. """
    AG_dict = {}
    assert pretrain_user_ids == user_ids, \
        "Order of pretrained and user ids should be same, otherwise SE will not be correct for avg the deltas. "

    for metric_name, user_val_list in final_stream_metric_userlists.items():
        assert metric_name in pretrain_final_stream_metric_userlists, \
            f"KEY:{metric_name} of user not in pretrain results!"

    # Get same results but delta with pretrain results
    delta_sign_map = get_delta_mappings()

    # make sure users have same order
    print("Assuming for ACC: new result - pretrain result")
    for metric_name, user_val_list in final_stream_metric_userlists.items():
        delta_sign = delta_sign_map[metric_name]
        pretrain_val_list = pretrain_final_stream_metric_userlists[metric_name]
        assert len(pretrain_val_list) == len(user_val_list)

        # Add as later used to calculate /mean and /SE on
        AG_dict[f"{metric_name}/PRETRAIN_abs"] = pretrain_val_list
        AG_dict[f"{metric_name}/adhoc_AG"] = [
            get_delta(delta_sign, user_val, pretrain_val)
            for pretrain_val, user_val in zip(pretrain_val_list, user_val_list)]

    return AG_dict


def upload_metric_dict_to_wandb(metricname_to_user_results_list: dict[str, list], group_name: str, run_filter=None,
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


def _collect_wandb_group_user_results_for_metrics(group_name, run_filter, metric_names, user_ids: list = None,
                                                  metrics_strict=True):
    # Get metrics over runs(users)
    final_stream_metrics_per_user = {metric_name: [] for metric_name in metric_names}

    def get_dynamic_user_results():
        """ Add user results in order of downloading from WandB."""
        # summary = final value (excludes NaN rows)
        # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter)):
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
        for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, run_filter=run_filter)):
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


def aggregate_test_results_over_user_streams(selected_group_names):
    """ After testing on a model with cfg.STREAM_EVAL_ONLY=True, aggregate over user stream results. """

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:

        # Filter runs in group that have finished
        run_filter = {
            "$and": [
                {"group": group_name},
                {"summary_metrics.finished_test_run": 1}
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

        upload_metric_dict_to_wandb(final_stream_metric_userlists, group_name, run_filter)


def _collect_user_test_results(group_name, run_filter):
    # Get metrics over runs(users)
    final_stream_metrics_per_user = {
        'test_action_batch/loss': [],
        'test_verb_batch/loss': [],
        'test_noun_batch/loss': [],

        'test_action_batch/balanced_loss': [],
        'test_verb_batch/balanced_loss': [],
        'test_noun_batch/balanced_loss': [],

        'test_action_batch/top1_acc': [],
        'test_verb_batch/top1_acc': [],
        'test_verb_batch/top5_acc': [],
        'test_noun_batch/top1_acc': [],
        'test_noun_batch/top5_acc': [],

        'test_action_batch/balanced_top1_acc': [],
        'test_verb_batch/balanced_top1_acc': [],
        'test_noun_batch/balanced_top1_acc': [],

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


def aggregate_online_calculated_OAG_over_user_streams(selected_group_names):
    """ Get OAG results for action/verb/noun per user, normalize cumulative metric over user stream samples, average and upload to wandb. """

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
    if single_group_name is None:
        csv_path = os.path.join(csv_dirname, csv_filename)
        assert os.path.isfile(csv_path), f"Non-existing: {csv_path}"

        # From WandB csv from overview, grouped by Group. Get all the names in the csv (these are the run group names).
        selected_group_names: list[str] = get_group_names_from_csv(csv_path)
    else:
        selected_group_names = [single_group_name]
    print(f"Group names={pprint.pformat(selected_group_names)}")

    locals()[MODE](selected_group_names)  # Call function
