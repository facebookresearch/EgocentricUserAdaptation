# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
All results saved in dump postprocessed here.
"""

import os
import pprint

import torch
import wandb

from continual_ego4d.metrics.offline_metrics import get_micro_macro_avg_acc
from continual_ego4d.metrics.offline_metrics import per_sample_metric_to_macro_avg
from continual_ego4d.processing.adhoc_metric_processing_wandb import avg_user_streams
from continual_ego4d.processing.utils import get_group_names_from_csv, get_group_run_iterator

api = wandb.Api()

# Adapt settings
csv_path = '/your/path/to/wandb_export_2022-09-19T12_08_04.153-07_00.csv'  # Iterate all WandB runs of the groupnames in this csv
single_group_name = None  # Or set to Groupname, e.g. "Finetuning_2022-10-14_13-11-14_UIDac6a2798-7fb0-4e9c-9896-c6c54de5237c"

train = True
pretrain_group_train = "FixedNetwork_2022-10-07_21-50-23_UID80e71950-cea4-44ba-ba16-7dddfe95be26"  # TODO set your pretrain groupname for training
pretrain_group_test = "FixedNetwork_2022-10-07_15-23-09_UID6e87e600-a447-438f-9a31-f5cae6dc9ed4"  # TODO set your pretrain groupname for testing

# Fixed Settings
PROJECT_NAME = "ContinualUserAdaptation"  # FIXME might have to pre-pend your wandb username followed by slash '/'

if train:
    train_users = ['68', '265', '324', '30', '24', '421', '104', '108', '27', '29']
    USER_LIST = train_users
    NB_EXPECTED_USERS = len(train_users)
    PRETRAIN_GROUP = pretrain_group_train

else:
    test_users = [
        "59", "23", "17", "37", "97", "22", "31", "10", "346", "359", "120", "19", "16", "283", "28", "20", "44", "38",
        "262", "25", "51", "278", "55", "39", "45", "33", "331", "452", "453", "21", "431", "116", "35", "105", "378",
        "74", "11", "126", "123", "436"]
    USER_LIST = test_users
    NB_EXPECTED_USERS = len(test_users)
    PRETRAIN_GROUP = pretrain_group_test

# New uploaded keys
NEW_METRIC_PREFIX = 'adhoc_users_aggregate'
NEW_METRIC_PREFIX_MEAN = 'mean'
NEW_METRIC_PREFIX_SE = 'SE'  # Unbiased standard error
USER_AGGREGATE_COUNT = f"{NEW_METRIC_PREFIX}/user_aggregate_count"  # Over how many users added, also used to check if processed


def main():
    # From WandB csv from overview, grouped by Group. Get all the names in the csv (these are the run group names).
    if single_group_name is None:
        assert os.path.isfile(csv_path), f"Non-existing: {csv_path}"
        selected_group_names: list[str] = get_group_names_from_csv(csv_path)

    # Process for single group
    else:
        selected_group_names = [single_group_name]
    print(f"Group names={pprint.pformat(selected_group_names)}")

    adhoc_metrics_from_csv_dump_to_wandb(selected_group_names)


def adhoc_metrics_from_csv_dump_to_wandb(selected_group_names, overwrite=True, skip_loss_metrics=True):
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

                if not skip_loss_metrics:
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

    print(f"Finished processing runs in group")
    metric_names_AG = sorted(list(metric_names_AG))
    metric_names_nonAG = sorted(list(metric_names_nonAG))

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:
        print(f"Aggregating over users in separate step: \n{pprint.pformat(metric_names_AG)}")
        avg_user_streams(PROJECT_NAME, group_name,
                         expected_user_ids=USER_LIST,
                         pretrain_group_name=PRETRAIN_GROUP,
                         metric_names=metric_names_AG)


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
                             decorrelation_window=1, correlated=False) -> dict[str, float]:
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


if __name__ == "__main__":
    main()
