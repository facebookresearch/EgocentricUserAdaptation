"""
Aggregate the EgoAdapt results of various user-streams into user-stream summarizing metrics.
Examples are the delta and the average over all users.
"""

import traceback


def avg_metrics_over_user_streams(group_name, expected_user_ids, ):
    """ """

    # Running ACC metrics
    metrics = [
        'train_action_batch/top1_acc_balanced_running_avg',
        'train_verb_batch/top1_acc_balanced_running_avg',
        'train_noun_batch/top1_acc_balanced_running_avg',
    ]

    # Running OAG metrics
    if:
        metrics += [
            'train_action_batch/AG_running_avg',
            'train_verb_batch/AG_running_avg',
            'train_noun_batch/AG_running_avg', ]

    # avg_and_delta_avg_results_over_user_streams(
    #     selected_group_names,
    #     metrics=metrics,
    #     skip_pretrain_delta=True
    # )

    # Filter user runs in group that have finished
    try:
        user_ids, final_stream_metric_userlists = _collect_wandb_group_user_results_for_metrics(
            group_name, metric_names=metrics, user_ids=expected_user_ids, run_filter=None,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Error in aggregating GROUP={group_name}")
        exit(1)

    # Checks
    if not is_user_runs_valid(user_ids, group_name):
        continue
    for metric_name, user_val_list in final_stream_metric_userlists.items():
        assert len(user_val_list) == len(expected_user_ids), \
            f"Metric '{metric_name}' has not all user values: {user_val_list}"

    print(f"Processing users: {len(user_ids)}/{len(expected_user_ids)} -> {user_ids}: Group ={group_name}")
    all_results = final_stream_metric_userlists

    # Upload for group the avg over runs from dict
    upload_metric_dict_to_wandb(all_results, group_name, run_filter=None)


def _collect_wandb_group_user_results_for_metrics(
        group_name,
        run_filter,
        metric_names,
        user_ids: list = None,
        metrics_strict=True,
):
    """
    """
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


def avg_and_delta_avg_results_over_user_streams(
        selected_group_names,
        metrics=None,
        skip_pretrain_delta=False,
        metric_to_pretrain_metric_map=None
):
    """ After training a model, aggregate the avg metrics over the stream such as ACC.
    Aggregate over user stream results. """

    default_metrics = [
        # 'train_action_batch/loss_running_avg',
        # 'train_verb_batch/loss_running_avg',
        # 'train_noun_batch/loss_running_avg',
        #
        # 'train_action_batch/top1_acc_running_avg',
        # 'train_verb_batch/top1_acc_running_avg',
        # 'train_noun_batch/top1_acc_running_avg',
        # 'train_verb_batch/top5_acc_running_avg',
        # 'train_noun_batch/top5_acc_running_avg',

        # Only for newer runs: Implemented at training time:

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
