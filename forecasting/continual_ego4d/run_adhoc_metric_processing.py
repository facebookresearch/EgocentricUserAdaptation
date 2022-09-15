"""
Given the metrics we stored per-user. Aggregate metrics over users into a single number.

Because run-selection can be cumbersome. We enable downloading a csv from wandb, and extracting for all groups the
user-results.

You can pull results directly from wandb with their API: https://docs.wandb.ai/guides/track/public-api-guide

1) Go to the table overview. Group runs based on 'Group'. Click download button and 'export as CSV'.
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

api = wandb.Api()

train_users = ['68', '265', '324', '30', '24', '421', '104', '108', '27', '29']

# Adapt settings
csv_filename = 'wandb_export_2022-09-15T09_58_01.353-07_00.csv'  # TODO copy file here and past name here
NB_EXPECTED_USERS = len(train_users)

# Fixed Settings
csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
PROJECT_NAME = "matthiasdelange/ContinualUserAdaptation"

# New uploaded keys
NEW_METRIC_PREFIX = 'adhoc_users_aggregate'
NEW_METRIC_PREFIX_MEAN = 'mean'
NEW_METRIC_PREFIX_SE = 'SE'  # Unbiased standard error
USER_AGGREGATE_COUNT = f"{NEW_METRIC_PREFIX}/user_aggregate_count"  # Over how many users added, also used to check if processed


def main(selected_group_names_csv_path):
    # From WandB csv from overview, grouped by Group. Get all the names in the csv (these are the run group names).
    selected_group_names: list[str] = get_selected_group_names(selected_group_names_csv_path)
    print(f"Group names={pprint.pformat(selected_group_names)}")

    # Iterate groups (a collective of independent user-runs)
    for group_name in selected_group_names:

        try:
            user_ids, total_iters_per_user, final_stream_metrics_to_avg = collect_users(group_name)
        except Exception as e:
            print(e)
            print(f"SKIPPING: contains error: Group ={group_name}")
            continue

        # Check if all users:
        if len(user_ids) < NB_EXPECTED_USERS:
            print(f"SKIPPING: MISSING users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")
            continue
        elif len(user_ids) > NB_EXPECTED_USERS:
            raise Exception(f"Might need to readjust EXPECTED USERS? -> user_ids={user_ids}")

        for name, user_val_list in final_stream_metrics_to_avg.items():
            assert len(user_val_list) == NB_EXPECTED_USERS, f"{name} has not all user values: {user_val_list}"

        print(f"Processing users: {len(user_ids)}/{NB_EXPECTED_USERS} -> {user_ids}: Group ={group_name}")

        # Change names for metrics
        updated_metrics_dict = {'total_iters_per_user': total_iters_per_user, 'user_id': user_ids}
        new_metric_names = []
        for k, v in final_stream_metrics_to_avg.items():
            new_metric_name = f"{NEW_METRIC_PREFIX}/{k}"
            new_metric_names.append(new_metric_name)
            updated_metrics_dict[new_metric_name] = v

        # Make averages over users from their final result on the stream, and normalize by steps in user-stream

        updated_metrics_df = pd.DataFrame.from_dict(updated_metrics_dict)
        updated_metrics_df_norm = updated_metrics_df[new_metric_names].div(
            updated_metrics_df.total_iters_per_user, axis=0)

        final_update_metrics_dict = {USER_AGGREGATE_COUNT: len(user_ids)}
        for col in updated_metrics_df_norm.columns.tolist():
            final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_MEAN}"] = updated_metrics_df_norm[col].mean()
            final_update_metrics_dict[f"{col}/{NEW_METRIC_PREFIX_SE}"] = updated_metrics_df_norm[col].sem()

        # Update all group entries:
        for user_run in get_group_run_iterator(group_name):
            for name, new_val in final_update_metrics_dict.items():
                user_run.summary[name] = new_val

            user_run.summary.update()  # UPLOAD


def get_group_run_iterator(group_name):
    """ Only get user-runs that finished processing stream (finished_run=True)."""
    group_runs = api.runs(PROJECT_NAME, {
        "$and": [
            {"group": group_name},
            {"summary_metrics.finished_run": True}
        ]
    })
    return group_runs


def collect_users(group_name):
    # Get metrics over runs(users)
    final_stream_metrics_to_avg = {
        'train_action_batch/AG_cumul': [],
        'train_verb_batch/AG_cumul': [],
        'train_noun_batch/AG_cumul': [],
    }
    user_ids = []  # all processed users
    total_iters_per_user = []  # nb of steps per user-stream

    # summary = final value (excludes NaN rows)
    # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
    for user_run in get_group_run_iterator(group_name):  # ACTS LIKE ITERATOR, CAN ONLY CALL LIKE THIS!
        user_ids.append(user_run.config['DATA.COMPUTED_USER_ID'])
        total_iters_per_user.append(user_run.summary['trainer/global_step'])

        for metric_name, val_list in final_stream_metrics_to_avg.items():
            user_metric_val = user_run.summary[metric_name]
            val_list.append(user_metric_val)

    return user_ids, total_iters_per_user, final_stream_metrics_to_avg


def get_selected_group_names(selected_group_names_csv_path):
    selected_group_names_df = pd.read_csv(selected_group_names_csv_path)
    group_names = selected_group_names_df['Name'].to_list()
    return group_names


if __name__ == "__main__":
    # selected_group_names_csv = StringIO(
    #
    #     cleanup_csv_str("""
    #     "Name","State","METHOD.METHOD_NAME","Notes","User","Tags","Created","Runtime","SOLVER.BASE_LR","SOLVER.MOMENTUM","SOLVER.OPTIMIZING_METHOD","OUTPUT_DIR"
    #     "Finetuning_2022-09-13_11-21-41_UID90dc9916-8bf1-42b2-aeca-867e5ac87c8b_MATT","running","Finetuning","-","matthiasdelange","","2022-09-13T18:23:32.000Z","12629","0.001","0.6","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-001_SOLVER-MOMENTUM=0-6_SOLVER-NESTEROV=True/2022-09-13_11-21-41_UID90dc9916-8bf1-42b2-aeca-867e5ac87c8b"
    #     "Finetuning_2022-09-13_11-17-06_UID7cbb3ae5-9ea5-429f-be0b-0308e7310316_MATT","running","Finetuning","-","matthiasdelange","","2022-09-13T18:18:39.000Z","12920","0.01","0.6","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-01_SOLVER-MOMENTUM=0-6_SOLVER-NESTEROV=True/2022-09-13_11-17-06_UID7cbb3ae5-9ea5-429f-be0b-0308e7310316"
    #     "Finetuning_2022-09-13_11-07-47_UID3570828c-3aa7-4983-ab21-d75dfbdb4226_MATT","running","Finetuning","-","matthiasdelange","","2022-09-13T18:09:13.000Z","13481","0.1","0.6","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-1_SOLVER-MOMENTUM=0-6_SOLVER-NESTEROV=True/2022-09-13_11-07-47_UID3570828c-3aa7-4983-ab21-d75dfbdb4226"
    #     "Finetuning_2022-09-13_09-33-33_UIDf1d94198-7bae-4ba6-ac99-a13c24c3fb14_MATT","finished","Finetuning","-","matthiasdelange","","2022-09-13T16:34:23.000Z","19154","0.1","0","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-1_SOLVER-MOMENTUM=0-0_SOLVER-NESTEROV=True/2022-09-13_09-33-33_UIDf1d94198-7bae-4ba6-ac99-a13c24c3fb14"
    #     "Finetuning_2022-09-13_10-53-52_UID958392f7-c477-4a09-a7ac-c72cc81251c2_MATT","running","Finetuning","-","matthiasdelange","","2022-09-13T17:55:20.000Z","14323","0.01","0","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-01_SOLVER-MOMENTUM=0-0_SOLVER-NESTEROV=True/2022-09-13_10-53-52_UID958392f7-c477-4a09-a7ac-c72cc81251c2"
    #     "Finetuning_2022-09-13_09-35-20_UID15afc9c2-2463-47fa-b6f9-59930f45a628_MATT","finished","Finetuning","-","matthiasdelange","","2022-09-13T16:37:57.000Z","18535","0.001","0.3","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-001_SOLVER-MOMENTUM=0-3_SOLVER-NESTEROV=True/2022-09-13_09-35-20_UID15afc9c2-2463-47fa-b6f9-59930f45a628"
    #     "Finetuning_2022-09-13_09-35-13_UID816d4af0-a434-4189-806a-4df2a917d615_MATT","finished","Finetuning","-","matthiasdelange","","2022-09-13T16:37:36.000Z","18985","0.01","0.3","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-01_SOLVER-MOMENTUM=0-3_SOLVER-NESTEROV=True/2022-09-13_09-35-13_UID816d4af0-a434-4189-806a-4df2a917d615"
    #     "Finetuning_2022-09-13_09-34-51_UID819b79e9-031e-4695-9d61-1d8f7dd89d50_MATT","finished","Finetuning","-","matthiasdelange","","2022-09-13T16:36:58.000Z","18994","0.1","0.3","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-1_SOLVER-MOMENTUM=0-3_SOLVER-NESTEROV=True/2022-09-13_09-34-51_UID819b79e9-031e-4695-9d61-1d8f7dd89d50"
    #     "Finetuning_2022-09-13_09-33-53_UIDa6fe4344-2d3e-4d95-9b80-bcc5005eba30_MATT","finished","Finetuning","-","matthiasdelange","","2022-09-13T16:34:43.000Z","19063","0.001","0","sgd","/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/final01_01_finetuning_sgd/../../../results/ego4d_action_recog/final01_01_finetuning_sgd/logs/GRID_SOLVER-BASE_LR=0-001_SOLVER-MOMENTUM=0-0_SOLVER-NESTEROV=True/2022-09-13_09-33-53_UIDa6fe4344-2d3e-4d95-9b80-bcc5005eba30"
    #     """)
    # )
    csv_path = os.path.join(csv_dirname, csv_filename)
    assert os.path.isfile(csv_path)

    main(csv_path)
