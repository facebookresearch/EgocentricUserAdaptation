"""
RUN REMOTE ONLY!
Iterate runs and if csv dump exists, the run is finished, add this to the wandb entry.
"""

import pandas as pd
import wandb
import pprint
import os
from continual_ego4d.processing.utils import get_group_names_from_csv, get_group_run_iterator

api = wandb.Api()

# Adapt settings
UPDATE_ALL_WANDB_RUNS = False
REMOTE = False

csv_filename = 'wandb_export_2022-09-19T12_08_04.153-07_00.csv'  # TODO copy file here and past name here
single_group_name = None
single_group_name = "Finetuning_2022-10-14_13-11-14_UIDac6a2798-7fb0-4e9c-9896-c6c54de5237c"

# Fixed settings
PROJECT_NAME = "matthiasdelange/ContinualUserAdaptation"
remote_csv_dirname = '/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
local_csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
csv_dirname = remote_csv_dirname if REMOTE else local_csv_dirname


def main():
    if not UPDATE_ALL_WANDB_RUNS:
        if single_group_name is not None:
            selected_group_names: list[str] = [single_group_name]
        else:
            print("Updating only groups in csv")
            selected_group_names_csv_path = os.path.join(csv_dirname, csv_filename)
            assert os.path.isfile(selected_group_names_csv_path)
            selected_group_names: list[str] = get_group_names_from_csv(selected_group_names_csv_path)
        print(f"Group names={pprint.pformat(selected_group_names)}")

        # Iterate groups (a collective of independent user-runs)
        finished_user_ids = []
        for group_name in selected_group_names:
            for idx, user_run in enumerate(get_group_run_iterator(PROJECT_NAME, group_name, finished_runs_only=False)): # TODO be carefull will indicate duplicate runs also as finished!
                finished_user_id = update_finished_run_entry(idx, user_run)
                if finished_user_id is not None:
                    finished_user_ids.append(finished_user_id)
    else:
        print("Updating all runs")
        # Update all group entries:
        for idx, user_run in enumerate(api.runs(PROJECT_NAME)):
            update_finished_run_entry(idx, user_run)

    print(f"finished_user_ids={finished_user_ids}")


def update_finished_run_entry(entry_idx, user_run):
    user_id = user_run.config['DATA.COMPUTED_USER_ID']
    parent_outdir = user_run.config['OUTPUT_DIR']
    user_dump_path = os.path.join(parent_outdir, 'user_logs', f"user_{user_id}", 'stream_info_dump.pth')

    if user_id == '436':
        import pdb;
        pdb.set_trace()

    if os.path.isfile(user_dump_path):
        user_run.summary['finished_run'] = True
        user_run.summary.update()  # UPLOAD
        print(f"{entry_idx}: Updated user {user_id} as finished: {user_dump_path}")
        return user_id
    else:
        print(f"{entry_idx}: SKIP: non-existing: {user_dump_path}")
        return None


if __name__ == "__main__":
    main()
