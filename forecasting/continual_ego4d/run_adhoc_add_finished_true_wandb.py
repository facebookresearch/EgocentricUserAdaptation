"""
RUN REMOTE ONLY!
Iterate runs and if csv dump exists, the run is finished, add this to the wandb entry.
"""

import pandas as pd
import wandb
import pprint
import os

api = wandb.Api()

PROJECT_NAME = "matthiasdelange/ContinualUserAdaptation"


def main():
    # Update all group entries:
    for idx, user_run in enumerate(api.runs(PROJECT_NAME)):
        user_id = user_run.config['DATA.COMPUTED_USER_ID']
        parent_outdir = user_run.config['OUTPUT_DIR']
        user_dump_path = os.path.join(parent_outdir, 'user_logs', f"user_{user_id}", 'stream_info_dump.pth')

        if os.path.isfile(user_dump_path):
            user_run.summary['finished_run'] = True
            user_run.summary.update()  # UPLOAD
            print(f"{idx}: Updated user {user_id} as finished: {user_dump_path}")
        else:
            print(f"{idx}: SKIP: non-existing: {user_dump_path}")


if __name__ == "__main__":
    main()
