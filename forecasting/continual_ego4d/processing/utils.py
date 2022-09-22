import pandas as pd
import wandb
import pprint
import os

api = wandb.Api()


def get_group_names_from_csv(selected_group_names_csv_path):
    """ Read from WandB downloaded CSV """
    selected_group_names_df = pd.read_csv(selected_group_names_csv_path)
    group_names = selected_group_names_df['Name'].to_list()
    return group_names


def get_group_run_iterator(project_name, group_name, finished_runs=True, run_filter=None):
    """ Only get user-runs that finished processing stream (finished_run=True)."""
    if run_filter is None:
        run_filter = {
            "$and": [
                {"group": group_name},
                {"summary_metrics.finished_run": finished_runs}
            ]
        }
    group_runs = api.runs(project_name, run_filter)
    return group_runs
