import pandas as pd
import wandb
import pprint
import os

api = wandb.Api()


def get_selected_group_names(selected_group_names_csv_path):
    """ Read from WandB downloaded CSV """
    selected_group_names_df = pd.read_csv(selected_group_names_csv_path)
    group_names = selected_group_names_df['Name'].to_list()
    return group_names


def get_group_run_iterator(project_name, group_name, finished_runs=True):
    """ Only get user-runs that finished processing stream (finished_run=True)."""
    group_runs = api.runs(project_name, {
        "$and": [
            {"group": group_name},
            {"summary_metrics.finished_run": finished_runs}
        ]
    })
    return group_runs
