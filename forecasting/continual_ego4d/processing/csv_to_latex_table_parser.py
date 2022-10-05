import os
import pandas as pd

local_csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
csv_dirname = local_csv_dirname


class LatexColumn:

    def __init__(
            self,
            pandas_col_mean_name,
            pandas_col_std_name=None,
            latex_col_report_name=None,
            latex_mean_std_format=r"${}\pm{}$",
            round_digits=1,
            format_fn_overwrite=None
    ):
        self.pandas_col_mean_name = pandas_col_mean_name
        self.pandas_col_std_name = pandas_col_std_name

        self.latex_col_report_name = latex_col_report_name
        if self.latex_col_report_name is None:
            self.latex_col_report_name = pandas_col_mean_name  # Use pandas name

        self.latex_mean_std_format = latex_mean_std_format
        self.round_digits = round_digits
        self.format_fn_overwrite = format_fn_overwrite

    def format_fn(self, col_vals):
        if self.format_fn_overwrite is not None:
            return self.format_fn_overwrite(col_vals)
        assert isinstance(col_vals, (tuple, list, pd.Series)), f"Type:{type(col_vals)}"

        if len(col_vals) == 2:
            return self.latex_mean_std_format.format(
                self.rounding_fn(col_vals[0]),
                self.rounding_fn(col_vals[1]),
            )
        else:
            return self.rounding_fn(col_vals[0])

    def rounding_fn(self, val):
        return round(val, ndigits=self.round_digits)


def parse_final01_01_momentum_table():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR', 'SOLVER.NESTEROV', 'SOLVER.MOMENTUM',
           'adhoc_users_aggregate/user_aggregate_count',

       'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
       'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',

       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean'],
    """
    # csv_filename = "wandb_export_2022-09-21T11_36_57.254-07_00.csv"
    # csv_filename = "wandb_export_2022-09-21T17_57_58.323-07_00.csv"  # Full results grid only
    csv_filename = "wandb_export_2022-09-24T16_35_35.285-07_00.csv"  # Full results all

    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 2

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001) & (orig_df['SOLVER.NESTEROV'] == True)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df = orig_df.loc[(orig_df['SOLVER.MOMENTUM'] == 0)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_report_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'SOLVER.MOMENTUM',
            latex_col_report_name=r"$\rho$",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_report_name=r"$\eta",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final03_01_fixed_feats():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR',
    ...
    """
    csv_filename = "wandb_export_2022-09-21T15_50_43.550-07_00.csv"
    csv_path = os.path.join(csv_dirname, csv_filename)

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.01) & (orig_df['SOLVER.NESTEROV'] == False)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_report_name=r"$\eta",
            format_fn_overwrite=lambda x: x
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$"
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$"
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$"
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$"
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$"
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$"
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final02_01_replay():
    """
    COLS:
Index(['Name', 'METHOD.REPLAY.STORAGE_POLICY',
       'METHOD.REPLAY.MEMORY_SIZE_SAMPLES', 'SOLVER.BASE_LR',
       'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
       'METHOD.METHOD_NAME',
    ...
    """
    # csv_filename = "wandb_export_2022-09-21T16_56_15.802-07_00.csv" # Missing hindsight metrics
    csv_filename = "wandb_export_2022-09-22T15_06_05.542-07_00.csv"
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 2
    final_excluded_colnames = ['Replay']

    orig_df = pd.read_csv(csv_path)

    # FILTER
    orig_df = orig_df.loc[(orig_df['METHOD.REPLAY.STORAGE_POLICY'] == 'reservoir_stream') & (
            orig_df['METHOD.REPLAY.MEMORY_SIZE_SAMPLES'] == 64)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]
    orig_df.sort_values(inplace=True, axis=0, by=[
        'METHOD.REPLAY.STORAGE_POLICY', 'METHOD.REPLAY.MEMORY_SIZE_SAMPLES'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        LatexColumn(
            'METHOD.REPLAY.STORAGE_POLICY',
            latex_col_report_name=r"Replay",
            format_fn_overwrite=lambda x: x
        ),
        LatexColumn(
            'METHOD.REPLAY.MEMORY_SIZE_SAMPLES',
            latex_col_report_name=r"$|\mathcal{M}|$",
            format_fn_overwrite=lambda x: x
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()

    print("\n\nAgain without excluded columns (printed above for sanity check")
    print_begin_table()
    print(latex_df.drop(final_excluded_colnames, axis=1).to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_exp04_01_momentum_video_reset_table():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR', 'SOLVER.NESTEROV', 'SOLVER.MOMENTUM',
           'adhoc_users_aggregate/user_aggregate_count',

       'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
       'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',

       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean'],
    """
    # csv_filename = "wandb_export_2022-09-26T09_46_26.767-07_00.csv"  # Full results all
    csv_filename = "wandb_export_2022-09-27T09_40_46.516-07_00.csv"  # Including lr 0.001

    csv_path = os.path.join(csv_dirname, csv_filename)
    orig_df = pd.read_csv(csv_path)
    round_digits = 2

    # FILTER
    orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == False)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_report_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'SOLVER.MOMENTUM',
            latex_col_report_name=r"$\rho$",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_report_name=r"$\eta",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final05_01_repaly_with_momentum_table():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR', 'SOLVER.NESTEROV', 'SOLVER.MOMENTUM',
           'adhoc_users_aggregate/user_aggregate_count',

       'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
       'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',

       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean'],
    """
    # csv_filename = "wandb_export_2022-09-21T11_36_57.254-07_00.csv"
    # csv_filename = "wandb_export_2022-09-21T17_57_58.323-07_00.csv"  # Full results grid only
    csv_filename = "wandb_export_2022-09-27T10_27_31.671-07_00.csv"  # Full results all
    caption = "Replay for different Nesterov momentum strengths with Reservoir Stream storage strategy and memory size 64, lr 0.01."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 2

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.01) & (orig_df['SOLVER.NESTEROV'] == True)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_report_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'SOLVER.MOMENTUM',
            latex_col_report_name=r"$\rho$",
            format_fn_overwrite=lambda x: x,
        ),

        # LatexColumn(
        #     'SOLVER.BASE_LR',
        #     latex_col_report_name=r"$\eta",
        #     format_fn_overwrite=lambda x: "{:.1g}".format(x)
        # ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final07_01_sgd_multi_iter():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR', 'SOLVER.NESTEROV', 'SOLVER.MOMENTUM',
           'adhoc_users_aggregate/user_aggregate_count',

       'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
       'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',

       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean'],
    """
    csv_filename = "wandb_export_2022-09-28T20_09_27.962-07_00.csv"  # Full results all
    caption = "SGD grid over multiple iterations and learning rates."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    orig_df.sort_values(inplace=True, axis=0, by=['TRAIN.INNER_LOOP_ITERS', 'SOLVER.BASE_LR', ])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_report_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'TRAIN.INNER_LOOP_ITERS',
            latex_col_report_name=r"iters",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_report_name=r"$\eta",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
        #     'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
        #     latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
        #     'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
        #     latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
        #     'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
        #     latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final11_01_async_lr():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR', 'SOLVER.NESTEROV', 'SOLVER.MOMENTUM',
           'adhoc_users_aggregate/user_aggregate_count',

       'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
       'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',

       'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
       'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',

       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',

       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
       'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean'],
    """
    csv_filename = "wandb_export_2022-09-30T19_50_49.808-07_00.csv"  # Full results all
    caption = ""
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR', 'SOLVER.CLASSIFIER_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_report_name=r"$\eta_{feat}$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        LatexColumn(
            'SOLVER.CLASSIFIER_LR',
            latex_col_report_name=r"$\eta_{classifier}$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_report_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final00_01_and_02_pretrain_performance():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR', 'SOLVER.NESTEROV', 'SOLVER.MOMENTUM',
           'adhoc_users_aggregate/user_aggregate_count',


    New metric results: [
    'adhoc_users_aggregate/user_aggregate_count/test',

    # LOSSES
     'adhoc_users_aggregate/test_action_batch/loss/mean',
     'adhoc_users_aggregate/test_action_batch/loss/SE',

     'adhoc_users_aggregate/test_verb_batch/loss/mean',
     'adhoc_users_aggregate/test_verb_batch/loss/SE',

     'adhoc_users_aggregate/test_noun_batch/loss/mean',
     'adhoc_users_aggregate/test_noun_batch/loss/SE',


     # ACTION ACC
     'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_action_batch/top1_acc/SE',

     # VERB ACC
     'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',

     'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
     'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',

     # NOUN ACC
     'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',

     'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
     'adhoc_users_aggregate/test_noun_batch/top5_acc/SE'
     ]

    """
    # csv_filename = "wandb_export_2022-10-04T17_12_15.398-07_00.csv"  # TRAIN USERS: Full results all
    csv_filename = "wandb_export_2022-10-04T18_26_17.293-07_00.csv"  # TEST USERS: Full results all
    caption = "Our pretrained vs original ego4d pretrained model."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR', 'SOLVER.CLASSIFIER_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'CHECKPOINT_FILE_PATH',
            latex_col_report_name=r"Path",
            format_fn_overwrite=lambda x: x,
        ),

        # LOSSES
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/loss/mean',
            'adhoc_users_aggregate/test_action_batch/loss/SE',
            latex_col_report_name=r"$\overline{\mathcal{L}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/loss/mean',
            'adhoc_users_aggregate/test_verb_batch/loss/SE',
            latex_col_report_name=r"$\overline{\mathcal{L}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/loss/mean',
            'adhoc_users_aggregate/test_noun_batch/loss/SE',
            latex_col_report_name=r"$\overline{\mathcal{L}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # ACTION ACC
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
            latex_col_report_name=r"$\overline{\text{ACC}}_{\text{top-1, action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
            latex_col_report_name=r"$\overline{\text{ACC}}_{\text{top-1, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',
            latex_col_report_name=r"$\overline{\text{ACC}}_{\text{top-1, noun}}$",
            round_digits=round_digits,
        ),

        # TOP-5 acc
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',
            latex_col_report_name=r"$\overline{\text{ACC}}_{\text{top-5, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/top5_acc/SE',
            latex_col_report_name=r"$\overline{\text{ACC}}_{\text{top-5, noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_report_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_report_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def print_begin_table(caption=""):
    print(r"\begin{table}[]")
    print(rf"\caption{{{caption}}}")
    print(r"\label{tab:HOLDER}")
    print(r"\centering")
    print(r"\resizebox{\linewidth}{!}{%")


def print_end_table():
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    parse_final00_01_and_02_pretrain_performance()