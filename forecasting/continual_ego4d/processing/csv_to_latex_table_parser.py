import os

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

local_csv_dirname = '/home/mattdl/projects/ContextualOracle_Matthias/adhoc_results'  # Move file in this dir
csv_dirname = local_csv_dirname


class LatexColumn:

    def __init__(
            self,
            pandas_col_mean_name,
            pandas_col_std_name=None,

            latex_col_header_name=None,  # Column header name
            latex_col_header_align='center',

            latex_mean_std_format=r"${}\pm{}$",
            round_digits=1,
            format_fn_overwrite=None
    ):
        self.pandas_col_mean_name = pandas_col_mean_name
        self.pandas_col_std_name = pandas_col_std_name

        self.latex_col_header_name = pandas_col_mean_name if latex_col_header_name is None else latex_col_header_name
        if latex_col_header_align == 'center':
            self.latex_col_header_name = rf"\multicolumn{{1}}{{c}}{{{self.latex_col_header_name}}}"  # Wrap

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

    New metric results: ['adhoc_users_aggregate/user_aggregate_count/test',

######### OAG ################
    # ABSOLUTE VALUES
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/mean',
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/SE',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/mean',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/SE',


 # PRETRAIN REFERENCE
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/PRETRAIN_abs/SE',

 # OAGs
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/mean', # Top1
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/SE',

 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/mean',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/SE',

 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/adhoc_AG/SE',

 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/adhoc_AG/mean', # Top5
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/adhoc_AG/SE',

 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/SE',
 ]

 ########### HAG ############

    # ABSOLUTE VALUES
    adhoc_users_aggregate/test_action_batch/loss/mean',
     'adhoc_users_aggregate/test_action_batch/loss/SE',
     'adhoc_users_aggregate/test_verb_batch/loss/mean',
     'adhoc_users_aggregate/test_verb_batch/loss/SE',
     'adhoc_users_aggregate/test_noun_batch/loss/mean',
     'adhoc_users_aggregate/test_noun_batch/loss/SE',
     'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
     'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
     'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
     'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',
     'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',
     'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
     'adhoc_users_aggregate/test_noun_batch/top5_acc/SE',

     # HAGs
     'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/mean', # TOP1
     'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/SE',

     'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/mean',
     'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/SE',

     'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/mean',
     'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/SE',

    'adhoc_users_aggregate/test_verb_batch/top5_acc/adhoc_hindsight_AG/mean', # TOP5
     'adhoc_users_aggregate/test_verb_batch/top5_acc/adhoc_hindsight_AG/SE',

     'adhoc_users_aggregate/test_noun_batch/top5_acc/adhoc_hindsight_AG/mean',
     'adhoc_users_aggregate/test_noun_batch/top5_acc/adhoc_hindsight_AG/SE']

    """
    csv_filename = "wandb_export_2022-10-10T10_21_32.541-07_00.csv"  # OAG -ACC results

    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001) & (orig_df['SOLVER.NESTEROV'] == True)]
    orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]  # TODO: Set to False or True to get both parts
    orig_df = orig_df.loc[(orig_df['SOLVER.MOMENTUM'] == 0)]  # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_header_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'SOLVER.MOMENTUM',
            latex_col_header_name=r"$\rho$",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_header_name=r"$\eta$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/mean',  # Top1
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final03_01_fixed_feats():
    """
    COLS:
    ['Name', 'SOLVER.BASE_LR',
    ...
    """
    csv_filename = "wandb_export_2022-10-09T17_12_44.119-07_00.csv"  # ACC-based
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

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
            latex_col_header_name=r"$\eta$",
            format_fn_overwrite=lambda x: x
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/mean',  # Top1
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

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
    csv_filename = "wandb_export_2022-10-09T16_49_03.226-07_00.csv"  # FINAL ACC HAG
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1
    final_excluded_colnames = ['Replay']

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['METHOD.REPLAY.STORAGE_POLICY'] == 'reservoir_stream') & (
    #         orig_df['METHOD.REPLAY.MEMORY_SIZE_SAMPLES'] == 64)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]
    orig_df.sort_values(inplace=True, axis=0, by=[
        'METHOD.REPLAY.STORAGE_POLICY', 'METHOD.REPLAY.MEMORY_SIZE_SAMPLES'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        LatexColumn(
            'METHOD.REPLAY.STORAGE_POLICY',
            latex_col_header_name=r"Replay",
            format_fn_overwrite=lambda x: x
        ),
        LatexColumn(
            'METHOD.REPLAY.MEMORY_SIZE_SAMPLES',
            latex_col_header_name=r"$|\mathcal{M}|$",
            format_fn_overwrite=lambda x: x
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/mean',  # Top1
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/mean',
        #     'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/mean',
        #     'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

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
        #     latex_col_header_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'SOLVER.MOMENTUM',
            latex_col_header_name=r"$\rho$",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_header_name=r"$\eta$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

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
        #     latex_col_header_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'SOLVER.MOMENTUM',
            latex_col_header_name=r"$\rho$",
            format_fn_overwrite=lambda x: x,
        ),

        # LatexColumn(
        #     'SOLVER.BASE_LR',
        #     latex_col_header_name=r"$\eta$",
        #     format_fn_overwrite=lambda x: "{:.1g}".format(x)
        # ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final07_01_sgd_multi_iter():
    """


    """
    csv_filename = "wandb_export_2022-10-10T11_54_32.411-07_00.csv"  # Full results all
    caption = "SGD grid over multiple iterations and learning rates."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR', 'TRAIN.INNER_LOOP_ITERS', ])
    # orig_df.sort_values(inplace=True, axis=0, by=['TRAIN.INNER_LOOP_ITERS', 'SOLVER.BASE_LR', ])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_header_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        LatexColumn(
            'TRAIN.INNER_LOOP_ITERS',
            latex_col_header_name=r"iters",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.BASE_LR',
            latex_col_header_name=r"$\eta$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/mean',  # Top1
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_LOSS_vs_ACC_vs_balancedLL_results_sgd_multi_iter():
    """
    Why different trends in delta? -> because log X - log Y = log X/Y
    Then, avging gives skewed results: \sum log X/Y

    To confirm this, we check the ABSOLUTE results for ACC and LOSS and see if they have the same trend.
    Then this would explain the difference with the delta-results.



    ######### OAG ################
    # ABSOLUTE VALUES ACC
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/mean',
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/SE',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/mean',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/SE',


    # ABSOLUTE VALUES LOSS
    'adhoc_users_aggregate/train_action_batch/loss_running_avg/mean',
    'adhoc_users_aggregate/train_action_batch/loss_running_avg/SE',
    'adhoc_users_aggregate/train_verb_batch/loss_running_avg/mean',
    'adhoc_users_aggregate/train_verb_batch/loss_running_avg/SE',
    'adhoc_users_aggregate/train_noun_batch/loss_running_avg/mean',
    'adhoc_users_aggregate/train_noun_batch/loss_running_avg/SE',


 # PRETRAIN REFERENCE
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/top5_acc_running_avg/PRETRAIN_abs/SE',
  'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/top5_acc_running_avg/PRETRAIN_abs/SE',

 # LL ABSOLUTE (balanced
'adhoc_users_aggregate/train_action_batch/balanced_LL/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/SE',

# LL OAG (delta)
 'adhoc_users_aggregate/train_action_batch/balanced_LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/adhoc_AG/SE']

    # Loss balanced (OAG)
     'adhoc_users_aggregate/train_action_batch/balanced_loss/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/SE',
 ########### HAG ############

    # ABSOLUTE VALUES
    adhoc_users_aggregate/test_action_batch/loss/mean',
     'adhoc_users_aggregate/test_action_batch/loss/SE',
     'adhoc_users_aggregate/test_verb_batch/loss/mean',
     'adhoc_users_aggregate/test_verb_batch/loss/SE',
     'adhoc_users_aggregate/test_noun_batch/loss/mean',
     'adhoc_users_aggregate/test_noun_batch/loss/SE',

     'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
     'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
          'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
     'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',

     'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
     'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',
     'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
     'adhoc_users_aggregate/test_noun_batch/top5_acc/SE',

     # LOSS:
     adhoc_users_aggregate/test_action_batch/loss/mean
     adhoc_users_aggregate/test_action_batch/loss/SE

     adhoc_users_aggregate/test_verb_batch/loss/mean
     adhoc_users_aggregate/test_verb_batch/loss/SE

     adhoc_users_aggregate/test_noun_batch/loss/mean
     adhoc_users_aggregate/test_noun_batch/loss/SE


     # TODO NEW ONES:
     ['adhoc_users_aggregate/user_aggregate_count/test',
 'adhoc_users_aggregate/train_verb_batch/LL/mean',
 'adhoc_users_aggregate/train_verb_batch/LL/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/SE',
 'adhoc_users_aggregate/train_action_batch/LL/mean',
 'adhoc_users_aggregate/train_action_batch/LL/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/SE',
 'adhoc_users_aggregate/train_noun_batch/LL/mean',
 'adhoc_users_aggregate/train_noun_batch/LL/SE',
 'adhoc_users_aggregate/train_verb_batch/LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_verb_batch/LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_verb_batch/LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_action_batch/LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_action_batch/LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_action_batch/LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/adhoc_AG/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_loss/adhoc_AG/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_verb_batch/balanced_LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_LL/adhoc_AG/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/adhoc_AG/mean',
 'adhoc_users_aggregate/train_action_batch/balanced_loss/adhoc_AG/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/balanced_loss/adhoc_AG/SE',
 'adhoc_users_aggregate/train_noun_batch/LL/PRETRAIN_abs/mean',
 'adhoc_users_aggregate/train_noun_batch/LL/PRETRAIN_abs/SE',
 'adhoc_users_aggregate/train_noun_batch/LL/adhoc_AG/mean',
 'adhoc_users_aggregate/train_noun_batch/LL/adhoc_AG/SE']


    """
    # csv_filename = "wandb_export_2022-10-10T20_28_09.281-07_00.csv"  # Full results all
    # csv_filename = "wandb_export_2022-10-11T21_06_08.210-07_00.csv"  # including balanced LL results
    # csv_filename = "wandb_export_2022-10-11T21_36_51.823-07_00.csv"  # including balanced loss results
    # csv_filename = "wandb_export_2022-10-12T13_43_44.601-07_00.csv"  # including balanced loss results
    csv_filename = "wandb_export_2022-10-12T14_35_29.228-07_00.csv"  # including balanced loss results
    caption = "SGD grid over multiple iterations and learning rates. " \
              "We report 3 metrics for action/verb/noun and balanced(macro)/unbalanced(micro) over actions/verbs/nouns."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1
    MODES = ['loss', 'loss_balanced', 'acc', 'acc_balanced', 'acc_pretrain', 'LL', 'LL_balanced', 'all_online']
    mode = MODES[-1]  # or acc
    col_headers_filter = None
    # col_headers_filter = {'action', 'iters', 'eta'}  # Show action cols only
    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.01)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR', 'TRAIN.INNER_LOOP_ITERS', ])
    # orig_df.sort_values(inplace=True, axis=0, by=['TRAIN.INNER_LOOP_ITERS', 'SOLVER.BASE_LR', ])

    # Place here in order you want the latex columns to be

    if mode == 'loss':
        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/loss_running_avg/mean',
                'adhoc_users_aggregate/train_action_batch/loss_running_avg/SE',
                latex_col_header_name=r"$\mathcal{L}^{micro}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/loss_running_avg/mean',
                'adhoc_users_aggregate/train_verb_batch/loss_running_avg/SE',
                latex_col_header_name=r"$\mathcal{L}^{micro}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/loss_running_avg/mean',
                'adhoc_users_aggregate/train_noun_batch/loss_running_avg/SE',
                latex_col_header_name=r"$\mathcal{L}^{micro}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # HISTORY AG
            # LatexColumn(
            #     'adhoc_users_aggregate/test_action_batch/loss/mean',
            #     'adhoc_users_aggregate/test_action_batch/loss/SE',
            #     latex_col_header_name=r"$\text{hindsight}-L_{\text{action}}$",
            #     round_digits=round_digits,
            # ),
            # LatexColumn(
            #     'adhoc_users_aggregate/test_verb_batch/loss/mean',
            #     'adhoc_users_aggregate/test_verb_batch/loss/SE',
            #     latex_col_header_name=r"$\text{hindsight}-L_{\text{verb}}$",
            #     round_digits=round_digits,
            # ),
            # LatexColumn(
            #     'adhoc_users_aggregate/test_noun_batch/loss/mean',
            #     'adhoc_users_aggregate/test_noun_batch/loss/SE',
            #     latex_col_header_name=r"$\text{hindsight}-L_{\text{noun}}$",
            #     round_digits=round_digits,
            # ),
        ]
    elif mode == 'loss_balanced':

        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/balanced_loss/mean',
                'adhoc_users_aggregate/train_action_batch/balanced_loss/SE',
                latex_col_header_name=r"$\mathcal{L}^{macro}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/balanced_loss/mean',
                'adhoc_users_aggregate/train_verb_batch/balanced_loss/SE',
                latex_col_header_name=r"$\mathcal{L}^{macro}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/balanced_loss/mean',
                'adhoc_users_aggregate/train_noun_batch/balanced_loss/SE',
                latex_col_header_name=r"$\mathcal{L}^{macro}_{\text{noun}}$",
                round_digits=round_digits,
            ),

        ]
    elif mode == 'LL_balanced':

        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/balanced_LL/mean',
                'adhoc_users_aggregate/train_action_batch/balanced_LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{macro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/balanced_LL/mean',
                'adhoc_users_aggregate/train_verb_batch/balanced_LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{macro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/balanced_LL/mean',
                'adhoc_users_aggregate/train_noun_batch/balanced_LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{macro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),
        ]
    elif mode == 'LL':

        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/LL/mean',
                'adhoc_users_aggregate/train_action_batch/LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{micro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/LL/mean',
                'adhoc_users_aggregate/train_verb_batch/LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{micro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/LL/mean',
                'adhoc_users_aggregate/train_noun_batch/LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{micro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),
        ]
    elif mode == 'acc_pretrain':

        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
                'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
                latex_col_header_name=r"$ACC_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
                'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
                latex_col_header_name=r"$ACC_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/PRETRAIN_abs/mean',
                'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/PRETRAIN_abs/SE',
                latex_col_header_name=r"$ACC_{\text{noun}}$",
                round_digits=round_digits,
            ),
        ]
    elif mode == 'acc':

        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
                'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{micro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
                'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{micro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
                'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{micro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # HISTORY AG
            # LatexColumn(
            #     'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
            #     'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
            #     latex_col_header_name=r"$\text{hindsight-ACC}_{\text{action}}$",
            #     round_digits=round_digits,
            # ),
            # LatexColumn(
            #     'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
            #     'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
            #     latex_col_header_name=r"$\text{hindsight-ACC}_{\text{verb}}$",
            #     round_digits=round_digits,
            # ),
            # LatexColumn(
            #     'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
            #     'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',
            #     latex_col_header_name=r"$\text{hindsight-ACC}_{\text{noun}}$",
            #     round_digits=round_digits,
            # ),
        ]
    elif mode == 'acc_balanced':

        ordered_cols = [

            # HPARAMS COL
            # LatexColumn(
            #     'SOLVER.MOMENTUM',
            #     'SOLVER.BASE_LR',
            #     latex_col_header_name=r"$\rho (\eta)$",
            #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
            # ),

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ONLINE AG
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/mean',
                'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{macro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/mean',
                'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{macro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/mean',
                'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{macro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),
        ]
    elif mode == 'all_online':

        ordered_cols = [

            # HPARAMS COL

            LatexColumn(
                'TRAIN.INNER_LOOP_ITERS',
                latex_col_header_name=r"iters",
                format_fn_overwrite=lambda x: x,
            ),

            LatexColumn(
                'SOLVER.BASE_LR',
                latex_col_header_name=r"$\eta$",
                format_fn_overwrite=lambda x: "{:.1g}".format(x)
            ),

            # ACC
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
                'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{micro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
                'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{micro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
                'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{micro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # ACC-Balanced
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/mean',
                'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{macro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/mean',
                'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{macro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/mean',
                'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/SE',
                latex_col_header_name=r"$\text{ACC}^{\text{macro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # C
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/LL/mean',
                'adhoc_users_aggregate/train_action_batch/LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{micro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/LL/mean',
                'adhoc_users_aggregate/train_verb_batch/LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{micro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/LL/mean',
                'adhoc_users_aggregate/train_noun_batch/LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{micro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # C-balanced
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/balanced_LL/mean',
                'adhoc_users_aggregate/train_action_batch/balanced_LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{macro}}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/balanced_LL/mean',
                'adhoc_users_aggregate/train_verb_batch/balanced_LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{macro}}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/balanced_LL/mean',
                'adhoc_users_aggregate/train_noun_batch/balanced_LL/SE',
                latex_col_header_name=r"$\mathcal{C}^{\text{macro}}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # LOSS
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/loss_running_avg/mean',
                'adhoc_users_aggregate/train_action_batch/loss_running_avg/SE',
                latex_col_header_name=r"$\mathcal{L}^{micro}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/loss_running_avg/mean',
                'adhoc_users_aggregate/train_verb_batch/loss_running_avg/SE',
                latex_col_header_name=r"$\mathcal{L}^{micro}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/loss_running_avg/mean',
                'adhoc_users_aggregate/train_noun_batch/loss_running_avg/SE',
                latex_col_header_name=r"$\mathcal{L}^{micro}_{\text{noun}}$",
                round_digits=round_digits,
            ),

            # LOSS-balanced
            LatexColumn(
                'adhoc_users_aggregate/train_action_batch/balanced_loss/mean',
                'adhoc_users_aggregate/train_action_batch/balanced_loss/SE',
                latex_col_header_name=r"$\mathcal{L}^{macro}_{\text{action}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_verb_batch/balanced_loss/mean',
                'adhoc_users_aggregate/train_verb_batch/balanced_loss/SE',
                latex_col_header_name=r"$\mathcal{L}^{macro}_{\text{verb}}$",
                round_digits=round_digits,
            ),
            LatexColumn(
                'adhoc_users_aggregate/train_noun_batch/balanced_loss/mean',
                'adhoc_users_aggregate/train_noun_batch/balanced_loss/SE',
                latex_col_header_name=r"$\mathcal{L}^{macro}_{\text{noun}}$",
                round_digits=round_digits,
            ),
        ]

    latex_df = pd.DataFrame()
    nb_cols = len(ordered_cols)
    col_format = 'r' * nb_cols

    for col in ordered_cols:
        if col_headers_filter is not None and \
                not any(f in col.latex_col_header_name for f in col_headers_filter):  # Skip based on header filter
            continue

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A', column_format=col_format), end='')
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
            latex_col_header_name=r"$\eta_{feat}$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        LatexColumn(
            'SOLVER.CLASSIFIER_LR',
            latex_col_header_name=r"$\eta_{classifier}$",
            format_fn_overwrite=lambda x: "{:.1g}".format(x)
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_action_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_verb_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/mean',
            'adhoc_users_aggregate/train_noun_batch/AG_cumul/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()
    nb_cols = len(ordered_cols)
    col_format = 'r' * nb_cols

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A', column_format=col_format), end='')
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
            latex_col_header_name=r"Path",
            format_fn_overwrite=lambda x: x,
        ),

        # LOSSES
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/loss/mean',
            'adhoc_users_aggregate/test_action_batch/loss/SE',
            latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/loss/mean',
            'adhoc_users_aggregate/test_verb_batch/loss/SE',
            latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/loss/mean',
            'adhoc_users_aggregate/test_noun_batch/loss/SE',
            latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # ACTION ACC
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, noun}}$",
            round_digits=round_digits,
        ),

        # TOP-5 acc
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-5, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/top5_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-5, noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final14_01_and_02_replay_classifier_retrain():
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
    csv_filename = "wandb_export_2022-10-05T18_47_30.747-07_00.csv"  # TEST USERS: Full results all
    caption = "Replay vs SGD classifier retrain on final fixed feature extractor."
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
            latex_col_header_name=r"Path",
            format_fn_overwrite=lambda x: x,
        ),

        # LOSSES
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/loss/mean',
            'adhoc_users_aggregate/test_action_batch/loss/SE',
            latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/loss/mean',
            'adhoc_users_aggregate/test_verb_batch/loss/SE',
            latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/loss/mean',
            'adhoc_users_aggregate/test_noun_batch/loss/SE',
            latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # ACTION ACC
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, noun}}$",
            round_digits=round_digits,
        ),

        # TOP-5 acc
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-5, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/top5_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-5, noun}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    with pd.option_context("max_colwidth", 1000):  # No truncating of strings
        print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()




def parse_final16_01_label_window_predictor():
    """
    """
    csv_filename = "wandb_export_2022-10-12T21_37_29.440-07_00.csv"  # TEST USERS: Full results all
    caption = "Label window predictor naive baseline."
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
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{action}}$",
            round_digits=round_digits,
        ),
    ]

    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table(caption)
    with pd.option_context("max_colwidth", 1000):  # No truncating of strings
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
    parse_final16_01_label_window_predictor()
