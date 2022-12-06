"""
Takes WandB table CSV export (e.g. download from the GUI), and transfers to formatted Latex table.
All the required metrics per table are enlisted in this script, and should be present in the CSV.
"""
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


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


def df_to_latex(ordered_cols: list[LatexColumn], df: pd.DataFrame):
    """
    Convert the dataframe to a printed latex table in stdout, with columns based on LatexColumns in 'ordered_cols'.
    """
    latex_df = pd.DataFrame()

    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()

    # Get lists for plots
    for col in ordered_cols:
        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)


def parse_SGD_momentum(csv_path, round_digits=1):
    """
    Displays OAG and HAG for action,verb,noun. Hindsight results (HAG) are obtained in a following run.
    """
    orig_df = pd.read_csv(csv_path)

    # FILTER to get results
    orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.MOMENTUM'] == 0)]
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [
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

        # ONLINE AG (OAG)
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG (HAG)
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    df_to_latex(ordered_cols, orig_df)


def parse_SGD_user_feature_adaptation(csv_path, round_digits=1):
    orig_df = pd.read_csv(csv_path)

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
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    df_to_latex(ordered_cols, orig_df)


def parse_replay(csv_path, round_digits=1):
    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['METHOD.REPLAY.STORAGE_POLICY'] == 'reservoir') & (
    #         orig_df['METHOD.REPLAY.MEMORY_SIZE_SAMPLES'] == 64)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]
    orig_df.sort_values(inplace=True, axis=0, by=[
        'METHOD.REPLAY.STORAGE_POLICY', 'METHOD.REPLAY.MEMORY_SIZE_SAMPLES'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        LatexColumn(
            'METHOD.REPLAY.STORAGE_POLICY',
            latex_col_header_name=r"Storage Policy",
            format_fn_overwrite=lambda x: x
        ),
        LatexColumn(
            'METHOD.REPLAY.MEMORY_SIZE_SAMPLES',
            latex_col_header_name=r"$M$",
            format_fn_overwrite=lambda x: x
        ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),
    ]

    df_to_latex(ordered_cols, orig_df)


def parse_final07_01_sgd_multi_iter():
    """
     'adhoc_users_aggregate/train_action_POST_UPDATE_BATCH/loss_running_avg/mean',
     'adhoc_users_aggregate/train_action_POST_UPDATE_BATCH/loss_running_avg/SE',

     'adhoc_users_aggregate/train_noun_POST_UPDATE_BATCH/top1_acc_balanced_running_avg/mean',
     'adhoc_users_aggregate/train_noun_POST_UPDATE_BATCH/top1_acc_balanced_running_avg/SE'




     # DECORRELATED ACC, window-size 4
      'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_keep/SE',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_total/mean',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_total/SE',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/percentage_kept/mean',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/percentage_kept/SE',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/num_samples_keep/SE',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/num_samples_total/mean',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/num_samples_total/SE',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/percentage_kept/mean',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/percentage_kept/SE',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/num_samples_keep/SE',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/num_samples_total/mean',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/num_samples_total/SE',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/percentage_kept/mean',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/percentage_kept/SE']

     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/mean',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/SE',

     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/mean',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/SE',

     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/mean',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/SE',

     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/PRETRAIN_abs/mean',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/PRETRAIN_abs/SE',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/mean',
     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/SE',

     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/PRETRAIN_abs/mean',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/PRETRAIN_abs/SE',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/adhoc_AG/mean',
     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/adhoc_AG/SE',

     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/PRETRAIN_abs/mean',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/PRETRAIN_abs/SE',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/adhoc_AG/mean',
     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/adhoc_AG/SE']


    """
    # csv_filename = "wandb_export_2022-10-14T15_52_32.086-07_00.csv"  # Full results all
    # csv_filename = "wandb_export_2022-10-21T16_40_32.034-07_00.csv"  # Full results all
    # csv_filename = "wandb_export_2022-10-21T16_57_03.896-07_00.csv"  # FIX OAG/HAG results!

    # Corr/decorr: in previous batch
    # csv_filename = "wandb_export_2022-10-25T10_31_19.778-07_00.csv"  # INCL corr/decorr results
    # csv_filename = "wandb_export_2022-10-25T10_58_55.415-07_00.csv" # SGD only

    # Corr/decorr: prev sample only
    # csv_filename = "wandb_export_2022-10-25T16_04_34.888-07_00.csv"  # Only preceding sample
    # csv_filename = "wandb_export_2022-10-25T16_10_07.514-07_00.csv"  # SGD

    # Corr/decorr: ALL
    csv_filename = "wandb_export_2022-10-26T15_27_02.501-07_00.csv"

    caption = "SGD grid over multiple iterations and learning rates."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 2
    NB_USERS = 10

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.01)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['TRAIN.INNER_LOOP_ITERS', ])
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

        # LatexColumn(
        #     'SOLVER.BASE_LR',
        #     latex_col_header_name=r"$\eta$",
        #     format_fn_overwrite=lambda x: "{:.1g}".format(x)
        # ),

        # ONLINE AG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=1,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=1,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # Training loss
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_POST_UPDATE_BATCH/loss_running_avg/mean',
        #     'adhoc_users_aggregate/train_action_POST_UPDATE_BATCH/loss_running_avg/SE',
        #     latex_col_header_name=r"$\mathcal{L}^{post}}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),

        # Decorrelated accuracy
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}^\text{decor.}_{\text{action}}$",
            round_digits=1,
        ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/mean',
        #     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/SE',
        #     latex_col_header_name=r"$\text{ACC}^{decor.}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/mean',
        #     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/SE',
        #     latex_col_header_name=r"$\text{ACC}^{decor.}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # Correlated accuracy
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}^\text{cor.}_{\text{action}}$",
            round_digits=1,
        ),

        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
        #     latex_col_header_name=r"nb samples (decorrelated)",
        #     format_fn_overwrite=lambda x: x * NB_USERS,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_keep/mean',
        #     latex_col_header_name=r"nb samples (correlated)",
        #     format_fn_overwrite=lambda x: x * NB_USERS,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_total/mean',
        #     latex_col_header_name=r"nb samples total)",
        #     format_fn_overwrite=lambda x: x * NB_USERS,
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

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final00_01_and_02_pretrain_performance():
    """
    Previously final00/eval00 -> Now abs00 (where we treat pretrain model like a fixed model method), collecting
    result over time.


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

        # # LOSSES
        # LatexColumn(
        #     'adhoc_users_aggregate/test_action_batch/loss/mean',
        #     'adhoc_users_aggregate/test_action_batch/loss/SE',
        #     latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_verb_batch/loss/mean',
        #     'adhoc_users_aggregate/test_verb_batch/loss/SE',
        #     latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_noun_batch/loss/mean',
        #     'adhoc_users_aggregate/test_noun_batch/loss/SE',
        #     latex_col_header_name=r"$\overline{\mathcal{L}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),
        #
        # # ACTION ACC
        # LatexColumn(
        #     'adhoc_users_aggregate/test_action_batch/top1_acc/mean',
        #     'adhoc_users_aggregate/test_action_batch/top1_acc/SE',
        #     latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_verb_batch/top1_acc/mean',
        #     'adhoc_users_aggregate/test_verb_batch/top1_acc/SE',
        #     latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_noun_batch/top1_acc/mean',
        #     'adhoc_users_aggregate/test_noun_batch/top1_acc/SE',
        #     latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-1, noun}}$",
        #     round_digits=round_digits,
        # ),

        # BALANCED ACTION ACC
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, noun}}$",
            round_digits=round_digits,
        ),

        # TOP-5 acc
        # LatexColumn(
        #     'adhoc_users_aggregate/test_verb_batch/top5_acc/mean',
        #     'adhoc_users_aggregate/test_verb_batch/top5_acc/SE',
        #     latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-5, verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_noun_batch/top5_acc/mean',
        #     'adhoc_users_aggregate/test_noun_batch/top5_acc/SE',
        #     latex_col_header_name=r"$\overline{\text{ACC}}_{\text{top-5, noun}}$",
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

    print_begin_table(caption)
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_abs00_01_and_02_pretrain_performance():
    """
    Previously final00/eval00 -> Now abs00 (where we treat pretrain model like a fixed model method), collecting
    result over time.


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
    # ABS00_01
    # csv_filename = "wandb_export_2022-10-14T18_16_04.607-07_00.csv"  # TRAIN USERS: Full results all
    # csv_filename = "wandb_export_2022-10-14T18_23_15.613-07_00.csv"  # TEST USERS: Full results all

    # ABS00_02
    caption = "Our pretrained vs original ego4d pretrained model."
    # csv_filename = "wandb_export_2022-10-20T15_09_44.872-07_00.csv"  # TRAIN USERS
    csv_filename = "wandb_export_2022-10-20T15_13_33.859-07_00.csv"  # TEST USERS
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

        # LatexColumn(
        #     'CHECKPOINT_FILE_PATH',
        #     latex_col_header_name=r"Path",
        #     format_fn_overwrite=lambda x: x,
        # ),

        # BALANCED ACTION ACC
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, noun}}$",
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




def parse_LWP_batched(csv_path, round_digits=1):
    orig_df = pd.read_csv(csv_path)
    orig_df.sort_values(inplace=True, axis=0, by=["ANALYZE_STREAM.WINDOW_SIZE_SAMPLES"])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'ANALYZE_STREAM.WINDOW_SIZE_SAMPLES',
            latex_col_header_name=r"window size",
            format_fn_overwrite=lambda x: x,
        ),

        # Balanced OAG:
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}_{\text{noun}}$",
            round_digits=round_digits,
        ),

    ]

    df_to_latex(ordered_cols, orig_df)


def parse_LWP_batched_hindsight(csv_path, round_digits=2):
    """
    Get OAG of the HindsightLabelWindowPredictor, which equals the HAG of the LabelWindowPredictor.
    This is because the running avg over the stream is calculated, in comparison to the pretrain model.
    """
    orig_df = pd.read_csv(csv_path)

    # FILTER
    orig_df.sort_values(inplace=True, axis=0, by=["ANALYZE_STREAM.WINDOW_SIZE_SAMPLES"])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'ANALYZE_STREAM.WINDOW_SIZE_SAMPLES',
            latex_col_header_name=r"window size",
            format_fn_overwrite=lambda x: x,
        ),

        # OAG of the HindsightLWP = HAG
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{HAG}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{HAG}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{HAG}_{\text{noun}}$",
            round_digits=round_digits,
        ),

    ]

    df_to_latex(ordered_cols, orig_df)


def parse_LWP_non_stationarity():
    """
    Also shows the correlated/decorrelated ACC results
    """
    csv_filename = "wandb_export_2022-10-26T20_07_50.954-07_00.csv"  # TRAIN: Batched
    # csv_filename = "wandb_export_2022-10-27T10_26_32.956-07_00.csv"  # TRAIN (decor/cor): Batched

    # csv_filename = "wandb_export_2022-10-27T10_11_48.448-07_00.csv"  # TEST: batched
    caption = "Label window predictor naive baseline."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    # orig_df.sort_values(inplace=True, axis=0, by=["ANALYZE_STREAM.WINDOW_SIZE_SAMPLES"])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # LatexColumn(
        #     'ANALYZE_STREAM.WINDOW_SIZE_SAMPLES',
        #     latex_col_header_name=r"window size",
        #     format_fn_overwrite=lambda x: x,
        # ),

        # BALANCED
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/mean',
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/SE',
        #     latex_col_header_name=r"$\text{ACC}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/mean',
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/SE',
        #     latex_col_header_name=r"$\text{ACC}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/mean',
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/SE',
        #     latex_col_header_name=r"$\text{ACC}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # UNBALANCED
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
        #     latex_col_header_name=r"$\text{ACC}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
        #     latex_col_header_name=r"$\text{ACC}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
        #     latex_col_header_name=r"$\text{ACC}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # OAG:
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # Decorrelated accuracy (OAG)
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}^\text{decor.}_{\text{action}}$",
        #     round_digits=1,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/mean',
        #     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/SE',
        #     latex_col_header_name=r"$\text{ACC}^{decor.}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/mean',
        #     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/SE',
        #     latex_col_header_name=r"$\text{ACC}^{decor.}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # Correlated accuracy (OAG)
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}^\text{cor.}_{\text{action}}$",
        #     round_digits=1,
        # ),

        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
        #     latex_col_header_name=r"nb samples (decorrelated)",
        #     format_fn_overwrite=lambda x: x * NB_USERS,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_keep/mean',
        #     latex_col_header_name=r"nb samples (correlated)",
        #     format_fn_overwrite=lambda x: x * NB_USERS,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_total/mean',
        #     latex_col_header_name=r"nb samples total)",
        #     format_fn_overwrite=lambda x: x * NB_USERS,
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
    csv_path = '/path/to/your/csvs/wandb_export_2022-10-20T16_04_10.497-07_00.csv'

    # Pick method for parsing
    mode = 'SGD_momentum'

    if mode == 'SGD_momentum':
        parse_SGD_momentum(csv_path)

    elif mode == 'SGD_user_feature_adaptation':
        parse_SGD_user_feature_adaptation(csv_path)

    elif mode == 'replay':
        parse_replay(csv_path)

    elif mode == 'LWP':
        parse_LWP_batched(csv_path)

    elif mode == 'LWP_hindsight':
        parse_LWP_batched_hindsight(csv_path)  # Hindsight performance

    else:
        raise ValueError()
