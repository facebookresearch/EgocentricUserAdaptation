"""
Takes WandB table CSV export (e.g. download from the GUI), and transfers to formatted Latex table.
All the required metrics per table are enlisted in this script, and should be present in the CSV (can be selected in WandB).
"""
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


def main(
        mode: str = 'SGD_momentum',
        csv_path: str = '/path/to/your/csvs/wandb_export_2022-10-20T16_04_10.497-07_00.csv',
):
    """
    Define which table to parse with the 'mode', and add the path
    to your local CSV file downloaded from your WandB results.
    """

    if mode == 'SGD_momentum':
        parse_SGD_momentum(csv_path)

    elif mode == 'SGD_user_feature_adaptation':
        parse_SGD_user_feature_adaptation(csv_path)

    elif mode == 'SGD_mult_iter':
        parse_SGD_multiple_iterations(csv_path)

    elif mode == 'replay':
        parse_replay(csv_path)

    elif mode == 'pretrain_user_performance':
        parse_pretrain_performance_on_user_streams(csv_path)

    elif mode == 'LWP':
        parse_LWP_batched(csv_path)

    elif mode == 'LWP_hindsight':
        parse_LWP_batched_hindsight(csv_path)  # Hindsight performance

    else:
        raise ValueError()


#####################################################
# UTILS
#####################################################
class LatexColumn:
    """
    This class represents a latex column in a latex table with possibly a mean and SE value.
    """

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


def print_begin_table(caption=""):
    print(r"\begin{table}[]")
    print(rf"\caption{{{caption}}}")
    print(r"\label{tab:HOLDER}")
    print(r"\centering")
    print(r"\resizebox{\linewidth}{!}{%")


def print_end_table():
    print(r"}")
    print(r"\end{table}")


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


#####################################################
# EXPERIMENT TABLES
# The used metric names are mentioned for each of the tables.
# Download the CSV from your result tables in WandB, then pass the csv-path as argument to obtain the Latex tables.
#####################################################
def parse_SGD_momentum(csv_path, round_digits=1):
    """
    Plain online finetuning and an ablation in momentum strengths.
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
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
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
    """
    Plain online finetuning, fixing the head or feature extractor or none.
    Displays OAG and HAG for action,verb,noun. Hindsight results (HAG) are obtained in a following run.
    """
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
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
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
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
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


def parse_SGD_multiple_iterations(csv_path, round_digits=1, nb_users=10):
    """
    The results for SGD with multiple iterations per batch.
    Note that the correlated/decorrelated ACC's are obtained by running the script:
    'src/continual_ego4d/processing/postprocess_metrics_dump.py'.
    """
    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.01)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    orig_df.sort_values(inplace=True, axis=0, by=['TRAIN.INNER_LOOP_ITERS', ])
    # orig_df.sort_values(inplace=True, axis=0, by=['TRAIN.INNER_LOOP_ITERS', 'SOLVER.BASE_LR', ])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
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
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{action}}$",
            round_digits=1,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
            latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
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

        # Decorrelated accuracy (Obtained after additional postprocessing step)
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}^\text{decor.}_{\text{action}}$",
            round_digits=1,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/decorrelated/SE',
            latex_col_header_name=r"$\text{ACC}^{decor.}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/decorrelated/SE',
            latex_col_header_name=r"$\text{ACC}^{decor.}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # Correlated accuracy (Obtained after additional postprocessing step)
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/adhoc_AG/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/adhoc_AG/SE',
            latex_col_header_name=r"$\text{OAG}^\text{cor.}_{\text{action}}$",
            round_digits=1,
        ),

        # Count number of samples in correlated vs decorrelated metric
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
            latex_col_header_name=r"nb samples (decorrelated)",
            format_fn_overwrite=lambda x: x * nb_users,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_keep/mean',
            latex_col_header_name=r"nb samples (correlated)",
            format_fn_overwrite=lambda x: x * nb_users,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_total/mean',
            latex_col_header_name=r"nb samples total)",
            format_fn_overwrite=lambda x: x * nb_users,
        ),
    ]

    df_to_latex(ordered_cols, orig_df)


def parse_pretrain_performance_on_user_streams(csv_path, round_digits=1):
    """
    After running 'reproduce/pretrain/eval_user_stream_performance',
    retrieve the class-balanced ACC of the fixed pretrained model.
    """
    orig_df = pd.read_csv(csv_path)

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # BALANCED ACTION ACC
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{bal,top-1, noun}}$",
            round_digits=round_digits,
        ),

    ]

    df_to_latex(ordered_cols, orig_df)


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


if __name__ == "__main__":
    main()
