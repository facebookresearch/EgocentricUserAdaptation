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
    csv_filename = "wandb_export_2022-10-14T15_17_33.711-07_00.csv"  #

    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001) & (orig_df['SOLVER.NESTEROV'] == True)]
    orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == False)]  # TODO: Set to False or True to get both parts
    # orig_df = orig_df.loc[(orig_df['SOLVER.MOMENTUM'] == 0)]  # TODO: Set to False or True to get both parts
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

    # Get lists for plots
    for col in ordered_cols:
        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = orig_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = orig_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)


def final01_02_grad_analysis():
    """
    'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/full_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/full_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/slow_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/slow_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/fast_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/fast_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_1/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_2/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_3/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_4/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_5/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_6/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/head_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/head_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_7/feat_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/feat_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_8/feat_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/feat_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_9/feat_grad_cos_sim/SE',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/feat_grad_cos_sim/mean',
     'adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_10/feat_grad_cos_sim/SE'
    :return:
    """

    csv_filename = "wandb_export_2022-10-23T15_13_53.325-07_00.csv"

    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 3
    model_parts = ["full", "slow", "fast", "head", "feat", ]
    # model_parts = ['head']

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001) & (orig_df['SOLVER.NESTEROV'] == True)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == False)]  # TODO: Set to False or True to get both parts
    # orig_df = orig_df.loc[(orig_df['SOLVER.MOMENTUM'] == 0)]  # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be

    # Reformat cols to have per metric, the SE and mean in separate columns, and 1->10 in rows, take 'steps' as another column
    df_dict_list = []
    for nb_steps_lookback in range(1, 11):  # 1 row per step
        single_df_row_dict = {'step': nb_steps_lookback, }

        for model_part in model_parts:
            mean = orig_df[
                f"adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_{nb_steps_lookback}/{model_part}_grad_cos_sim/mean"].to_list()[
                0]
            SE = orig_df[
                f"adhoc_users_aggregate/analyze_action_batch/LOOKBACK_STEP_{nb_steps_lookback}/{model_part}_grad_cos_sim/SE"].to_list()[
                0]

            single_df_row_dict[f"adhoc_users_aggregate/analyze_action_batch/{model_part}_grad_cos_sim/mean"] = mean
            single_df_row_dict[f"adhoc_users_aggregate/analyze_action_batch/{model_part}_grad_cos_sim/SE"] = SE

        # Update
        df_dict_list.append(single_df_row_dict)

    transformed_df = pd.DataFrame(df_dict_list)

    # Group in Latex mean \pm SE columns
    ordered_cols = [
        LatexColumn(
            f"step",
            latex_col_header_name=f"History steps",
            format_fn_overwrite=lambda x: x,
        ),
    ]
    for model_part in model_parts:
        ordered_cols.append(
            LatexColumn(
                f"adhoc_users_aggregate/analyze_action_batch/{model_part}_grad_cos_sim/mean",
                f"adhoc_users_aggregate/analyze_action_batch/{model_part}_grad_cos_sim/SE",
                latex_col_header_name=fr"$||\nabla_\text{{{model_part}}}||$",
                round_digits=round_digits,
            ),
        )

    latex_df = pd.DataFrame()
    for col in ordered_cols:

        if col.pandas_col_std_name is not None:
            latex_df[col.latex_col_header_name] = transformed_df.loc[:,
                                                  (col.pandas_col_mean_name, col.pandas_col_std_name)
                                                  ].apply(col.format_fn, axis=1)
        else:
            latex_df[col.latex_col_header_name] = transformed_df.loc[:, col.pandas_col_mean_name].apply(col.format_fn)

    print_begin_table()
    print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_eval01_eval02_eval03_eval13_eval15_test_users():
    """
    """
    # csv_filename = "wandb_export_2022-10-16T14_38_30.361-07_00.csv"  # Eval01 (FT)
    # csv_filename = "wandb_export_2022-10-16T14_43_11.032-07_00.csv"  # Eval02 (Replay)
    # csv_filename = "wandb_export_2022-10-18T21_46_13.587-07_00.csv"  # Eval02 (Replay - FIx reservoir-action)
    csv_filename = "wandb_export_2022-10-20T09_41_34.688-07_00.csv"  # Eval02_iter10 (ITER 10: ALL)
    # csv_filename = "wandb_export_2022-10-17T10_22_33.701-07_00.csv"  # Eval03 (FT classifier)
    # csv_filename = "wandb_export_2022-10-16T18_05_00.577-07_00.csv"  # Eval13 (FT-IID 1 epoch)
    # csv_filename = "wandb_export_2022-10-17T11_42_01.818-07_00.csv"  # Eval15 ()

    # 10 iters
    # csv_filename = "wandb_export_2022-10-17T21_29_31.574-07_00.csv"  # Eval01/03/13 iter10

    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.MOMENTUM'] == 0)]  # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     'SOLVER.BASE_LR',
        #     latex_col_header_name=r"$\rho (\eta)$",
        #     format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        # ),

        # LatexColumn(
        #     'SOLVER.MOMENTUM',
        #     latex_col_header_name=r"$\rho$",
        #     format_fn_overwrite=lambda x: x,
        # ),
        #
        # LatexColumn(
        #     'SOLVER.BASE_LR',
        #     latex_col_header_name=r"$\eta$",
        #     format_fn_overwrite=lambda x: "{:.1g}".format(x)
        # ),

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

        # ONLINE ABSOLUTE
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"online-$\overline{\text{ACC}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"online-$\overline{\text{ACC}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"online-$\overline{\text{ACC}}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # HISTORY ABSOLUTE
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"hindsight-$\overline{\text{ACC}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"hindsight-$\overline{\text{ACC}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"hindsight-$\overline{\text{ACC}}_{\text{noun}}$",
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
    # csv_filename = "wandb_export_2022-10-09T17_12_44.119-07_00.csv"  # ACC-based
    # csv_filename = "wandb_export_2022-10-14T11_03_03.510-07_00.csv"  # ACC-based
    csv_filename = "wandb_export_2022-10-21T23_00_22.814-07_00.csv"  # ACC-based
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
    # csv_filename = "wandb_export_2022-10-09T16_49_03.226-07_00.csv"  # FINAL ACC HAG
    # csv_filename = "wandb_export_2022-10-13T22_34_24.620-07_00.csv"  # FINAL ACC HAG
    csv_filename = "wandb_export_2022-10-18T16_06_38.622-07_00.csv"  # NEW RESULTS: Action-reservoir updated
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
    print(latex_df.drop(latex_df.columns[0], axis=1).to_latex(escape=False, index=False, na_rep='N/A'), end='')
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
    csv_filename = "wandb_export_2022-10-25T10_31_19.778-07_00.csv"  # INCL corr/decorr results
    # csv_filename = "wandb_export_2022-10-25T10_58_55.415-07_00.csv" # SGD only

    # Corr/decorr: prev sample only
    csv_filename = "wandb_export_2022-10-25T16_04_34.888-07_00.csv"  # Only preceding sample
    csv_filename = "wandb_export_2022-10-25T16_10_07.514-07_00.csv"  # SGD

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
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_verb_batch/balanced_top1_acc/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_noun_batch/balanced_top1_acc/adhoc_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{OAG}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

        # HISTORY AG
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
            latex_col_header_name=r"$\overline{\text{HAG}}_{\text{action}}$",
            round_digits=1,
        ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
        #     'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{HAG}}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/mean',
        #     'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/adhoc_hindsight_AG/SE',
        #     latex_col_header_name=r"$\overline{\text{HAG}}_{\text{noun}}$",
        #     round_digits=round_digits,
        # ),

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

        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/decorrelated/num_samples_keep/mean',
            latex_col_header_name=r"nb samples (decorrelated)",
            format_fn_overwrite=lambda x: x * NB_USERS,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_keep/mean',
            latex_col_header_name=r"nb samples (correlated)",
            format_fn_overwrite=lambda x: x * NB_USERS,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/balanced_top1_acc/correlated/num_samples_total/mean',
            latex_col_header_name=r"nb samples total)",
            format_fn_overwrite=lambda x: x * NB_USERS,
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
    # csv_filename = "wandb_export_2022-10-12T14_35_29.228-07_00.csv"  # including balanced loss results
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
    # csv_filename = "wandb_export_2022-09-30T19_50_49.808-07_00.csv"  # Full results all
    csv_filename = "wandb_export_2022-10-20T19_51_56.620-07_00.csv"  # BALANCED
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
    # csv_filename = "wandb_export_2022-10-05T18_47_30.747-07_00.csv"  # TEST USERS: Full results all
    # csv_filename = "wandb_export_2022-10-14T22_18_47.430-07_00.csv"  # TEST USERS: Full results all
    csv_filename = "wandb_export_2022-10-21T21_42_42.624-07_00.csv"  # TRAIN USERS: Full results all
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

        # BALANCED-ACC
        LatexColumn(
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/mean',  # TOP1
            'adhoc_users_aggregate/test_action_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_verb_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/mean',
            'adhoc_users_aggregate/test_noun_batch/balanced_top1_acc/SE',
            latex_col_header_name=r"$\overline{\text{ACC}}_{\text{noun}}$",
            round_digits=round_digits,
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
        #
        # # TOP-5 acc
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
    with pd.option_context("max_colwidth", 1000):  # No truncating of strings
        print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final16_eval16_01_label_window_predictor():
    """
    """
    # csv_filename = "wandb_export_2022-10-16T15_35_41.379-07_00.csv"  # TRAIN USERS: Full results all
    # csv_filename = "wandb_export_2022-10-17T20_39_59.956-07_00.csv"  # TRAIN USERS: Includes OAG
    # csv_filename = "wandb_export_2022-10-20T20_03_26.394-07_00.csv"  # TRAIN USERS: 1-window size
    # csv_filename = "wandb_export_2022-10-20T20_21_31.153-07_00.csv"  # TRAIN USERS: all in one
    csv_filename = "wandb_export_2022-10-20T20_45_59.875-07_00.csv"  # TRAIN USERS: all in one + top1 unbalanced
    # csv_filename = "wandb_export_2022-10-16T17_28_26.107-07_00.csv"  # TEST USERS
    caption = "Label window predictor naive baseline."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    orig_df.sort_values(inplace=True, axis=0, by=["ANALYZE_STREAM.WINDOW_SIZE_SAMPLES"])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'ANALYZE_STREAM.WINDOW_SIZE_SAMPLES',
            latex_col_header_name=r"window size",
            format_fn_overwrite=lambda x: x,
        ),

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
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{noun}}$",
            round_digits=round_digits,
        ),

        # OAG:
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}_{\text{noun}}$",
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
    with pd.option_context("max_colwidth", 1000):  # No truncating of strings
        print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()


def parse_final16_eval16_02_label_window_predictor_hindsight():
    """
    Take running avg mean of HindsightLabelWindowPredictor
    """
    csv_filename = "wandb_export_2022-10-25T22_09_45.528-07_00.csv"  # TRAIN USERS
    # csv_filename = "wandb_export_2022-10-25T21_00_30.781-07_00.csv"  # EVAL USERS

    caption = "Hindsight Label window predictor naive baseline."
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    orig_df.sort_values(inplace=True, axis=0, by=["ANALYZE_STREAM.WINDOW_SIZE_SAMPLES"])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'ANALYZE_STREAM.WINDOW_SIZE_SAMPLES',
            latex_col_header_name=r"window size",
            format_fn_overwrite=lambda x: x,
        ),

        # BALANCED
        LatexColumn(
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/mean',
            'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{action}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/mean',
            'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{verb}}$",
            round_digits=round_digits,
        ),
        LatexColumn(
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/mean',
            'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/SE',
            latex_col_header_name=r"$\text{ACC}_{\text{noun}}$",
            round_digits=round_digits,
        ),

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
        # LatexColumn(
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}_{\text{action}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}_{\text{verb}}$",
        #     round_digits=round_digits,
        # ),
        # LatexColumn(
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/mean',
        #     'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/adhoc_AG/SE',
        #     latex_col_header_name=r"$\text{OAG}_{\text{noun}}$",
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
    with pd.option_context("max_colwidth", 1000):  # No truncating of strings
        print(latex_df.to_latex(escape=False, index=False, na_rep='N/A'), end='')
    print_end_table()

def parse_final17_01_momentum_feat_head():
    """
    """
    # csv_filename = "wandb_export_2022-10-17T10_06_27.828-07_00.csv"
    csv_filename = "wandb_export_2022-10-20T16_04_10.497-07_00.csv"  # Also HAG
    caption = "Momentum head vs classifier"
    csv_path = os.path.join(csv_dirname, csv_filename)
    round_digits = 1

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.001)]
    # orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)] # TODO: Set to False or True to get both parts
    # orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.BASE_LR','TRAIN.INNER_LOOP_ITERS', ])
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM_HEAD', 'SOLVER.MOMENTUM_FEAT'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        LatexColumn(
            'SOLVER.MOMENTUM_HEAD',
            latex_col_header_name=r"$\rho_{\text{head}}$",
            format_fn_overwrite=lambda x: x,
        ),

        LatexColumn(
            'SOLVER.MOMENTUM_FEAT',
            latex_col_header_name=r"$\rho_{\text{feat}}$",
            format_fn_overwrite=lambda x: x,
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
    parse_final16_eval16_02_label_window_predictor_hindsight()
