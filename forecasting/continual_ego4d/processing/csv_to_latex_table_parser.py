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
    csv_filename = "wandb_export_2022-09-21T11_36_57.254-07_00.csv"
    csv_path = os.path.join(csv_dirname, csv_filename)

    orig_df = pd.read_csv(csv_path)

    # FILTER
    # orig_df = orig_df.loc[(orig_df['SOLVER.BASE_LR'] == 0.01) & (orig_df['SOLVER.NESTEROV'] == False)]
    orig_df = orig_df.loc[(orig_df['SOLVER.NESTEROV'] == True)]
    orig_df.sort_values(inplace=True, axis=0, by=['SOLVER.MOMENTUM', 'SOLVER.BASE_LR'])

    # Place here in order you want the latex columns to be
    ordered_cols = [

        # HPARAMS COL
        LatexColumn(
            'SOLVER.MOMENTUM',
            'SOLVER.BASE_LR',
            latex_col_report_name=r"$\rho (\eta)$",
            format_fn_overwrite=lambda x: f"{x[0]} ({x[1]})"
        ),

        # LatexColumn(
        #     'SOLVER.BASE_LR',
        #     latex_col_report_name=r"$\eta",
        #     format_fn_overwrite=lambda x: x
        # ),

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
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/SE',
            'adhoc_users_aggregate_history/pred_action_batch/loss/avg_history_AG/mean',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{action}}$"
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/SE',
            'adhoc_users_aggregate_history/pred_verb_batch/loss/avg_history_AG/mean',
            latex_col_report_name=r"$\overline{\text{HAG}}_{\text{verb}}$"
        ),
        LatexColumn(
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/SE',
            'adhoc_users_aggregate_history/pred_noun_batch/loss/avg_history_AG/mean',
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


def print_begin_table():
    print(r"\begin{table}[]")
    print(r"\caption{}")
    print(r"\label{tab:HOLDER}")
    print(r"\resizebox{\linewidth}{!}{%")


def print_end_table():
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    parse_final01_01_momentum_table()
