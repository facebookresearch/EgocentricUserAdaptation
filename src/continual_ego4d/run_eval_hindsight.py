"""
After training user-streams, evaluate on the final models with the full user-streams.

1) From a WandB group you are interested in copy the OUTPUT_DIR as input to this script:
/home/matthiasdelange/sftp_remote_projects/ContextualOracle_Matthias/exps/ego4d_action_recog/exp04_01_momentum_video_reset/../../../results/ego4d_action_recog/exp04_01_momentum_video_reset/logs/GRID_SOLVER-BASE_LR=0-01_SOLVER-MOMENTUM=0-9_SOLVER-NESTEROV=True/2022-09-16_17-34-18_UID8cae3077-fb05-4368-b3d6-95cb2813e823

2) Given the parent outputdir of the run over users, this script gets for each user:
- the checkpoint: in PARENT_DIR/checkpoints/<USER>/last.ckpt
- the final dump: in PARENT_DIR/user_logs/<USER>/dumpfile.pth

3) Iterate one datastream, 1 model for simplicity. But we can run multiple of these processes concurrently.
- Init a new ContinualMultiTaskClassificationTask for each new model-stream pair:
- Run predict-stream
- Flatten results, make DataFrame and save as csv file

4) POSTPROCESSING: When all csv's are collected: Aggregate results into the 2d-Matrix and upload as heatmap to WandB runs in group.
"""
import copy
import multiprocessing as mp
import os
import pprint
import traceback
from collections import Counter
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import torch
from fvcore.common.config import CfgNode
from pytorch_lightning import Trainer, seed_everything

from continual_ego4d.processing.run_adhoc_metric_processing_wandb import upload_metric_dict_to_wandb, \
    collect_wandb_group_user_results_for_metrics
from continual_ego4d.processing.utils import get_group_run_iterator, get_group_names_from_csv, get_delta, \
    get_delta_mappings
from continual_ego4d.run_train_user_streams import load_datasets_from_jsons, get_user_ids, get_device_ids
from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from continual_ego4d.tasks.continual_action_recog_task import PretrainState
from continual_ego4d.utils.checkpoint_loading import load_slowfast_model_weights, PathHandler
from continual_ego4d.utils.misc import makedirs
from continual_ego4d.utils.models import freeze_full_model, model_trainable_summary
from continual_ego4d.utils.scheduler import SchedulerConfig, RunConfig
from ego4d.config.defaults import set_cfg_by_name, cfg_add_non_existing_key_vals, convert_flat_dict_to_cfg, get_cfg
from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args

logger = logging.get_logger(__name__)


def parse_wandb_runs(project_name, group_name):
    """ For a given group name in a project, return the user-runs and get the training config. """
    user_to_config = {}

    # summary = final value (excludes NaN rows)
    # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
    for user_run in get_group_run_iterator(project_name, group_name, finished_runs_only=True):
        user_id = user_run.config['DATA.COMPUTED_USER_ID']
        user_to_config[user_id] = copy.deepcopy(user_run.config)

    return user_to_config


def eval_config_checks(eval_cfg):
    if eval_cfg.USER_SELECTION is None:
        if eval_cfg.TRANSFER_EVAL.PRETRAIN_REFERENCE_GROUP_WANDB == 'train':
            assert eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS == 10

        elif eval_cfg.TRANSFER_EVAL.PRETRAIN_REFERENCE_GROUP_WANDB == 'test':
            assert eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS == 40

        else:
            raise ValueError()


def main():
    """ Iterate users and aggregate. """
    print("Starting eval")
    args = parse_args()
    eval_cfg = load_config(args)
    eval_config_checks(eval_cfg)

    if eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH is not None:
        assert os.path.isfile(eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH), \
            f"Non-existing csv: {eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH}"
        group_names = get_group_names_from_csv(eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH)

        if eval_cfg.TRANSFER_EVAL.CSV_RANGE is not None:
            csv_range = eval_cfg.TRANSFER_EVAL.CSV_RANGE
            if isinstance(csv_range, str):
                csv_range = csv_range.split(',')

            start_idx, end_idx = csv_range
            group_names = group_names[start_idx:end_idx]
    else:
        group_names = [eval_cfg.TRANSFER_EVAL.WANDB_GROUP_TO_EVAL]

    print(f"Group names to process = {group_names}")
    project_name = eval_cfg.TRANSFER_EVAL.WANDB_PROJECT_NAME.strip()
    for group_name in group_names:

        try:
            process_group(copy.deepcopy(eval_cfg), group_name.strip(), project_name)
        except Exception as e:
            traceback.print_exc()
            print(f"GROUP FAILED: {group_name}")


def process_group(eval_cfg, group_name, project_name):
    """ Process single group. """
    logger.info(f"{'*' * 40} Starting processing of group {group_name} {'*' * 40}")

    # Retrieve all USERS and OUTPUT_DIRS
    user_to_flat_cfg = parse_wandb_runs(project_name, group_name)

    if len(user_to_flat_cfg) == 0:
        print("No finished user runs found in group")
        exit(0)

    # Filter
    # Should all have same outputdir in same group
    outputdirs = [os.path.abspath(user_config['OUTPUT_DIR']) for user_config in user_to_flat_cfg.values()]
    assert len(set(outputdirs)) == 1, f"Users not same group dir: {set(outputdirs)}"
    train_group_outputdir = outputdirs[0]

    user_to_train_runuid = {user: config['RUN_UID'] for user, config in user_to_flat_cfg.items()}

    # Get checkpoint paths
    user_to_checkpoint_path = {}
    train_path_handler = None
    for user in user_to_train_runuid.keys():
        train_path_handler = PathHandler(
            main_output_dir=train_group_outputdir,
            run_group_id=group_name,
            run_uid=user_to_train_runuid[user],
            is_resuming_run=True,
        )
        user_to_checkpoint_path[user] = train_path_handler.get_user_checkpoints_dir(
            user_id=user, include_ckpt_file='last.ckpt'
        )

    # Set current config with this config: Used to make trainer with same hyperparams on stream
    def_train_cfg: CfgNode = get_cfg()  # Defaults
    wandb_train_cfg: CfgNode = convert_flat_dict_to_cfg(
        next(iter(user_to_flat_cfg.values())),
        key_exclude_set={'DATA.COMPUTED_USER_ID',
                         'DATA.COMPUTED_USER_DS_ENTRIES',
                         'COMPUTED_USER_DUMP_FILE'}
    )
    train_cfg: CfgNode = cfg_add_non_existing_key_vals(wandb_train_cfg, def_train_cfg)  # For newly added ones
    user_to_checkpoint_path['pretrain'] = train_cfg['CHECKPOINT_FILE_PATH']  # Add pretrain as user

    # SET EVAL OUTPUT_DIR
    main_parent_dirname = 'balanced_transfer_eval' if not eval_cfg.FAST_DEV_RUN else \
        f"balanced_transfer_eval_DEBUG_CUTOFF_{eval_cfg.FAST_DEV_DATA_CUTOFF}"
    main_parent_dir = os.path.join(train_group_outputdir, main_parent_dirname)
    eval_cfg.TRANSFER_EVAL.COMPUTED_POSTPROCESS_RESULTDIR = os.path.join(main_parent_dir, 'postprocess_results')
    eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR = os.path.join(main_parent_dir, 'results')
    eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_LOGDIR = os.path.join(main_parent_dir, 'logs')

    makedirs(eval_cfg.TRANSFER_EVAL.COMPUTED_POSTPROCESS_RESULTDIR)
    makedirs(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR)
    makedirs(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_LOGDIR)

    # Main process logging
    logging.setup_logging(main_parent_dir, host_name='MASTER', overwrite_logfile=False)
    logger.info(f"Starting main script with OUTPUT_DIR={main_parent_dir}")

    # Dataset lists from json / users to process
    if eval_cfg.USER_SELECTION is not None:
        train_cfg.USER_SELECTION = eval_cfg.USER_SELECTION

    user_datasets, pretrain_dataset = load_datasets_from_jsons(train_cfg)
    processed_trained_user_ids, all_trained_user_ids = get_user_ids(train_cfg, user_datasets, train_path_handler)
    user_datasets['pretrain'] = pretrain_dataset  # Add (AFTER CHECKING USER IDS)

    assert len(processed_trained_user_ids) == len(all_trained_user_ids) == eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS, \
        f"Not all users were processed in training: {set(all_trained_user_ids) - set(processed_trained_user_ids)}\n" \
        f"Expected {eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS} users."

    # CFG overwrites and setup
    overwrite_config_transfer_eval(eval_cfg)
    logger.info("Run with eval_cfg:")
    logger.info(pprint.pformat(eval_cfg))

    # Sequential/parallel execution user jobs
    available_device_ids = get_device_ids(eval_cfg)
    assert len(available_device_ids) >= 1

    # Get run entries
    all_run_entries = []
    modeluser_streamuser_pairs = get_user_model_stream_pairs(eval_cfg, all_trained_user_ids)
    logger.info(f"Processing pairs: {modeluser_streamuser_pairs}")
    for (model_userid, stream_userid) in modeluser_streamuser_pairs:
        run_id = get_pair_id(model_userid, stream_userid)

        # STREAM config
        user_train_stream_cfg = copy.deepcopy(train_cfg)
        user_train_stream_cfg.DATA.COMPUTED_USER_ID = stream_userid
        user_train_stream_cfg.DATA.COMPUTED_USER_DS_ENTRIES = user_datasets[stream_userid]
        user_train_stream_cfg.COMPUTED_USER_DUMP_FILE = train_path_handler.get_user_streamdump_file(stream_userid)
        user_train_stream_cfg.STREAM_EVAL_ONLY = True  # Avoid training based checks

        # Debug mode
        user_train_stream_cfg.FAST_DEV_RUN = eval_cfg.FAST_DEV_RUN
        user_train_stream_cfg.FAST_DEV_DATA_CUTOFF = eval_cfg.FAST_DEV_DATA_CUTOFF

        # Prediction batch size
        user_train_stream_cfg.PREDICT_PHASE.NUM_WORKERS = eval_cfg.PREDICT_PHASE.NUM_WORKERS
        user_train_stream_cfg.PREDICT_PHASE.BATCH_SIZE = eval_cfg.PREDICT_PHASE.BATCH_SIZE

        all_run_entries.append(
            RunConfig(
                run_id=run_id,
                target_fn=eval_single_model_single_stream,
                fn_args=(
                    user_train_stream_cfg,  # To recreate data stream
                    eval_cfg,
                    user_to_checkpoint_path[model_userid],
                )
            )
        )

    processed_eval_run_ids = get_processed_run_ids(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR)
    logger.info(f"Processed run_ids = {processed_eval_run_ids}")

    scheduler_cfg = SchedulerConfig(
        run_entries=all_run_entries,
        processed_run_ids=processed_eval_run_ids,
        available_device_ids=available_device_ids,
        max_runs_per_device=eval_cfg.NUM_USERS_PER_DEVICE,
    )

    if scheduler_cfg.is_all_runs_processed():
        logger.info("All users already processed, skipping execution. "
                    f"All users={scheduler_cfg.all_run_ids}, "
                    f"processed={scheduler_cfg.processed_run_ids}")
    else:
        # LAUNCH SCHEDULING
        scheduler_cfg.schedule()

    logger.info("Finished processing all users")
    logger.info(f"All results over users can be found in OUTPUT-DIR={main_parent_dir}")

    # START POSTPROCESSING
    # Get all csvs from dir and only post-process if all are present
    completed_csv_filenames = get_processed_run_ids(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR, keep_extention=True)

    if len(completed_csv_filenames) != len(all_run_entries):
        logger.info(f"Skipping postprocessing as not all runs were completed successfully")
        return

    logger.info(f"Starting postprocessing")

    if eval_cfg.TRANSFER_EVAL.INCLUDE_PRETRAIN_MODEL:
        """ HAG processing, requires pretrain models. """
        postprocess_instance_counts_from_csv_files(eval_cfg, modeluser_streamuser_pairs, pretrain_dataset)

    """ Postprocess absolute values. Always possible, don't need pretrain for that. """
    postprocess_absolute_stream_results(eval_cfg, modeluser_streamuser_pairs, group_name)


def postprocess_absolute_stream_results(eval_cfg, modeluser_streamuser_pairs, group_name):
    """
    Get transfer matrix result, absolute (loss/acc) and relative (loss/acc w.r.t. pretrain)
    for measured metrics (action/verb/noun).

    The matrix is always calculated, even if only the diagonal is considered, unfilled values will contain np.inf.
    2d Matrix is dims, the model and stream users are considered identical (also in ordering):
    - rows (y-axis): model users
    - cols (x-axis): stream users

    Besides the matrix, the diagonal is used to calculate the hindsight average-over-users performance. (HAG)

    All results are uploaded to wandb after processing.
    """

    #####################################
    # INIT
    #####################################
    init_val = np.inf

    def init_matrix():
        return [[init_val for _ in range(len(streamusers))] for _ in range(len(modelusers))]

    # MATRIX
    modelusers = sorted(list(
        set([entry_pair[0] for entry_pair in modeluser_streamuser_pairs]) - {'pretrain'}
    ))
    streamusers = sorted(list(
        set([entry_pair[1] for entry_pair in modeluser_streamuser_pairs]) - {'pretrain'}
    ))
    assert modelusers == streamusers, "Only implemented for square matrix"

    modeluser_to_row = {user: idx for idx, user in enumerate(modelusers)}
    streamuser_to_col = {user: idx for idx, user in enumerate(streamusers)}

    # PRETRAIN METRICS MAPPING: pretrain results are collected during training (online)
    metric_name_to_pretrain_name_mapping = {

        # Micro-avg loss
        'test_action_batch/loss': 'train_action_batch/loss_running_avg',
        'test_verb_batch/loss': 'train_verb_batch/loss_running_avg',
        'test_noun_batch/loss': 'train_noun_batch/loss_running_avg',

        # Micro-avg ACC
        'test_action_batch/top1_acc': 'train_action_batch/top1_acc_running_avg',
        'test_verb_batch/top1_acc': 'train_verb_batch/top1_acc_running_avg',
        'test_noun_batch/top1_acc': 'train_noun_batch/top1_acc_running_avg',

        'test_verb_batch/top5_acc': 'train_verb_batch/top5_acc_running_avg',
        'test_noun_batch/top5_acc': 'train_noun_batch/top5_acc_running_avg',

        # Macro-avg ACC (class-balanced)
        'test_action_batch/balanced_top1_acc': 'adhoc_users_aggregate/train_action_batch/top1_acc_balanced_running_avg/mean',
        'test_verb_batch/balanced_top1_acc': 'adhoc_users_aggregate/train_verb_batch/top1_acc_balanced_running_avg/mean',
        'test_noun_batch/balanced_top1_acc': 'adhoc_users_aggregate/train_noun_batch/top1_acc_balanced_running_avg/mean',
    }

    #####################################
    # METRICS TO ACCUMULATE
    #####################################
    metric_name_to_delta_sign = get_delta_mappings()
    metrics_of_interest = list(metric_name_to_pretrain_name_mapping.keys())

    # DIAGONAL
    metric_to_avg_abs_per_user = defaultdict(list)  # delta values (relative to pretrain)
    metric_to_avg_AG_per_user = defaultdict(list)  # delta values (relative to pretrain)

    # MATRIX
    metric_to_avg_abs_matrix = defaultdict(init_matrix)  # Absolute values
    metric_to_avg_AG_matrix = defaultdict(init_matrix)  # delta values (relative to pretrain)
    # X and Y labels are based on modelusers and streamusers

    #####################################
    # PRETRAIN model, user stream results
    #####################################
    # Load results from pretrain run
    if eval_cfg.TRANSFER_EVAL.PRETRAIN_REFERENCE_GROUP_WANDB == 'train':
        pretrain_group = eval_cfg.TRANSFER_EVAL.PRETRAIN_TRAIN_USERS_GROUP_WANDB

    elif eval_cfg.TRANSFER_EVAL.PRETRAIN_REFERENCE_GROUP_WANDB == 'test':
        pretrain_group = eval_cfg.TRANSFER_EVAL.PRETRAIN_TEST_USERS_GROUP_WANDB
    else:
        raise ValueError()

    pretrain_metric_names = list(metric_name_to_pretrain_name_mapping.values())
    pretrain_user_ids, pretrain_final_stream_metric_userlists = collect_wandb_group_user_results_for_metrics(
        eval_cfg.TRANSFER_EVAL.WANDB_PROJECT_NAME,
        pretrain_group,
        metric_names=pretrain_metric_names,
        run_filter=None,
        user_ids=streamusers
    )
    assert len(pretrain_user_ids) == eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS
    assert pretrain_user_ids == streamusers, "Order of users should be same"
    pretrainuser_to_idx = {user: idx for idx, user in enumerate(pretrain_user_ids)}

    #####################################
    # ITERATE USER-PAIRS
    #####################################
    for model_userid, stream_userid in modeluser_streamuser_pairs:
        if 'pretrain' in [model_userid, stream_userid]:
            logger.info(f"SKIPPING: model_userid={model_userid}, stream_userid={stream_userid}")
            continue
        logger.info(f"Processing: model_userid={model_userid}, stream_userid={stream_userid}")

        # Pretrain model, user_stream results
        pretrainuser_idx = pretrainuser_to_idx[stream_userid]

        # Read CSV
        stream_df = get_df_from_csv(eval_cfg, model_userid, stream_userid)
        stream_avg_df = stream_df.iloc[[-1]]  # Only last row (final result/stream avg)

        # Iterate metrics, and add user result to each
        for metric_name in metrics_of_interest:

            # Absolute value
            user_stream_avg_result = stream_avg_df[metric_name].tolist()[0]  # Select stream avg

            # Deltas (AG): Only if available in pretrain as well
            if metric_name not in metric_name_to_pretrain_name_mapping:
                logger.info(f"Skipping metric {metric_name} for AG as not existing in pretrain")
                continue
            pretrain_key = metric_name_to_pretrain_name_mapping[metric_name]
            pretrain_stream_avg_result = pretrain_final_stream_metric_userlists[pretrain_key][
                pretrainuser_idx]  # Avg over pretrain users

            # Delta value
            delta_sign = metric_name_to_delta_sign[metric_name]
            user_stream_avg_delta = get_delta(delta_sign, user_stream_avg_result, pretrain_stream_avg_result)

            # Add matrix values (absolute and delta)
            row = modeluser_to_row[model_userid]
            col = streamuser_to_col[stream_userid]
            metric_to_avg_abs_matrix[f"TRANSFER_MATRIX/ABSOLUTE/{metric_name}"][row][col] = user_stream_avg_result
            metric_to_avg_AG_matrix[f"TRANSFER_MATRIX/AG/{metric_name}"][row][col] = user_stream_avg_delta

            # Add diagonal values (absolute and delta)
            if model_userid == stream_userid:
                metric_to_avg_abs_per_user[f"{metric_name}"].append(
                    user_stream_avg_result
                )
                metric_to_avg_AG_per_user[f"{metric_name}/adhoc_hindsight_AG"].append(
                    user_stream_avg_delta
                )

    #####################################
    #  AVG AND UPLOAD TO WANDB
    #####################################

    # Upload diagonal
    diagonal_dict = {
        **metric_to_avg_abs_per_user,
        **metric_to_avg_AG_per_user
    }
    logger.info(f"Collected DIAGONAL results for users {streamusers}:\n {diagonal_dict}")
    upload_metric_dict_to_wandb(
        diagonal_dict,
        project_name=eval_cfg.TRANSFER_EVAL.WANDB_PROJECT_NAME,
        group_name=group_name,
        mean=True
    )

    # Upload matrix (Based on diagonal mode, for eval is 40x40 users = 1.6k entries)
    if not eval_cfg.TRANSFER_EVAL.DIAGONAL_ONLY:
        matrix_dict = {
            **metric_to_avg_abs_matrix,
            **metric_to_avg_AG_matrix,
            'TRANSFER_MATRIX/USERS_IN_ORDER': streamusers
        }
        logger.info(f"Collected MATRIX results for users {streamusers}:\n {matrix_dict}")
        upload_metric_dict_to_wandb(
            matrix_dict,
            project_name=eval_cfg.TRANSFER_EVAL.WANDB_PROJECT_NAME,
            group_name=group_name,
            mean=False
        )  # Don't average, but report 2d array


def postprocess_instance_counts_from_csv_files(eval_cfg, modeluser_streamuser_pairs, pretrain_dataset):
    """ Postprocess and save in csv files for pretrain vs stream instance counts/action. """
    ###################################################
    # PRETRAIN vs INSTANCE COUNT ANALYSIS (includes HAG)
    ###################################################
    # Action instance counts in pretrain/stream and avgs over action-performances per user
    instance_count_df_dict: dict[str, pd.DataFrame] = get_pretrain_stream_instance_count_df(
        eval_cfg, modeluser_streamuser_pairs, pretrain_dataset
    )

    # Save to csv's
    for action_mode, action_mode_df in instance_count_df_dict.items():
        filepath = os.path.join(eval_cfg.TRANSFER_EVAL.COMPUTED_POSTPROCESS_RESULTDIR,
                                f"{action_mode}_instance_count.csv")
        action_mode_df.to_csv(filepath, index=False)


def get_pretrain_stream_instance_count_df(eval_cfg, modeluser_streamuser_pairs, pretrain_dataset) \
        -> dict[str, pd.DataFrame]:
    """
    Get dict {'action':DataFrame, 'verb':DataFrame,'noun':DataFrame}.

    Each DataFrame has following columns averaged over action/verb/noun per user:
    user, action/verb/noun, avg_stream_loss, avg_stream_HAG, stream_count, pretrain_count

    This is later user to calculate KDE map in the paper.
    """
    logger.info(f"PRETRAIN VS STREAM INSTANCE COUNT PROCESSING")
    ps = PretrainState(pretrain_dataset['user_action_sets']['user_agnostic'])

    # List of dicts (each row = dict) to convert to df
    out_df_lists: dict[str, list[dict]] = {
        'action': [],
        'verb': [],
        'noun': [],
    }

    loss_names = {
        'action': 'pred_action_batch/loss',
        'verb': 'pred_verb_batch/loss',
        'noun': 'pred_noun_batch/loss'
    }
    pretrain_freq_dicts = {
        'action': ps.pretrain_action_freq_dict,
        'verb': ps.pretrain_verb_freq_dict,
        'noun': ps.pretrain_noun_freq_dict
    }

    def verbnoun_to_action_cols_(df):
        df['action'] = [(v, n) for v, n in zip(df['verb'].tolist(), df['noun'].tolist())]

    for model_userid, stream_userid in modeluser_streamuser_pairs:
        if 'pretrain' in [model_userid, stream_userid]:  # TODO skipping pretrain cols/rows for now
            logger.info(f"SKIPPING: model_userid={model_userid}, stream_userid={stream_userid}")
            continue
        if model_userid != stream_userid:
            logger.info(
                f"SKIPPING: Only considering diagonal: model_userid={model_userid}, stream_userid={stream_userid}")
            continue
        logger.info(f"Processing: model_userid={model_userid}, stream_userid={stream_userid}")

        assert model_userid == stream_userid
        userid = model_userid

        # entire stream per-sample losses and action label
        stream_df = get_df_from_csv(eval_cfg, userid, userid)
        pretrain_df = get_df_from_csv(eval_cfg, 'pretrain', userid)  # On pretrain model, but same stream (used for AG)

        verbnoun_to_action_cols_(stream_df)
        verbnoun_to_action_cols_(pretrain_df)

        stream_freq_dicts = {
            'action': Counter(stream_df['action'].tolist()),
            'verb': Counter(stream_df['verb'].tolist()),
            'noun': Counter(stream_df['noun'].tolist()),
        }

        # Action/verb/noun
        for action_mode in ['action', 'verb', 'noun']:
            stream_freq_dict = stream_freq_dicts[action_mode]
            pretrain_freq_dict = pretrain_freq_dicts[action_mode]
            loss_name = loss_names[action_mode]  # DF col name
            out_df_list = out_df_lists[action_mode]

            # Iterate over uniques
            unique_actions = stream_df[action_mode].unique()
            for unique_action in unique_actions:
                action_pretrain_count = pretrain_freq_dict[unique_action] if unique_action in pretrain_freq_dict else 0
                action_stream_count = stream_freq_dict[unique_action]  # Has to be present in current stream

                # Select only selected action (Use stream df twice to be sure of idxs)
                df_action_subset = stream_df.loc[stream_df[action_mode] == unique_action]
                pretrain_df_action_subset = pretrain_df.loc[pretrain_df[action_mode] == unique_action]
                assert len(df_action_subset) == len(pretrain_df_action_subset), \
                    f"Both pretrain and stream dataframe should have same nb of samples in total and for each action"

                # Process values (Avg over nb actions to normalize stream length)
                avg_stream_action_loss = df_action_subset[loss_name].mean()
                avg_stream_action_HAG = (pretrain_df_action_subset[loss_name] - df_action_subset[loss_name]).mean()

                out_df_list.append({
                    'user': userid,
                    action_mode: unique_action,
                    'avg_stream_loss': avg_stream_action_loss,
                    'avg_stream_HAG': avg_stream_action_HAG,
                    'stream_count': action_stream_count,
                    'pretrain_count': action_pretrain_count,
                })
    return {action_mode: pd.DataFrame(out_df_list) for action_mode, out_df_list in out_df_lists.items()}


def get_df_from_csv(eval_cfg, model_userid, stream_userid):
    """ Get the csv path
    - summary_results_csv=False:  CSV with per-sample verb/noun/action losses for Transfer analysis.
    - summary_results_csv=True: CSV with final absolute test metrics averaged over stream (e.g. avg loss,...)
    """
    # Read csv
    csv_path = os.path.join(
        eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR,
        f"{get_pair_id(model_userid, stream_userid)}.csv"
    )

    # Parse table results
    df = pd.read_csv(csv_path).drop(['Unnamed: 0'], axis=1, errors='ignore')
    return df


def get_pair_id(modeluser, streamuser):
    return f"M={modeluser}_S={streamuser}"


def get_processed_run_ids(result_dir, keep_extention=False):
    result_file_ids = sorted([
        pair_id_resultfile.name if keep_extention else pair_id_resultfile.name.split('.')[0]
        for pair_id_resultfile in os.scandir(result_dir) if pair_id_resultfile.is_file()])
    return result_file_ids


def get_user_model_stream_pairs(eval_cfg, all_user_ids) -> list[tuple]:
    # ONLY USER-STREAM WITH SAME MODEL
    if eval_cfg.TRANSFER_EVAL.DIAGONAL_ONLY:
        modeluser_streamuser_pairs = [(user_id, user_id) for user_id in all_user_ids]

    # CARTESIAN
    else:
        modeluser_streamuser_pairs = list(product(all_user_ids, all_user_ids))

    # PRETRAIN
    if eval_cfg.TRANSFER_EVAL.INCLUDE_PRETRAIN_STREAM:
        modeluser_streamuser_pairs.extend(
            [(user_id, 'pretrain') for user_id in all_user_ids]
        )
    if eval_cfg.TRANSFER_EVAL.INCLUDE_PRETRAIN_MODEL:
        modeluser_streamuser_pairs.extend(
            [('pretrain', user_id) for user_id in all_user_ids]
        )

    return modeluser_streamuser_pairs


def overwrite_config_transfer_eval(eval_cfg):
    overwrite_dict = {
        "SOLVER.ACCELERATOR": "gpu",
        "NUM_SHARDS": 1,  # no DDP supported
        "CHECKPOINT_LOAD_MODEL_HEAD": True,  # From pretrain we also load model head
    }

    for hierarchy_k, v in overwrite_dict.items():
        set_cfg_by_name(eval_cfg, hierarchy_k, v)

    logger.debug(f"OVERWRITING CFG attributes for transfer eval:\n{pprint.pformat(overwrite_dict)}")


def eval_single_model_single_stream(
        # Scheduling args
        mp_queue: mp.Queue,
        device_id: int,
        run_id: str,

        # additional args
        user_train_cfg: CfgNode,  # To recreate data stream
        eval_cfg: CfgNode,  # For
        load_model_path: str,
) -> (bool, str, str):
    """ Run single user sequentially. Returns path to user results and interruption status."""
    seed_everything(eval_cfg.RNG_SEED)

    run_logdir = os.path.join(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_LOGDIR, run_id)  # Dir per run
    run_result_dir = eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR  # Shared

    # Loggers
    logging.setup_logging(  # Stdout logging
        [run_logdir],
        host_name=f'RUN-{run_id}|GPU-{device_id}|PID-{os.getpid()}',
        overwrite_logfile=False,
    )

    # Always load weights in CL-task for eval
    logger.info("Starting init Task")
    if user_train_cfg.DATA.TASK in ["continual_classification", "iid_classification"]:
        task = ContinualMultiTaskClassificationTask(user_train_cfg)

    else:
        raise ValueError(f"cfg.DATA.TASK={user_train_cfg.DATA.TASK} not supported")
    logger.info(f"Initialized task as {task}")

    # LOAD PRETRAINED
    try:
        load_slowfast_model_weights(
            load_model_path, task, eval_cfg.CHECKPOINT_LOAD_MODEL_HEAD,
        )
    except:
        task.configure_head(task.model, task.stream_state)  # Add outputhead masker so checkpoint can load the params
        load_slowfast_model_weights(
            load_model_path, task, eval_cfg.CHECKPOINT_LOAD_MODEL_HEAD,
        )

    # Freeze model fully
    freeze_full_model(task.model)
    model_trainable_summary(task.model)  # Print summary

    # There are no validation/testing phases!
    logger.info("Initializing Trainer")
    trainer = Trainer(
        # default_root_dir=main_output_dir,  # Default path for logs and weights when no logger/ckpt_callback passed
        accelerator="gpu",  # cfg.SOLVER.ACCELERATOR, only single device for now
        benchmark=True,
        replace_sampler_ddp=False,  # Disable to use own custom sampler
        fast_dev_run=False,  # For CL Should NOT define fast_dev_run in lightning! Doesn't log results then

        # Devices/distributed
        devices=[device_id],
        gpus=None,  # Determined by devices
        num_nodes=1,  # DDP specific

        callbacks=None,
        logger=False,
    )

    logger.info("PREDICTING (INFERENCE)")

    """ On fixed final model, collect loss of all samples in stream in hindsight. """
    sample_idx_to_loss_dict_list: list[dict[str, dict]] = trainer.predict(task)
    """
    Per batch contains mapping of sample_idx to dict of results.
    [{
    0: {'pred_action_batch/loss': 4.419626235961914, 'pred_verb_batch/loss': 0.3899345397949219, 'pred_noun_batch/loss': 4.029691696166992}, 
    1: {'pred_action_batch/loss': 4.606912136077881, 'pred_verb_batch/loss': 0.47056838870048523, 'pred_noun_batch/loss': 4.136343955993652}
    },
    {
    2: {'pred_action_batch/loss': 4.419626235961914, 'pred_verb_batch/loss': 0.3899345397949219, 'pred_noun_batch/loss': 4.029691696166992}, 
    3: {'pred_action_batch/loss': 4.606912136077881, 'pred_verb_batch/loss': 0.47056838870048523, 'pred_noun_batch/loss': 4.136343955993652}
    }]
    """

    # Flatten out batch dim (single dict)
    sample_idx_to_losses_dict: dict[str, dict] = {
        k: v for iter_dict in sample_idx_to_loss_dict_list
        for k, v in iter_dict.items()
    }

    # Dataframe for csv
    df = pd.DataFrame(sample_idx_to_losses_dict).transpose()
    """
    As sample idxs are keys, they will be columns, so then transpose resulting in:
        
           pred_action_batch/loss  pred_verb_batch/loss  pred_noun_batch/loss
    0               63.166710         -0.000000e+00             63.166710
    1               94.299492         -0.000000e+00             94.299492
    2               56.985905          4.014163e-04             56.985504
    3               56.100876          2.384186e-07             56.100876
    4               59.698318          1.567108e-02             59.682648
    ...

    """

    assert len(task.stream_state.sample_idx_to_action_list) == len(df), \
        f"Dataframe with predictions should contain preds for the entire stream."
    # CSV doesn't allow python object (e.g. tuples for action), instead save verb/noun separately
    df['verb'] = [action[0] for action in task.stream_state.sample_idx_to_action_list]
    df['noun'] = [action[1] for action in task.stream_state.sample_idx_to_action_list]

    # Postprocess predictions/labels and remove columns
    pred_list: list[tuple[torch.Tensor, torch.Tensor]] = df['prediction'].tolist()
    label_list: list[tuple[int, int]] = df['label'].tolist()

    """ On fixed final model, collect summary metrics of all samples in stream in hindsight. """
    test_results: dict = task.get_test_metrics(pred_list, label_list, task.loss_fun_unred)
    logger.info(f"Retrieved stream results: \n{pprint.pformat(test_results)}")
    for result_name, result_val in test_results.items():
        assert result_name not in df.columns.tolist(), f"Overwriting column {result_name} in df not allowed"
        df[result_name] = np.nan  # Empty col in csv
        df.iloc[-1, df.columns.get_loc(result_name)] = result_val  # Assign last row the stream avg

    # Save predictions/labels (Backup)
    prediction_backup_dir = os.path.join(run_result_dir, 'predictions')
    makedirs(prediction_backup_dir, exist_ok=True)
    torch.save(df, os.path.join(prediction_backup_dir, f"{run_id}.pth"))

    # Save CSV
    df.drop('prediction', axis=1, inplace=True)  # Clear csv: Can't save lists/tuples to restore
    df.drop('label', axis=1, inplace=True)
    df.to_csv(os.path.join(run_result_dir, f"{run_id}.csv"), index=False)

    # Cleanup process GPU-MEM allocation (Only process context will remain allocated)
    torch.cuda.empty_cache()

    ret = (False, device_id, run_id)
    if mp_queue is not None:
        mp_queue.put(ret)

    logger.info(f"Finished processing: {ret}")
    return ret  # For multiprocessing indicate which resources are free now


if __name__ == "__main__":
    main()
