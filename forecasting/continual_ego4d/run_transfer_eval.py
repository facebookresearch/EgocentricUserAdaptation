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
import traceback

from continual_ego4d.utils.misc import makedirs
import copy
from itertools import product
import pandas as pd
import numpy as np
from collections import defaultdict
from continual_ego4d.utils.checkpoint_loading import load_slowfast_model_weights, PathHandler
import multiprocessing as mp
from continual_ego4d.run_recog_CL import load_datasets_from_jsons, get_user_ids, get_device_ids

import pprint
import torch
from pytorch_lightning import Trainer, seed_everything

from ego4d.config.defaults import set_cfg_by_name, cfg_add_non_existing_key_vals, convert_flat_dict_to_cfg, get_cfg
from ego4d.utils import logging
from ego4d.utils.parser import load_config, parse_args
from continual_ego4d.tasks.continual_action_recog_task import ContinualMultiTaskClassificationTask
from continual_ego4d.utils.scheduler import SchedulerConfig, RunConfig

import os

from continual_ego4d.utils.models import freeze_full_model, model_trainable_summary
from continual_ego4d.processing.utils import get_group_run_iterator, get_group_names_from_csv

from fvcore.common.config import CfgNode

import wandb

logger = logging.get_logger(__name__)


def parse_wandb_runs(project_name, group_name):
    """ For a given group name in a project, return the user-runs and get the training config. """
    user_to_config = {}

    # summary = final value (excludes NaN rows)
    # history() = gives DF of all values (includes NaN entries for multiple logs per single train/global_step)
    for user_run in get_group_run_iterator(project_name, group_name, finished_runs=True):
        user_id = user_run.config['DATA.COMPUTED_USER_ID']
        user_to_config[user_id] = copy.deepcopy(user_run.config)

    return user_to_config


def main():
    """ Iterate users and aggregate. """
    print("Starting eval")
    args = parse_args()
    eval_cfg = load_config(args)

    if eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH is not None:
        assert os.path.isfile(eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH), \
            f"Non-existing csv: {eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH}"
        group_names = get_group_names_from_csv(eval_cfg.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH)

        if eval_cfg.TRANSFER_EVAL.CSV_RANGE is not None:
            start_idx, end_idx = eval_cfg.TRANSFER_EVAL.CSV_RANGE
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
    outputdirs = [user_config['OUTPUT_DIR'] for user_config in user_to_flat_cfg.values()]
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
    main_parent_dir = os.path.join(train_group_outputdir, 'transfer_eval')
    eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR = os.path.join(main_parent_dir, 'results')
    eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_LOGDIR = os.path.join(main_parent_dir, 'logs')

    makedirs(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_RESULTDIR)
    makedirs(eval_cfg.TRANSFER_EVAL.COMPUTED_EVAL_LOGDIR)

    # Main process logging
    logging.setup_logging(main_parent_dir, host_name='MASTER', overwrite_logfile=False)
    logger.info(f"Starting main script with OUTPUT_DIR={main_parent_dir}")

    # Dataset lists from json / users to process
    user_datasets, pretrain_dataset = load_datasets_from_jsons(train_cfg, return_pretrain=True)
    processed_trained_user_ids, all_trained_user_ids = get_user_ids(train_cfg, user_datasets, train_path_handler)
    assert len(processed_trained_user_ids) == len(all_trained_user_ids) == eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS, \
        f"Not all users were processed in training: {set(all_trained_user_ids) - set(processed_trained_user_ids)}\n" \
        f"Expected {eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS} users."
    user_datasets['pretrain'] = pretrain_dataset  # Add (AFTER CHECKING USER IDS)

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
    for (model_userid, stream_userid) in modeluser_streamuser_pairs:
        run_id = get_pair_id(model_userid, stream_userid)

        # STREAM config
        user_train_stream_cfg = copy.deepcopy(train_cfg)
        user_train_stream_cfg.DATA.COMPUTED_USER_ID = stream_userid
        user_train_stream_cfg.DATA.COMPUTED_USER_DS_ENTRIES = user_datasets[stream_userid]
        user_train_stream_cfg.COMPUTED_USER_DUMP_FILE = train_path_handler.get_user_streamdump_file(stream_userid)

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
                    eval_cfg,  # For
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
    postprocess_csv_files(eval_cfg, modeluser_streamuser_pairs, project_name, group_name)


def postprocess_csv_files(eval_cfg, modeluser_streamuser_pairs, project_name, group_name):
    """
    Get Avg loss per csv entry for

    - verb/noun/action.
    - absolute loss + relative to pretrain (AG)

    2d Matrix is dims:
    - rows (y-axis): model users
    - cols (x-axis): stream users
    """

    def init_matrix():
        return [[init_val for _ in range(len(streamusers))] for _ in range(len(modelusers))]

    init_val = np.inf

    modelusers = sorted(list(
        set([entry_pair[0] for entry_pair in modeluser_streamuser_pairs]) - {'pretrain'}
    ))
    streamusers = sorted(list(
        set([entry_pair[1] for entry_pair in modeluser_streamuser_pairs]) - {'pretrain'}
    ))

    # Mapping of idx in matrix to
    modeluser_to_row = {user: idx for idx, user in enumerate(modelusers)}
    streamuser_to_col = {user: idx for idx, user in enumerate(streamusers)}

    matrices = {'avg_loss': {}, 'avg_AG': {}}
    diagonal_AGs = defaultdict(list)
    for model_userid, stream_userid in modeluser_streamuser_pairs:
        if 'pretrain' in [model_userid, stream_userid]:  # TODO skipping pretrain cols/rows for now
            logger.info(f"SKIPPING: model_userid={model_userid}, stream_userid={stream_userid}")
            continue
        logger.info(f"Processing: model_userid={model_userid}, stream_userid={stream_userid}")

        df = get_df(eval_cfg, model_userid, stream_userid)
        pretrain_df = get_df(eval_cfg, 'pretrain', stream_userid)  # On pretrain model, but same stream

        # Action/verb/noun
        assert len(df.columns) == 3
        for loss_version in df.columns:

            # Init matrix
            if loss_version not in matrices['avg_loss']:
                matrices['avg_loss'][loss_version] = init_matrix()

            if loss_version not in matrices['avg_AG']:
                matrices['avg_AG'][loss_version] = init_matrix()

            # Process values
            avg_stream_loss = df[loss_version].mean()  # Avg over nb samples to normalize stream length
            avg_stream_AG = (pretrain_df[loss_version] - df[loss_version]).mean()

            # Fill in value
            row = modeluser_to_row[model_userid]
            col = streamuser_to_col[stream_userid]
            matrices['avg_loss'][loss_version][row][col] = avg_stream_loss
            matrices['avg_AG'][loss_version][row][col] = avg_stream_AG

            if model_userid == stream_userid and model_userid != 'pretrain':
                diagonal_AGs[loss_version].append((model_userid, avg_stream_AG))

    logger.info(f"Aggregated all results in matrices: \n{pprint.pformat(matrices, depth=4)}")

    # Iterate runs over group and upload heatmaps
    logger.info(f"Uploading to WandB runs in group")
    for user_run in get_group_run_iterator(project_name, group_name, finished_runs=True):

        # Log avg history
        for loss_version, diagonal_avgAGs in diagonal_AGs.items():
            assert len(diagonal_avgAGs) == len(modelusers) == len(
                streamusers) == eval_cfg.TRANSFER_EVAL.NUM_EXPECTED_USERS
            avg_stream_AGs_df = pd.DataFrame(diagonal_avgAGs)[1]  # Only the values, not the user-ids

            # Avg over all user-streams (avg of stream avgs)
            user_avg_AG_history = avg_stream_AGs_df.mean()
            user_SE_AG_history = avg_stream_AGs_df.sem()  # Get exact same result as current batch AG

            new_name_prefix = f"adhoc_users_aggregate_history/{loss_version}/avg_history_AG"
            user_run.summary[f"{new_name_prefix}/mean"] = user_avg_AG_history
            user_run.summary[f"{new_name_prefix}/SE"] = user_SE_AG_history

        user_run.summary.update()

        # Plot Matrices
        for heatmap_metric, subloss_dict in matrices.items():
            for loss_version, matrix in subloss_dict.items():
                logger.info(f"Uploading for user_run={user_run.config['DATA.COMPUTED_USER_ID']}")

                user_run.summary[f"TRANSFER_MATRIX/{heatmap_metric}/{loss_version}"] = matrix
                user_run.summary[f"TRANSFER_MATRIX/x_labels/{heatmap_metric}/{loss_version}/stream_users"] = streamusers
                user_run.summary[f"TRANSFER_MATRIX/y_labels/{heatmap_metric}/{loss_version}/model_users"] = modelusers
                user_run.summary.update()


def get_df(eval_cfg, model_userid, stream_userid):
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

    # Choose task type based on config.
    logger.info("Starting init Task")
    if user_train_cfg.DATA.TASK == "continual_classification":
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
        task.configure_head()  # Add outputhead masker so checkpoint can load the params
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
    sample_idx_to_loss_dict_list: list[dict] = trainer.predict(task)

    # Flatten
    sample_idx_to_losses_dict = {k: v for iter_dict in sample_idx_to_loss_dict_list for k, v in iter_dict.items()}

    # Dataframe for csv
    df = pd.DataFrame(sample_idx_to_losses_dict).transpose()
    """
        e.g.
           pred_action_batch/loss  pred_verb_batch/loss  pred_noun_batch/loss
    0               63.166710         -0.000000e+00             63.166710
    1               94.299492         -0.000000e+00             94.299492
    2               56.985905          4.014163e-04             56.985504
    3               56.100876          2.384186e-07             56.100876
    4               59.698318          1.567108e-02             59.682648
    ...

    """

    # Save
    df.to_csv(os.path.join(run_result_dir, f"{run_id}.csv"), index=False)

    # Cleanup process GPU-MEM allocation (Only process context will remain allocated)
    torch.cuda.empty_cache()

    ret = (False, device_id, run_id)
    if mp_queue is not None:
        mp_queue.put(ret)

    return ret  # For multiprocessing indicate which resources are free now


if __name__ == "__main__":
    main()
