"""

Train vs test
---------------------------------------------
TRAIN/EVAL are separate, set via: cfg.TRAIN.ENABLE: True
Can do both:     if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:



Pretraining
---------------------------------------------
cfg:
    CHECKPOINT_LOAD_MODEL_HEAD False \ -> Useful for using pretrained feats.
    MODEL.FREEZE_BACKBONE True \ -> Useful for using pretrained feat-net and freezing it
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \ -> Checkpoint for the MVIT or SlowFAST video net
    FORECASTING.AGGREGATOR "" \
    FORECASTING.DECODER


Flow
---------------------------------------------
run_lta.py: trainer.fit(task: LongTermAnticipationTask)
-> Select task (LTA/short-term recognition/...), initialize it with the config, restore checkpoint,
-> and launch pytorch ligthning trainer to fit the Task.

\ego4d\tasks\long_term_anticipation.py: class LongTermAnticipationTask(VideoTask):
-> Define {train/val/test}_{step/epoch_end}

\ego4d\tasks\video_task.py: class VideoTask(LightningModule)
-> The cfg is used all the way: cfg.MODEL.NUM_CLASSES
-> Build model/optimizer/dataloaders for the task/additional hooks (e.g. on_backwards)


"""

import os
import pickle
import pprint
import sys
import copy
import pathlib
import shutil
import submitit

from ego4d.utils import logging
import numpy as np
import pytorch_lightning
import torch
from continual_ego4d.tasks.continual_action_recog import ContinualMultiTaskClassificationTask
from ego4d.utils.c2_model_loading import get_name_convert_func
from ego4d.utils.misc import gpu_mem_usage
from ego4d.utils.parser import load_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from continual_ego4d.datasets.continual_action_recog_dataset import get_user_to_dataset_dict

from scripts.slurm import copy_and_run_with_config, init_and_run

logger = logging.get_logger(__name__)


def load_caffe_checkpoint(cfg, ckp_path, task):
    with open(ckp_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    state_dict = data["blobs"]
    fun = get_name_convert_func()
    state_dict = {
        fun(k): torch.from_numpy(np.array(v))
        for k, v in state_dict.items()
        if "momentum" not in k and "lr" not in k and "model_iter" not in k
    }

    if not cfg.CHECKPOINT_LOAD_MODEL_HEAD:
        state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
    print(task.model.load_state_dict(state_dict, strict=False))
    print(f"Checkpoint {ckp_path} loaded")


def load_mvit_checkpoint(cfg, ckp_path, task):
    data_parallel = False  # cfg.NUM_GPUS > 1 # Check this

    ms = task.model.module if data_parallel else task.model
    path = ckp_path if len(ckp_path) > 0 else cfg.DATA.CHECKPOINT_MODULE_FILE_PATH
    checkpoint = torch.load(
        path,
        map_location=lambda storage, loc: storage,
    )
    remove_model = lambda x: x[6:]
    if "model_state" in checkpoint.keys():
        pre_train_dict = checkpoint["model_state"]
    else:
        pre_train_dict = checkpoint["state_dict"]
        pre_train_dict = {remove_model(k): v for (k, v) in pre_train_dict.items()}

    model_dict = ms.state_dict()

    remove_prefix = lambda x: x[9:] if "backbone." in x else x
    model_dict = {remove_prefix(key): value for (key, value) in model_dict.items()}

    # Match pre-trained weights that have same shape as current model.
    pre_train_dict_match = {
        k: v
        for k, v in pre_train_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    # Weights that do not have match from the pre-trained model.
    not_load_layers = [
        k
        for k in model_dict.keys()
        if k not in pre_train_dict_match.keys()
    ]
    not_used_weights = [
        k
        for k in pre_train_dict.keys()
        if k not in pre_train_dict_match.keys()
    ]
    # Log weights that are not loaded with the pre-trained weights.
    if not_load_layers:
        for k in not_load_layers:
            logger.info("Network weights {} not loaded.".format(k))

    if not_used_weights:
        for k in not_used_weights:
            logger.info("Pretrained weights {} not being used.".format(k))

    if len(not_load_layers) == 0:
        print("Loaded all layer weights! Every. Single. One.")
    # Load pre-trained weights.
    ms.load_state_dict(pre_train_dict_match, strict=False)


def load_slowfast_checkpoint(cfg, task):
    # Load slowfast weights into backbone submodule
    ckpt = torch.load(
        cfg.DATA.CHECKPOINT_MODULE_FILE_PATH,
        map_location=lambda storage, loc: storage,
    )

    def remove_first_module(key):
        return ".".join(key.split(".")[1:])

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"

    state_dict = {
        remove_first_module(k): v
        for k, v in ckpt[key].items()
        if "head" not in k
    }

    if hasattr(task.model, 'backbone'):
        backbone = task.model.backbone
    else:
        backbone = task.model

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)

    # Ensure only head key is missing.w
    assert len(unexpected_keys) == 0
    assert all(["head" in x for x in missing_keys])

    for key in missing_keys:
        logger.info(f"Could not load {key} weights")


def load_any_checkpoint(cfg, ckp_path, task):
    # Get pretrained model
    pretrained = task.load_from_checkpoint(ckp_path)
    state_dict_for_child_module = {
        child_name: child_state_dict.state_dict()
        for child_name, child_state_dict in pretrained.model.named_children()
    }

    # Iterate current task model and load pretrained
    for child_name, child_module in task.model.named_children():
        if not cfg.CHECKPOINT_LOAD_MODEL_HEAD and "head" in child_name:
            continue

        logger.info(f"Loading in {child_name}")
        state_dict = state_dict_for_child_module[child_name]
        missing_keys, unexpected_keys = child_module.load_state_dict(state_dict)
        assert len(missing_keys) + len(unexpected_keys) == 0


def load_checkpoint(cfg, ckp_path, task):
    if cfg.CHECKPOINT_VERSION == "caffe2":
        load_caffe_checkpoint(cfg, ckp_path, task)

    elif cfg.MODEL.ARCH == "mvit" and cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":
        load_mvit_checkpoint(cfg, ckp_path, task)

    elif cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":
        load_slowfast_checkpoint(cfg, ckp_path)

    else:  # Load all child modules except for "head" if CHECKPOINT_LOAD_MODEL_HEAD is False.
        load_any_checkpoint(cfg, ckp_path, task)


def main(cfg):
    """ Iterate users and aggregate. """
    # Select user-split file based on config: Either train or test:
    assert cfg.DATA.USER_SUBSET in ['train', 'test'], \
        "Choose either 'train' or 'test' mode, TRAIN is the user-subset for hyperparam tuning, TEST is held-out final eval"
    usersplit_annotations = get_user_to_dataset_dict(cfg.DATA.PATH_TO_DATA_DIR, cfg.DATA.USER_SUBSET)

    # Checkpoint resume vars and checks
    if len(cfg.CHECKPOINT_USER_ID) > 0:  # Skip until first user ckp encountered (if defined)
        assert len(cfg.CHECKPOINT_FILE_PATH) > 0, "Must defined ckp_path "
        assert 'last.ckpt' in cfg.CHECKPOINT_FILE_PATH, \
            f'ckp_path is not last save (should be last.ckpt): {cfg.CHECKPOINT_FILE_PATH}'  # TODO last_userid_ckpt or in user_id dir
    cfg.LOADED_USER_CHECKPOINT = False

    # Iterate user datasets
    user_ids_s = sorted([u for u in usersplit_annotations.keys()])  # Deterministic user order
    for user_id in user_ids_s:
        cfg.DATA.USER_ID = user_id
        cfg.DATA.USER_DS_ENTRIES = usersplit_annotations[user_id]
        online_adaptation_single_user(cfg, user_id)

    # TODO aggregate metrics over user dumps


def online_adaptation_single_user(cfg, user_id):
    """ Run single user sequentially. """
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    assert cfg.DATA.TASK == "classification", "Only action recognition supported, no LTA"
    task = ContinualMultiTaskClassificationTask(cfg)

    # User-based Checkpointing
    if len(cfg.CHECKPOINT_USER_ID) > 0 and not cfg.LOADED_USER_CHECKPOINT:
        if user_id != cfg.CHECKPOINT_USER_ID:
            logger.info(f"Skipping user {user_id}, until found checkpoint-resuming user: {cfg.CHECKPOINT_USER_ID}")
            return
        else:  # Load model from checkpoint if checkpoint file path is given.
            logger.info(f"Loading ckpt for user: {user_id}")
            load_checkpoint(cfg, cfg.CHECKPOINT_FILE_PATH, task)
            cfg.LOADED_USER_CHECKPOINT = True  # Set state as resumed from this user

    # Save every N global training steps + save on the end of training (end of epoch)
    # Save_last will save an overwriting copy that we can easily resume from again
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=cfg.CHECKPOINT_step_freq, save_on_train_epoch_end=True, save_last=True, save_top_k=1
    )

    # Additional callbacks/logging on top of the default Tensorboard logger
    # TB logger is passed by default, stdout 'logger' in this script is a handler from the main Py-Lightning logger
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": [checkpoint_callback]}

    # There are no validation/testing phases!
    trainer = Trainer(
        gpus=1,  # cfg.NUM_GPUS
        num_nodes=1,  # cfg.NUM_SHARDS
        accelerator="gpu",  # cfg.SOLVER.ACCELERATOR, only single device for now
        max_epochs=1,  # cfg.SOLVER.MAX_EPOCH
        num_sanity_val_steps=0,  # Sanity check before starting actual training to make sure validation works
        benchmark=True,
        log_gpu_memory="min_max",
        replace_sampler_ddp=False,  # Disable to use own custom sampler
        fast_dev_run=cfg.FAST_DEV_RUN,  # Debug: Run defined batches (int) for train/val/test
        default_root_dir=cfg.OUTPUT_DIR,  # Default path for logs and weights when no logger/ckpt_callback passed
        # plugins=DDPPlugin(find_unused_parameters=False),
        **args,
    )

    trainer.fit(task, val_dataloaders=None)  # Skip validation


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    if args.on_cluster:
        copy_and_run_with_config(
            main,
            cfg,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="devlab,learnlab,learnfair",
            gpus_per_node=cfg.NUM_GPUS,
            ntasks_per_node=cfg.NUM_GPUS,
            cpus_per_task=10,
            mem="470GB",
            nodes=cfg.NUM_SHARDS,
            constraint="volta32gb",
        )
    else:  # local
        main(cfg)
