import os
import pickle
import numpy as np
import torch

from ego4d.utils import logging
from ego4d.utils.c2_model_loading import get_name_convert_func
import os.path as osp
import shutil
from pathlib import Path
from ego4d.config.defaults import get_cfg_by_name

logger = logging.get_logger(__name__)


class PathHandler:

    def __init__(self, cfg):
        self.main_output_dir, self.is_resuming_run = self.setup_main_output_dir(cfg)
        self.exp_uid = "{}_{}".format(
            cfg.METHOD.METHOD_NAME,
            cfg.RUN_UID,
        )

        # Full paths (user agnostic)
        self.meta_checkpoint_path = osp.join(self.main_output_dir, 'meta_checkpoint.pt')

        # USER DEPENDENT
        # Subdirnames
        self.results_dirname = 'user_logs'  # CSV/stdout
        self.tb_dirname = 'tb'
        self.wandb_dirname = 'wandb'
        self.wandb_project_name = "ContinualUserAdaptation"
        self.csv_dirname = self.results_dirname
        self.stdout_dirname = self.results_dirname
        self.checkpoint_dirname = 'checkpoints'

        # Filenames
        self.user_streamdump_filename = 'stream_info_dump.pth'

    @staticmethod
    def setup_main_output_dir(cfg) -> (str, bool):

        orig_path = Path(cfg.OUTPUT_DIR)  # Insert grid_dir as parent dir to group runs based on grid params
        grid_parent_dir = None
        if cfg.GRID_NODES is not None:
            grid_parent_dir_name = []
            for grid_node in cfg.GRID_NODES.split(','):
                grid_parent_dir_name.append(
                    f"{grid_node}={get_cfg_by_name(cfg, grid_node)}"
                )
            grid_parent_dir_name.sort()  # Make deterministic order
            grid_parent_dir = f"GRID_" + '_'.join(grid_parent_dir_name).replace('.', '-')

        # Resume run if specified, and output to same output dir
        output_dir = None
        is_resuming_run = len(cfg.RESUME_OUTPUT_DIR) > 0
        assert not (cfg.GRID_RESUME_LATEST and is_resuming_run), "Can only specify one of either"
        if is_resuming_run:

            # Check GRID_NODES and current resume-path are matching
            if grid_parent_dir is not None:
                assert Path(cfg.RESUME_OUTPUT_DIR).parent.name == grid_parent_dir, \
                    f"Defined resume path and gridsearch nodes are not matching. " \
                    f"Resume_dir={cfg.RESUME_OUTPUT_DIR}, grid_parent_dir={grid_parent_dir}"

            output_dir = cfg.RESUME_OUTPUT_DIR
            logger.info(f"RESUMING RUN: {output_dir}")

        elif cfg.GRID_RESUME_LATEST:  # Resume grid run
            # Resume latest run that already exists in the grid parent dir
            grid_parent_path = str(orig_path.parent.absolute() / grid_parent_dir)
            subdirs = sorted([user_subdir.name for user_subdir in os.scandir(grid_parent_path) if user_subdir.is_dir()])

            if len(subdirs) > 0:
                latest_subdir = subdirs[-1]
                output_dir = str(orig_path.parent.absolute() / grid_parent_dir / latest_subdir)
                is_resuming_run = True
                logger.info(f"RESUMING RUN (LATEST FROM GRID): {output_dir}")

        # Add gridsearch config nodes to add a grouping gridsearch parent dir
        if output_dir is None:
            if grid_parent_dir is not None:
                output_dir = str(orig_path.parent.absolute() / grid_parent_dir / orig_path.name)
            else:
                output_dir = cfg.OUTPUT_DIR

        # Create dir
        cfg.OUTPUT_DIR = output_dir
        PathHandler.makedirs(cfg.OUTPUT_DIR, exist_ok=True, mode=0o777)

        # Copy files to output dir for reproducing
        for reproduce_path in [cfg.PARENT_SCRIPT_FILE_PATH, cfg.CONFIG_FILE_PATH]:
            shutil.copy2(reproduce_path, cfg.OUTPUT_DIR)

        return cfg.OUTPUT_DIR, is_resuming_run

    def get_experiment_version(self, user_id):
        return self.userid_to_userdir(user_id)

    def get_user_checkpoints_dir(self, user_id=None):
        p = osp.join(self.main_output_dir, self.checkpoint_dirname)
        if user_id is not None:
            p = osp.join(p, self.get_experiment_version(user_id))
        return p

    def get_user_results_dir(self, user_id=None):
        p = osp.join(self.main_output_dir, self.results_dirname)
        if user_id is not None:
            p = osp.join(p, self.get_experiment_version(user_id))
        return p

    def get_user_wandb_dir(self, user_id=None, create=False):
        p = osp.join(self.main_output_dir, self.wandb_dirname)
        if user_id is not None:
            p = osp.join(p, self.get_experiment_version(user_id))
        if create:
            PathHandler.makedirs(p, exist_ok=True, mode=0o777)
        return p

    def get_user_wandb_name(self, user_id=None):
        wandb_name = self.exp_uid
        if user_id is not None:
            wandb_name = f"USER-{user_id}_{wandb_name}"

        return wandb_name

    def get_user_streamdump_file(self, user_id):
        return osp.join(self.get_user_results_dir(user_id), self.user_streamdump_filename)

    @staticmethod
    def userid_to_userdir(user_id: str):
        return f"user_{user_id.replace('.', '-')}"

    @staticmethod
    def userdir_to_userid(user_dirname):
        return user_dirname.replace('user_', '').replace('-', '.')

    def get_processed_users_from_final_dumps(self):
        """Scan user result dirs and find the ones that made the final dumpfile."""
        processed_user_ids = []

        for user_subdir in os.scandir(self.get_user_results_dir(user_id=None)):
            if not user_subdir.is_dir():
                continue
            user_dirname = user_subdir.name

            for user_file in os.scandir(user_subdir.path):
                if not user_file.is_file():
                    continue

                if user_file.name == self.user_streamdump_filename:
                    processed_user_ids.append(self.userdir_to_userid(user_dirname))
                    break

        return processed_user_ids

    @staticmethod
    def makedirs(path, mode=0o777, exist_ok=True):
        """Fix to change umask in order for makedirs to work. """
        try:
            original_umask = os.umask(0)
            os.makedirs(path, mode=mode, exist_ok=exist_ok)
        finally:
            os.umask(original_umask)


def save_meta_state(meta_checkpoint_path, user_id):
    # For easy debug
    meta_exists = osp.isfile(meta_checkpoint_path)
    with open(meta_checkpoint_path, 'a+') as meta_file:
        pretext = "\n" if meta_exists else ""
        meta_file.write(f"{pretext}{user_id}")
    logger.info(f"Saved meta state checkpoint to {meta_checkpoint_path}")
    # sys.stdout.flush()
    # logger.flush()

    # torch.save({'processed_user_ids': processed_user_ids}, meta_checkpoint_path, pickle_protocol=0)


def load_meta_state(meta_checkpoint_path):
    user_ids = []
    with open(meta_checkpoint_path, 'r') as meta_file:
        for line in meta_file.readlines():
            line_s = line.strip()
            if len(line_s) > 0:
                user_ids.append(line_s)

    # torch.load(meta_checkpoint_path)
    return {'processed_user_ids': user_ids}


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


def load_mvit_backbone(ckp_path, task):
    """ Never loads head, only backbone."""
    data_parallel = False  # cfg.NUM_GPUS > 1 # Check this

    ms = task.model.module if data_parallel else task.model
    checkpoint = torch.load(
        ckp_path,
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


def load_slowfast_backbone(ckpt_path, task):
    """ Load slowfast weights into backbone submodule. Never loads head. """
    ckpt = torch.load(
        ckpt_path,
        map_location=(lambda storage, loc: storage),
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


def load_lightning_model(cfg, ckp_path, task, ckpt_task_types):
    """
    Fully load pretrained model, then iterate current model and load_state_dict for all params.
    This allows to keep the hyperparams of our current model, and only adapting the weights.
    The head is loaded based on config.
    """
    # Get pretrained model (try valid types)
    for CheckpointTaskType in ckpt_task_types:
        try:
            pretrained = CheckpointTaskType.load_from_checkpoint(ckp_path)
        except:  # Try the different valid checkpointing types
            continue
        logger.info(f"Loading checkpoint type {CheckpointTaskType}")

    state_dict_for_child_module = {
        child_name: child_state_dict.state_dict()
        for child_name, child_state_dict in pretrained.model.named_children()
    }

    # Iterate current task model and load pretrained
    for child_name, child_module in task.model.named_children():
        if not cfg.CHECKPOINT_LOAD_MODEL_HEAD and "head" in child_name:
            logger.info(f"Skipping head: {child_name}")
            continue

        logger.info(f"Loading in {child_name}")
        state_dict = state_dict_for_child_module[child_name]
        missing_keys, unexpected_keys = child_module.load_state_dict(state_dict)
        assert len(missing_keys) + len(unexpected_keys) == 0


def load_pretrain_model(cfg, ckp_path, task, ckpt_task_types):
    logger.info(f"LOADING PRETRAINED MODEL")

    # For CAFFE backbone
    if cfg.CHECKPOINT_VERSION == "caffe2":
        load_caffe_checkpoint(cfg, ckp_path, task)
        logger.info(f"LOADED CAFFE PRETRAIN MODEL")

    # Pytorch pretrained model state-dict for backbone (Not Lightning), never loads head (Mainly used for LTA backbone)
    elif cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":
        if cfg.MODEL.ARCH == "mvit":
            load_mvit_backbone(cfg.DATA.CHECKPOINT_MODULE_FILE_PATH, task)
            logger.info(f"LOADED MVIT PRETRAIN MODEL")
        elif cfg.MODEL.ARCH == "slow":
            load_slowfast_backbone(cfg.DATA.CHECKPOINT_MODULE_FILE_PATH, task)
            logger.info(f"LOADED SLOWFAST PRETRAIN MODEL")
        else:
            raise NotImplementedError(f"Unkown ARCH in config: {cfg.MODEL.ARCH}")

    # For Lightning Checkpoint
    else:
        load_lightning_model(cfg, ckp_path, task, ckpt_task_types)
        logger.info(f"LOADED LIGHTNING PRETRAIN MODEL")
