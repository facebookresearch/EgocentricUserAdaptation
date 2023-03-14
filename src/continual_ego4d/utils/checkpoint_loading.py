import os
import torch

from ego4d.utils import logging
import os.path as osp
import shutil
from pathlib import Path
from ego4d.config.defaults import get_cfg_by_name
from continual_ego4d.utils.misc import makedirs
from pytorch_lightning.core import LightningModule

logger = logging.get_logger(__name__)


class PathHandler:

    def __init__(self,
                 # Define by config
                 cfg=None,

                 # Manually define
                 main_output_dir: str = None,
                 run_group_id: str = None,
                 run_uid: str = None,
                 is_resuming_run: bool = None
                 ):
        """
        We group runs based on the same parent OUTPUT_DIR.
        This grouping is also persisted to WandB logger.
        Runs can have different identifiers (with timestamp), but the group will remain the same when resuming in
        the same OUTPUT_DIR.
        """

        # Init from config
        if cfg is not None:
            self.main_output_dir, run_group_uid, self.is_resuming_run = self.setup_main_output_dir(cfg)

            # Use same group_id as the one resuming from
            self.run_group_uid = "{}_{}".format(
                cfg.METHOD.METHOD_NAME,
                run_group_uid,
            )

            # The single run_id can still be different from OUTPUT_DIR and run_group, e.g. for wandb
            self.run_uid = cfg.RUN_UID

        # Init from args (e.g. for adhoc eval)
        else:
            self.main_output_dir = main_output_dir
            self.run_group_uid = run_group_id
            self.run_uid = run_uid
            self.is_resuming_run = is_resuming_run

            assert self.main_output_dir is not None
            assert self.run_group_uid is not None
            assert self.run_uid is not None
            assert self.is_resuming_run is not None

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
        grid_parent_dir = "NO_GRID"  # For easy resuming with GRID_RESUME_LATEST
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

            subdirs = []
            if os.path.exists(grid_parent_path):
                subdirs = sorted(
                    [user_subdir.name for user_subdir in os.scandir(grid_parent_path) if user_subdir.is_dir()]
                )

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
        makedirs(cfg.OUTPUT_DIR, exist_ok=True, mode=0o777)

        # Copy files to output dir for reproducing
        for reproduce_path in [cfg.PARENT_SCRIPT_FILE_PATH, cfg.CONFIG_FILE_PATH]:
            try:
                shutil.copy2(reproduce_path, cfg.OUTPUT_DIR)
            except PermissionError as e:
                logger.exception(f"File may already exist, skipping copy to: {e}")

        run_id = os.path.basename(os.path.normpath(cfg.OUTPUT_DIR))

        return cfg.OUTPUT_DIR, run_id, is_resuming_run

    def get_experiment_version(self, user_id):
        return self.userid_to_userdir(user_id)

    def get_user_checkpoints_dir(self, user_id=None, include_ckpt_file=None):
        p = osp.join(self.main_output_dir, self.checkpoint_dirname)
        if user_id is not None:
            p = osp.join(p, self.get_experiment_version(user_id))
        if include_ckpt_file is not None:
            p = osp.join(p, include_ckpt_file)
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
            makedirs(p, exist_ok=True, mode=0o777)
        return p

    def get_wandb_group_name(self):
        """ When resuming, reuses the same group as the resuming run."""
        return self.run_group_uid

    def get_user_wandb_name(self, user_id=None):
        """ When resuming, has new identifier. """
        wandb_name = self.run_uid
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


def load_slowfast_model_weights(ckp_path: str, task: LightningModule, load_head: bool):
    """
    Load checkpoint weights into the current lightning module's model.
    We don't change/load other params into the LightningModule (even if checkpoint contains this info).

    :param ckp_path: Path to saved PL or Pytorch dict. Contains 'state_dict' key with model params.
    :param task: LightningModule to load checkpoint weights in.
    :param load_head: Also include the head, or exclude for weight loading.
    """
    logger.info(f"LOADING PRETRAINED MODEL: {ckp_path}")
    assert osp.isfile(ckp_path), f"Ckpt path not existing: {ckp_path}"

    ckpt = torch.load(
        ckp_path,
        map_location=(lambda storage, loc: storage),
    )

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"

    ckp_state_dict = {
        remove_first_module_name(k): v
        for k, v in ckpt[key].items()
    }

    if not load_head:  # Filter
        ckp_state_dict = {k: v for k, v in ckp_state_dict.items() if "head" not in k}

    if hasattr(task.model, 'backbone'):
        lightning_model_to_load = task.model.backbone
    else:
        lightning_model_to_load = task.model

    missing_keys, unexpected_keys = lightning_model_to_load.load_state_dict(
        ckp_state_dict, strict=False
    )
    logger.info(f'PRETRAIN LOADING: \nmissing {missing_keys}\nunexpected {unexpected_keys}')

    # Ensure only head keys is missing (or unexpected e.g. when checkpoint has different classifier head)
    if not load_head:
        assert all(["head" in x for x in missing_keys]), f"Missing keys: {missing_keys}"
        assert all(["head" in x for x in unexpected_keys]), f"Unexpected keys: {unexpected_keys}"
    else:
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    for key in missing_keys:
        logger.info(f"Could not load {key} weights")

    logger.info(f"LOADED LIGHTNING PRETRAIN MODEL: {ckp_path}")


def remove_first_module_name(key):
    return ".".join(key.split(".")[1:])
