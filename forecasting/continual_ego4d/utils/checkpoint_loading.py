import pickle
import numpy as np
import torch

from ego4d.utils import logging
from ego4d.utils.c2_model_loading import get_name_convert_func

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
