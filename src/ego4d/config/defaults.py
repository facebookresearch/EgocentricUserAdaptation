#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
import copy

from continual_ego4d.utils.misc import SoftCfgNode as CfgNode  # Wrapper

# Relative path to reproduce scripts
PROJECT_ROOT = "../../.."  # TODO add your absolute project path

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# Add paths
_C.CONFIG_FILE_PATH = ""
_C.PARENT_SCRIPT_FILE_PATH = ""
_C.RUN_UID = ""

# ---------------------------------------------------------------------------- #
# RUN MODE options
# ---------------------------------------------------------------------------- #
# Option used for CL task, skipping training and only evaluating the loaded pretrain model
# This ensures we use the same dataloaders and the exact same stream is used for evaluation.
# The stream is defined by cfg.DATA.USER_SUBSET: {train,test,pretrain}
_C.STREAM_EVAL_ONLY = False

# ---------------------------------------------------------------------------- #
# CONTEXT ADAPTATION options
# ---------------------------------------------------------------------------- #
_C.CONTEXT_ADAPT = CfgNode()
_C.CONTEXT_ADAPT.MEM_SIZE = 10
_C.CONTEXT_ADAPT.WRAPS_METHOD = "Finetuning"
_C.CONTEXT_ADAPT.HEAD_MODULE = 'GRU'  # Or 'attention'
_C.CONTEXT_ADAPT.GRU_LAYERS = 1
_C.CONTEXT_ADAPT.GRU_HIDDEN_SIZE = 64

# ---------------------------------------------------------------------------- #
# TRANSFER EVAL options
# ---------------------------------------------------------------------------- #
_C.TRANSFER_EVAL = CfgNode()
_C.TRANSFER_EVAL.WANDB_PROJECT_NAME = "ContinualUserAdaptation"

# TODO set below your resulting WandB group-names for train/test in 'reproduce/pretrain/eval_user_stream_performance', e.g. FixedNetwork_2023-03-13_15-28-42_UIDb2b1f8a3-ee28-4589-9dee-2747cf8f750a
_C.TRANSFER_EVAL.PRETRAIN_TRAIN_USERS_GROUP_WANDB = None
_C.TRANSFER_EVAL.PRETRAIN_TEST_USERS_GROUP_WANDB = None
_C.TRANSFER_EVAL.PRETRAIN_REFERENCE_GROUP_WANDB = 'train'  # or 'test'. Choose the train or test pretrain group for the reference performance (OAG/HAG).

_C.TRANSFER_EVAL.WANDB_GROUP_TO_EVAL = ""
_C.TRANSFER_EVAL.DIAGONAL_ONLY = True
_C.TRANSFER_EVAL.INCLUDE_PRETRAIN_STREAM = False
_C.TRANSFER_EVAL.INCLUDE_PRETRAIN_MODEL = True
_C.TRANSFER_EVAL.NUM_EXPECTED_USERS = 10
_C.TRANSFER_EVAL.WANDB_GROUPS_TO_EVAL_CSV_PATH = None
_C.TRANSFER_EVAL.CSV_RANGE = (0, None)  # all

# ---------------------------------------------------------------------------- #
# Analyze options.
# ---------------------------------------------------------------------------- #
_C.ANALYZE_STREAM = CfgNode()
_C.ANALYZE_STREAM.LOOKBACK_STRIDE_ITER = 1  # How many iters back to compare window with
_C.ANALYZE_STREAM.WINDOW_SIZE_SAMPLES = 10  # How many iters back to compare window with
_C.ANALYZE_STREAM.PARENT_DIR_FEAT_DUMP = ""  # Dir to take dumped feats from

# ---------------------------------------------------------------------------- #
# DEBUG options
# ---------------------------------------------------------------------------- #

# Run 1 train, val and test batch for debugging
_C.FAST_DEV_RUN = False

# For debugging on user-streams: How much data to consider max per user
_C.FAST_DEV_DATA_CUTOFF = 5

# ---------------------------------------------------------------------------- #
# GRIDSEARCH options
# ---------------------------------------------------------------------------- #
_C.GRID_NODES = None  # Add nodes that we gridsearch over to output path
_C.GRID_RESUME_LATEST = False  # Resume from latest in grid

_C.NUM_USERS_PER_DEVICE = 1  # How many user-processes per gpu
_C.USER_SELECTION = None  # Only process specific users, comma-seperated str

# ---------------------------------------------------------------------------- #
# CL STREAM options
# ---------------------------------------------------------------------------- #
# Include actions that have not been seen during pretraining
_C.ENABLE_FEW_SHOT = False

# ---------------------------------------------------------------------------- #
# WANDB options
# ---------------------------------------------------------------------------- #
_C.WANDB = CfgNode()
_C.WANDB.TAGS = None  # Split based on comma
_C.WANDB.MODE = "online"  # Split based on comma

# ---------------------------------------------------------------------------- #
# METHOD options
# ---------------------------------------------------------------------------- #
_C.METHOD = CfgNode()
_C.METHOD.METHOD_NAME = "Finetuning"
_C.METHOD.ANALYZE_GRADS_WINDOW = False  # Compare current grad with grad in prev
_C.METHOD.MAX_ANALYZE_GRADS_WINDOW_SIZE = 10  # If ANALYZE_GRADS_WINDOW=True, how many grads to go back to

_C.METHOD.REPLAY = CfgNode()
_C.METHOD.REPLAY.MEMORY_SIZE_SAMPLES = 1000
_C.METHOD.REPLAY.STORAGE_POLICY = "reservoir_stream"
_C.METHOD.REPLAY.ANALYZE_GRADS = False
_C.METHOD.REPLAY.RESAMPLE_MULTI_ITER = True
_C.METHOD.REPLAY.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

# BN epsilon.
_C.BN.EPSILON = 1e-5

# BN momentum.
_C.BN.MOMENTUM = 0.1

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SplitBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm3d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1

# ---------------------------------------------------------------------------- #
# PREDICT phase (before training) options.
# ---------------------------------------------------------------------------- #
_C.PREDICT_PHASE = CfgNode()

_C.PREDICT_PHASE.BATCH_SIZE = 10
_C.PREDICT_PHASE.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# CONTINUAL EVAL options.
# ---------------------------------------------------------------------------- #
_C.CONTINUAL_EVAL = CfgNode()

# Batch size for evaluation after each model prediction during training.
_C.CONTINUAL_EVAL.BATCH_SIZE = 8

# For continual eval dataloaders inference (no grads)
_C.CONTINUAL_EVAL.NUM_WORKERS = 8

# How much to sample from future/past stream
_C.CONTINUAL_EVAL.FUTURE_SAMPLE_CAPACITY = 64
_C.CONTINUAL_EVAL.PAST_SAMPLE_CAPACITY = 64
_C.CONTINUAL_EVAL.PAST_SAMPLER_MODE = 'windowed'

# Every how many update steps should evaluate
_C.CONTINUAL_EVAL.FREQ = -1

# When to plot figures for metrics
_C.CONTINUAL_EVAL.PLOTTING_FREQ = -1

# Measure online the OAG (First triggers prediction phase before training)
_C.CONTINUAL_EVAL.ONLINE_OAG = False

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "Kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 1

_C.TRAIN.INNER_LOOP_ITERS = 1

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# If True, adds final activation to model when evaluating
_C.TEST.NO_ACT = True

_C.TEST.EVAL_VAL = False

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = True

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"

# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [1, 2, 2],
    # Res3
    [1, 2, 2],
    # Res4
    [1, 2, 2],
    # Res5
    [1, 2, 2],
]

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slow"

# Model name
_C.MODEL.MODEL_NAME = "ResNet"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = [400]

# The number of verbs to predict for the model
_C.MODEL.NUM_VERBS = 125

# The number of nouns to predict for the model
_C.MODEL.NUM_NOUNS = 352

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Verb loss function.
_C.MODEL.VERB_LOSS_FUNC = "cross_entropy"

# Next-Active-Object classification loss function.
_C.MODEL.NAO_LOSS_FUNC = "bce_logit"

# TTC loss function.
_C.MODEL.TTC_LOSS_FUNC = "smooth_l1"

# STA loss weights.
_C.MODEL.STA_LOSS_WEIGHTS = [1, 1, 1]  # VERB, NOUN, TTI

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["c2d", "i3d", "slow"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation layer for the output verb head.
_C.MODEL.HEAD_VERB_ACT = "softmax"

# Activation layer for the output tti head.
_C.MODEL.HEAD_TTC_ACT = "softplus"
# TODO: LTA: _C.MODEL.HEAD_TTC_ACT = "scaled_sigmoid"

# The maximum TTC value the model should predict
# TODO: LTA: _C.MODEL.TTC_SCALE = 2

# Activation layer for the output noun head.
# TODO: LTA: _C.MODEL.HEAD_NAO_ACT = "sigmoid"

# Size of feature for each input clip (right three dims will be pooled with all
# other input clips).
_C.MODEL.MULTI_INPUT_FEATURES = 2048

# If True, freezes clip feature backbone.
_C.MODEL.FREEZE_BACKBONE = False  # Freeze feat extractor
_C.MODEL.FREEZE_MODEL = False  # Freeze entire model
_C.MODEL.FREEZE_HEAD = False  # Freeze classifier

# Transformer number of heads.
_C.MODEL.TRANSFORMER_ENCODER_HEADS = 8

# Transformer depth.
_C.MODEL.TRANSFORMER_ENCODER_LAYERS = 6

_C.MODEL.TRANSFORMER_DECODER_TGT_MASK = True

_C.MODEL.TRANSFORMER_FROM_PRETRAIN = True

_C.MODEL.TRANSFORMER_NOISE_TYPE = "masking"

_C.MODEL.TRANSFORMER_NOISE_PROB = 0.5

# Transformer subsequent src mask padding. If True, model encodes
# bidirectionally.
_C.MODEL.TRANSFORMER_ENCODER_SRC_MASK = False

# Transformer initial weight std dev.
_C.MODEL.TRANSFORMER_INIT_STD = 0.2

# LSTM hidden dimension size.
_C.MODEL.LSTM_HIDDEN_DIM = 2048

# LSTM depth.
_C.MODEL.LSTM_NUM_LAYERS = 1

_C.MODEL.BEAM_WIDTH = 5

# Teacher forcing probability for transformer and LSTM models.
_C.MODEL.TEACHER_FORCING_RATIO = 0.8

# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5

# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()

# Options include `conv`, `max` or "conv_unshared"
_C.MVIT.MODE = "conv"

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = True

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [3, 7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [2, 4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [1, 3, 3]

# If True, use 2d patch, otherwise use 3d patch.
_C.MVIT.PATCH_2D = False

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Normalization layer for the transformer. Only layernorm is supported now.
_C.MVIT.NORM = "layernorm"

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = []

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# If not None, overwrite the KV_KERNEL and Q_KERNEL size with POOL_KVQ_CONV_SIZ.
# Otherwise the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
_C.MVIT.POOL_KVQ_KERNEL = None

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = True

# If True, use norm after stem.
_C.MVIT.NORM_STEM = False

# If True, perform separate positional embedding.
_C.MVIT.SEP_POS_EMBED = False

# Dropout rate for the MViT backbone.
_C.MVIT.DROPOUT_RATE = 0.0

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = True

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = False

# If True, use relative positional embedding for temporal dimentions
_C.MVIT.REL_POS_TEMPORAL = False

_C.MVIT.POOL_FIRST = False

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

###############################################################################
# CONTINUAL EgoAdapt SPECIFIC

# User data splits for continual Ego4d
_C.DATA.USER_SUBSET = 'train'  # train or test split for users.
_C.DATA.PATH_TO_DATA_SPLIT_JSON = CfgNode()
_C.DATA.PATH_TO_DATA_SPLIT_JSON.PRETRAIN_SPLIT = f'{PROJECT_ROOT}/data/EgoAdapt/usersplits/ego4d_LTA_pretrain_incl_nanusers_usersplit_148users.json'
_C.DATA.PATH_TO_DATA_SPLIT_JSON.TRAIN_SPLIT = f'{PROJECT_ROOT}/data/EgoAdapt/usersplits/ego4d_LTA_train_usersplit_10users.json'
_C.DATA.PATH_TO_DATA_SPLIT_JSON.TEST_SPLIT = f'{PROJECT_ROOT}/data/EgoAdapt/usersplits/ego4d_LTA_test_usersplit_40users.json'

# Disable to only load labels (e.g. for stream analysis)
_C.DATA.RETURN_VIDEO = True

# Enables shuffling the dataset on creation, without changing loader idxs.
_C.DATA.SHUFFLE_DS_ORDER = False

# CL: Stride for next observed frame in sequential data stream (in a single sequential clip-video)
#  If batch size > STRIDE, then next step will contain seen samples (Although shifted)
_C.DATA.SEQ_OBSERVED_FRAME_STRIDE = None  # BY DEFAULT specified as None: a full new batch is observed

#####################################################################
# PRETRAIN SPECIFIC

# Default ego4d pretraining path, using default names for json files (overwrite by PATH_TO_DATA_FILE)
_C.DATA.PATH_TO_DATA_DIR = f'{PROJECT_ROOT}/data/Ego4D/v1/annotations'

# Custom data JSON path names for pretraining Ego4d (Train on pretrain data, and validate on U_train users)
_C.DATA.PATH_TO_DATA_FILE = CfgNode()
_C.DATA.PATH_TO_DATA_FILE.TRAIN = f'{PROJECT_ROOT}/data/EgoAdapt/usersplits/ego4d_LTA_pretrain_incl_nanusers_usersplit_148users.json'
_C.DATA.PATH_TO_DATA_FILE.VAL = f'{PROJECT_ROOT}/data/EgoAdapt/usersplits/ego4d_LTA_train_usersplit_10users.json'
_C.DATA.PATH_TO_DATA_FILE.TEST = None

#####################################################################
# COMMON

# Video path prefix: parent path of the .mp4 videos.
_C.DATA.PATH_PREFIX = f'{PROJECT_ROOT}/data/Ego4D/v1/clips'

# Model head path if any
_C.DATA.CHECKPOINT_MODULE_FILE_PATH = ""  # ego4d/models/

# The spatial crop size of the input clip.
_C.DATA.CROP_SIZE = 224

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculate the map as metric.
_C.DATA.TASK = "single-label"
# continual_classification: Learn in stream of data per user
# iid_classification: Use the exact same stream but shuffled (iid).
# classification: Standard ego4d iid setup. (one clip sampled per annotation).

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

_C.SOLVER.CLASSIFIER_LR = None  # Overwrite classifier LR

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 1

# Momentum.
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MOMENTUM_HEAD = -1.0  # If defined, overwrites default momentum
_C.SOLVER.MOMENTUM_FEAT = -1.0

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# Gradually warm up the SOLVER.BASE_LR over this number of steps.
_C.SOLVER.WARMUP_STEPS = 1000

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Which PyTorch Lightning accelerator to use
_C.SOLVER.ACCELERATOR = "ddp"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Path to the checkpoint to load the initial weight.
_C.CHECKPOINT_FILE_PATH = ""  # set pretrain model here to start EgoAdapt learning of user-streams
_C.CHECKPOINT_PATH_FORMAT_FOR_USER = False  # if ckpt file path is fmt for user_id to fill in, e.g. "{}".format(user_id)

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Can use a comma-split string to specify which GPU devices to use only. None sets to all.
_C.GPU_IDS = None

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Specify with outputdir if want to resume run, e.g. /path/to/logs/run_id/
_C.RESUME_OUTPUT_DIR = ""

# Path to the output results.pkl
_C.RESULTS_PKL = ""

# Path to the output results.json
_C.RESULTS_JSON = ""

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Whether to enable logging
_C.ENABLE_LOGGING = True

# Log gradient distributions at period. Don't log if None.
_C.LOG_GRADIENT_PERIOD = -1

# Whether the checkpoint follows the caffe2 format
_C.CHECKPOINT_VERSION = ""

# Whether to load model head or not. Useful for loading pretrained models.
_C.CHECKPOINT_LOAD_MODEL_HEAD = True

# Whether or not to run on fblearner
_C.FBLEARNER = False

# Make checkpoint every N iterations
_C.CHECKPOINT_step_freq = 300

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# ---------------------------------------------------------------------------- #
# Detection options.
# ---------------------------------------------------------------------------- #
_C.DETECTION = CfgNode()

# Aligned version of RoI. More details can be found at slowfast/models/head_helper.py
_C.DETECTION.ALIGNED = True

# Spatial scale factor.
_C.DETECTION.SPATIAL_SCALE_FACTOR = 16

# RoI tranformation resolution.
_C.DETECTION.ROI_XFORM_RESOLUTION = 7

# -----------------------------------------------------------------------------
# Forecasting options (LTA + STA)
# -----------------------------------------------------------------------------

_C.FORECASTING = CfgNode()
# _C.FORECASTING.BACKBONE = "SlowFast" # _C.MODEL.ARCH also has this info

# Concat Aggregator
_C.FORECASTING.AGGREGATOR = "ConcatAggregator"

# MultiHead Decoder
_C.FORECASTING.DECODER = "MultiHeadDecoder"

# The number of future actions to return from the Epic Kitchen src dataset.
_C.FORECASTING.NUM_ACTIONS_TO_PREDICT = 1
# TODO: LTA: _C.FORECASTING.NUM_ACTIONS_TO_PREDICT = 20

# The number of future action sequences to predict.
_C.FORECASTING.NUM_SEQUENCES_TO_PREDICT = 5

# Number of (~2s) input clips before the chosen action (only supported by src)
_C.FORECASTING.NUM_INPUT_CLIPS = 1  # Standard 2 in Ego4D

# -----------------------------------------------------------------------------
# AVA Dataset options
# -----------------------------------------------------------------------------
_C.AVA = CfgNode()

# Directory path of frames.
_C.AVA.FRAME_DIR = "/mnt/fair-flash3-east/ava_trainval_frames.img/"

# Directory path for files of frame lists.
_C.AVA.FRAME_LIST_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Directory path for annotation files.
_C.AVA.ANNOTATION_DIR = (
    "/mnt/vol/gfsai-flash3-east/ai-group/users/haoqifan/ava/frame_list/"
)

# Filenames of training samples list files.
_C.AVA.TRAIN_LISTS = ["train.csv"]

# Filenames of test samples list files.
_C.AVA.TEST_LISTS = ["val.csv"]

# Filenames of box list files for training. Note that we assume files which
# contains predicted boxes will have a suffix "predicted_boxes" in the
# filename.
_C.AVA.TRAIN_GT_BOX_LISTS = ["ava_train_v2.2.csv"]
_C.AVA.TRAIN_PREDICT_BOX_LISTS = []

# Filenames of box list files for test.
_C.AVA.TEST_PREDICT_BOX_LISTS = ["ava_val_predicted_boxes.csv"]

# This option controls the score threshold for the predicted boxes to use.
_C.AVA.DETECTION_SCORE_THRESH = 0.9

# If use BGR as the format of input frames.
_C.AVA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.AVA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.AVA.TRAIN_PCA_JITTER_ONLY = True

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.AVA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.AVA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.AVA.TEST_FORCE_FLIP = False

# Whether to use full test set for validation split.
_C.AVA.FULL_TEST_ON_VAL = False

# The name of the file to the ava label map.
_C.AVA.LABEL_MAP_FILE = "ava_action_list_v2.2_for_activitynet_2019.pbtxt"

# The name of the file to the ava exclusion.
_C.AVA.EXCLUSION_FILE = "ava_val_excluded_timestamps_v2.2.csv"

# The name of the file to the ava groundtruth.
_C.AVA.GROUNDTRUTH_FILE = "ava_val_v2.2.csv"

# Backend to process image, includes `pytorch` and `cv2`.
_C.AVA.IMG_PROC_BACKEND = "cv2"

# -----------------------------------------------------------------------------
# Epic Kitchen dataset options
# -----------------------------------------------------------------------------
_C.EPIC_KITCHEN = CfgNode()

# Whether to predict verb
_C.EPIC_KITCHEN.PREDICT_VERB = 1

# Whether to predict noun
_C.EPIC_KITCHEN.PREDICT_NOUN = 1

# Pickled train file name
_C.EPIC_KITCHEN.TRAIN_DATA_LIST = "EPIC_train_action_labels.pkl"

# Pickled val file name
_C.EPIC_KITCHEN.VAL_DATA_LIST = "EPIC_val_action_labels.pkl"

# Pickled test file name (default is to test on val)
_C.EPIC_KITCHEN.TEST_DATA_LIST = "EPIC_val_action_labels.pkl"

# Video path prefix for all filename in the pickled files above
_C.EPIC_KITCHEN.PATH_PREFIX = ""

# Pickled train file name
_C.EPIC_KITCHEN.MANIFOLD_TRAIN_DATA_LIST = (
    "manifold://ondevice_ai_data/tree/datasets/epic/mini_train_action_labels.pkl"
)

# Pickled val file name
_C.EPIC_KITCHEN.MANIFOLD_TEST_DATA_LIST = (
    "manifold://ondevice_ai_data/tree/datasets/epic/mini_test_action_labels.pkl"
)

# Video path prefix for all filename in the pickled files above
_C.EPIC_KITCHEN.MANIFOLD_TRAIN_PATH_PREFIX = (
    "manifold://fair_vision_data/tree/epic/epic_frames_train"
)

_C.EPIC_KITCHEN.MANIFOLD_TEST_PATH_PREFIX = (
    "manifold://fair_vision_data/tree/epic/epic_frames_test"
)

# If True, use epic-100 videos, otherwise use epic-55
_C.EPIC_KITCHEN.EPIC_100 = False

_C.EPIC_KITCHEN.RANDOM_STRIDE_RANGE = [0, 500]

# Specifies the number of frames between each input clip (only supported by src)
_C.EPIC_KITCHEN.INPUT_CLIP_STRIDE = 10

_C.EPIC_KITCHEN.STRIDE_TYPE = (
    "constant"  # other options distribution ("norm", "lognorm", "uniform")
)

_C.EPIC_KITCHEN.NUM_INPUT_CLIPS = 1

# TODO: LTA: Was this all required?  Or just retained as an artifact?
# -----------------------------------------------------------------------------
# EPIC_KITCHENS_STA Dataset options
# -----------------------------------------------------------------------------
_C.EPIC_KITCHENS_STA = CfgNode()

# Directory path of frames.
_C.EPIC_KITCHENS_STA.RGB_LMDB_DIR = "/home/furnari/SSD/ek55-sta/rgb/"
_C.EPIC_KITCHENS_STA.OBJ_DETECTIONS = "object_detections.pkl"
_C.EPIC_KITCHENS_STA.IMG_FNAME_TEMPLATE = "frame_{:010d}.jpg"

# Directory path for annotation files.
_C.EPIC_KITCHENS_STA.ANNOTATION_DIR = "/home/furnari/data/EK55-STA/"

# Filenames of training samples list files.
_C.EPIC_KITCHENS_STA.TRAIN_LISTS = ["training.pkl"]

# Filenames of test samples list files.
_C.EPIC_KITCHENS_STA.VAL_LISTS = ["validation.pkl"]

# This option controls the score threshold for the predicted boxes to use.
_C.EPIC_KITCHENS_STA.DETECTION_SCORE_THRESH = 0

# IOU threshold to determine if a detection is a next-active-object or not
_C.EPIC_KITCHENS_STA.NAO_IOU_THRESH = 0.5

# If use BGR as the format of input frames.
_C.EPIC_KITCHENS_STA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.EPIC_KITCHENS_STA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.EPIC_KITCHENS_STA.TRAIN_PCA_JITTER_ONLY = False

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.EPIC_KITCHENS_STA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.EPIC_KITCHENS_STA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.EPIC_KITCHENS_STA.TEST_FORCE_FLIP = False

# Backend to process image, includes `pytorch` and `cv2`.
_C.EPIC_KITCHENS_STA.IMG_PROC_BACKEND = "cv2"

# -----------------------------------------------------------------------------
# Ego4d Dataset options
# -----------------------------------------------------------------------------
_C.EGO4D = CfgNode()

# path to the verb labels on manifold
_C.EGO4D.MANIFOLD_VERB_LABELS_PATH = (
    "manifold://ondevice_ai_data/tree/datasets/ego4d/verb_label.txt"
)

# path to the noun labels on manifold
_C.EGO4D.MANIFOLD_NOUN_LABELS_PATH = (
    "manifold://ondevice_ai_data/tree/datasets/ego4d/noun_label.txt"
)

# -----------------------------------------------------------------------------
# Ego4d STA Dataset options
# -----------------------------------------------------------------------------
_C.EGO4D_STA = CfgNode()

# Directory path of frames.
_C.EGO4D_STA.VIDEOS_DIR = ""

# Directory path for annotation files.
_C.EGO4D_STA.ANNOTATION_DIR = ""

# Pre-extracted frames
_C.EGO4D_STA.RGB_LMDB_DIR = ""

# Frame key template
_C.EGO4D_STA.FRAME_KEY_TEMPLATE = "{video_id:s}_{frame_number:07d}"

# Object detections
_C.EGO4D_STA.OBJ_DETECTIONS = "object_detections.json"
# TODO: LTA: _C.EGO4D_STA.OBJ_DETECTIONS = ""

# Filenames of training samples list files.
_C.EGO4D_STA.TRAIN_LISTS = ["fho_sta_train.json"]
# TODO: STA: _C.EGO4D_STA.TRAIN_LISTS = ["train.pkl"]

# Filenames of test samples list files.
_C.EGO4D_STA.VAL_LISTS = ["fho_sta_val.json"]
# TODO: STA: _C.EGO4D_STA.VAL_LISTS = ["val.pkl"]

# Filenames of test samples list files.
_C.EGO4D_STA.TEST_LISTS = ["fho_sta_test_unannotated.json"]
# TODO: STA: _C.EGO4D_STA.TEST_LISTS = ["test.pkl"]

# This option controls the score threshold for the predicted boxes to use.
_C.EGO4D_STA.DETECTION_SCORE_THRESH = 0

# If True, augment proposals with ground-truth boxes during training
_C.EGO4D_STA.PROPOSAL_APPEND_GT = False

# If use BGR as the format of input frames.
_C.EGO4D_STA.BGR = False

# Training augmentation parameters
# Whether to use color augmentation method.
_C.EGO4D_STA.TRAIN_USE_COLOR_AUGMENTATION = False

# Whether to only use PCA jitter augmentation when using color augmentation
# method (otherwise combine with color jitter method).
_C.EGO4D_STA.TRAIN_PCA_JITTER_ONLY = False

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.EGO4D_STA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.EGO4D_STA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# Whether to do horizontal flipping during test.
_C.EGO4D_STA.TEST_FORCE_FLIP = False

# IOU threshold to deem if a detection is a next active object or not
_C.EGO4D_STA.NAO_IOU_THRESH = 0.5

_C.EGO4D_STA.VIDEO_LOAD_BACKEND = "lmdb"  # lmdb, pytorchvideo, decord, pyav


# TODO: STA: _C.EGO4D_STA.VIDEO_LOAD_BACKEND = "pytorchvideo" #lmdb, pytorchvideo, decord

def _assert_and_infer_cfg(cfg):
    # CL assertions
    if cfg.DATA.TASK == "continual_classification":
        assert cfg.SOLVER.MAX_EPOCH == 1, f"CL stream requires max 1 epoch, not {cfg.SOLVER.MAX_EPOCH}"
        assert not cfg.TEST.ENABLE, "CL pytorch lightning testing is not supported"

    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS == 3

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    if cfg.BN.NORM_TYPE == "sync_batchnorm":
        assert cfg.BN.NUM_SYNC_DEVICES % cfg.NUM_GPUS == 0

    return cfg


def set_cfg_by_name(cfg: CfgNode, hierarchy_cfg_name: str, cfg_val):
    """Maps cfg_val of METHOD.SOME.NODE to dict hierarchy: {METHOD:{SOME:{NODE:cfg_val}}}"""
    keys = hierarchy_cfg_name.split('.')
    target_obj = cfg
    for idx, key in enumerate(keys):  # If using dots, set in hierarchy of objects, not as single dotted-key
        if idx == len(keys) - 1:  # is last
            setattr(target_obj, key, cfg_val)
        else:
            target_obj = getattr(target_obj, key)


def get_cfg_by_name(cfg: CfgNode, hierarchy_cfg_name: str):
    """Gets with METHOD.SOME.NODE the value from dict hierarchy: {METHOD:{SOME:{NODE:cfg_val}}}"""
    keys = hierarchy_cfg_name.split('.')
    target_obj = cfg
    for idx, key in enumerate(keys):  # If using dots, set in hierarchy of objects, not as single dotted-key
        target_obj = getattr(target_obj, key)
    return target_obj


def convert_cfg_to_flat_dict(cfg: CfgNode, key_exclude_set: set = None):
    """ Dict hierarchy is transformed to '.'-separated string keys, to values."""
    if key_exclude_set is None:
        key_exclude_set = []

    def _get_leafnodes_dict(cfg_node, key_list, final_dict):
        if not isinstance(cfg_node, CfgNode):  # Final node
            final_dict[".".join(key_list)] = cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                if k not in key_exclude_set:
                    _get_leafnodes_dict(v, key_list + [k], final_dict)

    res = {}
    _get_leafnodes_dict(cfg, [], res)

    return res


def convert_flat_dict_to_cfg(flat_dict: dict, key_exclude_set: set = None):
    """ Dict hierarchy is transformed to '.'-separated string keys, to values."""

    init_dict = {}
    for flat_key, val in flat_dict.items():
        if flat_key in key_exclude_set:
            continue
        keylist = flat_key.split('.')
        leaf_key = keylist[-1]
        parent_keys = keylist[:-1]

        final_dict_ref = init_dict
        if parent_keys is not None:
            for parent_key in parent_keys:
                if parent_key not in final_dict_ref:
                    final_dict_ref[parent_key] = {}
                final_dict_ref = final_dict_ref[parent_key]

        final_dict_ref[leaf_key] = val

    cfg = CfgNode(init_dict)
    return cfg


def cfg_add_non_existing_key_vals(src_cfg, merge_cfg):
    """ Merge Keys in merge_cfg that are not in src_cfg, into src_cfg. """

    for key, val in merge_cfg.items():
        if key not in src_cfg:
            src_cfg[key] = copy.deepcopy(val)
        elif isinstance(val, CfgNode):
            cfg_add_non_existing_key_vals(src_cfg[key], merge_cfg[key])
    return src_cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
