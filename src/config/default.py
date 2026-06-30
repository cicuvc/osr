from yacs.config import CfgNode as CN

INFERENCE = False

_CN = CN()


##############  ↓  OSR Pipeline  ↓  ##############
_CN.OSR = CN()
_CN.OSR.RESOLUTION = (8, 2)  # options: [(8, 2)]
_CN.OSR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.OSR.MEDIUM_WINDOW_SIZE = 3  # window_size in fine_level, must be odd

# 1. OSR-backbone (local feature CNN) config
_CN.OSR.RESNET = CN()
_CN.OSR.RESNET.INITIAL_DIM = 128
_CN.OSR.RESNET.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# Dual backbone config
_CN.OSR.USE_DUAL_BACKBONE = False
_CN.OSR.DINOV3_PATH = ""
_CN.OSR.SARMAE_PATH = ""
_CN.OSR.DINOV3_LAYERS = 5
_CN.OSR.SARMAE_BLOCKS = 4
_CN.OSR.VIT_FUSE_DIM = 64
_CN.OSR.UNFREEZE_SARMAE = False
_CN.OSR.SARMAE_CROSS_ATTN_INDICES = [1, 2]
_CN.OSR.DINO_CROSS_ATTN_LAYERS = [1, 3]
_CN.OSR.SARMAE_LR_RATIO = 0.1

# 2. OSR-coarse module config
_CN.OSR.COARSE = CN()
_CN.OSR.COARSE.INFERENCE = INFERENCE
_CN.OSR.COARSE.D_MODEL = 256
_CN.OSR.COARSE.D_FFN = 256
_CN.OSR.COARSE.NHEAD = 8
_CN.OSR.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.OSR.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']


# 3. Coarse-Matching config
_CN.OSR.MATCH_COARSE = CN()
_CN.OSR.MATCH_COARSE.INFERENCE = INFERENCE
_CN.OSR.MATCH_COARSE.D_MODEL = 256
_CN.OSR.MATCH_COARSE.THR = 0.3
_CN.OSR.MATCH_COARSE.BORDER_RM = 2
_CN.OSR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax']
_CN.OSR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.OSR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.OSR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. OSR-fine module config
_CN.OSR.FINE = CN()
_CN.OSR.FINE.DENSER = False # if true, match all features in fine-level windows
_CN.OSR.FINE.INFERENCE = INFERENCE
_CN.OSR.FINE.DSMAX_TEMPERATURE = 0.1
_CN.OSR.FINE.THR = 0.1
_CN.OSR.FINE.MLP_HIDDEN_DIM_COEF = 2 # coef for  mlp hidden dim (hidden_dim = feat_dim * coef)
_CN.OSR.FINE.NHEAD_FINE_LEVEL = 8
_CN.OSR.FINE.NHEAD_MEDIUM_LEVEL = 7


# 5. OSR Losses

_CN.OSR.LOSS = CN()
_CN.OSR.LOSS.FOCAL_ALPHA = 0.25
_CN.OSR.LOSS.FOCAL_GAMMA = 2.0
_CN.OSR.LOSS.POS_WEIGHT = 1.0
_CN.OSR.LOSS.NEG_WEIGHT = 1.0

# -- # coarse-level
_CN.OSR.LOSS.COARSE_WEIGHT = 0.5
# -- # fine-level
_CN.OSR.LOSS.FINE_WEIGHT = 0.3
# -- # sub-pixel
_CN.OSR.LOSS.SUB_WEIGHT = 1 * 10**4



##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAIN_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_SOURCE = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = "rgb_thermal"  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

# VisTir options
_CN.DATASET.VISTIR_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.VISTIR_IMG_PAD = False  # pad img to square with size = VISTIR_IMG_RESIZE
_CN.DATASET.VISTIR_DF = 8

# RoadScenceDataset options
_CN.DATASET.RS_IMG_RESIZE = 64  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.RS_IMG_PAD = True  # pad img to square with size = RS_IMG_RESIZE
_CN.DATASET.RS_DF = 8

# OpticalSARDataset options
_CN.DATASET.OSAR_FMT = ["osd"]  # dataset format: ['osd', 'soma', '3mos', 'sarlo', 'ubc']
_CN.DATASET.OSAR_PATCH_SIZE = 384  # patch size for random crops
_CN.DATASET.OSAR_MAX_ANGLE_DEG = 15.0  # maximum rotation angle in degrees
_CN.DATASET.OSAR_MAX_TRANSLATION = 32.0  # maximum translation in pixels
_CN.DATASET.OSAR_IOU_THRESH = 0.7  # minimum IOU for valid transform samples
_CN.DATASET.OSAR_MAX_PAIRS = None  # optional limit on number of pairs
_CN.DATASET.OSAR_IMG_RESIZE = 64  # resize the longer side, zero-pad to square
_CN.DATASET.OSAR_IMG_PAD = True
_CN.DATASET.OSAR_DF = 8

# Pretrain dataset options
_CN.DATASET.PRETRAIN_IMG_RESIZE = 64 # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.PRETRAIN_IMG_PAD = True  # pad img to square with size = PRETRAIN_IMG_RESIZE
_CN.DATASET.PRETRAIN_DF = 8
_CN.DATASET.PRETRAIN_FRAME_GAP = 2 # the gap between video frames of Kaist dataset

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-5
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

_CN.TRAINER.USE_WANDB = False # use weight and biases

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1
_CN.TRAINER.SGD_DECAY = 0.
_CN.TRAINER.RMSPROP_DECAY = 0.0001

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'ExponentialLR'  # [MultiStepLR, CosineAnnealing, CosineAnnealingWarmRestarts, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosinAnnealing
_CN.TRAINER.COSA_ETA_MIN = 1e-6
_CN.TRAINER.COSWR_T0 = 50  # CosineAnnealingWarmRestarts: initial restart interval
_CN.TRAINER.COSWR_T_MULT = 2  # CosineAnnealingWarmRestarts: factor for restart interval
_CN.TRAINER.COSWR_ETA_MIN = 1e-6  # CosineAnnealingWarmRestarts: minimum learning rate
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = False
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 128     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 1.0  
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'normal'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not whether to allow the same sample to be sampled multiple times in one epoch
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not whether to shuffle again within each epoch
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data how many times to sample the same sample per epoch
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66

##############  Pretrain  ##############
_CN.PRETRAIN = CN()
_CN.PRETRAIN.PATCH_SIZE = 128 # patch sıze for masks
_CN.PRETRAIN.MASK_RATIO = 0.5 
_CN.PRETRAIN.MAE_MARGINS = [0, 0.4, 0, 0] # margins not to be masked (up bottom left right)
_CN.PRETRAIN.VAL_SEED = 42 # rng seed to crate the same masks for validation

_CN.OSR.PRETRAIN_PATCH_SIZE = _CN.PRETRAIN.PATCH_SIZE 

##############  Metrics  ##############
_CN.METRICS = CN()
_CN.METRICS.SR_THRESHOLD = 10000.0  # threshold for success rate in pixels
_CN.METRICS.USE_FULL_IMAGE = True  # whether to use full image for evaluation
_CN.METRICS.FLOW_SCALE = 1.0  # scale factor for flow values

##############  Test/Inference  ##############
_CN.TEST = CN()
_CN.TEST.IMG0_RESIZE = 64 # resize the longer side
_CN.TEST.IMG1_RESIZE = 64 # resize the longer side
_CN.TEST.DF = 8
_CN.TEST.PADDING = False  # pad img to square with size = IMG0_RESIZE, IMG1_RESIZE
_CN.TEST.COARSE_SCALE = 0.125

def get_cfg_defaults(inference=False):
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    if inference:
        _CN.OSR.COARSE.INFERENCE = True
        _CN.OSR.MATCH_COARSE.INFERENCE = True
        _CN.OSR.FINE.INFERENCE = True
    return _CN.clone()
