from src.config.default import _CN as cfg

# ====== OSR Model ======
cfg.OSR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.OSR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.OSR.MATCH_COARSE.BORDER_RM = 0

# Dual backbone
cfg.OSR.USE_DUAL_BACKBONE = True
cfg.OSR.DINOV3_PATH = "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/dinov3s/model.safetensors"
cfg.OSR.SARMAE_PATH = "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/SARMAE/SARMAE_vit_Base.pth"

# SARMAE cross-attention (unfreeze SARMAE, cross-attn with DINOv3)
cfg.OSR.UNFREEZE_SARMAE = True
cfg.OSR.SARMAE_LR_RATIO = 0.1
cfg.OSR.SARMAE_CROSS_ATTN_INDICES = [1, 2]
cfg.OSR.DINO_CROSS_ATTN_LAYERS = [1, 3]

# ====== Trainer ======
cfg.TRAINER.CANONICAL_LR = 3e-4
cfg.TRAINER.WARMUP_STEP = 1000
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.SCHEDULER = 'CosineAnnealing'
cfg.TRAINER.SCHEDULER_INTERVAL = 'step'
cfg.TRAINER.COSA_ETA_MIN = 1e-7
cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.TRAINER.USE_WANDB = True

# ====== Dataset ======
cfg.DATASET.TRAIN_DATA_SOURCE = "OpticalSARDataset"
cfg.DATASET.VAL_DATA_SOURCE = "OpticalSARDataset"
cfg.DATASET.TEST_DATA_SOURCE = "OpticalSARDataset"

cfg.DATASET.TRAIN_DATA_ROOT = ["/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/SARLO/train/chunk_000/unpacked", "/media/cicuvc/c63abdf1-0e56-4153-9228-95df5a2f239b/cicuvc/SARLO/train/chunk_001/unpacked_verified"]
cfg.DATASET.VAL_DATA_ROOT = list(cfg.DATASET.TRAIN_DATA_ROOT)
cfg.DATASET.TEST_DATA_ROOT = list(cfg.DATASET.TRAIN_DATA_ROOT)

cfg.DATASET.OSAR_FMT = ["sarlo", "sarlo"]
cfg.DATASET.OSAR_PATCH_SIZE = 384
cfg.DATASET.OSAR_MAX_ANGLE_DEG = 15.0
cfg.DATASET.OSAR_MAX_TRANSLATION = 32.0
cfg.DATASET.OSAR_IOU_THRESH = 0.7
cfg.DATASET.OSAR_MAX_PAIRS = None
cfg.DATASET.OSAR_IMG_RESIZE = 256
cfg.DATASET.OSAR_IMG_PAD = True
cfg.DATASET.OSAR_DF = 8
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

# ====== Metrics ======
cfg.METRICS.SR_THRESHOLD = 10000.0
cfg.METRICS.USE_FULL_IMAGE = True
cfg.METRICS.FLOW_SCALE = 1.0
