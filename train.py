
import math
import argparse
from pathlib import Path
from loguru import logger
import time 

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.config.default import get_cfg_defaults
from src.utils.misc import setup_gpus
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_osr import PL_OSR

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os


def main():
    # Simple argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='osr-train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pretrained_ckpt', type=str,default='/home/cicuvc/cs/project/XoFTR-old/weights_xoftr_640.ckpt')
    args = parser.parse_args() 
    
    print(f"Starting Training")
    print(f"   - Experiment: {args.exp_name}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - GPUs: {args.gpus}")
    print(f"   - Pretrained checkpoint: {args.pretrained_ckpt}")

    config = get_cfg_defaults()
    config.merge_from_file('configs/data/optical_sar.py')
    # Override configs
    config.TRAINER.MAX_EPOCHS = args.epochs
    
    pl.seed_everything(config.TRAINER.SEED)

    # Setup GPU
    if torch.cuda.is_available() and args.gpus > 0:
        device = 'gpu'
        accelerator = 'gpu'
        devices = args.gpus
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        accelerator = 'cpu'
        devices = 1
        print("Using CPU (GPU not available)")
    
    # Scale learning rate 
    config.TRAINER.WORLD_SIZE = devices
    config.TRAINER.TRUE_BATCH_SIZE = devices * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    print(f"raining configuration:")
    print(f"   - True batch size: {config.TRAINER.TRUE_BATCH_SIZE}")
    print(f"   - Learning rate: {config.TRAINER.TRUE_LR:.6f}")
    print(f"   - Warmup steps: {config.TRAINER.WARMUP_STEP}")

    # Create data module first (so we can estimate total steps)
    data_module = MultiSceneDataModule(args, config)
    data_module.setup('fit')
    train_size = len(data_module.train_dataset)
    steps_per_epoch = (train_size + config.TRAINER.TRUE_BATCH_SIZE - 1) // config.TRAINER.TRUE_BATCH_SIZE
    total_steps = steps_per_epoch * args.epochs
    config.TRAINER.COSA_TMAX = total_steps
    print(f"   - Dataset size: {train_size}")
    print(f"   - Steps/epoch: {steps_per_epoch}")
    print(f"   - Total steps (COSA_TMAX): {total_steps}")
    # Create model
    try:
        model = PL_OSR(config, pretrained_ckpt=args.pretrained_ckpt)
        print("Model loaded with pretrained weights")
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        model = PL_OSR(config)
        print("Model created without pretrained weights")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Data module created")
    
    # Setup logging and callbacks
    logger_tb = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name)
    loggers = [logger_tb]
    if config.TRAINER.USE_WANDB:
        from pytorch_lightning.loggers import WandbLogger
        logger_wb = WandbLogger(project='osr', name=args.exp_name, save_dir='logs/wandb')
        loggers.append(logger_wb)
    ckpt_dir = Path(logger_tb.log_dir) / 'checkpoints'
    
    callbacks = [ 
        
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='val_AEPE',
            save_top_k=15,
            mode='min',
            save_last=True,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
            dirpath=str(ckpt_dir),
            filename='{epoch:02d}-{val_AEPE:.3f}'
        )
    ]
    
    # Create trainer (Lightning 1.3.5 compatible)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=devices,
        accelerator=accelerator,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=1,
        precision='bf16-mixed',
        check_val_every_n_epoch=1,
        log_every_n_steps=1000
    )
    
    print("Trainer created")
    print(f"Logs will be saved to: {logger_tb.log_dir}")
    
    # Start training
    try:
        print("Starting training...")
        
        start_time = time.time()  
        
        trainer.fit(model, datamodule=data_module)
        
        end_time = time.time() 
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        

        avg_epoch_time = total_time / args.epochs
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        
        print("Training completed successfully!")
        print(f"Best model saved at: {ckpt_dir}")
        
        # Print final metrics
        if trainer.callback_metrics:
            print("\nFinal metrics:")
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    print(f"   - {key}: {value.item():.4f}")
                else:
                    print(f"   - {key}: {value}")
                    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
