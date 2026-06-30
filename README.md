# 遥感图像光学-SAR配准课程项目源码

## 数据准备

提供的数据包含SARLO-80数据集下筛选的6668对图像对供下载[Link](#COMING_SOON)，结构如下

```
├── slice0
|     ├──── parititon.json #数据集划分，若文件不存在则自动创建保存
|     ├──── 00000001.optic.png
|     ├──── 00000001.sar.png
|     ├──── ...
|
├── slice1
|     ├──── 00000001.optic.png
|     ├──── 00000001.sar.png
|     ├──── ...
```

训练需修改训练集路径

```
cfg.DATASET.TRAIN_DATA_ROOT = ["..../slice0", "..../slice1"]
cfg.DATASET.VAL_DATA_ROOT = list(cfg.DATASET.TRAIN_DATA_ROOT)
cfg.DATASET.TEST_DATA_ROOT = list(cfg.DATASET.TRAIN_DATA_ROOT)

cfg.DATASET.OSAR_FMT = ["sarlo", "sarlo"]
```

## 初始化权重

通过`scripts/extract_weights.py`从预训练的DINOv3s和SARMAE中转移权重并初始化到本框架模型结构。提供参考`init.ckpt`: [Link](#COMING_SOON)

## 训练脚本
```shell
python train.py --exp_name osr-sarlo --batch_size 16 --epochs 100 --gpus 1 --num_workers 8 --pretrained_ckpt checkpoints/init.ckpt
```
