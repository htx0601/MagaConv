# Efficient Indoor Depth Completion Network Using Mask-adaptive Gated Convolution

This repository contains the **official PyTorch implementation of MaG-Net**, an efficient and accurate network for **indoor depth completion**.

---

## Dataset Format

Place your dataset in the following folder structure:
```
your_dataset_root/
└── NYUDepth/
    ├── train/
    │   ├── 0001/
    │   │   ├── 0001_color.png
    │   │   ├── 0001_depth.png
    │   │   ├── 0001_gt.png
    │   ├── 0002/
    │   │   ├── 0002_color.png
    │   │   ├── 0002_depth.png
    │   │   ├── 0002_gt.png
    │   └── ...
    └── test/
        ├── 0101/
        │   ├── 0101_color.png
        │   ├── 0101_depth.png
        │   ├── 0101_gt.png
        ├── 0102/
        │   ├── 0102_color.png
        │   ├── 0102_depth.png
        │   ├── 0102_gt.png
        └── ...
```

Each sample consists of three images:
- `*_color.png`: the RGB input image
- `*_depth.png`: the raw depth input
- `*_gt.png`: the dense ground truth depth

## Training

Use the following command to train the model:

```bash
python train.py \
  --data_root /path/to/datasets \
  --project_root /path/to/project \
  --datasets NYUDepth \
  --inner_channels 64 \
  --layers 4 \
  --lr 0.001 \
  --num_epoch 100 \
  --batch_size_train 8 \
  --batch_size_eval 1
```

## Requirements

    Python 3.8+
    PyTorch ≥ 1.10
    torchvision
    tqdm
    PIL (Pillow)
    numpy

## Citation

If you use this code in your research, please cite:

```
@article{Magnet_aaai2025,
  title     = {Efficient Indoor Depth Completion Network Using Mask-adaptive Gated Convolution},
  author    = {Huang, Tingxuan and Miao, Jiacheng and Deng, Shizhuo and Jia, Tong and Chen, Dongyue},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {4},
  pages     = {3742--3750},
  year      = {2025},
  month     = {Apr.},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/32390},
  doi       = {10.1609/aaai.v39i4.32390}
}
```