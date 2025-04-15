import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(root, name, split):
    dir_path = os.path.join(root, name, split)
    rgb, depth, gt, d_name = [], [], [], []
    assert os.path.isdir(dir_path), f'{dir_path} is not a valid directory'
    for root_dir, _, fnames in sorted(os.walk(dir_path)):
        for fname in sorted(fnames):
            if fname[-6:-4] == 'or':  # num_color.png
                rgb.append(os.path.join(root_dir, fname))
            elif fname[-6:-4] == 'th':  # num_depth.png
                depth.append(os.path.join(root_dir, fname))
            elif fname[-6:-4] == 'gt':  # num_gt.png
                gt.append(os.path.join(root_dir, fname))
        d_name.append(name)
    return rgb, depth, gt, d_name


def pil_loader(path, ratio=1.):
    return transforms.ToTensor()(Image.open(path)) / ratio


class CompletionDataset(Dataset):
    def __init__(self, root, datasets='NYUDepth', split='train', data_len=0, loader=pil_loader):
        assert data_len % 3 == 0
        self.datasets = datasets
        self.loader = loader
        self.rgb, self.depth, self.gt, self.d_name = [], [], [], []

        dataset_list = [self.datasets]
        data_len_eval = data_len // 3

        for name in dataset_list:
            rgb_list, depth_list, gt_list, d_name_list = make_dataset(root, name, split)
            if data_len > 0:
                self.rgb += rgb_list[:data_len_eval]
                self.depth += depth_list[:data_len_eval]
                self.gt += gt_list[:data_len_eval]
                self.d_name += d_name_list[:data_len_eval]
            else:
                self.rgb += rgb_list
                self.depth += depth_list
                self.gt += gt_list
                self.d_name += d_name_list

        norm_size = [192, 288]
        if split == 'train':
            self.transforms_dict = self._build_train_transforms(norm_size)
        else:
            self.transforms_dict = self._build_eval_transforms(norm_size)

    def _build_train_transforms(self, norm_size):
        def build(size_base, mean=None, std=None):
            if mean is None:
                mean = [0.485, 0.456, 0.406]
            if std is None:
                std = [0.229, 0.224, 0.225]
            size = int(size_base * np.random.uniform(1.0, 1.2))
            return {
                'gt': transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(norm_size),
                    transforms.ConvertImageDtype(torch.float),
                ]),
                'rgb': transforms.Compose([
                    transforms.Resize(size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.CenterCrop(norm_size),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std),
                ]),
                'depth': transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(norm_size),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.RandomErasing(p=0.9, scale=(0.1, 0.6), value=0, inplace=True),
                ])
            }

        return {
            'DIML': build(192),
            'NYUDepth': build(240),
            'SUNRGBD': build(216),
        }

    def _build_eval_transforms(self, norm_size):
        def build(size, mean=None, std=None):
            if mean is None:
                mean = [0.485, 0.456, 0.406]
            if std is None:
                std = [0.229, 0.224, 0.225]
            return {
                'gt': transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(norm_size),
                    transforms.ConvertImageDtype(torch.float),
                ]),
                'rgb': transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(norm_size),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std),
                ]),
                'depth': transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(norm_size),
                    transforms.ConvertImageDtype(torch.float),
                ])
            }

        return {
            'DIML': build(192),
            'NYUDepth': build(240),
            'SUNRGBD': build(216),
        }

    def __getitem__(self, index):
        with torch.no_grad():
            dataset = self.d_name[index]
            transforms_set = self.transforms_dict[dataset]

            if dataset == 'DIML':
                ratio = 1000
            elif dataset == 'NYUDepth':
                ratio = 0.1
            elif dataset == 'SUNRGBD':
                ratio = 6553.5

            rgb = transforms_set['rgb'](self.loader(self.rgb[index]))
            gt = transforms_set['gt'](self.loader(self.gt[index], ratio))
            depth = transforms_set['depth'](self.loader(self.depth[index], ratio))

            # sparse input
            # mask = torch.rand_like(depth)
            # mask = transforms.ConvertImageDtype(torch.float)((mask <= 0.99))

            # dense input
            mask = transforms.ConvertImageDtype(torch.float)((depth <= 0.2))

            depth = gt * (1. - mask)
            return {'rgb': rgb, 'gt': gt, 'depth': depth, 'mask': mask}


    def __len__(self):
        return len(self.rgb)
