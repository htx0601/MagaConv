import math

import torch
import torch.nn as nn


def default_conv(in_ch, out_ch, act_func, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        act_func
    )


class BidAP(nn.Module):
    def __init__(self, in_h, in_w, inner_ch, layers=1):
        super().__init__()
        self.layers = layers
        self.fusion_blocks_1 = nn.ModuleList([])
        self.fusion_blocks_2 = nn.ModuleList([])

        for layer in range(self.layers):
            self.fusion_blocks_1.append(Cmap(in_h, in_w, inner_ch=inner_ch, n_heads=1))
            self.fusion_blocks_2.append(Cmap(in_h, in_w, inner_ch=inner_ch, n_heads=1))

    def forward(self, condition, depth):
        for layer in range(self.layers):
            condition = self.fusion_blocks_1[layer](depth, condition) + condition
            depth = self.fusion_blocks_2[layer](condition, depth) + depth
        return depth


class Cmap(nn.Module):
    def __init__(self, in_h, in_w, inner_ch=64, n_heads=1):
        super().__init__()
        self.proj_in_x = nn.Sequential(
            nn.Conv2d(inner_ch * 2, inner_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inner_ch),
            nn.SiLU(inplace=True)
        )

        inner_mlp_ch = in_h * in_w if in_h * in_w < 216 else 216
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_h * in_w, out_features=inner_mlp_ch),
            nn.BatchNorm1d(inner_mlp_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=inner_mlp_ch, out_features=inner_mlp_ch),
            nn.BatchNorm1d(inner_mlp_ch),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=inner_mlp_ch, out_features=in_h * in_w),
        )

        self.proj_gamma = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.proj_beta = zero_module(nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        ))

        self.batch_norm = nn.BatchNorm2d(inner_ch)

    def forward(self, query, target):
        b, c, *spatial = query.shape
        query = self.proj_in_x(torch.cat([query, target], dim=1))
        fusion = query.reshape(b*c, -1)
        fusion = self.mlp(fusion).reshape(b, c, *spatial)

        gamma = self.proj_gamma(fusion)
        beta = self.proj_beta(fusion)

        target = self.batch_norm(target)
        target = gamma * target + beta
        return target


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
