import torch
import torch.nn as nn
from model.DepthEncoder import DepthEncoder
from model.ColorEncoder import ColorEncoder
from model.DepthDecoder import DepthDecoder


class Network(nn.Module):
    def __init__(self, inner_ch=32, layers=4):
        super().__init__()
        self.layers = layers
        self.depth_encoder = DepthEncoder(inner_ch, layers)
        self.color_encoder = ColorEncoder(inner_ch, layers)
        self.depth_decoder = DepthDecoder(inner_ch, layers)

        self.depth_fused = nn.Sequential(
            nn.Conv2d(inner_ch * pow(2, layers), inner_ch * pow(2, layers-1), 3, 1, 1),
            nn.BatchNorm2d(inner_ch * pow(2, layers-1)),
            nn.SiLU(inplace=True),
        )

    def forward(self, color, depth, mask):
        color_out = self.color_encoder(color)
        depth_out = self.depth_encoder(depth, mask)
        depth_fused = self.depth_fused(torch.cat([depth_out[0], color_out[0]], dim=1)) + depth_out[0]
        predict = self.depth_decoder(color_out, depth_out, depth_fused)
        return predict
