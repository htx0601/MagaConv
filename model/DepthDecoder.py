import torch
import torch.nn as nn
from model.BidAP import BidAP


def default_conv(in_ch, out_ch, act_func, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        act_func
    )


class DepthDecoderAttentionLayer(nn.Module):
    def __init__(self, in_h, in_w, in_ch, out_ch):
        super().__init__()
        self.fusion_module = BidAP(in_h, in_w, in_ch)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        )

    def forward(self, depth, color_skip, depth_skip):
        depth = self.fuse(torch.cat([depth_skip, depth], dim=1)) + depth
        depth = self.fusion_module(color_skip, depth) + depth

        depth = self.up(depth)
        mask = self.mask(depth)
        depth = self.conv(depth) * mask
        return depth


class DepthDecoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.skip_conv_color = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )

        self.mask_fuse_color = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )

        self.skip_conv_depth = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )

        self.mask_fuse_depth = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, depth, color_skip, depth_skip):
        skip_color = self.skip_conv_color(color_skip) + color_skip
        mask_color = self.mask_fuse_color(torch.cat([color_skip, depth], dim=1))
        skip_color = skip_color * mask_color + skip_color

        skip_depth = self.skip_conv_depth(depth_skip) + depth_skip
        mask_depth = self.mask_fuse_depth(torch.cat([depth_skip, depth], dim=1))
        skip_depth = skip_depth * mask_depth + skip_color

        depth = self.fuse(torch.cat([skip_color, skip_depth, depth], dim=1)) + depth
        depth = self.up(depth)
        depth = self.conv(depth)
        return depth


class DepthDecoder(nn.Module):
    def __init__(self, inner_ch=64, layers=4):
        super().__init__()
        self.layers = layers
        channel_mults = [inner_ch]
        channel_mults = channel_mults + [inner_ch * pow(2, i) for i in range(layers)]
        channel_mults.reverse()

        self.decoder_blocks = nn.ModuleList([])

        in_h = [pow(2, i) * 12 for i in range(layers)]
        in_w = [pow(2, i) * 18 for i in range(layers)]

        for layer in range(self.layers):
            in_ch = channel_mults[layer]
            out_ch = channel_mults[layer + 1]
            if layer < 3:
                self.decoder_blocks.append(
                    DepthDecoderAttentionLayer(in_h[layer], in_w[layer], in_ch, out_ch))
            else:
                self.decoder_blocks.append(DepthDecoderLayer(in_ch, out_ch))

        self.final_conv = nn.Sequential(
            nn.Conv2d(inner_ch, inner_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(inner_ch, 1, kernel_size=3, padding=1),
        )

    def forward(self, color_out, depth_out, x):
        for layer in range(self.layers):
            x = self.decoder_blocks[layer](x, color_out[layer], depth_out[layer])
        return torch.clamp(self.final_conv(x), 1e-2, 10)
