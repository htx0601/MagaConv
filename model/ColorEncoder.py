import torch
import torch.nn as nn


def default_conv(in_ch, out_ch, act_func, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        act_func
    )


def default_deconv(in_ch, out_ch, act_func, kernel_size=3, stride=2, padding=1, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch,
                           kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm2d(out_ch),
        act_func
    )


class ColorEncoder(nn.Module):
    def __init__(self, inner_ch=64, layers=4):
        super().__init__()
        self.layers = layers
        channel_mults = [3]
        channel_mults = channel_mults + [inner_ch * pow(2, i) for i in range(layers)]

        self.c_conv = nn.ModuleList([])
        self.downsample = nn.ModuleList([])

        for layer in range(self.layers):
            in_ch = channel_mults[layer]
            out_ch = channel_mults[layer + 1]
            self.downsample.append(nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                default_conv(in_ch=in_ch, out_ch=out_ch, act_func=nn.SiLU(inplace=True),
                             kernel_size=3, stride=1, padding=1)))
            self.c_conv.append(default_conv(in_ch=out_ch, out_ch=out_ch, act_func=nn.SiLU(inplace=True)))
            self.c_conv.append(default_conv(in_ch=out_ch, out_ch=out_ch, act_func=nn.SiLU(inplace=True)))

    def forward(self, x):
        out = []
        for layer in range(self.layers):
            x = self.downsample[layer](x)
            x = self.c_conv[layer * 2](x) + x
            x = self.c_conv[layer * 2 + 1](x) + x
            out.append(x)
        out.reverse()
        return out


class ColorDecoder(nn.Module):
    def __init__(self, inner_ch=64, layers=4):
        super().__init__()
        self.layers = layers
        channel_mults = [inner_ch]
        channel_mults = channel_mults + [inner_ch * pow(2, i) for i in range(1, layers)]
        channel_mults.reverse()

        self.c_conv = nn.ModuleList([])

        for layer in range(self.layers - 1):
            in_ch = channel_mults[layer]
            out_ch = channel_mults[layer + 1]
            self.c_conv.append(default_conv(in_ch=in_ch, out_ch=out_ch, act_func=nn.nn.SiLU(inplace=True)))
            self.c_conv.append(default_conv(in_ch=out_ch * 2, out_ch=out_ch, act_func=nn.nn.SiLU(inplace=True)))

        self.bottom = default_conv(in_ch=inner_ch * pow(2, layers), out_ch=inner_ch * pow(2, layers - 1),
                                   act_func=nn.nn.SiLU(inplace=True))

    def forward(self, color_out, color_fused):
        x = self.bottom(torch.cat([color_out[0], color_fused], dim=1))
        out = [x]
        for layer in range(self.layers - 1):
            x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)
            x = self.c_conv[layer * 2](x)
            x = self.c_conv[layer * 2 + 1](torch.cat([x, color_out[layer + 1]], dim=1)) + x
            out.append(x)
        return out

