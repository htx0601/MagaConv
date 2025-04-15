import torch
import torch.nn as nn


class MagaConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1,
                 window_size=2, head=2, act_func=nn.SiLU(inplace=True)):
        super().__init__()
        self.in_ch = in_ch
        self.head_ch = out_ch * head
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=self.head_ch,
                              kernel_size=k, stride=s, padding=p, bias=False)
        self.bn_d = nn.Sequential(nn.BatchNorm2d(self.head_ch), act_func)
        self.avg_mask = nn.AvgPool2d(kernel_size=window_size, stride=1, padding=int((window_size - 1) / 2))
        self.mask_conv = nn.Sequential(
            nn.BatchNorm2d(self.head_ch),
            nn.Conv2d(in_channels=self.head_ch, out_channels=self.head_ch, kernel_size=1, stride=1, bias=False)
        )
        # self.borderline = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.borderline.data.fill_(0.5)
        self.borderline = nn.Sequential(
            nn.AdaptiveMaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=4 * self.head_ch, out_features=64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.act = act_func
        self.split = window_size
        self.relu = nn.ReLU(inplace=True)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.head_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, depth, mask):
        mask = torch.repeat_interleave(mask, self.in_ch, dim=1)

        mask_depth = torch.cat([mask, depth], dim=0)
        mask_depth = self.conv(mask_depth)
        mask, depth = mask_depth.chunk(2, dim=0)

        mask = self.mask_conv(mask)
        mask = torch.abs(mask)

        borderline = 0.1 + 0.9 * self.borderline(mask)
        borderline = borderline.unsqueeze(dim=1).unsqueeze(dim=2)
        mask = self.relu(torch.exp(-mask * borderline) - 0.5) * 2
        y = self.bn_d(depth) * mask
        y = self.fuse(y)
        return y


class DepthEncoderLayer(nn.Module):
    def __init__(self, in_ch, out_ch, win_1, win_2, win_3):
        super().__init__()
        self.conv_1_1 = MagaConv(in_ch, out_ch, window_size=win_1)
        self.conv_1_2 = MagaConv(in_ch, out_ch, window_size=win_2)
        self.conv_1_3 = MagaConv(in_ch, out_ch, window_size=win_3)
        self.conv_1_fuse = nn.Sequential(
            nn.Conv2d(in_channels=out_ch * 3, out_channels=out_ch,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        self.conv_2_1 = MagaConv(out_ch, out_ch, s=1, window_size=win_1)
        self.conv_2_2 = MagaConv(out_ch, out_ch, s=1, window_size=win_2)
        self.conv_2_3 = MagaConv(out_ch, out_ch, s=1, window_size=win_3)
        self.conv_2_fuse = nn.Sequential(
            nn.Conv2d(in_channels=out_ch * 3, out_channels=out_ch,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        self.mask_down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.mask_down.requires_grad_(False)
        self.mask_update = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.mask_update.requires_grad_(False)

    def mask_update_rule(self, mask, down=False):
        mask_ = 1 - mask
        mask_ = self.mask_update(mask_)
        if down:
            mask_ = self.mask_down(mask_)
        return 1 - mask_

    def forward(self, depth, mask):
        x1 = self.conv_1_1(depth, mask)
        x2 = self.conv_1_2(depth, mask)
        x3 = self.conv_1_3(depth, mask)
        depth = self.conv_1_fuse(torch.cat([x1, x2, x3], dim=1))
        mask = self.mask_update_rule(mask, True)

        x1 = self.conv_2_1(depth, mask)
        x2 = self.conv_2_2(depth, mask)
        x3 = self.conv_2_3(depth, mask)
        depth = self.conv_2_fuse(torch.cat([x1, x2, x3], dim=1))
        mask = self.mask_update_rule(mask)
        return depth, mask


class DepthEncoder(nn.Module):
    def __init__(self, inner_ch=64, layers=4):
        super().__init__()
        self.layers = layers
        channel_mults = [1]
        channel_mults = channel_mults + [inner_ch * pow(2, i) for i in range(layers)]
        self.encoder_blocks = nn.ModuleList([])
        self.mask_blocks = nn.ModuleList([])
        split_1 = [3, 3, 3, 3, 3]
        split_2 = [5, 5, 5, 5, 5]
        split_3 = [7, 7, 7, 7, 7]

        for layer in range(self.layers):
            in_ch = channel_mults[layer]
            out_ch = channel_mults[layer + 1]
            self.encoder_blocks.append(DepthEncoderLayer(
                in_ch, out_ch, split_1[layer], split_2[layer], split_3[layer]))

    def forward(self, depth, mask):
        out = []
        for layer in range(self.layers):
            depth, mask = self.encoder_blocks[layer](depth, mask)
            out.append(depth)
        out.reverse()
        return out
