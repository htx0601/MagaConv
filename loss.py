import torch
import torch.nn as nn


class Loss_with_laplace(nn.Module):
    def __init__(self):
        super(Loss_with_laplace, self).__init__()
        self.laplace = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3).cuda()

    def forward(self, output, gt, mask=None):
        b_gt = torch.conv2d(gt, self.laplace, stride=1, padding=1)
        b_out = torch.conv2d(output, self.laplace, stride=1, padding=1)
        b_loss = torch.sqrt(torch.mean((b_gt - b_out) ** 2))

        err = output - gt
        if mask is not None:
            mask_g = (mask == 0).detach() & (gt > 1e-5).detach()
        else:
            mask_g = (gt > 1e-5).detach()

        mse_loss = torch.sqrt(torch.mean((err[mask_g]) ** 2))

        return 0.8 * mse_loss + 0.2 * b_loss