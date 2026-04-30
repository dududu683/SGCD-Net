import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp
import numpy as np

class PerceptualLoss(nn.Module):

    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)

        self.layers = nn.Sequential(*list(vgg.children())[:23])
        for param in self.layers.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))

    def forward(self, pred, target):

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred_feat = self.layers(pred)
        target_feat = self.layers(target)
        return F.l1_loss(pred_feat, target_feat)


class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        window = self.window.to(pred.device)
        channel = pred.shape[1]
        if channel == self.channel and self.window.shape[1] == self.channel:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(pred.device)
        return 1 - self._ssim(pred, target, window, self.window_size, channel, self.size_average)


class CombinedLoss(nn.Module):

    def __init__(self, device='cuda', lambda_l1=1.0, lambda_perc=0.1, lambda_ssim=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.perceptual_loss = PerceptualLoss(device)
        self.ssim_loss = SSIMLoss()

    def predict_x0(self, x_t, noise_pred, t, alphas_cumprod):

        alpha_bar = alphas_cumprod[t].view(-1,1,1,1)
        pred_x0 = (x_t - noise_pred * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)

        pred_x0 = torch.clamp(pred_x0, -1, 1)
        return pred_x0

    def forward(self, noise_pred, noise, x_t, x0_target, t, alphas_cumprod):


        loss_l1 = F.l1_loss(noise_pred, noise)


        pred_x0 = self.predict_x0(x_t, noise_pred, t, alphas_cumprod)

        pred_x0_01 = (pred_x0 + 1) / 2
        x0_target_01 = (x0_target + 1) / 2

        loss_perc = self.perceptual_loss(pred_x0_01, x0_target_01)

        loss_ssim = self.ssim_loss(pred_x0_01, x0_target_01)

        total_loss = (self.lambda_l1 * loss_l1 +
                      self.lambda_perc * loss_perc +
                      self.lambda_ssim * loss_ssim)

        return total_loss, {
            'l1': loss_l1.item(),
            'perceptual': loss_perc.item(),
            'ssim': loss_ssim.item()
        }