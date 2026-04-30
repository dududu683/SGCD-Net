import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class DiffusionModel:
    def __init__(self, model, betas, device):
        self.model = model
        self.betas = betas.to(device)
        self.device = device
        self.num_timesteps = len(betas)

        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)


        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample_loop(self, cond, shape, num_steps=None, only_last=True, ddim=False, ddim_eta=0.0):

        self.model.eval()
        with torch.no_grad():
            if ddim:
                return self.ddim_sample_loop(cond, shape, num_steps, ddim_eta, only_last)
            else:
                return self.ddpm_sample_loop(cond, shape, num_steps, only_last)

    def ddpm_sample_loop(self, cond, shape, num_steps=None, only_last=True):

        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if num_steps is None:
            timesteps = list(range(self.num_timesteps))[::-1]
        else:

            timesteps = np.linspace(0, self.num_timesteps-1, num_steps, dtype=int)[::-1].tolist()

        imgs = []
        for i, t in enumerate(tqdm(timesteps, desc='DDPM sampling')):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            noise_pred = self.model(img, cond, t_batch)

            img = self.p_sample_ddpm(img, t_batch, noise_pred)
            if not only_last:
                imgs.append(img.cpu())
        if only_last:
            return img
        else:
            return imgs

    def p_sample_ddpm(self, x_t, t, noise_pred):

        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alphas_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)


        pred_x0 = (x_t - noise_pred * torch.sqrt(1 - alphas_cumprod_t)) / torch.sqrt(alphas_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        posterior_mean = (betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1 - alphas_cumprod_t)) * pred_x0 + \
                         ((1 - alphas_cumprod_prev_t) * torch.sqrt(self.alphas[t].view(-1,1,1,1)) / (1 - alphas_cumprod_t)) * x_t
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x_t) if t[0].item() > 0 else 0
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    def ddim_sample_loop(self, cond, shape, num_steps=50, eta=0.0, only_last=True):

        device = self.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        step_ratio = self.num_timesteps // num_steps
        timesteps = (np.arange(0, num_steps) * step_ratio).tolist()[::-1]

        alphas = self.alphas_cumprod
        alphas_prev = torch.cat([alphas[:1], alphas[:-1]], dim=0)  # alpha_{t-1}

        imgs = []
        for i, t in enumerate(tqdm(timesteps, desc='DDIM sampling')):
            t_cur = torch.full((b,), t, device=device, dtype=torch.long)

            noise_pred = self.model(img, cond, t_cur)

            alpha_cur = alphas[t]
            alpha_prev = alphas_prev[t] if t > 0 else alphas[0]

            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_cur)) * torch.sqrt(1 - alpha_cur / alpha_prev)

            pred_x0 = (img - noise_pred * torch.sqrt(1 - alpha_cur)) / torch.sqrt(alpha_cur)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * noise_pred

            noise = sigma * torch.randn_like(img) if eta > 0 else 0
            img = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
            if not only_last:
                imgs.append(img.cpu())
        if only_last:
            return img
        else:
            return imgs