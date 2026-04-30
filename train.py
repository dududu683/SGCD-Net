import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import numpy as np


from models.model import ConditionedUNet
from models.diffusion import DiffusionModel
from models.utils import get_named_beta_schedule
from loss import CombinedLoss


class PairedImageDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir, transform=None, image_size=256):
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.image_size = image_size

        self.filenames = sorted(os.listdir(degraded_dir))
        gt_filenames = sorted(os.listdir(gt_dir))
        assert len(self.filenames) == len(gt_filenames), "文件夹中文件数量不匹配"

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        degraded_path = os.path.join(self.degraded_dir, self.filenames[idx])
        gt_path = os.path.join(self.gt_dir, self.filenames[idx])

        degraded_img = Image.open(degraded_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        degraded_img = self.transform(degraded_img)
        gt_img = self.transform(gt_img)

        return {
            'degraded': degraded_img,
            'clear': gt_img,
            'filename': self.filenames[idx]
        }

# -------------------- 训练主函数 --------------------
def train(config):
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    train_dataset = PairedImageDataset(
        degraded_dir=config['degraded_dir'],
        gt_dir=config['gt_dir'],
        image_size=config['image_size']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    print(f"Dataset loaded: {len(train_dataset)} images")


    model = ConditionedUNet(
        in_channels=3,
        model_channels=config['model_channels'],
        out_channels=3,
        num_res_blocks=config['num_res_blocks'],
        channel_mult=config['channel_mult'],
        attention_resolutions=config['attention_resolutions']
    ).to(device)


    betas = get_named_beta_schedule(config['beta_schedule'], config['num_timesteps']).to(device)

    betas = betas.float()
    diffusion = DiffusionModel(model, betas, device)


    criterion = CombinedLoss(device=device,
                             lambda_l1=config['lambda_l1'],
                             lambda_perc=config['lambda_perc'],
                             lambda_ssim=config['lambda_ssim'])


    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=config['min_lr'])


    os.makedirs(config['save_dir'], exist_ok=True)


    best_loss = float('inf')
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        total_loss = 0.0
        loss_l1_total = 0.0
        loss_perc_total = 0.0
        loss_ssim_total = 0.0

        for batch_idx, batch in enumerate(train_loader):

            degraded = batch['degraded'].to(device).float()
            clear = batch['clear'].to(device).float()


            t = torch.randint(0, diffusion.num_timesteps, (clear.shape[0],), device=device).long()
            noise = torch.randn_like(clear)
            x_t = diffusion.q_sample(clear, t, noise)


            noise_pred = model(x_t, degraded, t)


            loss, loss_dict = criterion(noise_pred, noise, x_t, clear, t, diffusion.alphas_cumprod)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loss_l1_total += loss_dict['l1']
            loss_perc_total += loss_dict['perceptual']
            loss_ssim_total += loss_dict['ssim']

            if (batch_idx + 1) % config['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_l1 = loss_l1_total / (batch_idx + 1)
                avg_perc = loss_perc_total / (batch_idx + 1)
                avg_ssim = loss_ssim_total / (batch_idx + 1)
                print(f"Epoch {epoch}/{config['num_epochs']} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Total: {avg_loss:.6f} | L1: {avg_l1:.6f} | Perc: {avg_perc:.6f} | SSIM: {avg_ssim:.6f}")

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.6f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate updated to: {current_lr:.2e}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"Best model saved with loss {best_loss:.6f}")

        if epoch % config['save_interval'] == 0:
            torch.save(model.state_dict(), os.path.join(config['save_dir'], f'epoch_{epoch}.pth'))

    print("Training completed!")


if __name__ == '__main__':
    config = {

        'degraded_dir': 'UIEB_data/dataset/train/raw',
        'gt_dir': 'UIEB_data/dataset/train/target',
        'save_dir': 'checkpoint',


        'device': 'cuda',
        'batch_size': 8,
        'num_workers': 4,
        'image_size': 256,
        'num_epochs': 200,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-4,
        'log_interval': 50,
        'save_interval': 20,


        'lambda_l1': 1.0,
        'lambda_perc': 0.1,
        'lambda_ssim': 0.1,


        'beta_schedule': 'linear',
        'num_timesteps': 1000,


        'model_channels': 64,
        'num_res_blocks': 2,
        'channel_mult': (1, 2, 4, 8),
        'attention_resolutions': (16,),
    }

    train(config)