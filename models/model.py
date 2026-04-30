import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import timestep_embedding


class ColorStatisticalPrior(nn.Module):

    def __init__(self, in_channels=3, hidden_dim=64, stats_dim=12):
        super().__init__()
        self.stats_dim = stats_dim

        self.fc = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.proj = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        mean = x.view(B, C, -1).mean(dim=2)          # (B,C)

        var = x.view(B, C, -1).var(dim=2)            # (B,C)

        centered = x - mean.view(B, C, 1, 1)
        skew = (centered ** 3).view(B, C, -1).mean(dim=2) / (var ** 1.5 + 1e-8)

        kurt = (centered ** 4).view(B, C, -1).mean(dim=2) / (var ** 2 + 1e-8) - 3

        stats = torch.cat([mean, var, skew, kurt], dim=1)  # (B, 12)

        stats_feat = self.fc(stats)                   # (B, hidden_dim)

        stats_feat_map = stats_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, hidden_dim, H, W)
        stats_feat_map = self.proj(stats_feat_map)     # (B, hidden_dim, H, W)
        return stats_feat_map


# ---------- Cross Statistical-guided Attention ----------
class CrossStatisticalGuidedAttention(nn.Module):

    def __init__(self, channels, stat_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels

        self.stat_encoder = nn.Sequential(
            nn.Conv2d(stat_channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU()
        )

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.pos_encoder = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x, stat_map):
        B, C, H, W = x.shape

        stat_feat = self.stat_encoder(stat_map)
        stat_feat = stat_feat + self.pos_encoder(stat_feat)
        q = self.q_proj(stat_feat.flatten(2).transpose(1, 2))


        x_flat = x.flatten(2).transpose(1, 2)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = self.out_proj(out.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        return x + out



class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(channels).float()) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y



class FrequencyAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(x_fft)
        mag_pool = self.avg_pool(mag)
        mag_pool = mag_pool.view(B, C)
        attn = self.fc(mag_pool).view(B, C, 1, 1)
        return x * attn



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, use_scale_shift_norm=True, dropout=0.0):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * (2 if use_scale_shift_norm else 1)),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.out_layers[0](h) * (scale + 1) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        h = (attn @ v).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h



class Downsample(nn.Module):
    def __init__(self, channels, conv_resample=True):
        super().__init__()
        if conv_resample:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        return self.op(x)



class Upsample(nn.Module):
    def __init__(self, channels, conv_resample=True):
        super().__init__()
        if conv_resample:
            self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        else:
            self.op = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.op(x)



class ConditionedUNet(nn.Module):

    def __init__(self, in_channels=3, model_channels=64, out_channels=3, num_res_blocks=2,
                 attention_resolutions=(16,), channel_mult=(1, 2, 4, 8), conv_resample=True,
                 num_heads=4, use_scale_shift_norm=True):
        super().__init__()
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm

        # 时间步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )


        self.cond_encoder = nn.Sequential(
            nn.Conv2d(in_channels, model_channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(model_channels // 2, model_channels, 3, padding=1),
        )


        self.stat_prior = ColorStatisticalPrior(in_channels=in_channels, hidden_dim=model_channels)
        self.stat_proj = nn.Conv2d(model_channels, model_channels, 1)  # 用于融合


        self.input_conv = nn.Conv2d(in_channels + model_channels, model_channels, 3, padding=1)


        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        channels = model_channels
        ds = 1
        self.skip_channels = []
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(channels, out_ch, time_embed_dim,
                             use_scale_shift_norm=use_scale_shift_norm)
                ]
                channels = out_ch
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(channels, num_heads))
                self.down_blocks.append(nn.Sequential(*layers))
                self.down_attentions.append(
                    nn.Sequential(
                        ECAAttention(channels),
                        FrequencyAttention(channels)
                    )
                )
                self.skip_channels.append(channels)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(Downsample(channels, conv_resample))
                self.down_attentions.append(nn.Identity())
                ds *= 2
                self.skip_channels.append(channels)


        self.middle_block = nn.Sequential(
            ResBlock(channels, channels, time_embed_dim, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(channels, num_heads),
            ResBlock(channels, channels, time_embed_dim, use_scale_shift_norm=use_scale_shift_norm),
        )


        self.stat_guided_attn = CrossStatisticalGuidedAttention(
            channels=channels,
            stat_channels=model_channels,
            num_heads=num_heads
        )


        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        skip_index = len(self.skip_channels) - 1
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for _ in range(num_res_blocks + 1):
                enc_ch = self.skip_channels[skip_index]
                layers = [
                    ResBlock(channels + enc_ch, out_ch, time_embed_dim,
                             use_scale_shift_norm=use_scale_shift_norm)
                ]
                channels = out_ch
                skip_index -= 1
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(channels, num_heads))
                self.up_blocks.append(nn.Sequential(*layers))
                self.up_attentions.append(nn.Identity())
            if level != 0:
                self.up_blocks.append(Upsample(channels, conv_resample))
                self.up_attentions.append(nn.Identity())
                ds //= 2


        self.out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(self, x_t, cond, t):

        temb = timestep_embedding(t, self.model_channels)
        temb = self.time_embed(temb)


        stat_map = self.stat_prior(cond)          # (B, model_channels, H, W)
        stat_feat = self.stat_proj(stat_map)      # (B, model_channels, H, W)

        cond_feat = self.cond_encoder(cond)       # (B, model_channels, H, W)
        cond_feat = cond_feat + stat_feat


        inp = torch.cat([x_t, cond_feat], dim=1)
        h = self.input_conv(inp)

        skips = []


        for block, attn in zip(self.down_blocks, self.down_attentions):
            if isinstance(block, Downsample):
                h = block(h)
                skips.append(h)
            else:
                if isinstance(block, nn.Sequential):
                    for subblock in block:
                        if isinstance(subblock, ResBlock):
                            h = subblock(h, temb)
                        else:
                            h = subblock(h)
                else:
                    h = block(h, temb) if isinstance(block, ResBlock) else block(h)
                h = attn(h)
                skips.append(h)


        h = self.middle_block[0](h, temb)
        h = self.middle_block[1](h)
        h = self.stat_guided_attn(h, stat_map)
        h = self.middle_block[2](h, temb)


        for block, attn in zip(self.up_blocks, self.up_attentions):
            if isinstance(block, Upsample):
                h = block(h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                if isinstance(block, nn.Sequential):
                    for subblock in block:
                        if isinstance(subblock, ResBlock):
                            h = subblock(h, temb)
                        else:
                            h = subblock(h)
                else:
                    h = block(h, temb) if isinstance(block, ResBlock) else block(h)
                h = attn(h)

        out = self.out(h)
        return out