#! -*- coding: utf-8 -*-
# DDPM + GAU(Gated Attention Unit)Transformer 去噪器 —— PyTorch 版,对照 ../ddpm-gau.py
# 思路:8×8 patch 化 → 256 个 token(192 维)→ 24 层 GAU(带 2D-RoPE)→ 逆 patch 化
# 这其实是一个早期的"DiT":用注意力替代卷积 U-Net 做扩散去噪
# GAU 参考:苏剑林 FLASH 论文解读 https://kexue.fm/archives/8934
# 用法:DDPM_DATA_DIR=... python ddpm_gau.py;产物 model_gau.pt / samples_gau/

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm import (
    EMA, LAMB, FaceDataset, T, alpha, bar_alpha, bar_beta, beta, batch_size,
    data_dir, device, img_size, imwrite, list_pictures, piecewise_linear_lr,
    sigma, steps_per_epoch,
)

hidden_size = 768
num_layers = 24
patch = 8
n_tokens = (img_size // patch)**2          # 16*16 = 256
patch_dim = patch * patch * 3              # 192

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples_gau')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_gau.pt')
os.makedirs(SAMPLES_DIR, exist_ok=True)


def sinusoidal_embeddings(pos, dim, base=10000):
    """bert4keras 同款:交错排列 [sin0,cos0,sin1,cos1,...]"""
    indices = torch.pow(float(base), -2 * torch.arange(dim // 2).float() / dim)
    emb = pos[:, None].float() * indices[None]
    emb = torch.stack([emb.sin(), emb.cos()], dim=-1)
    return emb.flatten(-2)  # (n, dim)


def rope_2d():
    """2D-RoPE:行/列坐标各编 64 维正弦,拼成 128 维(对照 ../ddpm-gau.py rope_2d)"""
    w = img_size // patch
    pos = torch.arange(w**2)
    pos1, pos2 = pos // w, pos % w
    return torch.cat([sinusoidal_embeddings(pos1, 64, 1000),
                      sinusoidal_embeddings(pos2, 64, 1000)], dim=1)  # (256,128)


def apply_rotary(sinusoidal, x):
    """交错版 RoPE(bert4keras apply_rotary_position_embeddings 同款)"""
    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, dim=-1)
    x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos_pos + x2 * sin_pos


class RMSNorm(nn.Module):
    """LayerNormalization(zero_mean=False, offset=False):只除均方根、只留 scale
    eps=1e-7 对齐 bert4keras 默认(epsilon or K.epsilon() = 1e-7)
    """
    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GAU(nn.Module):
    """Gated Attention Unit(FLASH):单头注意力 + 门控,替代 Attention+FFN 两件套
    u, v: 门控/值分支(units 维,swish);z→q,k: 共享 128 维,各自做 scale-offset
    """
    def __init__(self, hidden, units, key_size=128):
        super().__init__()
        self.key_size = key_size
        self.i_dense = nn.Linear(hidden, units * 2 + key_size)
        self.o_dense = nn.Linear(units, hidden)
        self.q_scaleoffset = nn.Parameter(torch.stack([torch.ones(key_size), torch.zeros(key_size)]))
        self.k_scaleoffset = nn.Parameter(torch.stack([torch.ones(key_size), torch.zeros(key_size)]))
        self.units = units

    def forward(self, x, pos):
        # bert4keras 的 i_dense 对整个投影输出施加 swish(含 q/k 路径的 z),此处保持一致
        u, v, z = F.silu(self.i_dense(x)).split([self.units, self.units, self.key_size], dim=-1)
        q = z * self.q_scaleoffset[0] + self.q_scaleoffset[1]
        k = z * self.k_scaleoffset[0] + self.k_scaleoffset[1]
        q, k = apply_rotary(pos, q), apply_rotary(pos, k)
        attn = torch.softmax(q @ k.transpose(-1, -2) / self.key_size**0.5, dim=-1)
        return self.o_dense(u * (attn @ v))


class GAUDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(patch_dim, hidden_size, bias=False)
        self.t_embedding = nn.Embedding(T, hidden_size)
        self.register_buffer('pos', rope_2d())
        self.norms = nn.ModuleList([RMSNorm(hidden_size) for _ in range(num_layers)])
        self.gaus = nn.ModuleList([GAU(hidden_size, hidden_size * 2) for _ in range(num_layers)])
        self.out_norm = RMSNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, patch_dim, bias=False)

    def patchify(self, x):
        B = x.shape[0]
        w = img_size // patch
        x = x.permute(0, 2, 3, 1)                                # BHWC(对齐 Keras)
        x = x.reshape(B, w, patch, w, patch, 3)
        x = x.permute(0, 1, 3, 2, 4, 5)                          # (B,w,w,8,8,3)
        return x.reshape(B, n_tokens, patch_dim)

    def unpatchify(self, x):
        B = x.shape[0]
        w = img_size // patch
        x = x.reshape(B, w, w, patch, patch, 3)
        x = x.permute(0, 1, 3, 2, 4, 5)                          # (B,w,8,w,8,3)
        x = x.reshape(B, img_size, img_size, 3)
        return x.permute(0, 3, 1, 2)                             # BCHW

    def forward(self, x, t):
        x = self.in_proj(self.patchify(x))
        x = x + self.t_embedding(t.reshape(-1))[:, None]         # t 广播到所有 token
        for norm, gau in zip(self.norms, self.gaus):
            x = x + gau(norm(x), self.pos)                       # pre-RMSNorm 残差
        return self.unpatchify(self.out_proj(self.out_norm(x)))


gau_model = GAUDenoiser().to(device)


@torch.no_grad()
def sample(path=None, n=4, z_samples=None, t0=0, net=None):
    net = net or gau_model
    net.eval()
    if z_samples is None:
        z = torch.randn(n**2, 3, img_size, img_size, device=device)
    else:
        z = z_samples.clone()
    for step in tqdm(range(t0, T), ncols=0):
        t = T - step - 1
        bt = torch.full((z.shape[0],), t, dtype=torch.long, device=device)
        z -= beta[t]**2 / bar_beta[t] * net(z, bt)
        z /= alpha[t]
        z += torch.randn_like(z) * sigma[t]
    net.train()
    x = z.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()
    if path is None:
        return x
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = x[i * n + j]
    imwrite(path, figure)


def main():
    paths = list_pictures(os.path.join(data_dir, 'train'))
    paths += list_pictures(os.path.join(data_dir, 'valid'))
    assert paths, f'no images found under {data_dir}'
    print(f'{len(paths)} images, params: {sum(p.numel() for p in gau_model.parameters()):,}')

    loader = DataLoader(FaceDataset(paths), batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True)

    norm_p, other_p = [], []
    for name, p in gau_model.named_parameters():
        (norm_p if ('norm' in name.lower() or 'scaleoffset' in name.lower() or 'bias' in name.lower())
         else other_p).append(p)
    optimizer = LAMB([dict(params=other_p, layer_adaptation=True),
                      dict(params=norm_p, layer_adaptation=False)], lr=1e-3)

    ema = EMA(gau_model)
    base_lr = 1e-3
    initial_epoch = int(os.environ.get('DDPM_INITIAL_EPOCH', 0))
    global_step = initial_epoch * steps_per_epoch
    if initial_epoch > 0:
        ckpt = torch.load(CKPT, map_location=device)
        gau_model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])

    _ba = torch.from_numpy(bar_alpha).float().to(device)
    _bb = torch.from_numpy(bar_beta).float().to(device)
    epoch, running, seen = initial_epoch, 0.0, 0
    data_iter = iter(loader)
    gau_model.train()
    pbar = tqdm(total=steps_per_epoch, ncols=0, desc=f'Epoch {epoch + 1}')
    while True:
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)
        x0 = x0.to(device, non_blocking=True)
        bs = x0.shape[0]
        t = torch.randint(0, T, (bs,), device=device)
        noise = torch.randn_like(x0)
        xt = x0 * _ba[t][:, None, None, None] + noise * _bb[t][:, None, None, None]

        pred = gau_model(xt, t)
        loss = ((noise - pred)**2).sum(dim=(1, 2, 3)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for g in optimizer.param_groups:
            g['lr'] = base_lr * piecewise_linear_lr(global_step + 1)
        optimizer.step()
        ema.update(gau_model)

        global_step += 1
        running += loss.item() * bs
        seen += bs
        pbar.update(1)
        pbar.set_postfix(loss=f'{running / seen:.1f}')

        if global_step % steps_per_epoch == 0:
            pbar.close()
            epoch += 1
            torch.save({'model': gau_model.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch}, CKPT)
            sample(os.path.join(SAMPLES_DIR, '%05d.png' % epoch))
            running, seen = 0.0, 0
            pbar = tqdm(total=steps_per_epoch, ncols=0, desc=f'Epoch {epoch + 1}')


if __name__ == '__main__':
    main()
