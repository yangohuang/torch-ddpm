#! -*- coding: utf-8 -*-
# DDPM 参考代码2 —— PyTorch 版,对照 ../ddpm2.py
# 这版 U-Net 尽量贴近 DDPM 原论文(除了没加 Attention),效果更好,计算量也更大。
# 与 ddpm.py(简化版)的差异:
#   1. skip 连接用 Concat(原论文做法),不是 Add
#   2. 残差块是 pre-norm:GN→swish→conv→(+t)→GN→swish→conv(末层零初始化)
#   3. t 编码:固定正弦 Embedding + 两层 swish MLP(不是可学习查表)
#   4. 每级(除最后一级)都下采样,上行是 blocks+1 个残差块
# 博客:https://kexue.fm/archives/9152
# 用法:DDPM_DATA_DIR=... python ddpm2.py;产物 model2.pt / samples2/

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm import (
    EMA, LAMB, FaceDataset, T, alpha, bar_alpha, bar_beta, beta, batch_size,
    data_dir, device, imwrite, list_pictures, make_conv, make_dense,
    piecewise_linear_lr, sigma, steps_per_epoch, variance_scaling_,
)

img_size = 128
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
blocks = 2

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples2')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model2.pt')
os.makedirs(SAMPLES_DIR, exist_ok=True)


def sinusoidal_table(n, dim):
    """固定正弦位置编码表(对照 Keras 的 Sinusoidal initializer)"""
    pos = np.arange(n)[:, None]
    i = np.arange(dim // 2)[None]
    angle = pos / np.power(10000, 2 * i / dim)
    table = np.zeros((n, dim), dtype='float32')
    table[:, 0::2] = np.sin(angle)
    table[:, 1::2] = np.cos(angle)
    return torch.from_numpy(table)


class ResidualBlock2(nn.Module):
    """pre-norm 残差块(对照 ../ddpm2.py 的 residual_block):
    GN → swish → conv → (+t) → GN → swish → conv(零初始化) → +shortcut
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_dim, eps=1e-6)
        self.conv1 = make_conv(in_dim, out_dim)
        self.t_proj = make_dense(embedding_size * 4, out_dim)
        self.norm2 = nn.GroupNorm(32, out_dim, eps=1e-6)
        self.conv2 = make_conv(out_dim, out_dim, 0)  # init_scale=0:残差初始为恒等
        self.shortcut = None if in_dim == out_dim else make_dense(in_dim, out_dim)

    def forward(self, x, t_emb):
        if self.shortcut is None:
            xi = x
        else:
            xi = self.shortcut(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.t_proj(t_emb)[:, :, None, None]
        x = self.conv2(F.silu(self.norm2(x)))
        return x + xi


class UNet2(nn.Module):
    """贴近原论文的 U-Net(Concat skip)"""
    def __init__(self):
        super().__init__()
        self.register_buffer('t_table', sinusoidal_table(T, embedding_size))
        self.t_mlp = nn.Sequential(
            make_dense(embedding_size, embedding_size * 4), nn.SiLU(),
            make_dense(embedding_size * 4, embedding_size * 4), nn.SiLU(),
        )
        self.stem = make_conv(3, embedding_size)

        self.down_blocks = nn.ModuleList()
        self.down_plan = []
        dims = [embedding_size]
        dim = embedding_size
        for i, ch in enumerate(channels):
            for _ in range(blocks):
                out_dim = ch * embedding_size
                self.down_blocks.append(ResidualBlock2(dim, out_dim))
                self.down_plan.append('res')
                dim = out_dim
                dims.append(dim)
            if i != len(channels) - 1:
                self.down_plan.append('pool')
                dims.append(dim)
        self.middle = ResidualBlock2(dim, dim)

        self.up_blocks = nn.ModuleList()
        self.up_plan = []
        for i, ch in enumerate(channels[::-1]):
            for _ in range(blocks + 1):          # 上行每级 blocks+1 个块
                skip_dim = dims.pop()
                self.up_blocks.append(ResidualBlock2(dim + skip_dim, ch * embedding_size))
                self.up_plan.append('res')
                dim = ch * embedding_size
            if i != len(channels) - 1:
                self.up_plan.append('up')
        self.out_norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.out_conv = make_conv(dim, 3)

    def forward(self, x, t):
        t_emb = self.t_mlp(self.t_table[t.reshape(-1)])
        x = self.stem(x)
        stack = [x]
        di = 0
        for op in self.down_plan:
            if op == 'res':
                x = self.down_blocks[di](x, t_emb)
                di += 1
                stack.append(x)
            else:
                x = F.avg_pool2d(x, 2)
                stack.append(x)          # 池化结果也入栈(Keras 版 inputs.append 对应行)
        x = self.middle(x, t_emb)
        ui = 0
        for op in self.up_plan:
            if op == 'res':
                x = torch.cat([x, stack.pop()], dim=1)   # Concat skip(原论文风格)
                x = self.up_blocks[ui](x, t_emb)
                ui += 1
            else:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.out_conv(F.silu(self.out_norm(x)))


model2 = UNet2().to(device)


@torch.no_grad()
def sample(path=None, n=4, z_samples=None, t0=0, net=None):
    net = net or model2
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
    print(f'{len(paths)} images, params: {sum(p.numel() for p in model2.parameters()):,}')

    loader = DataLoader(FaceDataset(paths), batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True,
                        persistent_workers=True)

    norm_p, other_p = [], []
    for name, p in model2.named_parameters():
        (norm_p if 'norm' in name.lower() else other_p).append(p)
    optimizer = LAMB([dict(params=other_p, layer_adaptation=True),
                      dict(params=norm_p, layer_adaptation=False)], lr=1e-3)

    ema = EMA(model2)
    base_lr = 1e-3
    initial_epoch = int(os.environ.get('DDPM_INITIAL_EPOCH', 0))
    global_step = initial_epoch * steps_per_epoch
    if initial_epoch > 0:
        ckpt = torch.load(CKPT, map_location=device)
        model2.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])

    _ba = torch.from_numpy(bar_alpha).float().to(device)
    _bb = torch.from_numpy(bar_beta).float().to(device)
    epoch, running, seen = initial_epoch, 0.0, 0
    data_iter = iter(loader)
    model2.train()
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

        pred = model2(xt, t)
        loss = ((noise - pred)**2).sum(dim=(1, 2, 3)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for g in optimizer.param_groups:
            g['lr'] = base_lr * piecewise_linear_lr(global_step + 1)
        optimizer.step()
        ema.update(model2)

        global_step += 1
        running += loss.item() * bs
        seen += bs
        pbar.update(1)
        pbar.set_postfix(loss=f'{running / seen:.1f}')

        if global_step % steps_per_epoch == 0:
            pbar.close()
            epoch += 1
            torch.save({'model': model2.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch}, CKPT)
            sample(os.path.join(SAMPLES_DIR, '%05d.png' % epoch))
            ema_net = UNet2().to(device)
            sd = model2.state_dict()
            sd.update(ema.state_dict())
            ema_net.load_state_dict(sd)
            sample(os.path.join(SAMPLES_DIR, '%05d_ema.png' % epoch), net=ema_net)
            del ema_net
            running, seen = 0.0, 0
            pbar = tqdm(total=steps_per_epoch, ncols=0, desc=f'Epoch {epoch + 1}')


if __name__ == '__main__':
    main()
