#! -*- coding: utf-8 -*-
# Flow Matching (Rectified Flow 形式) —— 与 ddpm.py 同骨架的对照实现
# 复用 ddpm.py 的 UNet / 数据管道 / LAMB+EMA,只有三处不同：
#   1. 加噪方式：x_t = t * x1 + (1-t) * x0（线性插值,t=0 纯噪声 → t=1 数据）
#   2. 训练目标：回归恒定速度场 v = x1 - x0（而非预测噪声）
#   3. 采样方式：Euler 积分 dx = v * dt,默认 100 步（而非 1000 步随机去噪）
# 参考：Lipman et al. Flow Matching (2023) / Liu et al. Rectified Flow (2023)
# 中文推导：kexue.fm/archives/9497（生成扩散模型漫谈(十四)(十五)(十七)）
# 用法：DDPM_DATA_DIR=/path/to/CelebA-HQ python flow_matching.py

import os

import numpy as np
import torch
from tqdm import tqdm

from ddpm import (
    EMA, LAMB, FaceDataset, UNet, T, batch_size, data_dir, device, img_size,
    imwrite, list_pictures, piecewise_linear_lr, steps_per_epoch,
)
from torch.utils.data import DataLoader

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples_fm')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_fm.pt')
os.makedirs(SAMPLES_DIR, exist_ok=True)

fm_model = UNet().to(device)


def t_to_index(t):
    """连续 t∈[0,1] 映射到 UNet 的离散 Embedding 下标(复用 DDPM 的 t 编码)"""
    return (t * (T - 1)).round().long().clamp(0, T - 1)


@torch.no_grad()
def sample(path=None, n=4, z_samples=None, steps=100, net=None):
    """Euler 积分采样：从 t=0 的纯噪声走到 t=1 的数据"""
    net = net or fm_model
    net.eval()
    if z_samples is None:
        x = torch.randn(n**2, 3, img_size, img_size, device=device)
    else:
        x = z_samples.clone()
    dt = 1.0 / steps
    for i in tqdm(range(steps), ncols=0):
        t = torch.full((x.shape[0],), i * dt, device=device)
        v = net(x, t_to_index(t))
        x = x + v * dt
    net.train()
    x = x.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()
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
    print(f'{len(paths)} images')

    loader = DataLoader(
        FaceDataset(paths), batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
    )

    norm_params, other_params = [], []
    for name, p in fm_model.named_parameters():
        (norm_params if 'norm' in name.lower() else other_params).append(p)
    optimizer = LAMB([
        dict(params=other_params, layer_adaptation=True),
        dict(params=norm_params, layer_adaptation=False),
    ], lr=1e-3)

    ema = EMA(fm_model)
    base_lr = 1e-3
    initial_epoch = int(os.environ.get('DDPM_INITIAL_EPOCH', 0))
    global_step = initial_epoch * steps_per_epoch
    if initial_epoch > 0:
        ckpt = torch.load(CKPT, map_location=device)
        fm_model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f'resumed from {CKPT} @ epoch {initial_epoch}')

    epoch, running, seen = initial_epoch, 0.0, 0
    data_iter = iter(loader)
    fm_model.train()
    pbar = tqdm(total=steps_per_epoch, ncols=0, desc=f'Epoch {epoch + 1}')
    while True:
        try:
            x1 = next(data_iter)  # 数据端(t=1)
        except StopIteration:
            data_iter = iter(loader)
            x1 = next(data_iter)
        x1 = x1.to(device, non_blocking=True)
        bs = x1.shape[0]
        x0 = torch.randn_like(x1)                      # 噪声端(t=0)
        t = torch.rand(bs, device=device)              # 连续时刻
        xt = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0
        v_target = x1 - x0                             # 直线路径 → 恒定速度

        v_pred = fm_model(xt, t_to_index(t))
        loss = ((v_target - v_pred)**2).sum(dim=(1, 2, 3)).mean()  # 与 ddpm.py 同尺度
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for g in optimizer.param_groups:
            g['lr'] = base_lr * piecewise_linear_lr(global_step + 1)
        optimizer.step()
        ema.update(fm_model)

        global_step += 1
        running += loss.item() * bs
        seen += bs
        pbar.update(1)
        pbar.set_postfix(loss=f'{running / seen:.1f}')

        if global_step % steps_per_epoch == 0:
            pbar.close()
            epoch += 1
            torch.save({'model': fm_model.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch}, CKPT)
            sample(os.path.join(SAMPLES_DIR, '%05d.png' % epoch))
            ema_net = UNet().to(device)
            sd = fm_model.state_dict()
            sd.update(ema.state_dict())
            ema_net.load_state_dict(sd)
            sample(os.path.join(SAMPLES_DIR, '%05d_ema.png' % epoch), net=ema_net)
            del ema_net
            running, seen = 0.0, 0
            pbar = tqdm(total=steps_per_epoch, ncols=0, desc=f'Epoch {epoch + 1}')


if __name__ == '__main__':
    main()
