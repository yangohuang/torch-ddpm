#! -*- coding: utf-8 -*-
# Analytic-DPM —— PyTorch 版,对照 ../adpm.py
# 在 DDIM 上修改:不改训练,只把采样方差换成解析最优解
# 关键量 factors[t] = 1 - E||eps_pred||^2 —— 用数据估计每个 t 的方差修正项
# 博客:https://kexue.fm/archives/9245
# 用法:python adpm.py(需先训练 ddpm.py 得到 model.pt;factors 估计约需数分钟)

import os

import numpy as np
import torch
from tqdm import tqdm

from ddim import load_net
from ddpm import (
    FaceDataset, T, bar_alpha, bar_beta, data_dir, device, img_size, imwrite,
    list_pictures,
)
from torch.utils.data import DataLoader

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORS_CACHE = os.path.join(OUT_DIR, 'adpm_factors.npy')


@torch.no_grad()
def estimate_factors(net, n_batches=5, batch_size=64):
    """对每个 t:用真实图片造 x_t,估计 E||eps_pred||^2,得方差修正项 factors[t]"""
    paths = list_pictures(os.path.join(data_dir, 'train'))
    paths += list_pictures(os.path.join(data_dir, 'valid'))  # 与 Keras 版同源(train+valid)
    loader = DataLoader(FaceDataset(paths), batch_size=batch_size, shuffle=True,
                        num_workers=4, drop_last=True)
    # 预取 n_batches 份数据,循环给每个 t 用(样本量与 Keras 版 steps=5 等价;
    # 差异:Keras 每个 t 重新抽图,此处固定 5 个 batch 复用于所有 t,方差略小)
    batches = []
    for x0 in loader:
        batches.append(x0.to(device))
        if len(batches) >= n_batches:
            break
    factors = np.zeros(T)
    for t in tqdm(range(T), ncols=0, desc='estimating factors'):
        acc, cnt = 0.0, 0
        for x0 in batches:
            noise = torch.randn_like(x0)
            xt = x0 * bar_alpha[t] + noise * bar_beta[t]
            bt = torch.full((x0.shape[0],), t, dtype=torch.long, device=device)
            pred = net(xt, bt)
            acc += float((pred**2).mean()) * x0.shape[0]
            cnt += x0.shape[0]
        factors[t] = acc / cnt
    return np.clip(1 - factors, 0, 1)


@torch.no_grad()
def sample(net, factors, path=None, n=4, z_samples=None, stride=1, eta=1.0):
    """DDIM 采样 + Analytic-DPM 方差修正(比原版只多 gamma_/sigma_ 两行)"""
    bar_alpha_ = bar_alpha[::stride]
    bar_alpha_pre_ = np.pad(bar_alpha_[:-1], [1, 0], constant_values=1)
    bar_beta_ = np.sqrt(1 - bar_alpha_**2)
    bar_beta_pre_ = np.sqrt(1 - bar_alpha_pre_**2)
    alpha_ = bar_alpha_ / bar_alpha_pre_
    sigma_ = bar_beta_pre_ / bar_beta_ * np.sqrt(1 - alpha_**2) * eta
    epsilon_ = bar_beta_ - alpha_ * np.sqrt(bar_beta_pre_**2 - sigma_**2)
    gamma_ = epsilon_ * bar_alpha_pre_ / bar_alpha_              # Analytic-DPM 新增
    sigma_ = np.sqrt(sigma_**2 + gamma_**2 * factors[::stride])  # Analytic-DPM 新增
    T_ = len(bar_alpha_)
    if z_samples is None:
        z = torch.randn(n**2, 3, img_size, img_size, device=device)
    else:
        z = z_samples.clone().to(device)
    for step in tqdm(range(T_), ncols=0):
        t = T_ - step - 1
        bt = torch.full((z.shape[0],), t * stride, dtype=torch.long, device=device)
        z -= epsilon_[t] * net(z, bt)
        z /= alpha_[t]
        z += torch.randn_like(z) * sigma_[t]
    x = z.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()
    if path is None:
        return x
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = x[i * n + j]
    imwrite(path, figure)


if __name__ == '__main__':
    net = load_net()
    if os.path.exists(FACTORS_CACHE):
        factors = np.load(FACTORS_CACHE)
        print(f'loaded cached factors from {FACTORS_CACHE}')
    else:
        factors = estimate_factors(net)
        np.save(FACTORS_CACHE, factors)
    sample(net, factors, os.path.join(OUT_DIR, 'test_adpm.png'), n=8, stride=100, eta=1)
