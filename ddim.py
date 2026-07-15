#! -*- coding: utf-8 -*-
# DDIM 采样 —— PyTorch 版,对照 ../ddim.py
# DDIM 不改变训练,只修改采样过程:用待定系数法解出与 DDPM 训练目标兼容的
# 非马尔可夫反向过程族,eta 控制方差(eta=1 退化为 DDPM,eta=0 完全确定),stride 跳步加速
# 博客:https://kexue.fm/archives/9181
# 用法:python ddim.py(需先训练 ddpm.py 得到 model.pt)

import os

import numpy as np
import torch
from tqdm import tqdm

from ddpm import CKPT, UNet, T, bar_alpha, device, img_size, imwrite

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_net(use_ema=True):
    """加载训练好的去噪网络(默认用 EMA 权重,画质更好)"""
    net = UNet().to(device)
    ckpt = torch.load(CKPT, map_location=device)
    sd = ckpt['model']
    if use_ema:
        sd = dict(sd)
        sd.update(ckpt['ema'])
    net.load_state_dict(sd)
    net.eval()
    print(f"loaded {CKPT} @ epoch {ckpt.get('epoch', '?')} (ema={use_ema})")
    return net


@torch.no_grad()
def sample(net, path=None, n=4, z_samples=None, stride=1, eta=1.0):
    """DDIM 采样:eta 控制方差相对大小,stride 控制跳步(T/stride 步完成)"""
    # 采样参数(与 Keras 版逐行相同,纯 numpy 标量计算)
    bar_alpha_ = bar_alpha[::stride]
    bar_alpha_pre_ = np.pad(bar_alpha_[:-1], [1, 0], constant_values=1)
    bar_beta_ = np.sqrt(1 - bar_alpha_**2)
    bar_beta_pre_ = np.sqrt(1 - bar_alpha_pre_**2)
    alpha_ = bar_alpha_ / bar_alpha_pre_
    sigma_ = bar_beta_pre_ / bar_beta_ * np.sqrt(1 - alpha_**2) * eta
    epsilon_ = bar_beta_ - alpha_ * np.sqrt(bar_beta_pre_**2 - sigma_**2)
    T_ = len(bar_alpha_)
    # 采样过程
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


@torch.no_grad()
def sample_inter(net, path, n=4, k=8, stride=1):
    """球面插值采样:两个随机噪声间 slerp,eta=0 确定性生成,展示语义连续性"""
    figure = np.ones((img_size * n, img_size * k, 3))
    Z = torch.randn(n * 2, 3, img_size, img_size)
    z_samples = []
    for i in range(n):
        for j in range(k):
            theta = np.pi / 2 * j / (k - 1)
            z = Z[2 * i] * np.sin(theta) + Z[2 * i + 1] * np.cos(theta)
            z_samples.append(z)
    x = sample(net, z_samples=torch.stack(z_samples), stride=stride, eta=0)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size,
                   img_size * j:img_size * (j + 1)] = x[ij]
    imwrite(path, figure)


if __name__ == '__main__':
    net = load_net()
    sample(net, os.path.join(OUT_DIR, 'test.png'), n=4, stride=100, eta=0)
    sample_inter(net, os.path.join(OUT_DIR, 'test_inter.png'), n=8, k=15, stride=20)
