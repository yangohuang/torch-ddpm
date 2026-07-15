#! -*- coding: utf-8 -*-
# DDCM(Denoising Diffusion Codebook Models)—— PyTorch 版,对照 ../ddcm.py
# 不改训练,只改采样:把每步的随机噪声换成从固定 codebook 中选取
# → 随机选 = 生成;按"最贴近目标图残差"选 = 把图片编码成离散索引序列(压缩)
# 博客:https://kexue.fm/archives/10711
# 用法:python ddcm.py(需先训练 ddpm.py 得到 model.pt)

import os

import numpy as np
import torch
from tqdm import tqdm

from ddim import load_net
from ddpm import (
    T, alpha, bar_alpha, bar_beta, beta, data_dir, device, img_size, imread,
    imwrite, list_pictures, sigma,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
K = 64  # 每步的 Codebook 大小(每步编码 log2(64)=6 bit)

torch.manual_seed(42)  # codebook 是"公共随机数",编码端和解码端必须一致
codebook = torch.randn(T + 1, K, 3, img_size, img_size)


@torch.no_grad()
def sample(net, path, n=4):
    """随机采样:与 DDPM 唯一区别是噪声从 codebook 里抽,而不是现场 randn"""
    z = codebook[T][torch.randint(0, K, (n**2,))].to(device)
    for step in tqdm(range(T), ncols=0):
        t = T - step - 1
        bt = torch.full((z.shape[0],), t, dtype=torch.long, device=device)
        z -= beta[t]**2 / bar_beta[t] * net(z, bt)
        z /= alpha[t]
        z += codebook[t][torch.randint(0, K, (n**2,))].to(device) * sigma[t]
    x = z.clamp(-1, 1).permute(0, 2, 3, 1).cpu().numpy()
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = x[i * n + j]
    imwrite(path, figure)


@torch.no_grad()
def encode(net, path, n=4):
    """编码-重构:每步从 codebook 里选与 (目标图 - 当前x0估计) 内积最大的噪声
    选出的索引序列就是图片的离散编码;本函数同时输出 [原图|重构图] 对照
    """
    paths = list_pictures(os.path.join(data_dir, 'train'))
    paths += list_pictures(os.path.join(data_dir, 'valid'))  # 与 Keras 版同源(train+valid)
    picks = np.random.choice(len(paths), n**2, replace=False)
    x_target = torch.stack([
        torch.from_numpy(imread(paths[i])).permute(2, 0, 1) for i in picks
    ]).to(device)

    z = codebook[T][:1].repeat(n**2, 1, 1, 1).to(device)  # 固定起点
    for step in tqdm(range(T), ncols=0):
        t = T - step - 1
        bt = torch.full((z.shape[0],), t, dtype=torch.long, device=device)
        mp = net(z, bt)
        x0_est = (z - bar_beta[t] * mp) / bar_alpha[t]        # 当前对原图的估计
        residual = x_target - x0_est                          # 还差什么
        cb = codebook[t].to(device)                           # (K,3,H,W)
        sims = torch.einsum('kchw,bchw->kb', cb, residual)    # 每个码本项和残差的相似度
        idxs = sims.argmax(0)                                 # 贪心选最能弥补残差的噪声
        z -= beta[t]**2 / bar_beta[t] * mp
        z /= alpha[t]
        z += cb[idxs] * sigma[t]
    z = z.clamp(-1, 1)

    figure = np.zeros((img_size * n, img_size * n * 2, 3))
    xt_np = x_target.permute(0, 2, 3, 1).cpu().numpy()
    z_np = z.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(n):
        for j in range(n):
            ij = i * n + j
            figure[i * img_size:(i + 1) * img_size,
                   2 * j * img_size:(2 * j + 1) * img_size] = xt_np[ij]
            figure[i * img_size:(i + 1) * img_size,
                   (2 * j + 1) * img_size:(2 * j + 2) * img_size] = z_np[ij]
    imwrite(path, figure)


if __name__ == '__main__':
    net = load_net()
    sample(net, os.path.join(OUT_DIR, 'test_ddcm1.png'))
    encode(net, os.path.join(OUT_DIR, 'test_ddcm2.png'))
