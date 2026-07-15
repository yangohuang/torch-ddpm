#! -*- coding: utf-8 -*-
# 生成扩散模型DDPM —— PyTorch 版，逐行对照移植自 ../ddpm.py（苏剑林 Keras 原版）
# 保持一致的部分：U-Net 结构（skip 用 Add、无 Attention）、噪声调度、l2 损失（逐像素求和）、
#                LAMB 优化器 + 分段线性 lr（4000 步 warmup）+ EMA(0.9999)、DDPM 采样循环
# 有意的差异：GroupNorm 用 torch 原生实现（分组方式为连续分块，原版为隔 32 交错）；
#            数据加载用 DataLoader 多进程（原版单线程生成器）；
#            LAMB 为紧凑手写实现（非 bert4keras 逐行等价；且原版 exclude=['Norm','bias']
#            因大小写敏感匹配实际未生效，本版按论文意图真正排除了 norm 参数）；
#            EMA 影子权重用当前权重初始化、无偏差校正（bert4keras 为零初始化+apply 时校正，
#            两者在训练前几千步的 EMA 采样有差异，收敛后一致）；
#            未移植原版的线性插值 sample_inter（DDIM 版见 ddim.py 的球面插值）
# 用法：DDPM_DATA_DIR=/home/yg/data/CelebA-HQ python ddpm.py
#/home/yg/data/CelebA-HQ
# 环境变量：DDPM_DATA_DIR / DDPM_BATCH_SIZE / DDPM_INITIAL_EPOCH（断点续训）/ DDPM_DEVICE
'''
───────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
│                   断点位置                    │                                        断下后看什么                                        │
├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ torch-ddpm/ddpm.py:348(loss 计算行)           │ 训练循环心脏。Watch 面板加:loss.item()、xt.std().item()(应≈1)、t[:4]、pred.std().item()    │
├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ torch-ddpm/ddpm.py:179(UNet.forward)          │ 单步 F10 走 down/up plan,配合 Debug Console 敲 x.shape 看每层形状变化(昨天第 3 步的活体版) │
├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ torch-ddpm/ddpm.py:124(ResidualBlock.forward) │ 看 t 注入:self.t_proj(t_emb)[:, :, None, None].shape 如何广播到空间维                      │
├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ torch-ddpm/ddpm.py:285(采样去噪行)            │ 条件断点右键设 t < 5,直接跳到采样最后几步看 z.std()                                        │
├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ torch-ddpm/flow_matching.py:109               │ 对照 DDPM:这里 Watch v_target.std()(≈√2,因为是两个单位方差张量之差)
'''
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 基本配置（与 ../ddpm.py 相同）
data_dir = os.environ.get('DDPM_DATA_DIR', '/root/CelebA-HQ')
img_size = 128
batch_size = int(os.environ.get('DDPM_BATCH_SIZE', 64))
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
num_layers = len(channels) * 2 + 1
blocks = 2
min_pixel = 4
steps_per_epoch = 2000
device = os.environ.get('DDPM_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
os.makedirs(SAMPLES_DIR, exist_ok=True)

# 超参数选择（与 ../ddpm.py 相同）
T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()

# Min-SNR-gamma 加权(可选,对照 ../ddpm_yg.py):DDPM_MINSNR_GAMMA=5 开启,0/不设=原版行为
minsnr_gamma = float(os.environ.get('DDPM_MINSNR_GAMMA', 0))
snr = (bar_alpha / bar_beta)**2
loss_weight = np.minimum(1.0, minsnr_gamma / snr) if minsnr_gamma > 0 else np.ones(T)


def list_pictures(directory, exts=('.png', '.jpg', '.jpeg')):
    return sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(directory) for f in files
        if f.lower().endswith(exts)
    )


def imread(f, crop_size=None):
    """读取图片，中心裁剪 + 缩放 + 归一化到 [-1, 1]（与原版一致，BGR 顺序也保持）"""
    x = cv2.imread(f)
    height, width = x.shape[:2]
    if crop_size is None:
        crop_size = min([height, width])
    else:
        crop_size = min([crop_size, height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
    if x.shape[:2] != (img_size, img_size):
        x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    return x


def imwrite(path, figure):
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)


class FaceDataset(Dataset):
    """无限随机采样由 DataLoader 的 shuffle + 多 epoch 实现"""
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = imread(self.paths[i])
        return torch.from_numpy(x).permute(2, 0, 1)  # HWC -> CHW


def variance_scaling_(tensor, scale=1.0):
    """Keras VarianceScaling(scale, 'fan_avg', 'uniform') 的等价实现"""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan_avg = (fan_in + fan_out) / 2
    limit = math.sqrt(3.0 * max(scale, 1e-10) / fan_avg)
    with torch.no_grad():
        return tensor.uniform_(-limit, limit)


def make_conv(in_dim, out_dim, init_scale=1.0):
    conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False)
    variance_scaling_(conv.weight, init_scale)
    return conv


def make_dense(in_dim, out_dim, init_scale=1.0):
    fc = nn.Linear(in_dim, out_dim, bias=False)
    variance_scaling_(fc.weight, init_scale)
    return fc


class ResidualBlock(nn.Module):
    """对照 ../ddpm.py 的 residual_block：
    x += dense(t); x = conv-swish ×2; x += shortcut; GroupNorm
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.t_proj = make_dense(embedding_size, in_dim)
        self.conv1 = make_conv(in_dim, out_dim, 1 / num_layers**0.5)
        self.conv2 = make_conv(out_dim, out_dim, 1 / num_layers**0.5)
        self.shortcut = None if in_dim == out_dim else make_dense(in_dim, out_dim)
        self.norm = nn.GroupNorm(32, out_dim, eps=1e-6)

    def forward(self, x, t_emb):
        if self.shortcut is None:
            xi = x
        else:
            xi = self.shortcut(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.t_proj(t_emb)[:, :, None, None]
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return self.norm(x + xi)


class UNet(nn.Module):
    """结构与 ../ddpm.py 的函数式搭建逐层对应（skip 全部用加法）"""
    def __init__(self):
        super().__init__()
        self.t_embedding = nn.Embedding(T, embedding_size)
        self.stem = make_conv(3, embedding_size)

        # 静态推演一遍前向的通道/分辨率，登记各层（与原版 for 循环一致）
        self.down_blocks = nn.ModuleList()
        self.down_plan = []  # ('res', block_idx) / ('pool',)
        dims = [embedding_size]  # 模拟 skip 栈，只记录通道数
        dim, res = embedding_size, img_size
        skip_pooling = 0
        for ch in channels:
            for _ in range(blocks):
                out_dim = ch * embedding_size
                self.down_blocks.append(ResidualBlock(dim, out_dim))
                self.down_plan.append('res')
                dim = out_dim
                dims.append(dim)
            if res > min_pixel:
                self.down_plan.append('pool')
                res //= 2
                dims.append(dim)
            else:
                skip_pooling += 1
        self.skip_pooling = skip_pooling
        self.middle = ResidualBlock(dim, dim)
        dims.pop()

        self.up_blocks = nn.ModuleList()
        self.up_plan = []  # ('up',) / ('res',)
        for i, ch in enumerate(channels[::-1]):
            if i >= skip_pooling:
                self.up_plan.append('up')
                dim = dims.pop()  # upsample 后与栈顶相加，通道数不变
            for _ in range(blocks):
                skip_dim = dims.pop()
                self.up_blocks.append(ResidualBlock(dim, skip_dim))
                self.up_plan.append('res')
                dim = skip_dim
        self.out_norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.out_conv = make_conv(dim, 3)

    def forward(self, x, t):
        t_emb = self.t_embedding(t.reshape(-1))  # (B,) -> (B, emb)
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
                stack.append(x)
        x = self.middle(x, t_emb)
        stack.pop()
        ui = 0
        for op in self.up_plan:
            if op == 'up':
                x = F.interpolate(x, scale_factor=2, mode='nearest')
                x = x + stack.pop()
            else:
                xi = stack.pop()
                x = self.up_blocks[ui](x, t_emb) + xi
                ui += 1
        return self.out_conv(self.out_norm(x))


def piecewise_linear_lr(step):
    """bert4keras lr_schedule={4000:1, 20000:0.5, 40000:0.1} 的等价实现（返回倍率）"""
    pts = [(0, 0.0), (4000, 1.0), (20000, 0.5), (40000, 0.1)]
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if step < s1:
            return v0 + (v1 - v0) * (step - s0) / (s1 - s0)
    return pts[-1][1]


class LAMB(torch.optim.Optimizer):
    """Adam + layer adaptation（对照 bert4keras extend_with_layer_adaptation）
    exclude 的参数（Norm 的 scale/offset）退化为普通 Adam 更新
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                state['step'] += 1
                m, v, s = state['m'], state['v'], state['step']
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                m_hat = m / (1 - b1**s)
                v_hat = v / (1 - b2**s)
                update = m_hat / (v_hat.sqrt() + group['eps'])
                if group.get('layer_adaptation', True):
                    w_norm = p.norm()
                    u_norm = update.norm()
                    if w_norm > 0 and u_norm > 0:
                        update = update * (w_norm / u_norm)
                p.add_(update, alpha=-group['lr'])


class EMA:
    """指数滑动平均（对照 extend_with_exponential_moving_average, momentum=0.9999）"""
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.momentum).add_(v, alpha=1 - self.momentum)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


model = UNet().to(device)
_bar_alpha = torch.from_numpy(bar_alpha).float().to(device)
_bar_beta = torch.from_numpy(bar_beta).float().to(device)


@torch.no_grad()
def sample(path=None, n=4, z_samples=None, t0=0, net=None):
    """DDPM 原始采样（与 ../ddpm.py 的 sample 相同的更新公式）"""
    net = net or model
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
    print(f'{len(paths)} images')

    loader = DataLoader(
        FaceDataset(paths), batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True
    )

    # Norm 参数不做 layer adaptation（对照 exclude_from_layer_adaptation=['Norm','bias']）
    norm_params, other_params = [], []
    for name, p in model.named_parameters():
        (norm_params if 'norm' in name.lower() else other_params).append(p)
    optimizer = LAMB([
        dict(params=other_params, layer_adaptation=True),
        dict(params=norm_params, layer_adaptation=False),
    ], lr=1e-3)

    ema = EMA(model)
    base_lr = 1e-3
    initial_epoch = int(os.environ.get('DDPM_INITIAL_EPOCH', 0))
    global_step = initial_epoch * steps_per_epoch
    if initial_epoch > 0:
        ckpt = torch.load(CKPT, map_location=device)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f'resumed from {CKPT} @ epoch {initial_epoch}')

    epoch, running, seen = initial_epoch, 0.0, 0
    data_iter = iter(loader)
    model.train()
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
        xt = x0 * _bar_alpha[t][:, None, None, None] + noise * _bar_beta[t][:, None, None, None]

        pred = model(xt, t)
        w = torch.from_numpy(loss_weight).float().to(device)[t]  # Min-SNR 权重(默认全1)
        loss = (w * ((noise - pred)**2).sum(dim=(1, 2, 3))).mean()  # l2_loss：逐像素求和
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for g in optimizer.param_groups:
            g['lr'] = base_lr * piecewise_linear_lr(global_step + 1)
        optimizer.step()
        ema.update(model)

        global_step += 1
        running += loss.item() * bs
        seen += bs
        pbar.update(1)
        pbar.set_postfix(loss=f'{running / seen:.1f}')

        if global_step % steps_per_epoch == 0:
            pbar.close()
            epoch += 1
            torch.save({'model': model.state_dict(), 'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(), 'epoch': epoch}, CKPT)
            sample(os.path.join(SAMPLES_DIR, '%05d.png' % epoch))
            ema_model = UNet().to(device)
            sd = model.state_dict()
            sd.update(ema.state_dict())
            ema_model.load_state_dict(sd)
            sample(os.path.join(SAMPLES_DIR, '%05d_ema.png' % epoch), net=ema_model)
            del ema_model
            running, seen = 0.0, 0
            pbar = tqdm(total=steps_per_epoch, ncols=0, desc=f'Epoch {epoch + 1}')


if __name__ == '__main__':
    main()
