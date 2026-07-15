#!/bin/bash
# 自动训练队列：依次训练 4 个实现，全部完成后自动跑 3 个采样脚本
# 用法：bash train_all.sh（建议在 tmux 里跑）
# 进度总览：logs/queue.log；各阶段明细：logs/<name>.log
cd "$(dirname "$0")" || exit 1
export DDPM_DATA_DIR="${DDPM_DATA_DIR:-$HOME/data/CelebA-HQ}"
export DDPM_BATCH_SIZE="${DDPM_BATCH_SIZE:-32}"
mkdir -p logs

note() { echo "[$(date '+%F %T')] $*" >> logs/queue.log; }

last_epoch() {  # 从 checkpoint 读已完成的 epoch 数（无 ckpt 则 0）
    python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu').get('epoch',0))" "$1" 2>/dev/null || echo 0
}

train() {  # train <名字> <脚本> <ckpt文件> <目标epochs> [额外env，如 DDPM_MINSNR_GAMMA=5]
    local name=$1 script=$2 ckpt=$3 epochs=$4 extra=$5
    local log="logs/$name.log"
    note "START $name -> $epochs epochs (extra: ${extra:-none})"
    env $extra DDPM_EPOCHS="$epochs" python -u "$script" >> "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] && grep -qi "out of memory" "$log"; then
        local ep; ep=$(last_epoch "$ckpt")
        note "$name OOM at epoch $ep -> retry with batch 16, resume from $ep"
        env $extra DDPM_EPOCHS="$epochs" DDPM_INITIAL_EPOCH="$ep" DDPM_BATCH_SIZE=16 \
            python -u "$script" >> "$log" 2>&1
        rc=$?
    fi
    if [ $rc -eq 0 ]; then note "DONE  $name"; else note "FAIL  $name (exit $rc), continue queue"; fi
    return $rc
}

sample_step() {  # sample_step <名字> <脚本>（依赖 model.pt）
    local name=$1 script=$2
    note "START sample $name"
    if python -u "$script" >> "logs/$name.log" 2>&1; then note "DONE  sample $name"
    else note "FAIL  sample $name, continue"; fi
}

note "========== QUEUE START =========="
train ddpm  ddpm.py          model.pt     100 "DDPM_MINSNR_GAMMA=5"
train ddpm2 ddpm2.py         model2.pt     80 ""
train gau   ddpm_gau.py      model_gau.pt  80 ""
train fm    flow_matching.py model_fm.pt  100 ""

# 收尾：三个免训练采样脚本（复用 model.pt）
sample_step ddim ddim.py
sample_step adpm adpm.py
sample_step ddcm ddcm.py
note "========== ALL DONE =========="
