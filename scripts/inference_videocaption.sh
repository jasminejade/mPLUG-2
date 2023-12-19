#!/usr/bin/env bash
MASTER_ADDR=localhost
MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
WORLD_SIZE=1
RANK=0

GPU_NUM=1
TOTAL_GPU=$((WORLD_SIZE * GPU_NUM))

checkpoint_dir='mPLUG2_MSRVTT_Caption.pth'
output_dir='./output/blessed'

mkdir -p ${output_dir}
python -u -m torch.distributed.run --nproc_per_node=$GPU_NUM \
    --standalone\
    --nnodes=1\
    video_caption_mplug2.py \
    --config ./configs_video/VideoCaption_msrvtt_large.yaml \
    --text_encoder bert-large-uncased \
    --text_decoder bert-large-uncased \
    --output_dir ${output_dir} \
    --checkpoint ${checkpoint_dir} \
    --do_two_optim \
    --evaluate \
    --do_amp 2>&1 | tee ${output_dir}/train.log