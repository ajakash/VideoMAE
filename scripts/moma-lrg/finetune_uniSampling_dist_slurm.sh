#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=v100:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # There are 24 CPU cores on V100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --time=11:59:00
#SBATCH --account=def-mori
#SBATCH --output=log/slurm_output/slurm-%J.out
#SBATCH --error=log/slurm_output/error_%J.out

# Set the path to save checkpoints
OUTPUT_DIR='/home/aabdujyo/scratch/VideoMAE/checkpoints/'$1
# path to Kinetics set (train.csv/val.csv/test.csv)
LOG_DIR='/home/aabdujyo/scratch/VideoMAE/log/'$1
# path to pretrain model
MODEL_PATH='/home/aabdujyo/scratch/VideoMAE/'$2
# MODEL_PATH='/home/aabdujyo/scratch/VideoMAE/VideoMAE_pretrained_ckpts/checkpoint_ViT-S_SS_ep2400.pth'
# MODEL_PATH='/home/aabdujyo/scratch/VideoMAE/VideoMAE_pretrained_ckpts/checkpoint_ViT-B_SS_ep2400.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)

source /home/aabdujyo/scratch/VideoMAE/VidMAE/bin/activate
module load python/3.10
module load scipy-stack/2023b
module load cuda

echo 'Starting to run the script!'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
#     --master_port 12320 --nnodes=4  --node_rank=$1 --master_addr=$2 \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12320 --nnodes=1  --node_rank=0 --master_addr=127.0.0.1 \
    run_class_finetuning.py \
    --num_workers 10 \
    --model vit_base_patch16_224 \
    --data_set MOMA_sact_uniSampling \
    --nb_classes 91 \
    --finetune ${MODEL_PATH} \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $3 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 150 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --distributed \
    --dist_eval #\
    # --eval #\
    # --enable_deepspeed 
    # --data_path ${DATA_PATH} \
    # change num_wrokers for distributed training
