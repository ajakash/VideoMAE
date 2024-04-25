# Set the path to save checkpoints
OUTPUT_DIR='/home/aabdujyo/scratch/VideoMAE/checkpoints/MOMA_sact_default'
# path to Kinetics set (train.csv/val.csv/test.csv)
LOG_DIR='/home/aabdujyo/scratch/VideoMAE/log/MOMA_sact_default'
# path to pretrain model
MODEL_PATH='/home/aabdujyo/scratch/VideoMAE/VideoMAE_pretrained_ckpts/checkpoint_ViT-S_K400_ep1600.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)

echo 'Starting to run the script!'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
#     --master_port 12320 --nnodes=4  --node_rank=$1 --master_addr=$2 \
python run_class_finetuning.py \
    --num_workers 2 \
    --model vit_small_patch16_224 \
    --data_set MOMA_sact \
    --nb_classes 91 \
    --finetune ${MODEL_PATH} \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 10 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 150 \
    --test_num_segment 2 \
    --test_num_crop 2 \
    --eval #\
    # --dist_eval \
    # --enable_deepspeed 
    # --data_path ${DATA_PATH} \
    # change num_wrokers for distributed training
