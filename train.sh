#!/bin/bash
#SBATCH --job-name=level3_train
#SBATCH --time=2-00:00:00
#SBATCH -p gpu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/users/thz501/logs/bio/%x-%j.out
#SBATCH --error=/users/thz501/logs/bio/%x-%j.err
#SBATCH --chdir=/users/thz501

eval "$(conda shell.bash hook)"
conda activate /users/thz501/data/envs/selfdiv

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

cd /users/thz501/data/bio

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

torchrun --nproc_per_node=2 sft.py \
    --dataset_path "/users/thz501/fastscratch/bio/dataset/protein_dataset_level3_full" \
    --model_name "unsloth/Llama-3.1-8B-Instruct" \
    --max_length 1024 \
    --output_dir "/users/thz501/fastscratch/bio/models" 