#!/bin/bash
#SBATCH --job-name=level3_eval
#SBATCH --time=1-00:00:00
#SBATCH -p gpu-a-lowsmall
#SBATCH --gres=gpu:1
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

python -u evaluate.py \
    --dataset_path "/users/thz501/fastscratch/bio/dataset/gene_summary_uniprot_style" \
    --model_name_or_path "/users/thz501/fastscratch/bio/models/uniprot_trained" \
    --out_dir "/users/thz501/fastscratch/bio/models/uniprot_trained"  \
    --max_seq_length 4096 \
    --level 3 
    