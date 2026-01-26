# Protein Name Prediction
 
Fine-tuning LLMs for protein name prediction using Unsloth + LoRA on Llama-3.1-8B.
 
## Pipeline
 
```
read_data.ipynb  →  train.sh  →  evaluate.py
  (data prep)       (train)      (evaluate)
```
 
## 1. Data Preparation
 
Run `read_data.ipynb` to generate three hierarchical datasets:
 
| Level | Input Fields | Output |
|-------|-------------|--------|
| Level 1 | NAME | PRODUCT_NAME |
| Level 2 | NAME + organism | PRODUCT_NAME |
| Level 3 | NAME + organism + comment | PRODUCT_NAME |
 
Output directories:
- `protein_dataset_level1_full/`
- `protein_dataset_level2_full/`
- `protein_dataset_level3_full/`
 
## 2. Training
 
Submit training job:
 
```bash
sbatch train.sh
```
 
Or run directly:
 
```bash
torchrun --nproc_per_node=2 sft.py \
    --dataset_path protein_dataset_level3_full \
    --model_name unsloth/Llama-3.1-8B-Instruct \
    --max_length 1024 \
    --output_dir ./output
```
 
### Key Parameters
 
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 2e-4 | Learning rate |
| `--batch_size` | 16 | Per-device batch size |
| `--num_epochs` | 2 | Number of training epochs |
| `--lora_r` | 32 | LoRA rank |
| `--lora_target_modules` | q,k,v,o,gate,up,down_proj | LoRA target layers |
 
## 3. Evaluation
 
Evaluation runs automatically after training, or run manually:
 
```bash
python evaluate.py \
    --dataset_path protein_dataset_level3_full \
    --model_name_or_path ./output/model_checkpoint \
    --out_dir ./output \
    --level 3 \
    --split test
```
 
### Output Files
 
| File | Description |
|------|-------------|
| `summary2_test.json` | Metrics summary (BLEU-2, chrF, CosSim, LEN_RATIO) |
| `predictions2_test.jsonl` | Predictions (NAME, PRODUCT_NAME, PREDICTION) |
 
### Metrics
 
- **BLEU-2**: Sentence-level BLEU-2 score
- **chrF**: Character n-gram F-score, robust to spelling variations
- **COS_SIM**: Semantic similarity using BiomedBERT embeddings
- **LEN_RATIO**: Prediction length / reference length
 
## Dependencies
 
```
transformers
datasets
unsloth
torch
trl
wandb
pandas
numpy
sacrebleu (optional)
