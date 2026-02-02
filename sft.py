import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from unsloth import FastLanguageModel
import torch
import wandb
import sys
import logging
import argparse
import datetime
import subprocess
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_from_disk
from unsloth.chat_templates import train_on_responses_only, get_chat_template
from trl import SFTTrainer
from transformers import DataCollatorForSeq2Seq
from types import SimpleNamespace
local_rank = int(os.environ.get("LOCAL_RANK", 0))

def parse_args() -> SimpleNamespace:
    p = argparse.ArgumentParser(description="Train or evaluate Unsloth models")
    # 数据和模型路径
    p.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    p.add_argument("--model_name", type=str, required=True, help="Model name or path")
    p.add_argument("--output_dir", type=str, required=True, help="Path to output directory")

    # 序列长度
    p.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")

    # 训练超参数
    p.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    p.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")

    # LoRA 参数
    p.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated list of target modules for LoRA")

    # wandb 设置
    p.add_argument("--wandb_project", type=str, default="protein", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team/user)")
    p.add_argument("--wandb_run_name", type=str, default="llama3_sft", help="W&B run name")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    # 调试和评估
    p.add_argument("--DEBUG", action="store_true", help="Debug mode with smaller dataset")
    p.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training")

    return p.parse_args(namespace=SimpleNamespace())


# dataset1: 只有 NAME 和 PRODUCT_NAME
def format_instruction_level1(sample,tokenizer=None):
    instruction = "Predict the protein name based on the UniProt ID."
    input_text = sample["NAME"]
    output_text = sample["PRODUCT_NAME"]
    messages = [
        {"role": "user", "content": f"{instruction}\n\nUniProt ID: {input_text}"},
        {"role": "assistant", "content": f"Protein Name: {output_text}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
    return {"text": text, "instruction": instruction, "input": input_text, "output": output_text}
    
    

# dataset2: NAME, organism, PRODUCT_NAME
def format_instruction_level2(sample,tokenizer=None):
    instruction = "Predict the protein name based on the UniProt ID and the organism."
    input_text = f"UniProt ID: {sample['NAME']}\nOrganism: {sample['organism']}"
    output_text = sample["PRODUCT_NAME"]
    
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": f"Protein Name: {output_text}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "instruction": instruction, "input": input_text, "output": output_text}


# dataset3: NAME, organism, comment, PRODUCT_NAME (UniProt Level3)
def format_instruction_level3(sample, tokenizer=None):
    instruction = "Based on the gene summary, predict the standardized protein product name."
    comment_text = sample.get("comment", "") if sample.get("comment") else ""
    if not comment_text:
        comment_text = "No summary available."
    input_text = f"Gene ID: {sample['NAME']}\n\nSummary:\n{comment_text}"
    output_text = sample["PRODUCT_NAME"]
    
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": f"Product_Description: {output_text}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return {"text": text, "instruction": instruction, "input": input_text, "output": output_text}


# VEuPathDB Gene Summary Dataset: Gene_ID, user_prompt, PRODUCT_NAME
def format_instruction_veupathdb(sample, tokenizer=None):
    instruction = "Based on the gene summary, predict the standardized protein product name."
    gene_id = sample["Gene_ID"]
    user_prompt = sample["user_prompt"]
    if not user_prompt:
        user_prompt = "No summary available."
    output_text = sample["PRODUCT_NAME"]
    
    input_text = f"Gene ID: {gene_id}\n\nSummary:\n{user_prompt}"
    
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": f"Product_Description: {output_text}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    return {"text": text, "instruction": instruction, "input": input_text, "output": output_text}



def prepare_data(dataset_path, format_function):
    dataset = load_from_disk(dataset_path)
    # 只在主进程上处理，避免多进程OOM
    dataset = dataset.map(format_function, remove_columns=dataset["train"].column_names, num_proc=1)
    return dataset

def train():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    )
    log = logging.getLogger("main")
    
    args = parse_args()
    
    # wandb init (只在主进程初始化)
    if local_rank == 0 and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,  # None 表示使用默认 entity
            name=args.wandb_run_name,
            dir=args.output_dir,
            config={
                "model": args.model_name,
                "dataset": args.dataset_path,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_epochs": args.num_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            settings=wandb.Settings(init_timeout=120)
        )
    
    # add timestamp to output dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    # 每个进程加载模型到自己的GPU
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=torch.float16,
        load_in_4bit=False,
        device_map={'': local_rank},
    )
    tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",
            )
    # 解析 LoRA target modules
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    log.info(f"LoRA target modules: {target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=True,
        target_modules=target_modules,
    )
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    # Choose formatting function based on dataset structure
    if "level1" in args.dataset_path:
        format_function = lambda sample: format_instruction_level1(sample, tokenizer)
    elif "level2" in args.dataset_path:
        format_function = lambda sample: format_instruction_level2(sample, tokenizer)
    elif "gene_summary" in args.dataset_path:
        format_function = lambda sample: format_instruction_veupathdb(sample, tokenizer)
    elif "level3" in args.dataset_path:
        format_function = lambda sample: format_instruction_level3(sample, tokenizer)
    else:
        # Default to level3 for unknown formats
        format_function = lambda sample: format_instruction_level3(sample, tokenizer)
    
    log.info("Loading and preparing dataset...")
    dataset = prepare_data(args.dataset_path, format_function)
    if args.DEBUG:
        train_size = min(20000, len(dataset["train"]))
        dev_size = min(1000, len(dataset["dev"]))
        dataset["train"] = dataset["train"].select(range(train_size))
        dataset["dev"] = dataset["dev"].select(range(dev_size))
    log.info("Dataset loaded and prepared.")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    # 计算有效 batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    log.info(f"Effective batch size: {effective_batch_size} (per_device={args.batch_size} x grad_accum={args.gradient_accumulation_steps})")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
        max_grad_norm=1.0,
        report_to=["wandb"] if (not args.no_wandb and local_rank == 0) else [],
        # DDP 相关参数
        ddp_find_unused_parameters=False,
        ddp_timeout=30 * 60,  # 30 分钟超时
        local_rank=local_rank,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        dataset_text_field="text",
        max_seq_length=args.max_length,
        data_collator=data_collator,
        args=training_args,
        dataset_num_proc=1,  # 减少内存使用
        packing=False,  
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    trainer.train()
    
    # Save the final model (只在主进程保存)
    if local_rank == 0:
        model.save_pretrained(args.output_dir)
        print("Training completed and model saved to", args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    
    # Run evaluation after training (只在主进程执行)
    if not args.skip_eval and local_rank == 0:
        log.info("="*60)
        log.info("Starting post-training evaluation...")
        log.info("="*60)
        
        # Determine level from dataset path
        if "level1" in args.dataset_path:
            level = 1
        elif "level2" in args.dataset_path:
            level = 2
        elif "gene_summary" in args.dataset_path:
            level = 3  # VEuPathDB uses level 3 evaluation logic
        elif "level3" in args.dataset_path:
            level = 3
        else:
            level = 3
        
        # Build evaluation command
        eval_script = os.path.join(os.path.dirname(__file__), "evaluate.py")
        eval_cmd = [
            sys.executable, "-u", eval_script,
            "--dataset_path", args.dataset_path,
            "--model_name_or_path", args.output_dir,
            "--out_dir", args.output_dir,
            "--max_seq_length", str(args.max_length+500),
            "--level", str(level),
            "--split", "test"
        ]
        
        if args.DEBUG:
            eval_cmd.append("--DEBUG")
        
        log.info(f"Running evaluation command: {' '.join(eval_cmd)}")
        
        try:
            # Run evaluation as subprocess to ensure proper logging
            result = subprocess.run(eval_cmd, check=True, capture_output=False, text=True)
            log.info("Evaluation completed successfully!")
        except subprocess.CalledProcessError as e:
            log.error(f"Evaluation failed with error: {e}")
            log.error("Continuing without evaluation results...")
    else:
        log.info("Skipping evaluation (--skip_eval flag set)")
    
    if local_rank == 0 and not args.no_wandb:
        wandb.finish()
    return model, tokenizer

if __name__ == "__main__":
    train()
    
    
    
    
    
    



