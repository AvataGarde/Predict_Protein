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
    p.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    p.add_argument("--model_name", type=str, required=True, help="Model name or path")
    p.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    p.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
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


# dataset3: NAME, organism, comment, PRODUCT_NAME
def format_instruction_level3(sample,tokenizer=None):
    instruction = "Predict the protein name based on the UniProt ID, organism, and comments."
    comment_text = sample.get("comment", "No comment") if sample.get("comment") else "No comment"
    input_text = f"UniProt ID: {sample['NAME']}\nOrganism: {sample['organism']}\nComment: {comment_text}"
    output_text = sample["PRODUCT_NAME"]
    
    messages = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": f"Protein Name: {output_text}"}
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
    if local_rank == 0:
        wandb.init(project="protein", 
                    entity="diversity_dpo",
                    name=f"llama3_sft",
                    dir  = args.output_dir,
                    settings=wandb.Settings(init_timeout=120))
    
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
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        use_rslora=True,
        target_modules=["q_proj", "v_proj"],
    )
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    # Choose formatting function based on dataset structure
    if "level1" in args.dataset_path:
        format_function = lambda sample: format_instruction_level1(sample, tokenizer)
    elif "level2" in args.dataset_path:
        format_function = lambda sample: format_instruction_level2(sample, tokenizer)
    else:
        format_function = lambda sample: format_instruction_level3(sample, tokenizer)
    
    dataset = prepare_data(args.dataset_path, format_function)
    if args.DEBUG:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["dev"] = dataset["dev"].select(range(200))
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,  # 从32降到16，减少内存使用
        gradient_accumulation_steps=4,  # 从2增加到4，保持有效batch size=64
        warmup_steps=20,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        save_total_limit=2,
        max_grad_norm=1.0,
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
    
    if local_rank == 0:
        wandb.finish()
    return model, tokenizer

if __name__ == "__main__":
    train()
    
    
    
    
    
    



