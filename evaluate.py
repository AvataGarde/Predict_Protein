import os
import re
import json
import math
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from unsloth import FastLanguageModel

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm

from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer, AutoModel


# Optional: Try to import sacrebleu for better BLEU computation
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not found. Using custom BLEU implementation.")

# Setup logging with immediate flush for sbatch output
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,  # Override any existing configuration
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicit stdout handler
    ]
)
logger = logging.getLogger(__name__)

# Force flush after each log
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    handler.flush = lambda: sys.stdout.flush()


_PUNCT_SPLIT_RE = re.compile(r"([,;:\(\)\[\]\{\}\-\/])")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    # unify spacing around punctuation a bit
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s*\)\s*", ") ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_for_metrics(s: str) -> List[str]:
    s = normalize_text(s)
    s = _PUNCT_SPLIT_RE.sub(r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split(" ")

def strip_model_prefix(s: str) -> str:
    """If your model outputs 'Protein Name: xxx', keep only xxx (first line). or Product_Description: xxx"""
    if s is None or s == "":
        return ""
    s = str(s).strip()
    if not s:
        return ""
    
    lines = s.splitlines()
    if not lines:
        return ""
    
    s = lines[0].strip()
    # common prefix patterns
    m = re.search(r"protein\s*name\s*:\s*(.*)$", s, flags=re.IGNORECASE)
    
    if m:
        s = m.group(1).strip()

    m = re.search(r"product[_\s]*description\s*:\s*(.*)$", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    return normalize_text(s)

def embed_texts(
    texts: List[str],
    embed_tokenizer,
    embed_model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = embed_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = embed_model(**enc)
        # mean pooling with attention mask
        last = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).type_as(last)  # [B, T, 1]
        summed = (last * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        mean = summed / denom
        mean = torch.nn.functional.normalize(mean, dim=-1)
        embs.append(mean.detach().cpu().numpy())
    return np.vstack(embs)

# ---------- BLEU-2 (sentence-level average) ----------
def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts

def sentence_bleu2(reference: List[str], hypothesis: List[str], smooth: float = 1.0) -> float:
    """
    Sentence-level BLEU-2 with smoothing:
    BLEU = BP * exp( (1/2)(log p1 + log p2) )
    where p_n is clipped precision for n-grams; with add-k smoothing.
    Note: For better results, consider using sacrebleu: pip install sacrebleu
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0.0

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(ref_len) / float(max(hyp_len, 1)))

    # p1
    ref1 = _ngram_counts(reference, 1)
    hyp1 = _ngram_counts(hypothesis, 1)
    match1 = 0
    total1 = sum(hyp1.values())
    for k, v in hyp1.items():
        match1 += min(v, ref1.get(k, 0))
    
    if total1 == 0:
        return 0.0
    
    p1 = (match1 + smooth) / (total1 + smooth)

    # p2
    ref2 = _ngram_counts(reference, 2)
    hyp2 = _ngram_counts(hypothesis, 2)
    match2 = 0
    total2 = sum(hyp2.values())
    for k, v in hyp2.items():
        match2 += min(v, ref2.get(k, 0))
    p2 = (match2 + smooth) / (total2 + smooth)

    bleu = bp * math.exp(0.5 * (math.log(p1) + math.log(p2)))
    return float(bleu)


# ---------- chrF (character n-gram F-score) ----------
# Minimal chrF (n=6, beta=2) similar spirit to sacrebleu.
# For robust reporting, you can replace with sacrebleu if you prefer.

def _char_ngrams(s: str, n: int) -> Dict[str, int]:
    s = normalize_text(s)
    # use space-preserving string
    s = s.replace(" ", "▁")  # keep word boundary info
    if len(s) < n:
        return {}
    out = {}
    for i in range(len(s) - n + 1):
        g = s[i:i+n]
        out[g] = out.get(g, 0) + 1
    return out


def chrf_score(ref: str, hyp: str, n: int = 6, beta: float = 2.0) -> float:
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    if not hyp:
        return 0.0
    if not ref:
        return 0.0

    precisions = []
    recalls = []
    for k in range(1, n + 1):
        ref_ng = _char_ngrams(ref, k)
        hyp_ng = _char_ngrams(hyp, k)
        if not hyp_ng:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
        match = 0
        for g, c in hyp_ng.items():
            match += min(c, ref_ng.get(g, 0))
        p = match / max(sum(hyp_ng.values()), 1)
        r = match / max(sum(ref_ng.values()), 1)
        precisions.append(p)
        recalls.append(r)

    p_bar = float(np.mean(precisions))
    r_bar = float(np.mean(recalls))
    if p_bar == 0.0 and r_bar == 0.0:
        return 0.0
    beta2 = beta * beta
    f = (1 + beta2) * p_bar * r_bar / (beta2 * p_bar + r_bar + 1e-12)
    return float(f)

def make_messages(sample: Dict, level: int, max_comment_len: int = 3000, ignore_comment: bool = False) -> List[Dict[str, str]]:
    """Create messages for inference, with comment truncation to avoid overly long inputs."""
    if level == 1:
        instruction = "Predict the protein name based on the UniProt ID."
        user = f"{instruction}\n\nUniProt ID: {sample['NAME']}"
    elif level == 2:
        instruction = "Predict the protein name based on the UniProt ID and the organism."
        user = f"{instruction}\n\nUniProt ID: {sample['NAME']}\nOrganism: {sample.get('organism','')}"
    else: # level == 3 Gene Summary Dataset
        user_prompt = sample.get("user_prompt", sample.get("comment", ""))
        
        # Force empty comment if ignore_comment is True
        if ignore_comment:
            user_prompt = ""
        
        if not user_prompt:
            user_prompt = "No summary available."
        else:
            # Truncate overly long summaries to avoid exceeding max_seq_length
            if len(user_prompt) > max_comment_len:
                user_prompt = user_prompt[:max_comment_len] + "..."
        
        instruction = "Based on the gene summary, predict the standardized protein product name."
        user = f"{instruction}\n\nSummary:\n{user_prompt}"
        
    return [{"role": "user", "content": user}]


# ---------- Main evaluation ----------

@dataclass
class Args:
    dataset_path: str
    model_name_or_path: str
    out_dir: str
    chat_template: str
    level: int
    split: str
    max_seq_length: int
    max_new_tokens: int
    batch_size: int
    embed_model: str
    debug: bool
    embed_batch_size: int
    embed_max_length: int
    temperature: float
    top_p: float
    ignore_comment: bool

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True, type=str)
    p.add_argument("--model_name_or_path", required=True, type=str)
    p.add_argument("--out_dir", default="eval_out", type=str)
    p.add_argument("--chat_template", default="llama-3", type=str)
    p.add_argument("--level", default=1, type=int, choices=[1, 2, 3])
    p.add_argument("--split", default="test", type=str, choices=["train", "dev", "test"])
    p.add_argument("--max_seq_length", default=512, type=int)
    p.add_argument("--max_new_tokens", default=128, type=int)
    p.add_argument("--batch_size", default=32, type=int)

    p.add_argument("--embed_model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", type=str)
    p.add_argument("--embed_batch_size", default=64, type=int)
    p.add_argument("--embed_max_length", default=128, type=int)

    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--top_p", default=1.0, type=float)
    
    p.add_argument("--ignore_comment", action="store_true", help="Ignore comment field and use empty comment (for level 3 only)")
    p.add_argument("--DEBUG", action="store_true", help="Debug mode with smaller dataset")
    a = p.parse_args()
    
    return Args(
        dataset_path=a.dataset_path,
        model_name_or_path=a.model_name_or_path,
        out_dir=a.out_dir,
        chat_template=a.chat_template,
        level=a.level,
        split=a.split,
        max_seq_length=a.max_seq_length,
        max_new_tokens=a.max_new_tokens,
        batch_size=a.batch_size,
        embed_model=a.embed_model,
        debug=a.DEBUG,
        embed_batch_size=a.embed_batch_size,
        embed_max_length=a.embed_max_length,
        temperature=a.temperature,
        top_p=a.top_p,
        ignore_comment=a.ignore_comment,
    )
    
def generate_predictions(
    model,
    tokenizer,
    samples: List[Dict],
    level: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_seq_length: int = 1024,
    ignore_comment: bool = False,
) -> List[str]:
    """Generate predictions one sample at a time to avoid padding issues."""
    preds = []
    
    # 生成参数设置
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False
    
    logger.info(f"Generating predictions for {len(samples)} samples (one at a time)...,with IGNORE_COMMENT={ignore_comment}")
    for s in tqdm(samples, desc="Generating"):
        msgs = make_messages(s, level=level, ignore_comment=ignore_comment)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize with explicit max_length to ensure truncation happens correctly
        enc = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=max_seq_length - max_new_tokens  # Leave room for generation
        ).to(model.device)
        
        gen = model.generate(**enc, **gen_kwargs)
        
        # Decode only the generated part (skip input tokens)
        input_len = enc["input_ids"].shape[1]
        out_ids = gen[0][input_len:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        # print generated text for debugging
        logger.debug(f"Generated text: {text}")
        
        preds.append(strip_model_prefix(text))
    
    return preds


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Starting Evaluation")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model_name_or_path}")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Output Dir: {args.out_dir}")
    logger.info(f"  Level: {args.level}")
    logger.info(f"  Split: {args.split}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Ignore Comment: {args.ignore_comment}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Output directory created/verified: {args.out_dir}")
    
    # ----- load dataset -----
    logger.info(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)
    split_ds = ds[args.split]
    if args.debug:
        split_ds = split_ds.select(range(min(1000, len(split_ds))))
        logger.info("Debug mode: using a smaller subset of the dataset")
    logger.info(f"Loaded {len(split_ds)} samples from {args.split} split.")
    
    # Convert dataset to list of samples
    logger.info("Converting dataset to sample list...")
    samples = [split_ds[i] for i in range(len(split_ds))]
    golds = [normalize_text(s["PRODUCT_NAME"]) for s in samples]
    logger.info(f"Prepared {len(samples)} samples with gold labels")
    
    # ----- load SFT model -----
    logger.info(f"Loading model from {args.model_name_or_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=False,
    )
    logger.info("Setting model to inference mode...")
    FastLanguageModel.for_inference(model)
    
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    tokenizer.padding_side = "left"  # Ensure left padding for decoder-only models
    logger.info("Tokenizer padding side set to 'left' for generation")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Model moved to CUDA device")
    logger.info(f"Model loaded successfully from {args.model_name_or_path}")
    # ----- generate -----
    logger.info("="*60)
    logger.info("Starting prediction generation...")
    logger.info(f"Processing {len(samples)} samples one at a time (no batching)")
    preds = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        level=args.level,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_length=args.max_seq_length,
        ignore_comment=args.ignore_comment,
    )
    logger.info(f"Generated {len(preds)} predictions")
    
    # save raw predictions
    pred_out_path = os.path.join(args.out_dir, f"predictions_{args.split}.jsonl")
    logger.info(f"Saving predictions to {pred_out_path}...")
    with open(pred_out_path, "w") as f:
        for i in range(len(samples)):
            rec = {
                "Gene_ID": samples[i].get("Gene_ID", samples[i].get("NAME", "")),
                "PMID": samples[i].get("PMID", ""),
                "LLM_Product": preds[i] + ", putative Evidence_Code: ISS",
            }
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Predictions saved to {pred_out_path}")
    
    # ----- evaluate -----
    logger.info("="*60)
    logger.info("Computing evaluation metrics...")
    bleu2_list = []
    chrf_list = []
    len_ratio_list = []

    gold_tok_lens = []
    pred_tok_lens = []
    
    # Use sacrebleu if available for BLEU-2
    if HAS_SACREBLEU:
        logger.info("Using sacrebleu for BLEU-2 computation")
    else:
        logger.info("Using custom BLEU-2 implementation")
    
    for gold, pred in zip(golds, preds):
        gold_toks = tokenize_for_metrics(gold)
        pred_toks = tokenize_for_metrics(pred)

        gold_tok_lens.append(len(gold_toks))
        pred_tok_lens.append(len(pred_toks))

        # BLEU-2: Use sacrebleu if available, otherwise custom implementation
        if HAS_SACREBLEU:
            # sacrebleu.sentence_bleu expects hypothesis string and references list
            bleu_score = sacrebleu.sentence_bleu(
                hypothesis=pred,
                references=[gold],
                smooth_method='add-k',
                smooth_value=1.0,
                use_effective_order=True
            ).score / 100.0  # sacrebleu returns 0-100, normalize to 0-1
            bleu2_list.append(bleu_score)
        else:
            bleu2_list.append(sentence_bleu2(gold_toks, pred_toks, smooth=1.0))
        
        chrf_list.append(chrf_score(gold, pred, n=6, beta=2.0))

        ratio = (len(pred_toks) / len(gold_toks)) if len(gold_toks) > 0 else 0.0
        len_ratio_list.append(ratio)
    
    bleu2 = np.array(bleu2_list, dtype=np.float64)
    chrf = np.array(chrf_list, dtype=np.float64)
    len_ratio = np.array(len_ratio_list, dtype=np.float64)
    logger.info(f"Computed BLEU-2, chrF, and length ratio for {len(samples)} samples")
    
    # ----- embedding cosine -----
    logger.info("="*60)
    logger.info("Computing semantic similarity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading embedding model: {args.embed_model}")
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    embed_model = AutoModel.from_pretrained(args.embed_model).to(device)
    embed_model.eval()
    logger.info("Embedding model loaded")
    
    logger.info("Embedding gold labels...")
    gold_emb = embed_texts(
        golds, embed_tokenizer, embed_model, device,
        batch_size=args.embed_batch_size, max_length=args.embed_max_length
    )
    logger.info("Embedding predictions...")
    pred_emb = embed_texts(
        preds, embed_tokenizer, embed_model, device,
        batch_size=args.embed_batch_size, max_length=args.embed_max_length
    )
    logger.info("Computing cosine similarities...")
    cos_sims = (gold_emb * pred_emb).sum(axis=-1)
    logger.info(f"Computed semantic similarity for {len(samples)} samples")
    
    # ----- save results -----
    logger.info("="*60)
    logger.info("Saving evaluation results...")
    out_path = os.path.join(args.out_dir, f"VEuPathDB_{args.split}.jsonl")
    with open(out_path, "w") as f:
        for i in range(len(samples)):
            rec = {
                "Gene_ID": samples[i].get("Gene_ID", samples[i].get("NAME", "")),
                "PMID": samples[i].get("PMID", ""),
                "PRODUCT_NAME": golds[i],
                "PREDICTION": preds[i],
                "BLEU-2": bleu2_list[i],
                "chrF": chrf_list[i],
                "LEN_RATIO": len_ratio_list[i],
                "COS_SIM": float(cos_sims[i]),
            }
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Detailed results saved to {out_path}")
    
    # summary
    summary_path = os.path.join(args.out_dir, f"summary_{args.split}.json")
    summary = {
        "split": args.split,
        "num_samples": len(samples),
        "metrics": {
            "BLEU-2": {"mean": float(bleu2.mean()), "std": float(bleu2.std())},
            "chrF": {"mean": float(chrf.mean()), "std": float(chrf.std())},
            "LEN_RATIO": {"mean": float(len_ratio.mean()), "std": float(len_ratio.std())},
            "COS_SIM": {"mean": float(cos_sims.mean()), "std": float(cos_sims.std())},
        },
        "config": {
            "model": args.model_name_or_path,
            "dataset": args.dataset_path,
            "level": args.level,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "ignore_comment": args.ignore_comment,
            "used_sacrebleu": HAS_SACREBLEU,
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")
    
    # print overall metrics
    logger.info("\n" + "="*60)
    logger.info(f"Evaluation Results - {args.split} split")
    logger.info("="*60)
    logger.info(f"Samples: {len(samples)}")
    logger.info("\nMetrics (mean ± std):")
    logger.info(f"  BLEU-2:     {bleu2.mean():.4f} ± {bleu2.std():.4f}")
    logger.info(f"  chrF:       {chrf.mean():.4f} ± {chrf.std():.4f}")
    logger.info(f"  Len Ratio:  {len_ratio.mean():.4f} ± {len_ratio.std():.4f}")
    logger.info(f"  CosSim:     {cos_sims.mean():.4f} ± {cos_sims.std():.4f}")
    logger.info(f"\nAll results saved to: {args.out_dir}")
    logger.info("="*60)
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
    