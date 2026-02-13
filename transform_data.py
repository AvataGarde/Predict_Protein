import os
import re
import time
import argparse
import random
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, Dataset

# ==================== Prompt 模板 ====================

SYSTEM_PROMPT = """You are an expert bioinformatician. Your task is to convert narrative protein summaries from one format to another.

INPUT FORMAT (VEuPathDB style):
- Uses bullet points (•)
- Contains experimental details, citations, specific numbers
- Narrative prose style

OUTPUT FORMAT (UniProt/Swiss-Prot style):
- Uses section labels: FUNCTION, SUBUNIT, SUBCELLULAR LOCATION, SIMILARITY, CATALYTIC ACTIVITY, COFACTOR, PATHWAY, etc.
- Sections separated by " ; "
- Telegraphic style, no bullet points
- Removes experimental details and citations but keeps all functional information

CRITICAL RULES:
1. PRESERVE all protein/gene names mentioned (e.g., DDX47, CRZ1, MSP1, Rrp8p)
2. PRESERVE protein type keywords (helicase, kinase, factor, synthase, etc.)
3. PRESERVE family/domain information  
4. REMOVE specific experimental numbers, citations, and detailed methodology
5. Output length should match the detail level of the provided examples (can be 100-1500 characters)"""

USER_PROMPT_TEMPLATE = """Convert the following VEuPathDB-style summary to UniProt-style format.

### UniProt Style Examples:
{examples}

### Input (VEuPathDB style):
{summary}

### Output (UniProt style):"""

# ==================== 辅助函数 ====================

class FewShotPool:
    """
    Few-shot 样本池，支持每次随机抽取不同的 examples
    """
    def __init__(self, uniprot_path: str, pool_size: int = 200):
        self.pool = []
        self.fallback = [
            "FUNCTION: RNA helicase involved in ribosome biogenesis ; SIMILARITY: Belongs to the DEAD box helicase family",
            "FUNCTION: Transcription factor that regulates gene expression ; SUBUNIT: Homodimer ; SIMILARITY: Contains zinc finger domains",
            "FUNCTION: Catalyzes the attachment of amino acid to tRNA ; SIMILARITY: Belongs to the class-I aminoacyl-tRNA synthetase family",
        ]
        
        print(f"Loading UniProt dataset for few-shot pool from {uniprot_path}...")
        try:
            ds = load_from_disk(uniprot_path)
            train_data = ds["train"]
            
            # 随机抽取候选样本
            indices = random.sample(range(len(train_data)), min(len(train_data), pool_size * 5))
            
            for idx in indices:
                sample = train_data[idx]
                comment = sample.get("comment", "").strip()
                
                # 筛选条件：
                # 1. 有内容
                # 2. 包含 FUNCTION 或 SIMILARITY（典型 UniProt 格式）
                # 3. 长度适中 (50-2500 字符，涵盖大部分样本)
                if (comment 
                    and 50 < len(comment) < 2500 
                    and ("FUNCTION" in comment or "SIMILARITY" in comment)):
                    self.pool.append(comment)
                
                if len(self.pool) >= pool_size:
                    break
            
            # 统计信息
            if self.pool:
                lengths = [len(ex) for ex in self.pool]
                print(f"  Loaded {len(self.pool)} examples into few-shot pool")
                print(f"  Length statistics:")
                print(f"    Min: {min(lengths)} chars")
                print(f"    Max: {max(lengths)} chars")
                print(f"    Avg: {sum(lengths)/len(lengths):.0f} chars")
            else:
                print("  Warning: No examples loaded, using fallback")
                self.pool = self.fallback
            
        except Exception as e:
            print(f"Warning: Failed to load UniProt dataset ({e}). Using fallback examples.")
            self.pool = self.fallback
    
    def sample(self, num_shots: int = 5) -> str:
        """每次随机抽取 num_shots 个 examples"""
        if len(self.pool) < num_shots:
            examples = self.pool + self.fallback[:num_shots - len(self.pool)]
        else:
            examples = random.sample(self.pool, num_shots)
        
        # 格式化输出
        formatted = "\n\n".join([f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples)])
        return formatted
    
    def show_samples(self, num_shots: int = 3):
        """显示几个样本（用于验证）"""
        examples = self.sample(num_shots)
        print("\n" + "="*60)
        print(f"Sample Few-Shot Examples ({num_shots} random samples):")
        print("="*60)
        print(examples)
        print("="*60 + "\n")


# ==================== OpenAI 客户端 ====================

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = openai.OpenAI()
        self.model = model
    
    def rewrite(self, summary: str, examples: str) -> str:
        """
        将 VEuPathDB summary 转换为 UniProt style
        纯格式转换，不涉及 protein description
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                        examples=examples, 
                        summary=summary
                    )}
                ],
                temperature=0.2,  # 低温度保持格式一致
                max_tokens=1500,  # 适应较长输出
            )
            content = response.choices[0].message.content.strip()
            
            # 清理：移除可能的前缀
            content = re.sub(r"^(Output|UniProt[- ]?style|Result)[\s:]*", "", content, flags=re.IGNORECASE).strip()
            content = re.sub(r"^```.*?```$", "", content, flags=re.DOTALL).strip()
            
            return content
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # 降级处理：简单规则转换
            return self._fallback_rewrite(summary)
    
    def _fallback_rewrite(self, summary: str) -> str:
        """API 失败时的降级处理"""
        # 移除 bullet points
        text = re.sub(r'[•\-\*]\s*', '', summary)
        # 取前两句
        sentences = re.split(r'[.!?]', text)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:2]
        if key_sentences:
            return f"FUNCTION: {'. '.join(key_sentences)}"
        return f"FUNCTION: {text[:200]}"


class AnthropicClient:
    """Anthropic Claude API 客户端（备选）"""
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.client = anthropic.Anthropic()
        self.model = model
    
    def rewrite(self, summary: str, examples: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,  # 适应较长输出
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                        examples=examples,
                        summary=summary
                    )}
                ]
            )
            content = response.content[0].text.strip()
            content = re.sub(r"^(Output|UniProt[- ]?style|Result)[\s:]*", "", content, flags=re.IGNORECASE).strip()
            return content
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return f"FUNCTION: {summary[:200]}"


# ==================== 主处理逻辑 ====================

def get_client(provider: str, model: str):
    """获取 LLM 客户端"""
    if provider == "openai":
        return OpenAIClient(model)
    elif provider == "anthropic":
        return AnthropicClient(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def process_single_sample(args):
    """处理单个样本"""
    idx, sample, client, few_shot_pool, num_shots = args
    original = sample.get("user_prompt", "")
    
    if not original or original == "No summary available." or len(original) < 20:
        return idx, "FUNCTION: No functional information available"
    
    # 每次随机抽取不同的 few-shot examples
    examples = few_shot_pool.sample(num_shots)
    rewritten = client.rewrite(original, examples)
    return idx, rewritten


def rewrite_dataset(
    input_path: str,
    output_path: str,
    uniprot_path: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    num_shots: int = 5,
    pool_size: int = 200,
    max_samples: int = None,
    num_workers: int = 8,
    rate_limit_delay: float = 0.05
):
    """重写整个数据集"""
    
    # 1. 初始化 Few-Shot Pool（一次加载，每次随机抽取）
    few_shot_pool = FewShotPool(uniprot_path, pool_size=pool_size)
    few_shot_pool.show_samples(num_shots)  # 显示几个样本验证
    
    # 2. 初始化客户端
    client = get_client(provider, model)
    print(f"Using {provider} with model: {model}")
    print(f"Each API call will use {num_shots} randomly sampled few-shot examples\n")
    
    # 3. 加载数据集
    print(f"Loading input dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    
    transformed_splits = {}
    
    for split_name in dataset.keys():
        print(f"\nProcessing {split_name} split...")
        split_data = dataset[split_name]
        
        # 限制样本数（用于测试）
        if max_samples and len(split_data) > max_samples:
            indices = list(range(max_samples))
            print(f"  Limited to {max_samples} samples (total: {len(split_data)})")
        else:
            indices = list(range(len(split_data)))
            
        results = {}
        
        print(f"  Rewriting {len(indices)} samples using {num_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for idx in indices:
                sample = split_data[idx]
                # 传递 few_shot_pool 而不是固定的 examples
                futures.append(executor.submit(
                    process_single_sample, 
                    (idx, sample, client, few_shot_pool, num_shots)
                ))
                time.sleep(rate_limit_delay)  # Rate limiting
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, rewritten = future.result()
                results[idx] = rewritten
        
        # 显示样本验证
        print(f"\n[Sample Verification for {split_name}]")
        sample_idx = indices[0] if indices else 0
        if sample_idx in results:
            original = split_data[sample_idx]['user_prompt']
            rewritten = results[sample_idx]
            print(f"Original ({len(original)} chars):")
            print(f"  {original[:200]}...")
            print(f"Rewritten ({len(rewritten)} chars):")
            print(f"  {rewritten}")
            print()
                
        # 构建新数据
        new_data = []
        for idx in range(len(split_data)):
            sample = dict(split_data[idx])
            sample["user_prompt_original"] = sample.get("user_prompt", "")
            if idx in results:
                sample["user_prompt"] = results[idx]
            new_data.append(sample)
            
        transformed_splits[split_name] = Dataset.from_list(new_data)
    
    # 4. 保存结果
    print(f"\nSaving transformed dataset to {output_path}...")
    new_ds = DatasetDict(transformed_splits)
    new_ds.save_to_disk(output_path)
    
    # 5. 统计信息
    print("\n" + "="*60)
    print("Transformation Statistics")
    print("="*60)
    for split_name in new_ds.keys():
        split_data = new_ds[split_name]
        orig_lens = [len(s.get("user_prompt_original", "")) for s in split_data]
        new_lens = [len(s["user_prompt"]) for s in split_data]
        print(f"\n{split_name}:")
        print(f"  Samples: {len(split_data)}")
        print(f"  Original avg length: {sum(orig_lens)/len(orig_lens):.0f} chars")
        print(f"  Transformed avg length: {sum(new_lens)/len(new_lens):.0f} chars")
        print(f"  Compression ratio: {sum(new_lens)/max(sum(orig_lens),1):.1%}")
    
    print("\nDone!")
    return new_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform VEuPathDB summaries to UniProt style")
    parser.add_argument("--input_path", type=str, required=True, help="Input VEuPathDB dataset path")
    parser.add_argument("--output_path", type=str, required=True, help="Output dataset path")
    parser.add_argument("--uniprot_path", type=str, required=True, help="UniProt dataset path for few-shot examples")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--num_shots", type=int, default=5, help="Number of few-shot examples per API call")
    parser.add_argument("--pool_size", type=int, default=1000, help="Size of few-shot example pool")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split (for testing)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--rate_limit_delay", type=float, default=0.05, help="Delay between API calls")
    
    args = parser.parse_args()
    
    rewrite_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        uniprot_path=args.uniprot_path,
        provider=args.provider,
        model=args.model,
        num_shots=args.num_shots,
        pool_size=args.pool_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        rate_limit_delay=args.rate_limit_delay
    )