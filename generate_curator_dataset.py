"""
从 curator_score.csv 生成数据集
格式: Gene_ID, PMID, user_prompt (summary), PRODUCT_NAME (VEuPathDB product description)

可直接运行，也可粘贴到 read_data.ipynb 中执行。
"""
import pandas as pd
from datasets import Dataset, DatasetDict

# ============= 读取 curator_score.csv =============
csv_path = "curator_score.csv"
df = pd.read_csv(csv_path)

print(f"原始数据: {len(df)} 条记录")
print(f"列名: {df.columns.tolist()}")

# ============= 重命名列，映射到目标格式 =============
# gene_ID -> Gene_ID
# pmid -> PMID
# summary -> user_prompt
# VEuPathDB product description -> PRODUCT_NAME
df_dataset = df.rename(columns={
    "gene_ID": "Gene_ID",
    "pmid": "PMID",
    "summary": "user_prompt",
    "VEuPathDB product description": "PRODUCT_NAME",
})

# 只保留目标列
df_dataset = df_dataset[["Gene_ID", "PMID", "user_prompt", "PRODUCT_NAME"]].copy()

# 清理：去除缺失值
df_dataset = df_dataset.dropna()
print(f"清理后: {len(df_dataset)} 条记录")

# PMID 转为 int（原始数据中是整数）
df_dataset["PMID"] = df_dataset["PMID"].astype(int)

# ============= 显示示例 =============
print("\n" + "=" * 80)
print("数据集示例:")
print("=" * 80)
sample = df_dataset.iloc[0]
print(f"Gene_ID: {sample['Gene_ID']}")
print(f"PMID: {sample['PMID']}")
print(f"PRODUCT_NAME: {sample['PRODUCT_NAME']}")
print(f"user_prompt (前200字符): {sample['user_prompt'][:200]}...")

# ============= 保存为 HuggingFace Dataset =============
dataset = Dataset.from_pandas(df_dataset, preserve_index=False)

output_dir = "/users/thz501/fastscratch/bio/dataset/curator_score_dataset"
dataset_dict = DatasetDict({"test": dataset})
dataset_dict.save_to_disk(output_dir)

print(f"\n数据集已保存到: {output_dir}/")
print(f"数据集结构: {dataset_dict}")
print(f"字段: {dataset.column_names}")