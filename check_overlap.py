"""
列出 gene_summary_dataset train set 与 curator_score_dataset test set 的重叠样本。
匹配键：Gene_ID + PMID
输出：overlap_samples.jsonl（每行一条重叠样本）
"""
import json
from datasets import load_from_disk

TRAIN_DS = "/users/thz501/fastscratch/bio/dataset/gene_summary_dataset"
TEST_DS  = "/users/thz501/fastscratch/bio/dataset/curator_score_dataset"
OUT_FILE = "/users/thz501/fastscratch/bio/overlap_samples.jsonl"

train = load_from_disk(TRAIN_DS)["train"]
test  = load_from_disk(TEST_DS)["test"]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# 以 Gene_ID + PMID 为键建立 test set 索引
test_keys = set(
    (str(s["Gene_ID"]), str(s["PMID"])) for s in test
)

# 找重叠
overlaps = [
    dict(s) for s in train
    if (str(s["Gene_ID"]), str(s["PMID"])) in test_keys
]

print(f"重叠样本数: {len(overlaps)}")

# 写出
with open(OUT_FILE, "w") as f:
    for s in overlaps:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"已写出到: {OUT_FILE}")

# 打印 Gene_ID + PMID 列表，便于快速查看
print("\n--- 重叠列表 (Gene_ID, PMID) ---")
for s in overlaps:
    print(f"  {s['Gene_ID']}\t{s['PMID']}")
