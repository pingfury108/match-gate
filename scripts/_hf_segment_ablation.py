"""对比 HF 在「真实 token_type_ids」与「全部置 0」下的 logits。

用来量化 llama.cpp BERT 图把 token type 硬编码为 0 带来的偏差。

用法：~/venv-conv/bin/python scripts/_hf_segment_ablation.py
"""
import torch
from transformers import BertForSequenceClassification, BertTokenizer

tok = BertTokenizer.from_pretrained("/home/pi/models/hf")
model = BertForSequenceClassification.from_pretrained("/home/pi/models/hf")
model.eval()

PAIRS = [
    ("中考", "高中考试"),
    ("猫在垫子上", "狗在垫子上"),
    ("专科毕业", "本科及以上学历"),
    ("十八岁", "未满十六岁"),
    ("会使用 Python", "会使用 Java"),
    ("持有 C1 驾驶证", "要求有驾驶证"),
]

with torch.no_grad():
    for premise, hypothesis in PAIRS:
        enc = tok(premise, hypothesis, return_tensors="pt")
        real = model(**enc).logits[0]
        zero_types = dict(enc)
        zero_types["token_type_ids"] = torch.zeros_like(enc["token_type_ids"])
        zero = model(**zero_types).logits[0]
        print(f"{premise} || {hypothesis}")
        print("  real types logits", [round(x, 4) for x in real.tolist()])
        print("  zero types logits", [round(x, 4) for x in zero.tolist()])
        print("  delta            ", [round((a - b).item(), 4) for a, b in zip(real, zero)])
