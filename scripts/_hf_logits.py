"""打印 HF 模型的原始 logits 与 softmax，用于和 Rust/llama.cpp 逐项对比。

用法：~/venv-conv/bin/python scripts/_hf_logits.py "premise" "hypothesis" ["p2" "h2" ...]
"""
import sys

import torch
from transformers import BertForSequenceClassification, BertTokenizer

tok = BertTokenizer.from_pretrained("/home/pi/models/hf")
model = BertForSequenceClassification.from_pretrained("/home/pi/models/hf")
model.eval()

args = sys.argv[1:]
pairs = list(zip(args[0::2], args[1::2]))
with torch.no_grad():
    for premise, hypothesis in pairs:
        enc = tok(premise, hypothesis, return_tensors="pt")
        out = model(**enc)
        logits = out.logits[0].tolist()
        probs = torch.softmax(out.logits[0], dim=-1).tolist()
        print(f"{premise!r} || {hypothesis!r}")
        print("  ids   ", enc["input_ids"][0].tolist())
        print("  types ", enc["token_type_ids"][0].tolist())
        print("  logits", [round(x, 4) for x in logits])
        print("  probs ", [round(x, 4) for x in probs])
