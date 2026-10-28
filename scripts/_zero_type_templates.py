"""在「token_type_ids 全 0」（llama.cpp 的语义）下对比多种句对模板。

llama.cpp 的 BERT 图把 token type 硬编码为 0，第二句的 segment embedding 丢失。
这里验证：能否用文本模板把「第二句」的信息补回来。

用法：~/venv-conv/bin/python scripts/_zero_type_templates.py
"""
import torch
from transformers import BertForSequenceClassification, BertTokenizer

MODEL = "/home/pi/models/hf"
CASES = "tests/fixtures/nli_cases.tsv"
LABELS = ["contradiction", "neutral", "entailment"]

TEMPLATES = {
    "raw": lambda p, h: (p, h),
    "premise_hypo": lambda p, h: (f"前提：{p}", f"假设：{h}"),
    "known_judge": lambda p, h: (f"已知：{p}", f"需要判断：{h}"),
    "sent_a_b": lambda p, h: (f"句子A：{p}", f"句子B：{h}"),
    "situation": lambda p, h: (f"候选人的情况是：{p}", f"候选人满足以下条件：{h}"),
    "therefore": lambda p, h: (f"{p}。", f"因此，{h}。"),
    "info_cond": lambda p, h: (f"信息：{p}", f"条件：{h}"),
    "first_second": lambda p, h: (f"第一句：{p}", f"第二句：{h}"),
}


def read_cases():
    rows = []
    for line in open(CASES, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        rows.append((f[0], f[1], f[2], [x.strip() for x in f[3].split("|")]))
    return rows


def main():
    tok = BertTokenizer.from_pretrained(MODEL)
    model = BertForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    cases = read_cases()

    with torch.no_grad():
        for name, tpl in TEMPLATES.items():
            ok = 0
            fails = []
            for cid, premise, hypothesis, expected in cases:
                a, b = tpl(premise, hypothesis)
                enc = tok(a, b, return_tensors="pt", truncation=True, max_length=512)
                enc["token_type_ids"] = torch.zeros_like(enc["token_type_ids"])
                p = torch.softmax(model(**enc).logits[0], dim=-1).tolist()
                label = LABELS[max(range(3), key=lambda i: p[i])]
                if label in expected:
                    ok += 1
                else:
                    fails.append(f"{cid}:{label}")
            print(f"{name:<14} {ok}/{len(cases)} = {ok / len(cases) * 100:5.1f}%   fails: {' '.join(fails)}")


if __name__ == "__main__":
    main()
