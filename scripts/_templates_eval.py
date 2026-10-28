"""对比几种句对模板对同一批 NLI 测试集的影响。

用法：~/venv-conv/bin/python scripts/_templates_eval.py
"""
import torch
from transformers import BertForSequenceClassification, BertTokenizer

MODEL = "/home/pi/models/hf"
CASES = "tests/fixtures/nli_cases.tsv"
LABELS = ["contradiction", "neutral", "entailment"]

TEMPLATES = {
    "raw": lambda c, cond: (c, cond),
    "situation": lambda c, cond: (f"候选人的情况是：{c}", f"候选人满足以下条件：{cond}"),
    "therefore": lambda c, cond: (f"{c}。", f"因此，{cond}。"),
    "known": lambda c, cond: (f"已知：{c}", f"判断：{cond}"),
    "is_a": lambda c, cond: (c, cond),
    "reverse": lambda c, cond: (cond, c),
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
            for cid, c, cond, expected in cases:
                a, b = tpl(c, cond)
                enc = tok(a, b, return_tensors="pt", truncation=True, max_length=512)
                p = torch.softmax(model(**enc).logits[0], dim=-1).tolist()
                label = LABELS[max(range(3), key=lambda i: p[i])]
                if label in expected:
                    ok += 1
                else:
                    fails.append(f"{cid}:{label}({p[2]:.2f})")
            print(f"{name:<12} {ok}/{len(cases)} = {ok / len(cases) * 100:5.1f}%   fails: {' '.join(fails)}")


if __name__ == "__main__":
    main()
