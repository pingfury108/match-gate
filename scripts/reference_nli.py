#!/usr/bin/env python3
"""用 transformers 跑同一份测试集，作为 GGUF 推理的参考结果。

用法：
    ~/venv-conv/bin/python scripts/reference_nli.py \
        --model ~/models/hf --cases tests/fixtures/nli_cases.tsv --out /tmp/ref.json

输出每条样本的三个概率和标签，并在 stderr 打印准确率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]


def read_cases(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise SystemExit(f"坏行：{line!r}")
        yield fields[0], fields[1], fields[2], [x.strip().lower() for x in fields[3].split("|")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models/hf"))
    ap.add_argument("--cases", default="tests/fixtures/nli_cases.tsv")
    ap.add_argument("--out", default="/tmp/nli_reference.json")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    rows = []
    correct = 0
    with torch.no_grad():
        for case_id, premise, hypothesis, expected in read_cases(Path(args.cases)):
            batch = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            logits = model(**batch).logits[0]
            probs = torch.softmax(logits, dim=-1).tolist()
            idx = int(torch.argmax(logits).item())
            label = LABELS[idx].lower()
            ok = label in expected
            correct += int(ok)
            rows.append(
                {
                    "id": case_id,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "expected": expected,
                    "label": label,
                    "ok": ok,
                    "scores": {
                        "contradiction": probs[0],
                        "neutral": probs[1],
                        "entailment": probs[2],
                    },
                }
            )
            print(
                f"{'ok  ' if ok else 'FAIL'} {case_id:<10} -> {label:<13} "
                f"entail {probs[2]:.4f} neutral {probs[1]:.4f} contra {probs[0]:.4f}",
                file=sys.stderr,
            )

    Path(args.out).write_text(
        json.dumps({"total": len(rows), "correct": correct, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"accuracy {correct}/{len(rows)} = {correct / len(rows) * 100:.2f}%", file=sys.stderr)
    print(f"written {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
