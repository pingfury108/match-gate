"""用审计数据抽查：决策模型能否从试卷标题选出教材版本（ver）。

数据来自 ../bedu-jiaofu-douzi/.audit/report.json 的 problems：
只取「规则判出了 ver 且该值确实在 options.editions 里」的样本，规则值当伪标签。

用法：
    python3 scripts/build_edition_probe.py ../bedu-jiaofu-douzi/.audit/report.json \
        --limit 25 --out tests/fixtures/edition_probe.jsonl
"""
import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--out", default="tests/fixtures/edition_probe.jsonl")
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()

    data = json.load(open(args.report, encoding="utf-8"))
    rows = []
    for subject, payload in data["subjects"].items():
        for p in payload.get("problems", []):
            ver = p.get("ver") or {}
            opts = (p.get("options") or {}).get("editions") or []
            if ver.get("state") != "ok" or not ver.get("rule"):
                continue
            if ver["rule"] not in opts:
                continue
            rows.append(
                {
                    "id": f"{subject}-{len(rows):03d}",
                    "subject": subject,
                    "fileName": p["name"],
                    "options": opts,
                    "expected": ver["rule"],
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写入 {len(rows)} 条 -> {args.out}")
    for row in rows[:3]:
        print(" ", row["expected"], "|", row["fileName"][:50], "| options", len(row["options"]))


if __name__ == "__main__":
    main()
