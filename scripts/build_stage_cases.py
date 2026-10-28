"""从 bedu-jiaofu-douzi/.audit/crawl.json 生成「试卷 → 学段」门禁用例。

真实数据里，很多试卷被归到了错误的学段（例如「高中数学」目录下混进四年级试卷）。
这里用标题里的年级词推断试卷真实学段，作为 ground truth：

    小学：一年级..六年级
    初中：七年级..九年级 / 初一..初三 / 中考
    高中：高一..高三 / 高考

条件是目录名（如「高中数学」），候选项是试卷标题，期望值 = 标题学段是否等于目录学段。

用法：
    python3 scripts/build_stage_cases.py ../bedu-jiaofu-douzi/.audit/crawl.json \
        --per-label 50 --out tests/fixtures/stage_cases.tsv
"""
import argparse
import json
import random
import re

PRIMARY = ("一年级", "二年级", "三年级", "四年级", "五年级", "六年级")
JUNIOR = ("七年级", "八年级", "九年级", "初一", "初二", "初三")
SENIOR = ("高一", "高二", "高三")

STAGE_OF_BUCKET = {"高中": "高中", "初中": "初中", "小学": "小学"}


def stages_in(title):
    hits = set()
    if any(g in title for g in PRIMARY):
        hits.add("小学")
    if any(g in title for g in JUNIOR):
        hits.add("初中")
    if any(g in title for g in SENIOR):
        hits.add("高中")
    if "中考" in title:
        hits.add("初中")
    if "高考" in title:
        hits.add("高中")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crawl", help="crawl.json 路径")
    ap.add_argument("--per-label", type=int, default=50, help="每个学段取多少正/负例")
    ap.add_argument("--out", default="tests/fixtures/stage_cases.tsv")
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()

    data = json.load(open(args.crawl, encoding="utf-8"))
    rng = random.Random(args.seed)

    buckets = []
    for bucket, payload in data["files"].items():
        prefix = bucket[:2]
        if prefix not in STAGE_OF_BUCKET:
            continue
        buckets.append((bucket, STAGE_OF_BUCKET[prefix], payload["list"]))

    rows = []
    for bucket, bucket_stage, items in buckets:
        pos, neg = [], []
        seen = set()
        for item in items:
            title = item["name"].strip()
            if title in seen:
                continue
            seen.add(title)
            hits = stages_in(title)
            if len(hits) != 1:
                continue
            actual = next(iter(hits))
            (pos if actual == bucket_stage else neg).append(title)

        rng.shuffle(pos)
        rng.shuffle(neg)
        take = args.per_label
        # 条件只用学段（高中/初中/小学），与标题里的年级词做语义对齐；
        # 目录名里的学科不参与判定（抓取数据本身就有学科串桶）。
        for i, title in enumerate(pos[:take]):
            rows.append((f"{bucket}-p{i:03d}", title, bucket_stage, 1))
        for i, title in enumerate(neg[:take]):
            rows.append((f"{bucket}-n{i:03d}", title, bucket_stage, 0))

    rng.shuffle(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# 由 scripts/build_stage_cases.py 从真实抓取数据生成\n")
        f.write("# id\tcandidate(试卷标题)\tcondition(学段)\texpected(1=学段一致)\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")

    pos = sum(1 for r in rows if r[3] == 1)
    print(f"写入 {len(rows)} 条（正例 {pos} / 负例 {len(rows) - pos}）-> {args.out}")
    for stage in ("高中", "初中", "小学"):
        n = sum(1 for r in rows if r[2] == stage)
        print(f"  {stage:<4} {n}")


if __name__ == "__main__":
    main()
