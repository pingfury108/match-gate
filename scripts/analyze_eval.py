"""对比多个 eval-gate --json 结果：准确率、阈值扫描、AUC、延迟。

用法：
    python3 scripts/analyze_eval.py /tmp/stage_nli.json /tmp/stage_rerank.json /tmp/stage_decision.json
"""
import json
import sys


def load(path):
    data = json.load(open(path))
    rows = data["results"]
    pos = [r["p_true"] for r in rows if r["expect_true"]]
    neg = [r["p_true"] for r in rows if not r["expect_true"]]
    return data, rows, pos, neg


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def best_threshold(rows):
    best = (0.5, 0.0)
    for i in range(1, 100):
        th = i / 100
        acc = sum(1 for r in rows if (r["p_true"] >= th) == r["expect_true"]) / len(rows)
        if acc > best[1]:
            best = (th, acc)
    return best


def main(paths):
    print(f"{'file':<28} {'n':>4} {'acc@.5':>7} {'best':>13} {'AUC':>6} {'ms':>7} {'FP':>3} {'FN':>3}")
    for path in paths:
        data, rows, pos, neg = load(path)
        acc = sum(1 for r in rows if (r["p_true"] >= 0.5) == r["expect_true"]) / len(rows)
        th, bacc = best_threshold(rows)
        fp = sum(1 for r in rows if not r["expect_true"] and r["p_true"] >= 0.5)
        fn = sum(1 for r in rows if r["expect_true"] and r["p_true"] < 0.5)
        name = path.split("/")[-1]
        print(
            f"{name:<28} {len(rows):>4} {acc * 100:>6.2f}% "
            f"{th:>5.2f}->{bacc * 100:>5.1f}% {auc(pos, neg):>6.3f} "
            f"{data['mean_latency_ms']:>7.0f} {fp:>3} {fn:>3}"
        )

    # 每个文件的错误样例（按置信度排序，最多 8 条）
    for path in paths:
        data, rows, _, _ = load(path)
        errs = [r for r in rows if (r["p_true"] >= 0.5) != r["expect_true"]]
        errs.sort(key=lambda r: -abs(r["p_true"] - 0.5))
        print(f"\n-- {path.split('/')[-1]} 前 8 个错误 --")
        for r in errs[:8]:
            tag = "FP" if r["p_true"] >= 0.5 else "FN"
            print(f"  {tag} p={r['p_true']:.3f} [{r['condition']}] {r['candidate'][:58]}")


if __name__ == "__main__":
    main(sys.argv[1:])
