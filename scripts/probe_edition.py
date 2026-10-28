"""跑 edition_probe.jsonl：用决策模型的 choice 题型选教材版本。

用法（pi400，仓库根目录）：
    python3 scripts/probe_edition.py tests/fixtures/edition_probe.jsonl
"""
import json
import subprocess
import sys

BIN = "./target/release/match-gate"
INSTRUCTIONS = "这份试卷使用的是哪个教材版本？"


def norm(s):
    return s.replace("(新)", "").replace("（新）", "").strip()


def main(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    exact = 0
    loose = 0
    total_ms = 0
    details = []

    for row in rows:
        cmd = [
            BIN,
            "--threads",
            "4",
            "--json",
            "decide",
            "--state",
            row["fileName"],
            "--instructions",
            INSTRUCTIONS,
            "--type",
            "choice",
        ]
        for opt in row["options"]:
            cmd += ["--option", f"{opt}={opt}"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print("ERR", row["id"], out.stderr[-200:])
            continue
        data = json.loads(out.stdout)
        choice = data["result"].get("choice")
        ms = 0.0
        ok = choice == row["expected"]
        ok_loose = choice is not None and norm(choice) == norm(row["expected"])
        exact += ok
        loose += ok_loose
        details.append((row["id"], row["expected"], choice, ok_loose))

    for cid, exp, got, ok in details:
        print(f"{'ok  ' if ok else 'FAIL'} {cid:<18} expect={exp:<14} got={got}")
    n = len(details)
    print(f"\nexact {exact}/{n} = {exact / n * 100:.1f}%   loose {loose}/{n} = {loose / n * 100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/edition_probe.jsonl")
