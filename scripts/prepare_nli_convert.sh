#!/usr/bin/env bash
# 把 IDEA-CCNL/Erlangshen-Roberta-110M-NLI 转成 llama.cpp 能跑的 GGUF。
#
# 前置：
#   1. scripts/fetch_models.sh 已把 config.json / vocab.txt / pytorch_model.bin
#      下到 ~/models/hf
#   2. scripts/fetch_convert_tools.sh 已把与 llama-cpp-sys-2 0.1.160 完全对应的
#      convert_hf_to_gguf.py + conversion/ + gguf-py/ 下到 ~/models/convert
#   3. ~/venv-conv 里有 torch / transformers / gguf
#
# 关键补丁：上游 conversion/bert.py 假定 BERT 只做 embedding，会把 pooler.dense 丢掉。
# 但 BertForSequenceClassification 的分类头是 pooler(dense+tanh) -> classifier，
# llama.cpp 的 RANK 池化图正是 cls -> tanh -> cls_out。所以这里：
#   * 把 filter_tensors 从 classmethod 改成实例方法（需要读 self.hparams）；
#   * 把 pooler.dense.{weight,bias} 改名为 classifier.dense.{weight,bias} 保留下来。
set -euo pipefail

VENV="${VENV:-$HOME/venv-conv}"
CONVERT_DIR="${CONVERT_DIR:-$HOME/models/convert}"
SRC_DIR="${SRC_DIR:-$HOME/models/hf}"
OUT="${OUT:-$HOME/models/match-gate/erlangshen-nli-110m-q8_0.gguf}"

test -x "$VENV/bin/python" || { echo "缺少 $VENV/bin/python"; exit 1; }
test -f "$SRC_DIR/pytorch_model.bin" || { echo "缺少 $SRC_DIR/pytorch_model.bin"; exit 1; }
test -f "$CONVERT_DIR/convert_hf_to_gguf.py" || { echo "缺少 $CONVERT_DIR/convert_hf_to_gguf.py"; exit 1; }

echo "== 1. 打补丁：保留 BERT 分类头的 pooler =="
"$VENV/bin/python" - "$CONVERT_DIR/conversion/bert.py" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text(encoding="utf-8")

# 1) filter_tensors 需要读 self.hparams，改成实例方法。
old_sig = '''    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("bert."):'''
new_sig = '''    def filter_tensors(self, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("bert."):'''
if old_sig in src:
    src = src.replace(old_sig, new_sig, 1)
elif new_sig not in src:
    raise SystemExit("没找到 filter_tensors 签名，convert 脚本可能变了，请手工检查")

# 2) 保留 pooler 并改名成 cls（llama.cpp 的 classifier.dense）。
new_block = '''        if name == "embeddings.position_ids":
            return None

        # BertForSequenceClassification 的分类头是 pooler(dense + tanh) -> classifier，
        # llama.cpp 用 `cls` 张量表示 pooler（graph: cls -> tanh -> cls_out）。
        # 上游默认丢弃 pooler，会让分类 logits 与 HF 不一致，这里保留并改名。
        # 注意：filter_tensors 在 super().__init__() 里就会被调用，
        # 那时 self.cls_out_labels 还没赋值，所以直接读 hparams。
        if name in ("pooler.dense.weight", "pooler.dense.bias"):
            labels = self.hparams.get("id2label") or {}
            if len(labels) < 2 or (len(labels) == 2 and labels.get("0") == "LABEL_0"):
                return None
            name = "classifier.dense" + name[len("pooler.dense"):]
'''

if new_block in src:
    print("pooler 补丁已存在")
else:
    patterns = [
        # 上游原版
        r'        # we are only using BERT for embeddings.*?\n'
        r'        if name in \("embeddings\.position_ids".*?\n'
        r'            return None\n',
        # 早期打过一半的版本
        r'        if name == "embeddings\.position_ids":\n'
        r'            return None\n\n'
        r'        # BertForSequenceClassification.*?\n'
        r'        if name in \("pooler\.dense\.weight".*?\n'
        r'            name = "classifier\.dense".*?\n',
    ]
    for pattern in patterns:
        if re.search(pattern, src, re.S):
            src = re.sub(pattern, new_block, src, count=1, flags=re.S)
            break
    else:
        raise SystemExit("没找到 pooler 相关代码块，请手工检查")

path.write_text(src, encoding="utf-8")
print("补丁已应用")
PY

echo "== 2. 转换 =="
mkdir -p "$(dirname "$OUT")"
cd "$CONVERT_DIR"
"$VENV/bin/python" convert_hf_to_gguf.py "$SRC_DIR" \
    --outtype q8_0 \
    --outfile "$OUT"

echo "== 3. 结果 =="
ls -l "$OUT"
df -h / | tail -1
