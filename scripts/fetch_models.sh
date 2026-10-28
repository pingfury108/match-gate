#!/usr/bin/env bash
# 下载模型到 ~/models/match-gate。
#
# * 决策模型：ModelScope 的 quincyqiang/chinese-decision-0.8b-gguf（Q8_0，812MB）
# * 重排模型：hf-mirror 的 gpustack/bge-reranker-v2-m3-GGUF（Q8_0，607MB）
# * NLI 原始权重：hf-mirror 上的 IDEA-CCNL/Erlangshen-Roberta-110M-NLI（409MB，用于自己转换）
#
# 用法：
#   scripts/fetch_models.sh decision   # 只下决策模型（+ 配置文件）
#   scripts/fetch_models.sh rerank     # 只下重排模型
#   scripts/fetch_models.sh nli-src    # 只下 NLI 原始权重
#   scripts/fetch_models.sh all
set -euo pipefail

MODELS="${MODELS:-$HOME/models}"
DEST="$MODELS/match-gate"
HF_SRC="$MODELS/hf"
MS_BASE="https://www.modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf/resolve/master"
HF_BASE="https://hf-mirror.com/IDEA-CCNL/Erlangshen-Roberta-110M-NLI/resolve/main"
RR_BASE="https://hf-mirror.com/gpustack/bge-reranker-v2-m3-GGUF/resolve/main"

fetch() {
    local url="$1" out="$2"
    if [ -s "$out" ]; then
        echo "已存在，跳过：$out"
        return
    fi
    echo "下载 $url"
    curl -fL --retry 5 --retry-delay 3 --max-time 3600 -o "$out.part" "$url"
    mv "$out.part" "$out"
}

fetch_decision() {
    mkdir -p "$DEST"
    fetch "$MS_BASE/chinese-decision-0.8b-Q8_0.gguf" "$DEST/chinese-decision-0.8b-Q8_0.gguf"
    for f in decision_config.json tokenizer.json tokenizer_config.json chat_template.jinja README.md; do
        fetch "$MS_BASE/$f" "$DEST/$f"
    done
    echo "决策模型 SHA256（应为 6a71cc35bd83f8b050499567faec59e9ae05778b75b1e2119d27d45b80f7e68b）："
    sha256sum "$DEST/chinese-decision-0.8b-Q8_0.gguf"
}

fetch_nli_src() {
    mkdir -p "$HF_SRC"
    for f in config.json vocab.txt pytorch_model.bin; do
        fetch "$HF_BASE/$f" "$HF_SRC/$f"
    done
}

fetch_rerank() {
    mkdir -p "$DEST"
    fetch "$RR_BASE/bge-reranker-v2-m3-Q8_0.gguf" "$DEST/bge-reranker-v2-m3-Q8_0.gguf"
    echo "重排模型 SHA256（应为 a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3）："
    sha256sum "$DEST/bge-reranker-v2-m3-Q8_0.gguf"
}

case "${1:-all}" in
    decision) fetch_decision ;;
    rerank) fetch_rerank ;;
    nli-src) fetch_nli_src ;;
    all) fetch_decision; fetch_rerank; fetch_nli_src ;;
    *) echo "用法：$0 [decision|rerank|nli-src|all]"; exit 1 ;;
esac

df -h / | tail -1
