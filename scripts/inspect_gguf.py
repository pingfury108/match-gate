"""检查转换出来的 GGUF 元数据是否符合 llama.cpp BERT 分类器的要求。"""
import sys

from gguf import GGUFReader

path = sys.argv[1] if len(sys.argv) > 1 else "/home/pi/models/match-gate/erlangshen-nli-110m-q8_0.gguf"
r = GGUFReader(path)

print("== metadata ==")
for key in (
    "general.architecture",
    "general.name",
    "bert.context_length",
    "bert.embedding_length",
    "bert.block_count",
    "bert.attention.head_count",
    "bert.pooling_type",
    "bert.classifier_pooling_type",
    "classifier.output_labels",
    "tokenizer.ggml.model",
    "tokenizer.ggml.token_type_count",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
):
    f = r.get_field(key)
    if f is None:
        print(f"  {key:<40} <missing>")
        continue
    try:
        val = f.contents()
    except Exception as exc:  # noqa: BLE001
        val = f"<error: {exc}>"
    text = repr(val)
    if len(text) > 120:
        text = text[:117] + "..."
    print(f"  {key:<40} {text}")

print("== tensors (head) ==")
names = [t.name for t in r.tensors]
for n in names:
    if n.startswith(("cls", "classifier", "pooler", "token_types", "position_embd")):
        print("  ", n)
print("total tensors:", len(names))
