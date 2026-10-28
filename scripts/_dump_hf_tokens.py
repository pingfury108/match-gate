"""打印 HF BertTokenizer 对给定文本的分词结果，用于和 llama.cpp 对比。

用法：~/venv-conv/bin/python scripts/_dump_hf_tokens.py "中考" "高中考试" ...
"""
import sys

from transformers import BertTokenizer

tok = BertTokenizer.from_pretrained("/home/pi/models/hf")
for text in sys.argv[1:]:
    ids = tok.encode(text, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"{text!r}: {ids}  {pieces}")
