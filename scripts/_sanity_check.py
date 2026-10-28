import torch
from transformers import BertForSequenceClassification, BertTokenizer

tok = BertTokenizer.from_pretrained("/home/pi/models/hf")
model = BertForSequenceClassification.from_pretrained("/home/pi/models/hf")
model.eval()

pairs = [
    ("今天的饭不好吃", "今天心情不好"),
    ("猫在垫子上", "猫在垫子上"),
    ("猫在垫子上", "狗在垫子上"),
    ("专科毕业", "本科及以上学历"),
    ("中考", "高中考试"),
    ("十八岁", "未满十六岁"),
]
print("id2label:", model.config.id2label)
with torch.no_grad():
    for a, b in pairs:
        enc = tok(a, b, return_tensors="pt")
        out = model(**enc)
        p = torch.softmax(out.logits, dim=-1)[0].tolist()
        ids = enc["input_ids"][0][:12].tolist()
        print(a, "||", b, "->", [round(x, 4) for x in p],
              "argmax=", int(out.logits[0].argmax()), "ids=", ids)
