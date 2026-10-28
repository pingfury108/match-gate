# match-gate

中文语义门禁：判断「候选项是否满足某个条件」，纯 CPU、离线、单进程。
提供三条技术路线，并在 Raspberry Pi 400 上做了完整实测（215 条真实试卷数据）：

| 路线 | 模型 | 体积 | 准确率（真实试卷 215 条） | 延迟 | 结论 |
|---|---|---|---|---|---|
| **决策模型** | `chinese-decision-0.8b` Q8_0 | 812 MB | **86.1%**（AUC 0.949） | 5.3 s/题 | ✅ 推荐 |
| NLI 语义蕴含 | `Erlangshen-Roberta-110M-NLI` Q8_0 | 110 MB | 65.1%（AUC 0.802） | 0.36 s/题 | ⚠️ 只能做粗筛 |
| 交叉编码器重排 | `bge-reranker-v2-m3` Q8_0 | 636 MB | 54.9%（AUC 0.607） | 1.2 s/题 | ❌ 不适合单词级条件 |

**核心结论：110M 中文 NLI 无法区分「中考 / 高中考试」这类包含关系，即使换成 HF 原生推理
（76.5%）也会判错；0.8B 决策模型能稳定判对（P(满足)=0.15 / 0.04）。**
换更"擅长匹配"的 encoder（bge-reranker-v2-m3）也不行：短条件对长标题的相关性打分不可靠。
因此默认引擎是 `decision`。

注意边界：**能用代码规则判的，不要用模型**。标题里已有明确年级词（占 94.9%）时规则更快更准；
模型只该做规则判不了的兜底（约 5%）和「短条件 vs 候选项」的语义判定（规则做不了），详见第 5 节。

---

## 环境

实测机器：Raspberry Pi 400（BCM2711，4×Cortex-A72 @ 1.8GHz，3.7 GB RAM，Debian 11 aarch64）。

- Rust stable，`llama-cpp-2 = 0.1.160`（vendored llama.cpp，纯 CPU，`openmp`）
- 需要 `libclang-dev`（bindgen）
- 模型：GGUF，Q8_0

## 构建

```bash
# pi400 / Debian
sudo apt-get install -y build-essential cmake libclang-dev
LIBCLANG_PATH=/usr/lib/llvm-11/lib CMAKE_BUILD_PARALLEL_LEVEL=3 cargo build --release
# 或直接
scripts/build.sh
```

首次构建约 17 分钟（llama.cpp 的 C++ 部分占绝大部分），增量构建约 13 秒。

## 用法

```bash
# 0) 下载模型（决策 812MB；重排 636MB；NLI 源权重 409MB，可选）
scripts/fetch_models.sh decision
scripts/fetch_models.sh rerank
scripts/fetch_models.sh nli-src

# 1) 推荐：门禁，逐候选判断是否满足条件（默认 decision 引擎）
match-gate gate --condition "高中考试" \
  --candidate 中考 --candidate 高中考试 --candidate 高考
# REJECT   中考        p_true 0.1508  (4143ms)
# MATCH    高中考试     p_true 0.8292  (4168ms)
# MATCH    高考        p_true 0.7184  (4074ms)

# 也可以从文件读候选（一行一个）
match-gate gate --condition "本科及以上学历" --candidate-file candidates.txt

# 2) 多候选项条件匹配：一次前向给出所有候选项的概率
match-gate decide --state "客户留言：付款后一直没收到发票，我已经催了三次。" \
  --instructions "应该由哪个部门处理？" --type choice \
  --option "billing=支付、发票及退款" \
  --option "technical=故障和系统问题" \
  --option "sales=购买咨询与升级"

# 3) 单条 NLI / 重排打分（对照用）
match-gate nli --premise 中考 --hypothesis 高中考试
match-gate rerank --query "高中考试" --passage "中考"

# 4) 评测（三个引擎共用同一份 TSV）
match-gate eval-gate --cases tests/fixtures/gate_cases.tsv --min-accuracy 0.8
match-gate eval-gate --engine rerank --threshold 0.3 --cases tests/fixtures/gate_cases.tsv
match-gate eval-gate --engine nli --cases tests/fixtures/gate_cases.tsv
match-gate eval --cases tests/fixtures/nli_cases.tsv --min-accuracy 0.5

# 5) 排查分词（应与 HF BertTokenizer 完全一致）
match-gate tokens --text "中考"
```

全局参数：`--nli-model` / `--decision-model` / `--rerank-model`（或 `MATCH_GATE_*` 环境变量）、
`--threads`、`--json`。`gate --engine nli|rerank` 可切换路线（`--strict` / `--no-reverse` 只对 NLI 有效）。

重排引擎的 query/passage 模板可调：

```bash
# 默认 {condition}试卷 / {candidate}；也可换成更长的 query 提升召回
match-gate gate --engine rerank --threshold 0.3 \
  --rerank-query "{condition}试题" --rerank-passage "{candidate}" ...
```

## 模型准备

### 决策模型（现成 GGUF，无需转换）

`quincyqiang/chinese-decision-0.8b-gguf` 的 `chinese-decision-0.8b-Q8_0.gguf`：

- 811,843,936 bytes
- SHA256 `6a71cc35bd83f8b050499567faec59e9ae05778b75b1e2119d27d45b80f7e68b`
- 校准温度 `T = 1.10813307762146`（`decision_config.json`）
- 架构 `qwen35`（Qwen3.5 混合线性注意力，每题前必须清空 recurrent state）

### NLI 模型（必须自己转 GGUF）

HF / ModelScope 都**没有** Erlangshen 的现成 GGUF，只有 `pytorch_model.bin`。转换流程：

```bash
scripts/bootstrap_venv.sh        # torch(cpu) + transformers + gguf
scripts/fetch_convert_tools.sh   # 按 llama-cpp-sys-2 0.1.160 的 commit 拉 conversion/
scripts/fetch_models.sh nli-src  # config.json / vocab.txt / pytorch_model.bin
scripts/prepare_nli_convert.sh   # 打补丁 + 转 Q8_0
```

**关键补丁**（`scripts/prepare_nli_convert.sh` 自动完成）：

1. 上游 `conversion/bert.py` 假定 BERT 只做 embedding，会丢掉 `pooler.dense.{weight,bias}`；
   但 `BertForSequenceClassification` 的分类头是 `pooler(dense+tanh) → classifier`，
   llama.cpp 的 RANK 池化图正是 `cls → tanh → cls_out`。不保留 pooler，分类 logits 全错。
2. `filter_tensors` 从 `@classmethod` 改成实例方法，并把 `pooler.dense.*` 改名为
   `classifier.dense.*`。必须读 `self.hparams` 而不是 `self.cls_out_labels`，因为
   `super().__init__()` 内部就会调用 `filter_tensors`，那时 `cls_out_labels` 尚未赋值。

转换结果：201 tensors，含 `cls.weight/bias`、`cls.output.weight/bias`，
`bert.classifier.output_labels = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]`，
`tokenizer.ggml.token_type_count = 2`。

### 重排模型（现成 GGUF）

`gpustack/bge-reranker-v2-m3-GGUF` 的 `bge-reranker-v2-m3-Q8_0.gguf`：

- 635,676,416 bytes，SHA256 `a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3`
- 基座 XLM-RoBERTa，`type_vocab_size = 1`，**不依赖句段 embedding**，因此在 llama.cpp 下
  没有 Erlangshen 那个 token type 被硬编码为 0 的问题（`cls` / `cls.output` 张量齐全）。
- 输出 1 个 logit，本实现取 sigmoid。句对格式按 XLM-R：`<s> query </s></s> passage </s>`。
- 实测：**不适合「一个词的条件 + 很长的试卷标题」**，见下文第 5 节。

## 实测结果

### 1. 决策模型复现模型卡示例

同一段客户留言，`T = 1.1081`，pi400 4 线程：

| 题目 | 模型卡（Q8_0 / Windows） | 本实现（pi400） | 模型卡（BF16） |
|---|---|---|---|
| 紧迫性 noul | 0.8804 | **0.8614** | 0.8654 |
| 部门 choice | billing 0.9369 | **billing 0.9366** | 0.9406 |
| 不满程度 score | 1.3752 | **1.3763** | 1.3768 |

与 Q8_0 参考的差距主要来自 CPU/线程差异；noul 一题更接近 BF16 参考（差 0.004），
说明 prompt 拼接、字母 token 读取、温度校准、softmax 都对齐了。

### 2. 门禁用例（`tests/fixtures/gate_cases.tsv`，15 条）

`match-gate eval-gate`，门限 0.5：

```
ok   nli-ok-01  p_true=0.7184  高考 | 高中考试
FAIL nli-ok-02  p_true=0.1373  初中学业水平考试 | 中考        ← 模型不知道这是同义说法
ok   nli-ok-03  p_true=0.8379  硕士研究生 | 本科及以上学历
ok   nli-ok-04  p_true=0.9585  持有 C1 驾驶证 | 有驾驶证
ok   nli-ok-05  p_true=0.0150  没有教师资格证 | 持有教师资格证
ok   hard-01    p_true=0.1508  中考 | 高中考试              ← 目标场景
ok   hard-02    p_true=0.0390  高中考试 | 中考              ← 反向也对
ok   hard-03    p_true=0.1714  专科毕业 | 本科及以上学历
ok   hard-04    p_true=0.0857  十八岁 | 未满十六岁
ok   hard-05    p_true=0.0552  会使用 Python | 会使用 Java
ok   hard-06    p_true=0.0041  应届毕业生 | 有五年以上工作经验
ok   hard-07    p_true=0.0172  英语四级 | 英语六级
ok   hard-08    p_true=0.0130  小学教师资格证 | 高中教师资格证
ok   hard-09    p_true=0.0289  在职公务员 | 非公务员
ok   hard-10    p_true=0.0124  有三年工作经验 | 有五年以上工作经验

accuracy 14/15 = 93.33%   mean 4207ms/case
```

唯一失败的是「初中学业水平考试 = 中考」这种**别名知识**，不是逻辑判断问题；
需要的话可以在输入侧加同义词归一化。

同一份 15 条上另外两个引擎：`nli` 9/15 = 60.0%（137 ms）、`rerank` 13/15 = 86.7%（482 ms）。
人工用例太短、太干净，真实数据上的差距见第 5 节。

### 3. 110M NLI：为什么不可用

同一批 34 条中文 NLI 用例（`tests/fixtures/nli_cases.tsv`）：

| 推理路径 | 准确率 |
|---|---|
| HF 原生（真实 `token_type_ids`） | 26/34 = 76.5% |
| HF，`token_type_ids` 全 0（模拟 llama.cpp） | 20/34 = 58.8% |
| **llama.cpp / 本实现** | **20/34 = 58.8%** |
| HF 全 0 + 最佳模板（"候选人的情况是 / 候选人满足以下条件"） | 24/34 = 70.6% |

三个独立问题：

1. **语义能力不足**：即使 HF 原生推理，「中考 ⇒ 高中考试」的 entailment 也有 0.7631
   （反向 0.9015），「专科毕业 ⇒ 本科及以上学历」0.96，「十八岁 ⇒ 未满十六岁」0.75。
   这是 110M 模型对包含/上下位关系的判断偏差，换模板（situation / therefore / known /
   句子A/B 等 8 种）最多到 76.5%，无法解决。
2. **llama.cpp 丢失句段 embedding**：`src/models/bert.cpp` 明确把 token type 硬编码为 0
   （"Sentence A"），而 Erlangshen 的 `type_vocab_size=2`，第二句的 segment embedding
   被丢弃。实测把 HF 的 `token_type_ids` 全置 0，logits 与 llama.cpp 输出逐项吻合
   （差值 ≤0.06，纯 Q8 量化误差），准确率从 76.5% 掉到 58.8%。
3. **分词没有问题**：`match-gate tokens` 与 HF `BertTokenizer` 的 token id 完全一致
   （中文逐字切分、英文小写、`[CLS]/[SEP]` 均正确）。

结论：llama.cpp 的 BERT 图不支持 segment embedding，110M NLI 在这条链路上先天吃亏；
即便修好也会被模型本身的语义能力卡住。**不要再把 110M NLI 当作门禁主力。**

### 4. 延迟与内存（pi400，4 线程）

| 模型 | 单题延迟（短文本） | 单题延迟（真实长标题） | 加载时间 | 峰值 RSS |
|---|---|---|---|---|
| 决策 0.8B Q8_0 | 4.1 s | 5.3 s | ~2.2 s | 945 MB |
| NLI 110M Q8_0 | 145 ms | 363 ms | <0.1 s | 122 MB |
| 重排 568M Q8_0 | 482 ms | 1075 ms | ~3 s | 970 MB |

4 秒/题是 pi400 的硬件上限：Cortex-A72 没有 `dotprod`/`i8mm`，Q8_0 点积走标量路径。
同一进程内复用 session 只省加载时间，不改变单题耗时；结果可复现（同一输入重复跑，
概率完全一致）。若要批量处理大量候选，建议放到有 ARMv8.2 dotprod 或 x86 AVX2 的机器上。

### 5. 真实试卷数据（215 条，来自 `bedu-jiaofu-douzi/.audit`）

从 `../bedu-jiaofu-douzi/.audit/crawl.json`（14 个学科目录、共 5,922 份试卷）生成：
标题里的年级词推断真实学段（一年级..六年级→小学，七/八/九年级、中考→初中，
高一..高三、高考→高中），条件是学段（高中/初中/小学），期望值 = 标题学段是否一致。
这正是该 audit 里的 `stageMismatch`（例如「高中数学」目录混进四年级试卷）。

生成方式（可复现，固定 seed）：

```bash
python3 scripts/build_stage_cases.py ../bedu-jiaofu-douzi/.audit/crawl.json \
  --per-label 8 --out tests/fixtures/stage_cases.tsv
# 215 条：高中 103 / 初中 96 / 小学 16（正例 112 / 负例 103）
```

> 该文件由真实抓取数据生成，含 215 条试卷标题，**不随仓库提供**（见 `.gitignore`）。
> 克隆后按上面的命令在本地重新生成即可复现本节数字。

三个引擎在同一份 215 条数据上（门限 0.5，pi400 4 线程）：

| 引擎 | acc@0.5 | 最佳阈值 | AUC | 延迟 | FP / FN |
|---|---|---|---|---|---|
| **decision** | **86.05%** | 0.41 → 88.4% | **0.949** | 5.3 s | 7 / 23 |
| nli | 65.12% | 0.14 → 72.1% | 0.802 | 0.36 s | 1 / 74 |
| rerank（bge-v2-m3） | 54.88% | 0.16 → 60.0% | 0.607 | 1.2 s | 12 / 85 |
| rerank（bce-base_v1） | 50.23% | 0.53 → 53.5% | 0.508 | 0.33 s | 66 / 41 |

- **decision**：唯一能在 0.5 门限下直接用的引擎；AUC 0.949，误报只有 7 条。
  典型错误是「高一化学试卷」被判 0.16、九年级语文被判 0.69 这类长标题噪声。
- **nli**：几乎从不误报（1 FP），但极保守（74 FN），0.5 门限下只有 65%；
  要把阈值压到 0.14 才能到 72%。适合做「一票否决」的粗筛，不适合做最终判定。
- **rerank**：条件是一个词（「高中」）时，与长标题的相关性分数被压到 0.03 左右，
  AUC 0.607；裸条件 0.645、「{condition}试题」0.662，换措辞也救不回来 ——
  encoder reranker 不适合这种「短条件 vs 长标题」的蕴含判断。

**融合的可能性**：在学科目录名做条件时（更接近真实查询），NLI 与 rerank 在阈值 0.3 上
「都同意」的 124 条准确率 85.5%（覆盖率 58%）。也就是说，双 encoder 共识可以挡掉
一半以上的调用，剩余交给决策模型；但 85.5% 是否够用取决于业务容错。

**和现有代码规则的对比（重要）**：这套 ground truth 本身就是「标题里的年级词」，
所以规则基线在这个集合上是 100%——上面的 86% 衡量的是「模型能否复现规则」，不是
「模型比规则更准」。全量 5,423 份试卷里：

- 94.86% 的标题有唯一学段词 → 规则直接判，零成本且确定；
- 4.28% 有多个学段词（`高一年级` 同时命中「高一」和「一年级」、`中小学`、`Ji高一筹`）；
- 0.87% 完全没有年级词（`小学5年级`、`三年数学` 这类写法）。

也就是说：**规则能覆盖的 95%，模型不如规则；模型的价值只在剩下 5% 和「短条件语义匹配」。**
`tests/fixtures/stage_hard_probe.tsv` 抽了 6 条规则判不了的标题，决策模型 4/6 判对
（`高一年级`、`高二年级`、`小学5年级` 对；`Ji高一筹`、`三年数学` 错）。
这个样本很小，只能说明方向，不能当基准。

正确的架构是「规则优先 + 模型兜底」：规则给出的高置信结果直接采用，`rule-none` /
`rule-mismatch` / 短条件语义匹配（中考 vs 高中考试）才调用决策模型。

### 6. 有没有更好的模型？选型结论

| 候选 | 能否直接用 | 结论 |
|---|---|---|
| Erlangshen-Roberta-330M-NLI | 需自己转 GGUF | 同架构，仍有 segment embedding 丢失 + 110M 的语义短板，提升有限，放弃 |
| bge-reranker-v2-m3（已实测） | 现成 GGUF（Apache-2.0） | 快（1.1 s）但短条件不可用，只适合「描述型 query 检索」 |
| bce-reranker-base_v1（已实测） | 现成 GGUF（Apache-2.0，279M） | 更快（0.33 s），但真实试卷集 AUC 0.508 ≈ 随机，不能用 |
| jina-reranker-v2-base（未测） | 无官方 GGUF，CC-BY-NC-4.0 | 中文 AirBench 61.33 vs bge 61.28，基本打平；**不能商用** |
| Qwen3-Reranker-0.6B | 有官方 GGUF（`ggml-org/...`） | 解码式 yes/no，能力接近决策模型，但速度同为 3–4 s，没有收益 |
| xlm-roberta-large-xnli | 需自己转 GGUF | 真正的 3 分类 NLI，`type_vocab_size=1` 无 segment 问题；560M，中文 XNLI 约 80%，仍低于决策模型，且要重建转换环境 |
| mDeBERTa-v3 / XNLI 系列 | 否 | llama.cpp 不支持 DeBERTa 架构 |
| Qwen2.5/Qwen3 1.5B–4B Instruct | 现成 GGUF | 可能更准，但 pi400 上 10–40 s/题，内存也吃紧，不划算 |

**结论：在 pi400 + llama.cpp 的约束下，做语义判定时 0.8B 决策模型已经是当前最优解；
但它不是规则的替代品，而是规则判不了时的兜底。**
继续投入应该放在「别名归一化」（如 初中学业水平考试 = 中考）和「长标题清洗」上，
而不是换 encoder。

### 7. 教材版本选择（另一类任务，抽查）

`bedu-jiaofu-douzi` 的 AI 兜底是 `/llm/choose`：给文件名 + 规则说明 + 选项，让模型选
`version`（教材版本，2–18 个选项）和 `bookEdition`（分册，最多 41 个选项）。那个 audit 的
`ai` 指标只统计「答案是否在选项里」，不统计对错，且最近一次报告 `--ai` 没开（asked=0），
所以没有可直接引用的准确率。

用同一份 report 里「规则已判出且值在选项内」的 25 条做伪标签，测我们的 choice：

```bash
python3 scripts/build_edition_probe.py ../bedu-jiaofu-douzi/.audit/report.json --limit 25
python3 scripts/probe_edition.py tests/fixtures/edition_probe.jsonl
# exact 5/25 = 20.0%   loose（忽略 "(新)" 后缀）22/25 = 88.0%
```

> `edition_probe.jsonl` 同样含真实试卷标题，不随仓库提供。

- 严格 20% 是因为选项里同时有「人教版」和「人教版(新)」，伪标签只认其中一个；
  去掉 `(新)` 后 88%。
- 3 个真错都是「标题里没有版本词、需要靠地区推断」：沪科版→人教版、沪粤版→人教版、
  浙教版→鲁教五四版。**模型不确定时会默认「人教版」**，这正好是 AI 兜底最容易翻车的场景。
- 分册选项最多 41 个，而 `decide` 上限是 26 个字母，需要分两级选（先必修/选择性必修，再册次）。

结论：版本选择这类「多选一 + 区域知识」不是 0.8B 的强项；更划算的做法是先用
「学科 + 地区」把选项过滤到 3–5 个，再让模型选，而不是把 18 个选项直接丢给它。

## 已知局限

- `gate --engine decision` 每个候选一次前向，5 秒/候选；没有做 batch/并行。
- 决策模型的输出是**校准概率**，不是事实；`confidence` 只表示分布尖锐程度。
- **候选项措辞会改变概率**：同一题用 `false=否 / true=是` 得到 P(满足)=0.1943，
  用 `false=不满足 / true=满足` 得到 0.1508。批量判定时要固定措辞，
  `gate` / `eval-gate` 内部统一用「不满足 / 满足」。
- NLI 路线受 llama.cpp BERT 图限制（无 segment embedding），且 110M 语义能力有限。
- 重排引擎的分数依赖 query 措辞和输入长度，跨数据集不可直接复用同一个阈值。
- 用例集：34 条 NLI + 15 条人工门禁 + 215 条真实试卷；真实试卷的 ground truth
  来自标题里的年级词，标题本身有噪声（学校名、学制、"中小学"等）。
- `scripts/_*.py` 是转换/对照实验用的脚本，需要先跑 `bootstrap_venv.sh` 和
  `fetch_models.sh nli-src`；不属于运行时依赖。
- `scripts/analyze_eval.py` 可以对比多份 `eval-gate --json` 结果（准确率/AUC/阈值扫描）。
