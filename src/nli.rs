//! 中文 NLI（语义蕴含）推理。
//!
//! 模型：`IDEA-CCNL/Erlangshen-Roberta-110M-NLI`（BertForSequenceClassification）。
//! 输出三个 logits，顺序由 GGUF 的 `classifier.output_labels` 决定，默认
//! `CONTRADICTION / NEUTRAL / ENTAILMENT`。
//!
//! 输入按 BERT 句对格式拼成 `[CLS] premise [SEP] hypothesis [SEP]`。
//!
//! 已知偏差：llama.cpp 的 BERT 图把 token type 硬编码为 0（"Sentence A"），
//! 而 HF 训练时第二句用 type id 1。这个差异只能用真实数据验证，见 README。

use anyhow::{bail, Context as _, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use crate::engine::{encoder_params, LoadedModel};

/// BERT 的位置编码上限，也是句对的最大 token 数。
pub const MAX_PAIR_TOKENS: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NliLabel {
    Contradiction,
    Neutral,
    Entailment,
}

impl NliLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Contradiction => "contradiction",
            Self::Neutral => "neutral",
            Self::Entailment => "entailment",
        }
    }

    fn from_hf_name(name: &str) -> Option<Self> {
        match name.trim().trim_matches('"').to_ascii_uppercase().as_str() {
            "CONTRADICTION" | "CONTRADICT" => Some(Self::Contradiction),
            "NEUTRAL" => Some(Self::Neutral),
            "ENTAILMENT" | "ENTAIL" => Some(Self::Entailment),
            _ => None,
        }
    }
}

impl std::fmt::Display for NliLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 三个类别的概率（和为 1）。
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct NliScores {
    pub contradiction: f32,
    pub neutral: f32,
    pub entailment: f32,
}

impl NliScores {
    pub fn prob(&self, label: NliLabel) -> f32 {
        match label {
            NliLabel::Contradiction => self.contradiction,
            NliLabel::Neutral => self.neutral,
            NliLabel::Entailment => self.entailment,
        }
    }

    pub fn label(&self) -> NliLabel {
        let mut best = NliLabel::Contradiction;
        let mut best_p = self.contradiction;
        if self.neutral > best_p {
            best = NliLabel::Neutral;
            best_p = self.neutral;
        }
        if self.entailment > best_p {
            best = NliLabel::Entailment;
        }
        best
    }

    pub fn confidence(&self) -> f32 {
        self.prob(self.label())
    }
}

/// 已加载的 NLI 模型。
pub struct NliModel {
    inner: LoadedModel,
    /// `label_order[i]` 是第 i 个输出 logit 对应的类别。
    label_order: [NliLabel; 3],
}

impl NliModel {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = LoadedModel::load(path)?;
        let model = inner.model();

        let n_cls_out = model.n_cls_out();
        anyhow::ensure!(
            n_cls_out == 3,
            "NLI 模型应有 3 个分类输出（CONTRADICTION/NEUTRAL/ENTAILMENT），实际 n_cls_out={n_cls_out}；\
             请确认加载的是 Erlangshen-Roberta-110M-NLI 的 GGUF"
        );

        let label_order = match read_classifier_labels(model) {
            Some(labels) if labels.len() == 3 => {
                let mut order = [NliLabel::Contradiction; 3];
                for (i, name) in labels.iter().enumerate() {
                    order[i] = NliLabel::from_hf_name(name).with_context(|| {
                        format!(
                            "无法识别的 NLI 标签 {name:?}（期望 CONTRADICTION/NEUTRAL/ENTAILMENT）"
                        )
                    })?;
                }
                order
            }
            _ => [
                NliLabel::Contradiction,
                NliLabel::Neutral,
                NliLabel::Entailment,
            ],
        };

        Ok(Self { inner, label_order })
    }

    pub fn path(&self) -> &std::path::Path {
        self.inner.path()
    }

    /// 模型里写死的标签顺序（输出 logit i 的类别）。
    pub fn label_order(&self) -> [NliLabel; 3] {
        self.label_order
    }

    /// 用模型自带词表分词，返回 `(token, 文本片段)`，用于和 HF tokenizer 对比。
    pub fn tokenize(&self, text: &str) -> Vec<(LlamaToken, String)> {
        let vocab = self.inner.model().vocab();
        vocab
            .tokenize(text.as_bytes(), false, false)
            .into_iter()
            .map(|token| {
                let piece = vocab.token_to_piece(token, true, None);
                (token, String::from_utf8_lossy(&piece).into_owned())
            })
            .collect()
    }

    pub fn session(&self, n_threads: i32) -> Result<NliSession<'_>> {
        let model = self.inner.model();
        let ctx = model
            .new_context(
                self.inner.backend(),
                encoder_params(MAX_PAIR_TOKENS, n_threads),
            )
            .context("创建 NLI 上下文失败")?;

        let cls = special_token(model, "[CLS]")?;
        let sep = special_token(model, "[SEP]")?;

        Ok(NliSession {
            ctx,
            model,
            label_order: self.label_order,
            cls,
            sep,
        })
    }
}

/// 一次推理会话，可复用上下文连续判断多组句对。
pub struct NliSession<'a> {
    ctx: LlamaContext<'a>,
    model: &'a LlamaModel,
    label_order: [NliLabel; 3],
    cls: LlamaToken,
    sep: LlamaToken,
}

impl NliSession<'_> {
    /// 判断 `premise` 是否蕴含 `hypothesis`。
    ///
    /// 做「候选项是否满足条件」时，premise 传候选项（证据），hypothesis 传条件（结论）。
    pub fn predict(&mut self, premise: &str, hypothesis: &str) -> Result<NliScores> {
        let vocab = self.model.vocab();
        let mut p_tokens = vocab.tokenize(premise.as_bytes(), false, false);
        let mut h_tokens = vocab.tokenize(hypothesis.as_bytes(), false, false);

        let budget = MAX_PAIR_TOKENS as usize - 3; // [CLS] p [SEP] h [SEP]
        trim_pair(&mut p_tokens, &mut h_tokens, budget);

        let mut tokens = Vec::with_capacity(p_tokens.len() + h_tokens.len() + 3);
        tokens.push(self.cls);
        tokens.extend_from_slice(&p_tokens);
        tokens.push(self.sep);
        tokens.extend_from_slice(&h_tokens);
        tokens.push(self.sep);

        let n_tokens = tokens.len();
        if n_tokens < 3 {
            bail!("句对分词结果为空");
        }

        let debug = std::env::var_os("MATCH_GATE_DEBUG").is_some();
        if debug {
            let pieces: Vec<String> = tokens
                .iter()
                .map(|token| {
                    let piece = vocab.token_to_piece(*token, false, None);
                    format!("{}({})", token.0, String::from_utf8_lossy(&piece))
                })
                .collect();
            eprintln!("tokens: {}", pieces.join(" "));
        }

        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, token) in tokens.iter().enumerate() {
            // 池化要求「所有 token 都输出」，否则 llama.cpp 直接报错。
            batch
                .add(*token, i as i32, &[0], true)
                .context("构造 NLI batch 失败")?;
        }

        self.ctx.encode(&mut batch).context("NLI 前向失败")?;
        let logits = self
            .ctx
            .embeddings_seq_ith(0)
            .context("读取 NLI 分类头输出失败")?;
        if logits.len() != 3 {
            bail!("NLI 输出维度应为 3，实际 {}", logits.len());
        }

        if debug {
            eprintln!("logits: {:?}", logits);
        }

        let probs = softmax(&logits[..3]);
        let mut scores = NliScores::default();
        for (i, label) in self.label_order.iter().enumerate() {
            match label {
                NliLabel::Contradiction => scores.contradiction = probs[i],
                NliLabel::Neutral => scores.neutral = probs[i],
                NliLabel::Entailment => scores.entailment = probs[i],
            }
        }
        Ok(scores)
    }
}

fn special_token(model: &LlamaModel, text: &str) -> Result<LlamaToken> {
    let vocab = model.vocab();
    let ids = vocab.tokenize(text.as_bytes(), false, true);
    if let [id] = ids.as_slice() {
        if vocab.token_to_piece(*id, true, None) == text.as_bytes() {
            return Ok(*id);
        }
    }
    // 兜底：某些 BERT 词表没有把 [CLS]/[SEP] 标成 control token，直接扫一遍。
    for (id, piece) in model.tokens(true) {
        if piece == text.as_bytes() {
            return Ok(id);
        }
    }
    bail!("词表里找不到特殊符号 {text:?}")
}

/// 把句对裁到 `budget` 个 token 以内，优先保留两边的前半部分。
fn trim_pair(premise: &mut Vec<LlamaToken>, hypothesis: &mut Vec<LlamaToken>, budget: usize) {
    if premise.len() + hypothesis.len() <= budget {
        return;
    }
    // 按长度比例分配预算，两边都保留开头的内容。
    let total = (premise.len() + hypothesis.len()).max(1);
    let p_keep = ((budget * premise.len()) / total).min(premise.len());
    premise.truncate(p_keep);
    let h_keep = budget.saturating_sub(premise.len());
    hypothesis.truncate(h_keep);
}

/// 数值稳定的 softmax。
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

/// 从 GGUF 元数据读分类标签，读不到返回 None。
fn read_classifier_labels(model: &LlamaModel) -> Option<Vec<String>> {
    let n = model.meta_count();
    for i in 0..n {
        let Ok(key) = model.meta_key_by_index(i) else {
            continue;
        };
        // llama.cpp 的 key 带架构前缀（bert.classifier.output_labels）。
        if key != "classifier.output_labels" && !key.ends_with(".classifier.output_labels") {
            continue;
        }
        let Ok(value) = model.meta_val_str_by_index(i) else {
            continue;
        };
        let labels = parse_label_list(&value);
        if !labels.is_empty() {
            return Some(labels);
        }
    }
    None
}

/// 解析 llama.cpp 对字符串数组的渲染格式，例如
/// `["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]`。
fn parse_label_list(value: &str) -> Vec<String> {
    value
        .trim()
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_classifier_labels() {
        assert_eq!(
            parse_label_list(r#"["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]"#),
            vec!["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
        );
    }

    #[test]
    fn label_names_are_case_insensitive() {
        assert_eq!(
            NliLabel::from_hf_name("entailment"),
            Some(NliLabel::Entailment)
        );
        assert_eq!(NliLabel::from_hf_name(" LABEL_0 "), None);
    }
}
