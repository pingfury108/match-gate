//! 交叉编码器重排（reranker）：判断「候选项」与「条件」的相关性。
//!
//! 模型：`BAAI/bge-reranker-v2-m3`（XLM-RoBERTa，568M，GGUF Q8_0）。
//!
//! 与 NLI 路线的关键区别：XLM-R 的 `type_vocab_size = 1`，训练和推理都只用
//! token type 0，因此 llama.cpp「把 token type 硬编码为 0」不会造成精度损失。
//! 这也是选它替代 Erlangshen 的原因。
//!
//! 输入按 XLM-R 句对格式拼成 `<s> query </s></s> passage </s>`，
//! 输出一个 logit，取 sigmoid 作为相关性分数。

use anyhow::{bail, Context as _, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

use crate::engine::{encoder_params, LoadedModel};

/// 句对最大 token 数。
pub const MAX_PAIR_TOKENS: u32 = 512;

/// 已加载的重排模型。
pub struct RerankModel {
    inner: LoadedModel,
}

impl RerankModel {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let inner = LoadedModel::load(path)?;
        let n_cls_out = inner.model().n_cls_out();
        anyhow::ensure!(
            n_cls_out == 1,
            "重排模型应只有 1 个分类输出，实际 n_cls_out={n_cls_out}；\
             请确认加载的是 bge-reranker-v2-m3 的 GGUF"
        );
        Ok(Self { inner })
    }

    pub fn path(&self) -> &std::path::Path {
        self.inner.path()
    }

    pub fn session(&self, n_threads: i32) -> Result<RerankSession<'_>> {
        let model = self.inner.model();
        let ctx = model
            .new_context(
                self.inner.backend(),
                encoder_params(MAX_PAIR_TOKENS, n_threads),
            )
            .context("创建重排上下文失败")?;

        let cls = special_token(model, &["[CLS]", "<s>"])?;
        let sep = special_token(model, &["[SEP]", "</s>"])?;

        Ok(RerankSession {
            ctx,
            model,
            cls,
            sep,
        })
    }
}

/// 一次推理会话，可复用上下文连续打分。
pub struct RerankSession<'a> {
    ctx: LlamaContext<'a>,
    model: &'a LlamaModel,
    cls: LlamaToken,
    sep: LlamaToken,
}

impl RerankSession<'_> {
    /// 给 `(query, passage)` 打相关性分，返回 0..1。
    ///
    /// 做门禁时 query 传条件、passage 传候选项。
    pub fn score(&mut self, query: &str, passage: &str) -> Result<f32> {
        let vocab = self.model.vocab();
        let mut q = vocab.tokenize(query.as_bytes(), false, false);
        let mut p = vocab.tokenize(passage.as_bytes(), false, false);

        // [CLS] q [SEP] [SEP] p [SEP]
        let budget = MAX_PAIR_TOKENS as usize - 4;
        trim_pair(&mut q, &mut p, budget);

        let mut tokens = Vec::with_capacity(q.len() + p.len() + 4);
        tokens.push(self.cls);
        tokens.extend_from_slice(&q);
        tokens.push(self.sep);
        tokens.push(self.sep);
        tokens.extend_from_slice(&p);
        tokens.push(self.sep);

        let n_tokens = tokens.len();
        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, token) in tokens.iter().enumerate() {
            // 池化要求所有 token 都输出。
            batch
                .add(*token, i as i32, &[0], true)
                .context("构造重排 batch 失败")?;
        }

        self.ctx.encode(&mut batch).context("重排前向失败")?;
        let logits = self.ctx.embeddings_seq_ith(0).context("读取重排分数失败")?;
        if logits.len() != 1 {
            bail!("重排输出维度应为 1，实际 {}", logits.len());
        }

        // 训练时用 sigmoid 归一化；这里保持同样的口径。
        Ok(1.0 / (1.0 + (-logits[0]).exp()))
    }
}

fn special_token(model: &LlamaModel, candidates: &[&str]) -> Result<LlamaToken> {
    let vocab = model.vocab();
    for text in candidates {
        let ids = vocab.tokenize(text.as_bytes(), false, true);
        if let [id] = ids.as_slice() {
            if vocab.token_to_piece(*id, true, None) == text.as_bytes() {
                return Ok(*id);
            }
        }
        for (id, piece) in model.tokens(true) {
            if piece == text.as_bytes() {
                return Ok(id);
            }
        }
    }
    bail!("词表里找不到特殊符号 {candidates:?}")
}

/// 按长度比例裁剪，优先保留两边开头。
fn trim_pair(query: &mut Vec<LlamaToken>, passage: &mut Vec<LlamaToken>, budget: usize) {
    if query.len() + passage.len() <= budget {
        return;
    }
    let total = (query.len() + passage.len()).max(1);
    let q_keep = ((budget * query.len()) / total).min(query.len());
    query.truncate(q_keep);
    let p_keep = budget.saturating_sub(query.len());
    passage.truncate(p_keep);
}

#[cfg(test)]
mod tests {
    #[test]
    fn sigmoid_is_monotonic() {
        // 只验证数值口径，不依赖模型。
        let s = |x: f32| 1.0 / (1.0 + (-x).exp());
        assert!(s(0.0) > 0.49 && s(0.0) < 0.51);
        assert!(s(-5.0) < 0.01);
        assert!(s(5.0) > 0.99);
    }
}
