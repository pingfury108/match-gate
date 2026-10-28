//! 中文语义蕴含门禁 + 多候选项条件匹配。
//!
//! 两个能力分别由两个 GGUF 模型提供：
//!
//! * [`nli`]：中文 NLI（Erlangshen-Roberta-110M-NLI），用来判断
//!   「候选项」是否真的满足「条件」，解决「中考」被「高中考试」这类字符串匹配误判。
//! * [`decision`]：中文决策模型（chinese-decision-0.8b），一次前向给出多个候选项的
//!   概率，用于 noul / choice / score 三种题型。
//! * [`rerank`]：交叉编码器重排（bge-reranker-v2-m3，XLM-R，无句段 embedding 依赖），
//!   给「条件 / 候选项」打相关性分。
//!
//! 三者都通过 llama.cpp 在 CPU 上运行，不依赖 PyTorch。

pub mod decision;
pub mod engine;
pub mod nli;
pub mod prompt;
pub mod rerank;

pub use decision::{Answer, DecisionModel, DecisionOutcome, DecisionSession, QuestionType};
pub use engine::LoadedModel;
pub use nli::{NliLabel, NliModel, NliScores, NliSession};
pub use rerank::{RerankModel, RerankSession};
