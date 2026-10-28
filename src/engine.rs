//! 模型加载与上下文参数的公共部分。

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

/// llama.cpp backend + 一个已加载的模型。
///
/// `LlamaContext` 借用 `LlamaModel`，所以上下文不放在这里，而是由
/// [`LoadedModel::model`] 的借用者在需要时创建（见 `NliSession` / `DecisionSession`）。
pub struct LoadedModel {
    backend: LlamaBackend,
    model: LlamaModel,
    path: PathBuf,
}

impl LoadedModel {
    /// 从 GGUF 文件加载模型，全部放在 CPU 上（`n_gpu_layers = 0`）。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        anyhow::ensure!(
            path.is_file(),
            "模型文件不存在：{}（用 --nli-model / --decision-model 或 MATCH_GATE_* 环境变量指定）",
            path.display()
        );

        let backend = LlamaBackend::init().context("初始化 llama.cpp backend 失败")?;
        let params = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(&backend, path, &params)
            .with_context(|| format!("加载模型失败：{}", path.display()))?;

        Ok(Self {
            backend,
            model,
            path: path.to_path_buf(),
        })
    }

    pub fn backend(&self) -> &LlamaBackend {
        &self.backend
    }

    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// 编码类模型（BERT）用的上下文参数。
///
/// * `with_embeddings(true)`：必须开，否则 llama.cpp 的 `build_pooling` 直接返回，
///   拿不到分类头输出。
/// * `LlamaPoolingType::Rank`：分类头（`cls` -> tanh -> `cls_out`）只在 RANK 池化下构建，
///   输出宽度是 `n_cls_out`。
/// * `n_ubatch >= n_tokens`：非因果注意力要求整个序列在一次 ubatch 里。
pub fn encoder_params(n_ctx: u32, n_threads: i32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx)
        .with_n_ubatch(n_ctx)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
        .with_embeddings(true)
        .with_pooling_type(LlamaPoolingType::Rank)
}

/// 生成类模型（决策模型）用的上下文参数。
///
/// `n_batch = n_ctx` 让整段 prompt 一次 decode 完；`n_ubatch` 保持 512，
/// 避免 4k 上下文的计算缓冲区把 pi400 的内存吃光（llama_decode 会自动分块）。
pub fn decoder_params(n_ctx: u32, n_threads: i32) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx)
        .with_n_ubatch(512)
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
}

/// 默认线程数：机器核数，上限 8（pi400 是 4 核）。
pub fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8) as i32
}
