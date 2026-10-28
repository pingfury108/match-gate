use std::io::BufRead;
use std::path::PathBuf;

use anyhow::{bail, Context as _, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;

use match_gate::decision::{DecisionModel, DecisionOption, QuestionType, DEFAULT_N_CTX};
use match_gate::engine::default_threads;
use match_gate::nli::{NliLabel, NliModel, NliScores};
use match_gate::rerank::RerankModel;

#[derive(Parser)]
#[command(
    name = "match-gate",
    about = "中文语义蕴含门禁 + 多候选项条件匹配（llama.cpp / GGUF，纯 CPU）",
    version
)]
struct Cli {
    /// NLI 模型 GGUF（默认 $HOME/models/match-gate/erlangshen-nli-110m-q8_0.gguf）
    #[arg(long, global = true, env = "MATCH_GATE_NLI_MODEL")]
    nli_model: Option<PathBuf>,

    /// 决策模型 GGUF（默认 $HOME/models/match-gate/chinese-decision-0.8b-Q8_0.gguf）
    #[arg(long, global = true, env = "MATCH_GATE_DECISION_MODEL")]
    decision_model: Option<PathBuf>,

    /// 重排模型 GGUF（默认 $HOME/models/match-gate/bge-reranker-v2-m3-Q8_0.gguf）
    #[arg(long, global = true, env = "MATCH_GATE_RERANK_MODEL")]
    rerank_model: Option<PathBuf>,

    /// 重排 query 模板，`{condition}` 会被替换（gate / eval-gate --engine rerank）
    #[arg(long, global = true, default_value = "{condition}试卷")]
    rerank_query: String,

    /// 重排 passage 模板，`{candidate}` 会被替换
    #[arg(long, global = true, default_value = "{candidate}")]
    rerank_passage: String,

    /// 线程数（默认取机器核数，上限 8）
    #[arg(long, global = true, env = "MATCH_GATE_THREADS")]
    threads: Option<i32>,

    /// 输出 JSON
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 判断 premise 是否蕴含 hypothesis
    Nli {
        #[arg(long)]
        premise: String,
        #[arg(long)]
        hypothesis: String,
    },

    /// 打印文本在模型词表下的分词结果（排查分词差异）
    Tokens {
        #[arg(long)]
        text: String,
    },

    /// 交叉编码器打分：query（条件）与 passage（候选项）的相关性
    Rerank {
        #[arg(long)]
        query: String,
        #[arg(long)]
        passage: String,
    },

    /// 门禁：逐个判断候选项是否满足条件（候选项 => 条件）
    Gate {
        /// 条件/要求（被蕴含的结论）
        #[arg(long)]
        condition: String,

        /// 候选项，可重复
        #[arg(long = "candidate")]
        candidates: Vec<String>,

        /// 从文件读取候选项，一行一个（空行与 # 开头忽略）
        #[arg(long)]
        candidate_file: Option<PathBuf>,

        /// 判定引擎：decision（0.8B 决策模型，准但慢）/ nli（110M，快但精度有限）
        #[arg(long, value_enum, default_value = "decision")]
        engine: EngineArg,

        /// 蕴含概率阈值
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,

        /// 严格模式：还要求「条件 => 候选项」不构成矛盾
        #[arg(long)]
        strict: bool,

        /// 跳过反向判断（更快）
        #[arg(long)]
        no_reverse: bool,
    },

    /// 多候选项条件匹配：一次前向给出每个候选项的概率
    Decide {
        /// 待判断的上下文
        #[arg(long)]
        state: String,

        /// 问题/判断指令
        #[arg(long)]
        instructions: String,

        /// 题型：noul（是/否）、choice（单选）、score（有序等级）
        #[arg(long = "type", value_enum, default_value = "choice")]
        question_type: QTypeArg,

        /// 候选项：`key=描述` 或只有描述（score 题型会自动编号 0,1,2...）
        #[arg(long = "option")]
        options: Vec<String>,

        /// 覆盖校准温度（默认 1.10813307762146）
        #[arg(long)]
        temperature: Option<f32>,

        /// 上下文 token 上限
        #[arg(long, default_value_t = DEFAULT_N_CTX)]
        n_ctx: u32,
    },

    /// 跑门禁用例（候选项是否满足条件），可切换 decision / rerank / nli 引擎
    EvalGate {
        /// TSV：id <TAB> 候选项 <TAB> 条件 <TAB> 期望（1=满足，0=不满足）
        #[arg(long, default_value = "tests/fixtures/gate_cases.tsv")]
        cases: PathBuf,

        /// 判定引擎：decision / rerank / nli
        #[arg(long, value_enum, default_value = "decision")]
        engine: EngineArg,

        /// P(满足) 判定门限
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,

        /// 准确率低于该值则以非零退出
        #[arg(long, default_value_t = 0.8)]
        min_accuracy: f32,
    },

    /// 跑中文 NLI 测试集，报告逐条结果、准确率与延迟
    Eval {
        /// TSV：id <TAB> premise <TAB> hypothesis <TAB> 期望标签（多个用 | 分隔）
        #[arg(long, default_value = "tests/fixtures/nli_cases.tsv")]
        cases: PathBuf,

        /// 准确率低于该值则以非零退出
        #[arg(long, default_value_t = 0.8)]
        min_accuracy: f32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let threads = cli.threads.unwrap_or_else(default_threads);

    match &cli.command {
        Command::Nli {
            premise,
            hypothesis,
        } => {
            let model = NliModel::load(nli_model_path(cli.nli_model))?;
            let mut session = model.session(threads)?;
            let scores = session.predict(premise, hypothesis)?;
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": scores.label().as_str(),
                        "scores": scores,
                    }))?
                );
            } else {
                println!(
                    "{}  (entailment {:.4} / neutral {:.4} / contradiction {:.4})",
                    scores.label(),
                    scores.entailment,
                    scores.neutral,
                    scores.contradiction
                );
            }
            Ok(())
        }

        Command::Tokens { text } => {
            let model = NliModel::load(nli_model_path(cli.nli_model))?;
            for (token, piece) in model.tokenize(text) {
                println!("{:>6}  {}", token.0, piece);
            }
            Ok(())
        }

        Command::Rerank { query, passage } => {
            let model = RerankModel::load(rerank_model_path(cli.rerank_model))?;
            let mut session = model.session(threads)?;
            let start = std::time::Instant::now();
            let score = session.score(query, passage)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "query": query,
                        "passage": passage,
                        "score": score,
                        "latency_ms": elapsed_ms,
                    }))?
                );
            } else {
                println!("score {score:.4}  ({elapsed_ms:.0}ms)");
            }
            Ok(())
        }

        Command::Gate {
            condition,
            candidates,
            candidate_file,
            engine,
            threshold,
            strict,
            no_reverse,
        } => {
            let mut all: Vec<String> = candidates.clone();
            if let Some(path) = candidate_file {
                all.extend(read_lines(path)?);
            }
            if all.is_empty() {
                bail!("至少给一个 --candidate 或 --candidate-file");
            }

            match engine {
                EngineArg::Decision => {
                    run_gate_decision(
                        &all,
                        condition,
                        *threshold,
                        threads,
                        cli.decision_model.clone(),
                        cli.json,
                    )?;
                }
                EngineArg::Nli => {
                    run_gate_nli(
                        &all,
                        condition,
                        *threshold,
                        *strict,
                        *no_reverse,
                        threads,
                        cli.nli_model.clone(),
                        cli.json,
                    )?;
                }
                EngineArg::Rerank => {
                    run_gate_rerank(
                        &all,
                        condition,
                        *threshold,
                        threads,
                        cli.rerank_model.clone(),
                        &cli.rerank_query,
                        &cli.rerank_passage,
                        cli.json,
                    )?;
                }
            }
            Ok(())
        }

        Command::Decide {
            state,
            instructions,
            question_type,
            options,
            temperature,
            n_ctx,
        } => {
            let qtype: QuestionType = (*question_type).into();
            let options = build_options(qtype, options)?;
            let mut model = DecisionModel::load(decision_model_path(cli.decision_model))?;
            if let Some(t) = temperature {
                model = model.with_temperature(*t)?;
            }
            let mut session = model.session(*n_ctx, threads)?;
            let outcome = session.decide(state, instructions, &options, qtype)?;

            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "type": qtype.as_str(),
                        "temperature": model.temperature(),
                        "result": outcome,
                    }))?
                );
            } else {
                let probs: Vec<String> = outcome
                    .probabilities
                    .iter()
                    .map(|p| format!("{} {:.4}", p.key, p.probability))
                    .collect();
                println!("{{{}}}", probs.join(", "));
                println!("confidence {:.4}", outcome.confidence);
                match &outcome.answer {
                    match_gate::Answer::Noul(v) => println!("noul {v:.4}"),
                    match_gate::Answer::Choice(v) => println!("choice {v}"),
                    match_gate::Answer::Score(v) => println!("score {v:.4}"),
                }
            }
            Ok(())
        }

        Command::EvalGate {
            cases,
            engine,
            threshold,
            min_accuracy,
        } => {
            let rows = read_gate_cases(cases)?;
            if rows.is_empty() {
                bail!("测试集为空：{}", cases.display());
            }

            let (model_label, correct, total_latency_ms, results) = match engine {
                EngineArg::Decision => {
                    let model =
                        DecisionModel::load(decision_model_path(cli.decision_model.clone()))?;
                    let mut session = model.session(DEFAULT_N_CTX, threads)?;
                    let options = [
                        DecisionOption::new("false", "不满足"),
                        DecisionOption::new("true", "满足"),
                    ];
                    let (correct, ms, results) =
                        eval_gate_loop(&rows, *threshold, cli.json, |case| {
                            let state = format!("候选人的情况：{}", case.candidate);
                            let instructions =
                                format!("该候选人是否满足条件「{}」？", case.condition);
                            let outcome = session.decide(
                                &state,
                                &instructions,
                                &options,
                                QuestionType::Noul,
                            )?;
                            Ok(outcome.probability_of("true").unwrap_or(0.0))
                        })?;
                    (
                        format!("decision (T={:.4})", model.temperature()),
                        correct,
                        ms,
                        results,
                    )
                }
                EngineArg::Rerank => {
                    let model = RerankModel::load(rerank_model_path(cli.rerank_model.clone()))?;
                    let mut session = model.session(threads)?;
                    let (correct, ms, results) =
                        eval_gate_loop(&rows, *threshold, cli.json, |case| {
                            session.score(
                                &fill_template(&cli.rerank_query, &case.condition, &case.candidate),
                                &fill_template(
                                    &cli.rerank_passage,
                                    &case.condition,
                                    &case.candidate,
                                ),
                            )
                        })?;
                    (
                        "rerank (bge-reranker-v2-m3)".to_string(),
                        correct,
                        ms,
                        results,
                    )
                }
                EngineArg::Nli => {
                    let model = NliModel::load(nli_model_path(cli.nli_model.clone()))?;
                    let mut session = model.session(threads)?;
                    let (correct, ms, results) =
                        eval_gate_loop(&rows, *threshold, cli.json, |case| {
                            let scores = session.predict(&case.candidate, &case.condition)?;
                            Ok(scores.entailment)
                        })?;
                    ("nli (Erlangshen-110M)".to_string(), correct, ms, results)
                }
            };

            let total = rows.len();
            let accuracy = correct as f64 / total as f64;
            let mean_ms = total_latency_ms / total as f64;

            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "model": model_label,
                        "threshold": threshold,
                        "total": total,
                        "correct": correct,
                        "accuracy": accuracy,
                        "mean_latency_ms": mean_ms,
                        "results": results,
                    }))?
                );
            } else {
                println!(
                    "\naccuracy {correct}/{total} = {:.2}%   mean {:.0}ms/case",
                    accuracy * 100.0,
                    mean_ms
                );
            }

            anyhow::ensure!(
                accuracy >= f64::from(*min_accuracy),
                "准确率 {:.2}% 低于阈值 {:.2}%",
                accuracy * 100.0,
                min_accuracy * 100.0
            );
            Ok(())
        }

        Command::Eval {
            cases,
            min_accuracy,
        } => {
            let rows = read_cases(cases)?;
            if rows.is_empty() {
                bail!("测试集为空：{}", cases.display());
            }

            let model = NliModel::load(nli_model_path(cli.nli_model))?;
            let mut session = model.session(threads)?;

            let mut correct = 0usize;
            let mut total_latency_ms = 0f64;
            let mut results = Vec::with_capacity(rows.len());

            for case in &rows {
                let start = std::time::Instant::now();
                let scores = session.predict(&case.premise, &case.hypothesis)?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                total_latency_ms += elapsed_ms;

                let label = scores.label();
                let ok = case.expected.iter().any(|e| e.as_str() == label.as_str());
                if ok {
                    correct += 1;
                }

                if !cli.json {
                    println!(
                        "{} {:<10} {:<20} -> {:<13} entail {:.4} neutral {:.4} contra {:.4}  {:.0}ms",
                        if ok { "ok  " } else { "FAIL" },
                        case.id,
                        truncate(&case.premise, 20),
                        label.as_str(),
                        scores.entailment,
                        scores.neutral,
                        scores.contradiction,
                        elapsed_ms
                    );
                }

                results.push(json!({
                    "id": case.id,
                    "premise": case.premise,
                    "hypothesis": case.hypothesis,
                    "expected": case.expected,
                    "label": label.as_str(),
                    "ok": ok,
                    "scores": scores,
                    "latency_ms": elapsed_ms,
                }));
            }

            let total = rows.len();
            let accuracy = correct as f64 / total as f64;
            let mean_ms = total_latency_ms / total as f64;

            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string(&json!({
                        "model": model.path(),
                        "total": total,
                        "correct": correct,
                        "accuracy": accuracy,
                        "mean_latency_ms": mean_ms,
                        "results": results,
                    }))?
                );
            } else {
                println!(
                    "\naccuracy {correct}/{total} = {:.2}%   mean {:.0}ms/case",
                    accuracy * 100.0,
                    mean_ms
                );
            }

            anyhow::ensure!(
                accuracy >= f64::from(*min_accuracy),
                "准确率 {:.2}% 低于阈值 {:.2}%",
                accuracy * 100.0,
                min_accuracy * 100.0
            );
            Ok(())
        }
    }
}

struct GateCase {
    id: String,
    candidate: String,
    condition: String,
    expect_true: bool,
}

fn read_gate_cases(path: &PathBuf) -> Result<Vec<GateCase>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("打开测试集失败：{}", path.display()))?;
    let mut cases = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split('\t').collect();
        if fields.len() != 4 {
            bail!(
                "{}:{} 应有 4 列（id/candidate/condition/expected），实际 {}",
                path.display(),
                lineno + 1,
                fields.len()
            );
        }
        let expect_true = match fields[3].trim() {
            "1" | "true" | "yes" => true,
            "0" | "false" | "no" => false,
            other => bail!(
                "{}:{} 期望值应为 0/1，实际 {other:?}",
                path.display(),
                lineno + 1
            ),
        };
        cases.push(GateCase {
            id: fields[0].trim().to_string(),
            candidate: fields[1].trim().to_string(),
            condition: fields[2].trim().to_string(),
            expect_true,
        });
    }
    Ok(cases)
}

struct EvalCase {
    id: String,
    premise: String,
    hypothesis: String,
    expected: Vec<String>,
}

fn read_cases(path: &PathBuf) -> Result<Vec<EvalCase>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("打开测试集失败：{}", path.display()))?;
    let mut cases = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split('\t').collect();
        if fields.len() != 4 {
            bail!(
                "{}:{} 应有 4 列（id/premise/hypothesis/expected），实际 {}",
                path.display(),
                lineno + 1,
                fields.len()
            );
        }
        cases.push(EvalCase {
            id: fields[0].trim().to_string(),
            premise: fields[1].trim().to_string(),
            hypothesis: fields[2].trim().to_string(),
            expected: fields[3]
                .split('|')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect(),
        });
    }
    Ok(cases)
}

/// 按字符数截断，避免中文在终端里把表格撑乱。
fn truncate(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn verdict(
    forward: NliScores,
    reverse: Option<NliScores>,
    threshold: f32,
    strict: bool,
) -> &'static str {
    if forward.label() == NliLabel::Contradiction && forward.contradiction >= threshold {
        return "no_match";
    }
    if forward.label() == NliLabel::Entailment && forward.entailment >= threshold {
        if strict {
            if let Some(reverse) = reverse {
                if reverse.label() == NliLabel::Contradiction {
                    return "no_match";
                }
            }
        }
        return "match";
    }
    "uncertain"
}

fn build_options(question_type: QuestionType, raw: &[String]) -> Result<Vec<DecisionOption>> {
    if raw.is_empty() {
        if question_type == QuestionType::Noul {
            return Ok(vec![
                DecisionOption::new("false", "否"),
                DecisionOption::new("true", "是"),
            ]);
        }
        bail!("至少给两个 --option");
    }

    let keyed = raw.iter().all(|o| o.contains('='));
    let options: Vec<DecisionOption> = raw
        .iter()
        .enumerate()
        .map(|(i, o)| match o.split_once('=') {
            Some((k, v)) if keyed => DecisionOption::new(k.trim(), v.trim()),
            _ => DecisionOption::new(i.to_string(), o.trim()),
        })
        .collect();

    if question_type == QuestionType::Noul && options.len() != 2 {
        bail!("noul 题型必须正好两个候选项");
    }
    Ok(options)
}

/// 门禁用例的评测循环：每个用例调用一次 `score`，返回 (正确数, 总延迟, 明细)。
fn eval_gate_loop(
    rows: &[GateCase],
    threshold: f32,
    json_output: bool,
    mut score: impl FnMut(&GateCase) -> Result<f32>,
) -> Result<(usize, f64, Vec<serde_json::Value>)> {
    let mut correct = 0usize;
    let mut total_latency_ms = 0f64;
    let mut results = Vec::with_capacity(rows.len());

    for case in rows {
        let start = std::time::Instant::now();
        let p_true = score(case)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_latency_ms += elapsed_ms;

        let verdict = p_true >= threshold;
        let ok = verdict == case.expect_true;
        if ok {
            correct += 1;
        }

        if !json_output {
            println!(
                "{} {:<10} p={:<7.4} expect={}  {} | {}  ({:.0}ms)",
                if ok { "ok  " } else { "FAIL" },
                case.id,
                p_true,
                u8::from(case.expect_true),
                truncate(&case.candidate, 18),
                truncate(&case.condition, 22),
                elapsed_ms
            );
        }

        results.push(json!({
            "id": case.id,
            "candidate": case.candidate,
            "condition": case.condition,
            "expect_true": case.expect_true,
            "p_true": p_true,
            "verdict": verdict,
            "ok": ok,
            "latency_ms": elapsed_ms,
        }));
    }

    Ok((correct, total_latency_ms, results))
}

/// 用交叉编码器重排模型逐个判断候选项是否满足条件。
fn run_gate_rerank(
    candidates: &[String],
    condition: &str,
    threshold: f32,
    threads: i32,
    rerank_model: Option<PathBuf>,
    query_tpl: &str,
    passage_tpl: &str,
    json_output: bool,
) -> Result<()> {
    let model = RerankModel::load(rerank_model_path(rerank_model))?;
    let mut session = model.session(threads)?;

    let mut rows = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let start = std::time::Instant::now();
        // 加自然语言框架后，reranker 才更像「蕴含判断」而不是「主题相关」：
        // 裸字符串会把「中考 / 高中考试」判成高相关（0.80），加框架后降到 0.01。
        let score = session.score(
            &fill_template(query_tpl, condition, candidate),
            &fill_template(passage_tpl, condition, candidate),
        )?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let matched = score >= threshold;

        if !json_output {
            println!(
                "{:<8} {:<28} score {:.4}  ({:.0}ms)",
                if matched { "MATCH" } else { "REJECT" },
                truncate(candidate, 28),
                score,
                elapsed_ms
            );
        }

        rows.push(json!({
            "candidate": candidate,
            "matched": matched,
            "score": score,
            "latency_ms": elapsed_ms,
        }));
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "engine": "rerank",
                "condition": condition,
                "threshold": threshold,
                "results": rows,
            }))?
        );
    }
    Ok(())
}

/// 替换重排模板里的 `{condition}` / `{candidate}`。
fn fill_template(template: &str, condition: &str, candidate: &str) -> String {
    template
        .replace("{condition}", condition)
        .replace("{candidate}", candidate)
}

fn read_lines(path: &PathBuf) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("打开候选项文件失败：{}", path.display()))?;
    let mut out = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        out.push(trimmed.to_string());
    }
    Ok(out)
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn nli_model_path(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| home_dir().join("models/match-gate/erlangshen-nli-110m-q8_0.gguf"))
}

fn decision_model_path(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| home_dir().join("models/match-gate/chinese-decision-0.8b-Q8_0.gguf"))
}

fn rerank_model_path(explicit: Option<PathBuf>) -> PathBuf {
    explicit.unwrap_or_else(|| home_dir().join("models/match-gate/bge-reranker-v2-m3-Q8_0.gguf"))
}

/// 门禁判定引擎。
#[derive(Debug, Clone, Copy, ValueEnum)]
enum EngineArg {
    /// 0.8B 决策模型：每候选一次前向，准确率高
    Decision,
    /// 110M NLI：快，但 llama.cpp 丢失句段 embedding，精度有限
    Nli,
    /// bge-reranker-v2-m3：交叉编码器，快且不受句段 embedding 影响
    Rerank,
}

/// 用决策模型逐个判断候选项是否满足条件。
fn run_gate_decision(
    candidates: &[String],
    condition: &str,
    threshold: f32,
    threads: i32,
    decision_model: Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    let model = DecisionModel::load(decision_model_path(decision_model))?;
    let mut session = model.session(DEFAULT_N_CTX, threads)?;
    let options = [
        DecisionOption::new("false", "不满足"),
        DecisionOption::new("true", "满足"),
    ];

    let mut rows = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let state = format!("候选人的情况：{candidate}");
        let instructions = format!("该候选人是否满足条件「{condition}」？");
        let start = std::time::Instant::now();
        let outcome = session.decide(&state, &instructions, &options, QuestionType::Noul)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let p_true = outcome.probability_of("true").unwrap_or(0.0);
        let matched = p_true >= threshold;

        if !json_output {
            println!(
                "{:<8} {:<28} p_true {:.4}  ({:.0}ms)",
                if matched { "MATCH" } else { "REJECT" },
                truncate(candidate, 28),
                p_true,
                elapsed_ms
            );
        }

        rows.push(json!({
            "candidate": candidate,
            "matched": matched,
            "p_true": p_true,
            "latency_ms": elapsed_ms,
        }));
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "engine": "decision",
                "condition": condition,
                "threshold": threshold,
                "temperature": model.temperature(),
                "results": rows,
            }))?
        );
    }
    Ok(())
}

/// 用 110M NLI 逐个判断候选项是否满足条件。
fn run_gate_nli(
    candidates: &[String],
    condition: &str,
    threshold: f32,
    strict: bool,
    no_reverse: bool,
    threads: i32,
    nli_model: Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    let model = NliModel::load(nli_model_path(nli_model))?;
    let mut session = model.session(threads)?;

    let mut rows = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        // 证据 = 候选项，结论 = 条件：候选项是否真的满足条件。
        let forward = session.predict(candidate, condition)?;
        let reverse = if no_reverse {
            None
        } else {
            Some(session.predict(condition, candidate)?)
        };
        let verdict = verdict(forward, reverse, threshold, strict);

        if !json_output {
            let extra = match reverse {
                Some(r) => format!("   reverse: {} (entail {:.4})", r.label(), r.entailment),
                None => String::new(),
            };
            println!(
                "{:<8} {:<28} entail {:.4} / neutral {:.4} / contra {:.4}{}",
                verdict,
                truncate(candidate, 28),
                forward.entailment,
                forward.neutral,
                forward.contradiction,
                extra
            );
        }

        rows.push(json!({
            "candidate": candidate,
            "verdict": verdict,
            "scores": forward,
            "reverse": reverse,
        }));
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string(&json!({
                "engine": "nli",
                "condition": condition,
                "threshold": threshold,
                "strict": strict,
                "results": rows,
            }))?
        );
    }
    Ok(())
}

/// clap 侧的题型参数，避免把 clap 依赖带进库。
#[derive(Debug, Clone, Copy, ValueEnum)]
enum QTypeArg {
    Noul,
    Choice,
    Score,
}

impl From<QTypeArg> for QuestionType {
    fn from(value: QTypeArg) -> Self {
        match value {
            QTypeArg::Noul => QuestionType::Noul,
            QTypeArg::Choice => QuestionType::Choice,
            QTypeArg::Score => QuestionType::Score,
        }
    }
}
