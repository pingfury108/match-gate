//! 提示词模板。
//!
//! 决策模型的模板必须与训练/评测时逐 token 对齐，否则概率会漂移。
//! 这里照抄 `chinese-decision-0.8b-gguf` 模型卡的实现（`prompt_version = hmm-zh-v1`）。

/// 候选项字母，最多 26 个。
pub const LETTERS: &[u8; 26] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// 决策模型 prompt 版本（与 `decision_config.json` 一致）。
pub const DECISION_PROMPT_VERSION: &str = "hmm-zh-v1";

/// 拼出决策模型的一次性输入。
///
/// `options` 是 `(key, description)`，顺序即字母顺序 A/B/C...
pub fn decision_prompt(state: &str, instructions: &str, options: &[(String, String)]) -> String {
    assert!(!options.is_empty() && options.len() <= LETTERS.len());

    let option_lines: Vec<String> = options
        .iter()
        .enumerate()
        .map(|(i, (key, desc))| format!("{}: {key} — {desc}", LETTERS[i] as char))
        .collect();

    let content = format!(
        "待判断的上下文：\n{state}\n\n问题：\n{instructions}\n\n候选项：\n{}\n只返回候选项对应的字母。",
        option_lines.join("\n")
    );

    // Qwen3.5 chat template, enable_thinking = False
    format!(
        "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noul_prompt_matches_model_card() {
        let options = vec![
            ("false".to_string(), "否".to_string()),
            ("true".to_string(), "是".to_string()),
        ];
        let prompt = decision_prompt("客户留言：发票没收到。", "客户是否表达了紧迫性？", &options);
        assert_eq!(
            prompt,
            "<|im_start|>user\n待判断的上下文：\n客户留言：发票没收到。\n\n问题：\n客户是否表达了紧迫性？\n\n候选项：\nA: false — 否\nB: true — 是\n只返回候选项对应的字母。<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n"
        );
    }

    #[test]
    fn letters_are_used_in_order() {
        let options = vec![
            ("billing".to_string(), "支付、发票及退款".to_string()),
            ("technical".to_string(), "故障和系统问题".to_string()),
        ];
        let prompt = decision_prompt("s", "i", &options);
        assert!(prompt.contains("A: billing — 支付、发票及退款\nB: technical — 故障和系统问题"));
    }
}
