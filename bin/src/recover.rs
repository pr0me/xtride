use std::cmp::Ordering;
use std::fs;

use anyhow::{anyhow, Result};
use hashbrown::{HashMap, HashSet};
use xtride::entry::{Entry, LabelType};
use xtride::ngram::VarKind;
use xtride::predict::{Prediction, Predictor};
use xtride::tokenizer::tokenize;

use crate::eval::extract_type_name;
use crate::{append_file_stem_suffix, XtrideCommands};

#[derive(Debug, Clone)]
struct RecoverCandidate {
    display_type: String,
    score: f64,
}

#[derive(Debug, Clone)]
struct RecoverRow {
    symbol: String,
    kind: VarKind,
    count: usize,
    candidates: Vec<RecoverCandidate>,
}

pub(crate) fn recover(args: &XtrideCommands) -> Result<()> {
    let XtrideCommands::Recover {
        input,
        vocab,
        fn_vocab,
        db_dir,
        flanking,
        strip,
        threshold,
        top_k,
    } = args
    else {
        return Err(anyhow!("invalid recover invocation"));
    };

    if *top_k == 0 {
        return Err(anyhow!("top-k must be greater than zero"));
    }

    let source = fs::read_to_string(input)?;
    let targets = discover_targets(&source);
    if targets.is_empty() {
        return Err(anyhow!(
            "no recoverable symbols found in input; expected decompiler-style local names"
        ));
    }

    let mut entry = Entry::from_decompiled_text(&source, targets.iter().cloned(), *strip);

    let mut builder = Predictor::builder()
        .vocab(vocab)
        .db_dir(db_dir)
        .label_type(LabelType::Type)
        .flanking(*flanking);

    let fn_vocab = fn_vocab.clone().or_else(|| {
        let derived = append_file_stem_suffix(vocab, ".fn");
        if derived.exists() {
            Some(derived)
        } else {
            None
        }
    });
    if let Some(path) = fn_vocab {
        builder = builder.fn_vocab(path);
    }

    let predictor = builder.build()?;
    let predictions = predictor.predict_n(&mut entry, *top_k)?;
    let detected_targets = targets.len();
    let with_predictions = predictions.len();
    let rows = rows_from_predictions(predictions);

    let threshold = if *threshold >= 1.0 {
        None
    } else {
        Some(*threshold)
    };
    let (mut rows, suppressed_by_threshold) = apply_threshold(rows, threshold);
    sort_rows(&mut rows);

    println!("[*] recover mode");
    println!("    input: {}", input.display());
    println!("    detected symbols: {}", detected_targets);
    println!("    symbols with model output: {}", with_predictions);
    println!("    top-k: {}", top_k);
    println!(
        "    threshold: {}",
        threshold
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "disabled".into())
    );

    if rows.is_empty() {
        println!("\n[!] no predictions to report after filtering");
    } else {
        println!("\n[*] recovered predictions:");
        for row in &rows {
            let kind = if row.kind == VarKind::Function {
                "function"
            } else {
                "variable"
            };

            println!("\n- {} [{}] (occurrences: {})", row.symbol, kind, row.count);
            for (idx, candidate) in row.candidates.iter().enumerate() {
                println!(
                    "  {:>2}. score={:.4}  type={}",
                    idx + 1,
                    candidate.score,
                    candidate.display_type
                );
            }
        }
    }

    let no_model_output = detected_targets.saturating_sub(with_predictions);
    println!("\n[*] summary");
    println!("    reported symbols: {}", rows.len());
    println!("    filtered by threshold: {}", suppressed_by_threshold);
    println!("    no model output: {}", no_model_output);

    Ok(())
}

fn rows_from_predictions(predictions: HashMap<String, Vec<Prediction>>) -> Vec<RecoverRow> {
    predictions
        .into_iter()
        .filter_map(|(symbol, preds)| {
            let first = preds.first()?;
            let kind = first.kind().to_owned();
            let count = first.count();
            let candidates = preds
                .into_iter()
                .map(|pred| RecoverCandidate {
                    display_type: extract_type_name(pred.pred()),
                    score: pred.score(),
                })
                .collect::<Vec<_>>();

            Some(RecoverRow {
                symbol,
                kind,
                count,
                candidates,
            })
        })
        .collect()
}

fn apply_threshold(rows: Vec<RecoverRow>, threshold: Option<f64>) -> (Vec<RecoverRow>, usize) {
    let mut suppressed = 0usize;
    let rows = rows
        .into_iter()
        .filter_map(|mut row| {
            if let Some(threshold) = threshold {
                row.candidates
                    .retain(|candidate| candidate.score >= threshold);
            }
            if row.candidates.is_empty() {
                suppressed += 1;
                None
            } else {
                Some(row)
            }
        })
        .collect::<Vec<_>>();
    (rows, suppressed)
}

fn sort_rows(rows: &mut [RecoverRow]) {
    rows.sort_by(|left, right| {
        let left_best = left.candidates.first().map_or(0.0, |value| value.score);
        let right_best = right.candidates.first().map_or(0.0, |value| value.score);

        right_best
            .partial_cmp(&left_best)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.symbol.cmp(&right.symbol))
    });
}

fn discover_targets(source: &str) -> HashSet<String> {
    let known_variables = HashSet::<String>::new();
    tokenize(source, &known_variables)
        .into_iter()
        .filter(|token| is_candidate_target(token))
        .collect()
}

fn is_candidate_target(token: &str) -> bool {
    is_identifier(token) && !is_keyword(token) && looks_like_recover_symbol(token)
}

fn is_identifier(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn is_keyword(token: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "const", "return", "if", "else", "while", "for", "switch", "case", "break", "continue",
        "do", "goto", "int", "char", "float", "double", "void", "unsigned", "signed", "short",
        "long", "struct", "union", "enum", "sizeof",
    ];
    KEYWORDS.contains(&token)
}

fn looks_like_recover_symbol(token: &str) -> bool {
    looks_like_sub_function(token)
        || has_digit_after_prefix(token, "var")
        || has_digit_after_prefix(token, "param")
        || has_digit_after_prefix(token, "stack")
        || looks_like_ghidra_stack(token)
        || looks_like_ghidra_var(token)
}

fn looks_like_sub_function(token: &str) -> bool {
    if !token.starts_with("sub_") {
        return false;
    }
    let suffix = &token[4..];
    !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn has_digit_after_prefix(token: &str, prefix: &str) -> bool {
    if !token.starts_with(prefix) {
        return false;
    }
    let suffix = &token[prefix.len()..];
    !suffix.is_empty()
        && suffix.chars().any(|ch| ch.is_ascii_digit())
        && suffix
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn looks_like_ghidra_stack(token: &str) -> bool {
    token.contains("Stack_")
        && token.chars().any(|ch| ch.is_ascii_digit())
        && token
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn looks_like_ghidra_var(token: &str) -> bool {
    token.contains("Var")
        && token.chars().any(|ch| ch.is_ascii_digit())
        && token
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

#[cfg(test)]
mod tests {
    use hashbrown::HashSet;

    use super::{apply_threshold, discover_targets, sort_rows, RecoverCandidate, RecoverRow};
    use xtride::ngram::VarKind;

    #[test]
    fn discovers_common_decompiler_symbols() {
        let source = r#"
        int __fastcall sub_401000(int param1) {
            int var2;
            long local_18;
            iVar3 = var2 + param1;
            return sub_401100(iVar3);
        }
        "#;

        let discovered = discover_targets(source);
        let expected = HashSet::from_iter(
            ["sub_401000", "sub_401100", "param1", "var2", "iVar3"]
                .iter()
                .map(|&s| s.into()),
        );

        assert!(expected.is_subset(&discovered));
        assert!(!discovered.contains("return"));
    }

    #[test]
    fn threshold_filters_and_sorts_rows() {
        let mut rows = vec![
            RecoverRow {
                symbol: "var2".into(),
                kind: VarKind::Variable,
                count: 3,
                candidates: vec![
                    RecoverCandidate {
                        display_type: "A".into(),
                        score: 0.61,
                    },
                    RecoverCandidate {
                        display_type: "B".into(),
                        score: 0.58,
                    },
                ],
            },
            RecoverRow {
                symbol: "param1".into(),
                kind: VarKind::Variable,
                count: 2,
                candidates: vec![RecoverCandidate {
                    display_type: "C".into(),
                    score: 0.90,
                }],
            },
            RecoverRow {
                symbol: "stack3".into(),
                kind: VarKind::Variable,
                count: 1,
                candidates: vec![RecoverCandidate {
                    display_type: "D".into(),
                    score: 0.40,
                }],
            },
        ];

        let (filtered, suppressed) = apply_threshold(rows.split_off(0), Some(0.6));
        let mut filtered = filtered;
        sort_rows(&mut filtered);

        assert_eq!(suppressed, 1);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].symbol, "param1");
        assert_eq!(filtered[1].symbol, "var2");
        assert_eq!(filtered[1].candidates.len(), 1);
    }
}
