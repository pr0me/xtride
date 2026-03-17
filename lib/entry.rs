use std::str::FromStr;

use hashbrown::HashMap;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::ngram::{ngram_hash, Ngram, VarKind};

static GHIDRA_STACK: Lazy<Regex> = Lazy::new(|| Regex::new(r"[a-z]*Stack_[0-9]+").unwrap());
static GHIDRA_VAR: Lazy<Regex> = Lazy::new(|| Regex::new(r"[a-z]*Var[0-9]+").unwrap());
static GHIDRA_ADDR_STRING: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"s_[a-zA-Z0-9_]+[a-fA-F0-9]{8}").unwrap());
static GHIDRA_ADDR_PTR: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"PTR_[a-zA-Z0-9_]+[a-fA-F0-9]{8}").unwrap());
static HEXNUM: Lazy<Regex> = Lazy::new(|| Regex::new(r"0x[0-9a-fA-F]+").unwrap());
static PREFIXES: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        // IDA (for inter-op with DIRT-format datasets)
        "sub_",
        "loc_",
        "unk_",
        "off_",
        "asc_",
        "stru_",
        "funcs_",
        "byte_",
        "word_",
        "dword_",
        "qword_",
        "xmmword_",
        "ymmword_",
        "LABEL_",
        // Ghidra
        "FUN_",
        "thunk_FUN_",
        "LAB_",
        "joined_r0x",
        "DAT_",
        "_DAT_",
        "code_r0x",
        "uRam",
        "switchD_",
        "switchdataD_",
        "caseD_",
    ]
});
static FULL_STRIP: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "?",
        "Number",
        "String",
        "L",
        "__int8",
        "__int16",
        "__int32",
        "__int64",
        "LODWORD",
        "const",
        "_BYTE",
        "_WORD",
        "_DWORD",
        "_QWORD",
        "char",
        "float",
        "double",
        "__fastcall",
        "unsigned",
        "void",
        "break",
        "if",
        "else",
        "while",
        "int",
        "void",
        "goto",
        "[",
        "]",
        "(",
        ")",
        "{",
        "}",
        "+",
        "-",
        ",",
        ";",
        "*",
        "*=",
        "<",
        ">",
        "=",
        "<=",
        ">=",
        "==",
        "!=",
        "++",
        "--",
        "+=",
        "-=",
        "<<",
        ">>",
        "<<=",
        ">>=",
        "!",
        "|",
        "||",
        "|=",
        "&",
        "&&",
        "&=",
        "/",
        "/=",
        "^",
        "^=",
        "%",
        "%=",
    ]
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelType {
    Name,
    Type,
}

impl FromStr for LabelType {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "type" => Ok(LabelType::Type),
            "name" => Ok(LabelType::Name),
            _ => Err("invalid label type"),
        }
    }
}

/// Label used for dataset annotation.
/// This can be used to hold groundtruth information in evaluation datasets and originates from the
/// DIRT format and benchmarks used in the original STRIDE paper.
#[derive(Debug, Deserialize, Serialize)]
pub struct Label {
    human: bool,
    label: String,
}

impl Label {
    pub fn is_human(&self) -> bool {
        self.human
    }

    pub fn get_label(&self) -> &str {
        &self.label
    }
}

#[derive(Debug, Deserialize)]
pub struct Labels {
    #[serde(flatten)]
    variables: HashMap<String, Label>,
}

impl Labels {
    pub fn get(&self, label: &str) -> Option<&Label> {
        self.variables.get(label)
    }

    pub fn all_human_labels(&self) -> Vec<&str> {
        self.variables
            .values()
            .filter(|l| l.human)
            .map(|l| l.label.as_str())
            .collect()
    }

    pub fn vars(&self) -> &HashMap<String, Label> {
        &self.variables
    }
}

pub fn strip_tokens(tokens: &Vec<String>, full_strip: bool) -> Option<Vec<String>> {
    let mut stripped = Vec::with_capacity(tokens.len());

    for tok in tokens {
        let mut norm = tok.to_owned();

        for &prefix in PREFIXES.iter() {
            if tok.starts_with(prefix) {
                norm = format!("{}XXX", prefix);
                break;
            }
        }

        if tok.starts_with('"') && tok.ends_with('"') {
            norm = "<STRING>".into();
        } else if GHIDRA_STACK.find(tok).map_or(false, |m| m.as_str() == tok) {
            norm = "<ghidra_stack>".into();
        } else if GHIDRA_VAR.find(tok).map_or(false, |m| m.as_str() == tok) {
            norm = "<ghidra_var>".into();
        } else if GHIDRA_ADDR_STRING
            .find(tok)
            .map_or(false, |m| m.as_str() == tok)
        {
            norm = tok.get(..tok.len() - 8)?.into();
        } else if GHIDRA_ADDR_PTR
            .find(tok)
            .map_or(false, |m| m.as_str() == tok)
        {
            norm = tok.get(..tok.len() - 8)?.into();
        } else if let Some(_captures) = HEXNUM.captures(tok) {
            if let Ok(val) = i64::from_str_radix(&tok.get(2..)?, 16) {
                if val >= 0x100 {
                    let hex_digits = format!("{:x}", val).len();
                    norm = format!("<NUM_{}>", hex_digits);
                } else {
                    norm = format!("0x{:x}", val);
                }
            }
        } else if let Ok(val) = tok.parse::<i64>() {
            if val >= 0x100 {
                let hex_digits = format!("{:x}", val).len();
                norm = format!("<NUM_{}>", hex_digits);
            } else {
                norm = format!("0x{:x}", val);
            }
        }

        stripped.push(norm);
    }

    if full_strip {
        stripped = stripped
            .into_iter()
            .map(|x| {
                if FULL_STRIP.contains(&x.as_str()) {
                    x
                } else {
                    "?".into()
                }
            })
            .collect();
    }

    Some(stripped)
}

#[derive(Debug, Deserialize)]
pub struct Entry {
    tokens: Vec<String>,
    labels: HashMap<LabelType, Labels>,
    #[serde(default)]
    meta: HashMap<String, serde_json::Value>,
    #[serde(skip)]
    stripped_tokens: Option<Vec<String>>,
    #[serde(skip)]
    full_strip: bool,
}

impl Entry {
    pub fn new(raw: serde_json::Value, full_strip: bool) -> serde_json::Result<Self> {
        let mut entry = serde_json::from_value::<Entry>(raw)?;
        entry.full_strip = full_strip;
        Ok(entry)
    }

    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    pub fn stripped_tokens(&mut self) -> &[String] {
        if self.stripped_tokens.is_none() {
            self.stripped_tokens = strip_tokens(&self.tokens, self.full_strip);
        }

        self.stripped_tokens.as_ref().unwrap()
    }

    pub fn meta(&self) -> &HashMap<String, serde_json::Value> {
        &self.meta
    }

    pub fn labels(&self, label_type: LabelType) -> Option<&Labels> {
        self.labels.get(&label_type)
    }

    pub fn var_counts(&self) -> Option<HashMap<&str, usize>> {
        let mut counts = HashMap::new();
        for tok in &self.tokens {
            if tok.starts_with("@@") && tok.ends_with("@@") {
                let var = tok.get(2..tok.len() - 2)?;
                *counts.entry(var).or_insert(0) += 1;
            }
        }
        Some(counts)
    }

    pub fn iter_ngrams(
        &mut self,
        size: usize,
        flanking: bool,
    ) -> Option<impl Iterator<Item = Ngram<'_>> + '_> {
        let padding = vec!["??".into(); size];
        let tokens = self.stripped_tokens();
        let mut padded = padding.to_owned();
        padded.extend_from_slice(tokens);
        padded.extend(padding);

        // find ipc boundaries in original tokens
        let ipc_marker = "/*ipc*/";
        let tokens_ref = &self.tokens;
        let boundaries = tokens_ref
            .iter()
            .enumerate()
            .filter_map(|(idx, tok)| if tok == ipc_marker { Some(idx) } else { None })
            .collect::<Vec<usize>>();

        Some(
            tokens_ref
                .iter()
                .enumerate()
                .filter(|(_, tok)| tok.starts_with("@@") && tok.ends_with("@@"))
                .flat_map(move |(i, tok)| {
                    let var = tok.get(2..tok.len() - 2)?;
                    // define discriminators based on var type
                    let (center_disc, left_disc, right_disc, kind) = if var.starts_with("sub_") {
                        (
                            b"fn" as &[u8],
                            b"left_fn" as &[u8],
                            b"right_fn" as &[u8],
                            VarKind::Function,
                        )
                    } else {
                        (
                            b"var" as &[u8],
                            b"left_var" as &[u8],
                            b"right_var" as &[u8],
                            VarKind::Variable,
                        )
                    };

                    // compute segment [seg_start, seg_end] for current var index i
                    let (seg_start, seg_end) = {
                        let mut start = -1isize;
                        let mut end = tokens_ref.len() as isize;
                        for &b in boundaries.iter() {
                            if (b as isize) < i as isize {
                                start = b as isize;
                            } else if (b as isize) > i as isize {
                                end = b as isize;
                                break;
                            }
                        }
                        (start + 1, end - 1)
                    };

                    // helper to build a span with boundary-aware masking
                    let build_span = |start_padded: usize, len: usize| -> Vec<String> {
                        let mut out = Vec::with_capacity(len);
                        for off in 0..len {
                            let pidx = start_padded + off;
                            let orig_idx = pidx as isize - size as isize;
                            let is_outside_segment = orig_idx < seg_start || orig_idx > seg_end;
                            let is_ipc = orig_idx >= 0
                                && (orig_idx as usize) < tokens_ref.len()
                                && tokens_ref[orig_idx as usize] == ipc_marker;
                            if is_outside_segment || is_ipc {
                                out.push("??".into());
                            } else {
                                out.push(padded[pidx].clone());
                            }
                        }
                        out
                    };

                    if !flanking {
                        let span_vec = build_span(i, size * 2 + 1);
                        let h = ngram_hash(&span_vec, center_disc);
                        Some(vec![Ngram::new(h, span_vec, i, false, var, kind)])
                    } else {
                        let left_vec = build_span(i, size);
                        let right_vec = build_span(i + size + 1, size);
                        let left_h = ngram_hash(&left_vec, left_disc);
                        let right_h = ngram_hash(&right_vec, right_disc);
                        Some(vec![
                            Ngram::new(left_h, left_vec, i, false, var, kind.to_owned()),
                            Ngram::new(right_h, right_vec, i, true, var, kind),
                        ])
                    }
                })
                .flatten(),
        )
    }
}
