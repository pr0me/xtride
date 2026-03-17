use hashbrown::HashMap;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::entry::{Entry, LabelType};

/// Inner LabelCountMap: Label String -> Count
pub type LabelCountMap = HashMap<String, usize>;
/// Processor Output: Hash -> (IsFunction, LabelCountMap)
pub type ProcessorOutput = HashMap<[u8; 12], (bool, LabelCountMap)>;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum VarKind {
    Variable,
    Function,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalVar<'a> {
    var_name: &'a str,
    kind: VarKind,
}

impl<'a> LocalVar<'a> {
    pub fn new(var_name: &'a str, kind: VarKind) -> Self {
        Self { var_name, kind }
    }

    pub fn name(&self) -> &str {
        self.var_name
    }

    pub fn kind(&self) -> &VarKind {
        &self.kind
    }

    pub fn is_variable(&self) -> bool {
        matches!(self.kind, VarKind::Variable)
    }

    pub fn is_function(&self) -> bool {
        matches!(self.kind, VarKind::Function)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NgramIndex {
    i: usize,
    flank: bool,
}

#[derive(Debug, Clone)]
pub struct Ngram<'a> {
    hash: [u8; 12],
    token_span: Vec<String>,
    idx: NgramIndex,
    var: LocalVar<'a>,
}

impl<'a> Ngram<'a> {
    pub fn new(
        hash: [u8; 12],
        token_span: impl Into<Vec<String>>,
        idx: usize,
        flank: bool,
        var: &'a str,
        kind: VarKind,
    ) -> Self {
        let new_idx = NgramIndex { i: idx, flank };
        let var = LocalVar::new(var, kind);

        Self {
            hash,
            token_span: token_span.into(),
            idx: new_idx,
            var,
        }
    }

    pub fn hash(&self) -> &[u8; 12] {
        &self.hash
    }

    pub fn token_span(&self) -> &[String] {
        &self.token_span
    }

    pub fn idx(&self) -> &NgramIndex {
        &self.idx
    }

    pub fn var(&self) -> &LocalVar<'_> {
        &self.var
    }
}

pub fn ngram_hash(tokens: &[String], discriminator: &[u8]) -> [u8; 12] {
    let mut name_idx = 0;
    let mut normalised_names = HashMap::new();
    let mut hasher = Sha256::new();

    for (i, tok) in tokens.iter().enumerate() {
        if i > 0 {
            hasher.update(&[0xFF]);
        }

        if tok.len() > 4 && tok.starts_with("@@") && tok.ends_with("@@") {
            // by construction, this slice access is safe
            let var = &tok[2..tok.len() - 2];
            let normalised = normalised_names.entry(var).or_insert_with(|| {
                let new_name = format!("@@var_{}@@", name_idx);
                name_idx += 1;
                new_name
            });
            hasher.update(normalised.as_bytes());
        } else {
            hasher.update(tok.as_bytes());
        }
    }
    hasher.update(discriminator);

    let result = hasher.finalize();
    let mut output = [0u8; 12];
    // result is 32 byte for sha256 finalize
    output.copy_from_slice(&result[..12]);
    output
}

pub struct Processor {
    label_type: LabelType,
    size: usize,
    flanking: bool,
}

impl Processor {
    pub fn new(label: LabelType, size: usize, flanking: bool) -> Self {
        Self {
            label_type: label,
            size,
            flanking,
        }
    }

    pub fn process(&self, entry: &mut Entry) -> Option<ProcessorOutput> {
        let mut hmap = ProcessorOutput::new();

        // get all label info first to avoid double borrow
        let label_info = entry
            .labels(self.label_type)?
            .vars()
            .iter()
            .filter(|(_, label)| label.is_human())
            .map(|(var, label)| (var.to_owned(), label.get_label().into()))
            .collect::<HashMap<String, String>>();

        for curr_ngram in entry.iter_ngrams(self.size as usize, self.flanking)? {
            if let Some(target) = label_info.get(curr_ngram.var().name()) {
                let is_function = curr_ngram.var().is_function();
                let (_flag, counts) = hmap
                    .entry(curr_ngram.hash().to_owned())
                    .or_insert_with(|| (is_function, LabelCountMap::new()));

                counts
                    .entry(target.to_owned())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        Some(hmap)
    }
}
