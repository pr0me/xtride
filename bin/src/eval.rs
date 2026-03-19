use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use hashbrown::{HashMap, HashSet};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::ParallelIterator;
use serde::{Deserialize, Serialize};
use xtride::entry::Entry;
use xtride::ngram::VarKind;
use xtride::predict::{Prediction, Predictor, PredictorError};
use xtride::vocab::Vocab;

use crate::corpus::{Corpus, CorpusError};
use crate::{append_file_stem_suffix, XtrideCommands};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Io(anyhow::Error),
    #[error(transparent)]
    Predictor(anyhow::Error),
    #[error(transparent)]
    Serialise(anyhow::Error),
    #[error(transparent)]
    Corpus(#[from] CorpusError),
}

impl EvalError {
    pub fn io<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Io(anyhow::Error::new(e))
    }

    pub fn predictor<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Predictor(anyhow::Error::new(e))
    }

    pub fn serialise<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Serialise(anyhow::Error::new(e))
    }
}

#[derive(Serialize)]
pub struct PredictionRecord {
    var: String,
    pred: String,
    label: String,
    count: usize,
    /// Additional (optional) meta info. This is a leftover of the original python implementation
    /// and can, e.g., signal if a database was created with `flanking`.
    #[serde(flatten)]
    meta: HashMap<String, serde_json::Value>,
}

/// DIRT dataset type representation
#[derive(Debug, Clone, Deserialize)]
struct DirtType {
    #[serde(rename = "T")]
    type_id: u8,
    #[serde(rename = "n", default)]
    name: Option<String>,
    #[serde(rename = "l", default)]
    fields: Option<Vec<DirtField>>,
    #[serde(rename = "s", default)]
    size: Option<usize>,
    #[serde(rename = "t", default)]
    type_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct DirtField {
    #[serde(rename = "T")]
    type_id: u8,
    #[serde(rename = "n", default)]
    name: Option<String>,
    #[serde(rename = "t", default)]
    type_name: Option<String>,
    #[serde(rename = "s", default)]
    size: Option<usize>,
}

impl DirtType {
    fn is_struct(&self) -> bool {
        self.type_id == 6
    }

    fn is_pointer(&self) -> bool {
        self.type_id == 3
    }

    fn is_union(&self) -> bool {
        self.type_id == 7
    }

    fn get_pointed_type_name(&self) -> Option<&str> {
        if self.is_pointer() {
            self.type_name.as_deref()
        } else {
            None
        }
    }

    fn calculate_offsets(&self) -> Option<HashMap<String, usize>> {
        if !self.is_struct() {
            return None;
        }

        let fields = self.fields.as_ref()?;
        let mut offsets = HashMap::new();
        let mut current_offset = 0;

        for field in fields {
            // Handle fields with size (including padding)
            if let Some(field_size) = field.size {
                // Only store named fields in the offsets map
                if let Some(field_name) = &field.name {
                    if !field.is_padding() {
                        offsets.insert(field_name.clone(), current_offset);
                    }
                    if let Some(t) = &field.type_name {
                        // anonymous types skew the struct layout in DIRT types
                        if t.starts_with("$") {
                            return None;
                        }
                    }
                }
                // Always advance the offset for any field with a size
                current_offset += field_size;
            }
        }

        Some(offsets)
    }
}

impl DirtField {
    fn is_padding(&self) -> bool {
        self.type_id == 5
    }
}

// minimal header view to quickly inspect DIRT type category without full parsing
#[derive(Debug, Clone, Deserialize)]
struct DirtTypeHeader {
    #[serde(rename = "T")]
    type_id: u8,
    #[serde(rename = "t", default)]
    type_name: Option<String>,
}

// minimal array view for DIRT arrays
#[derive(Debug, Clone, Deserialize)]
#[allow(unused)]
struct DirtArray {
    #[serde(rename = "T")]
    type_id: u8,
    #[serde(rename = "n")]
    num_elems: usize,
    #[serde(rename = "s")]
    elem_size: usize,
}

// collect field sizes for a struct; ignores padding fields
fn collect_struct_field_sizes(t: &DirtType) -> Option<Vec<usize>> {
    if !t.is_struct() {
        return None;
    }

    let mut sizes = Vec::new();
    let fields = t.fields.as_ref()?;
    for f in fields {
        if f.is_padding() {
            continue;
        }
        let Some(sz) = f.size else {
            return None;
        };
        sizes.push(sz);
    }
    Some(sizes)
}

// collect field sizes for a union; ignores padding fields
fn collect_union_field_sizes(t: &DirtType) -> Option<Vec<usize>> {
    if !t.is_union() {
        return None;
    }
    let mut sizes = Vec::new();
    let fields = t.fields.as_ref()?;
    for f in fields {
        if f.is_padding() {
            continue;
        }
        let Some(sz) = f.size else {
            return None;
        };
        sizes.push(sz);
    }
    Some(sizes)
}

/// Collect (offset, width) pairs for a struct; ignores padding fields.
/// Offsets computed by walking fields sequentially.
fn collect_field_offset_widths(t: &DirtType) -> Option<HashSet<(usize, usize)>> {
    if !t.is_struct() {
        return None;
    }

    let fields = t.fields.as_ref()?;
    let mut result = HashSet::new();
    let mut current_offset = 0usize;

    for f in fields {
        // padding fields might lack size; skip them for offset tracking if so
        let sz = f.size.unwrap_or(0);
        if !f.is_padding() && sz > 0 {
            result.insert((current_offset, sz));
        }
        current_offset += sz;
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Per-field layout matching result: (TP, FP, FN) counts
struct FieldMatchResult {
    tp: usize,
    fp: usize,
    fn_: usize,
}

/// Compare predicted vs label struct layouts at field level.
/// Returns (TP, FP, FN) based on (offset, width) pair matching.
/// Handles both direct structs and pointers-to-structs.
/// Only returns Some when BOTH pred and label resolve to structs.
fn field_level_compare(
    pred_str: &str,
    label_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> Option<FieldMatchResult> {
    // resolve prediction to struct definitions (handles both direct structs and pointers)
    let pred_defs = resolve_to_structs(pred_str, struct_types_multi)
        .into_iter()
        .filter(|d| d.is_struct())
        .collect::<Vec<_>>();

    if pred_defs.is_empty() {
        return None;
    }

    // resolve label to struct definitions
    let label_defs = resolve_to_structs(label_str, struct_types_multi)
        .into_iter()
        .filter(|d| d.is_struct())
        .collect::<Vec<_>>();

    if label_defs.is_empty() {
        return None;
    }

    // find best matching combination of pred and label definitions
    let mut best_result = None;
    let mut best_tp = 0usize;

    for pred_t in &pred_defs {
        let Some(pred_fields) = collect_field_offset_widths(pred_t) else {
            continue;
        };

        for label_t in &label_defs {
            let Some(label_fields) = collect_field_offset_widths(label_t) else {
                continue;
            };

            let tp = pred_fields.intersection(&label_fields).count();
            let fp = pred_fields.difference(&label_fields).count();
            let fn_ = label_fields.difference(&pred_fields).count();

            if tp > best_tp || best_result.is_none() {
                best_tp = tp;
                best_result = Some(FieldMatchResult { tp, fp, fn_ });
            }
        }
    }

    best_result
}

// resolve a type to its underlying unions if it's a pointer or a direct union
fn resolve_to_unions(
    type_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> Vec<DirtType> {
    let Ok(dirt_type) = serde_json::from_str::<DirtType>(type_str) else {
        return Vec::new();
    };

    if dirt_type.is_union() {
        vec![dirt_type]
    } else if dirt_type.is_pointer() {
        if let Some(type_name) = dirt_type.get_pointed_type_name() {
            for candidate in normalize_dirt_type_name(type_name) {
                if let Some(defs) = struct_types_multi.get(&candidate) {
                    return defs.iter().cloned().filter(|d| d.is_union()).collect();
                }
            }
        }
        Vec::new()
    } else {
        Vec::new()
    }
}

// compare predicted vs label layout and report category and match result (DIRT-format)
// only returns `Some`` when BOTH pred and label are the same composite category
fn layout_compare(
    pred_str: &str,
    label_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> Option<(&'static str, bool)> {
    let header = serde_json::from_str::<DirtTypeHeader>(pred_str).ok()?;
    match header.type_id {
        6 => {
            // struct - only evaluate if label is also a struct
            let pred_t = serde_json::from_str::<DirtType>(pred_str).ok()?;
            let pred_sizes = collect_struct_field_sizes(&pred_t)?;
            let label_defs = resolve_to_structs(label_str, struct_types_multi)
                .into_iter()
                .filter(|d| d.is_struct())
                .collect::<Vec<_>>();
            if label_defs.is_empty() {
                return None;
            }
            let is_match = label_defs.into_iter().any(|lt| {
                collect_struct_field_sizes(&lt)
                    .map(|ls| ls == pred_sizes)
                    .unwrap_or(false)
            });
            Some(("struct", is_match))
        }
        7 => {
            // union - only evaluate if label is also a union
            let pred_t = serde_json::from_str::<DirtType>(pred_str).ok()?;
            let mut pred_sizes = collect_union_field_sizes(&pred_t)?;
            pred_sizes.sort_unstable();
            let label_defs = resolve_to_unions(label_str, struct_types_multi);
            if label_defs.is_empty() {
                return None;
            }
            let is_match = label_defs.into_iter().any(|lt| {
                collect_union_field_sizes(&lt)
                    .map(|mut ls| {
                        ls.sort_unstable();
                        ls == pred_sizes
                    })
                    .unwrap_or(false)
            });
            Some(("union", is_match))
        }
        3 => {
            // pointer - decide if to struct or union via resolution
            let pred_struct_defs = resolve_to_structs(pred_str, struct_types_multi)
                .into_iter()
                .filter(|d| d.is_struct())
                .collect::<Vec<_>>();
            if !pred_struct_defs.is_empty() {
                let label_defs = resolve_to_structs(label_str, struct_types_multi)
                    .into_iter()
                    .filter(|d| d.is_struct())
                    .collect::<Vec<_>>();
                if label_defs.is_empty() {
                    return None;
                }
                let is_match = pred_struct_defs.into_iter().any(|ps| {
                    let Some(ps_sizes) = collect_struct_field_sizes(&ps) else {
                        return false;
                    };
                    label_defs.iter().any(|lt| {
                        collect_struct_field_sizes(lt)
                            .map(|ls| ls == ps_sizes)
                            .unwrap_or(false)
                    })
                });
                return Some(("ptr-struct", is_match));
            }

            let pred_union_defs = resolve_to_unions(pred_str, struct_types_multi);
            if !pred_union_defs.is_empty() {
                let label_defs = resolve_to_unions(label_str, struct_types_multi);
                if label_defs.is_empty() {
                    return None;
                }
                let is_match = pred_union_defs.into_iter().any(|pu| {
                    let Some(mut pu_sizes) = collect_union_field_sizes(&pu) else {
                        return false;
                    };
                    pu_sizes.sort_unstable();
                    label_defs.iter().any(|lt| {
                        collect_union_field_sizes(lt)
                            .map(|mut ls| {
                                ls.sort_unstable();
                                ls == pu_sizes
                            })
                            .unwrap_or(false)
                    })
                });
                return Some(("ptr-union", is_match));
            }
            None
        }
        2 => {
            // array - only evaluate if label is also an array
            let pred_arr = serde_json::from_str::<DirtArray>(pred_str).ok()?;
            let label_hdr = serde_json::from_str::<DirtTypeHeader>(label_str).ok()?;
            if label_hdr.type_id != 2 {
                return None;
            }
            let label_arr = serde_json::from_str::<DirtArray>(label_str).ok()?;
            let total_pred = pred_arr.num_elems.saturating_mul(pred_arr.elem_size);
            let total_label = label_arr.num_elems.saturating_mul(label_arr.elem_size);
            let is_match = pred_arr.num_elems == label_arr.num_elems && total_pred == total_label;
            Some(("array", is_match))
        }
        _ => None,
    }
}

// classify predicted type string into composite category
fn pred_category(
    pred_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> Option<&'static str> {
    let header = serde_json::from_str::<DirtTypeHeader>(pred_str).ok()?;
    match header.type_id {
        6 => Some("struct"),
        7 => Some("union"),
        2 => Some("array"),
        3 => {
            // pointer: resolve to struct or union
            let has_struct = !resolve_to_structs(pred_str, struct_types_multi)
                .into_iter()
                .filter(|d| d.is_struct())
                .collect::<Vec<_>>()
                .is_empty();
            if has_struct {
                return Some("ptr-struct");
            }
            let has_union = !resolve_to_unions(pred_str, struct_types_multi).is_empty();
            if has_union {
                return Some("ptr-union");
            }
            None
        }
        _ => None,
    }
}

// classify label type string into composite category
fn label_category(
    label_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> Option<&'static str> {
    // try to parse header directly
    if let Ok(header) = serde_json::from_str::<DirtTypeHeader>(label_str) {
        return match header.type_id {
            6 => Some("struct"),
            7 => Some("union"),
            2 => Some("array"),
            3 => {
                // pointer to struct/union?
                let has_struct = !resolve_to_structs(label_str, struct_types_multi)
                    .into_iter()
                    .filter(|d| d.is_struct())
                    .collect::<Vec<_>>()
                    .is_empty();
                if has_struct {
                    return Some("ptr-struct");
                }
                let has_union = !resolve_to_unions(label_str, struct_types_multi).is_empty();
                if has_union {
                    return Some("ptr-union");
                }
                None
            }
            _ => None,
        };
    }
    None
}

/// Check if a DIRT type string is a struct
fn is_struct(type_str: &str) -> bool {
    if let Ok(dirt_type) = serde_json::from_str::<DirtType>(type_str) {
        dirt_type.is_struct()
    } else {
        false
    }
}

/// Check if a DIRT type string has category 10 (target type for variables missing from ground truth)
fn is_type_category_10(type_str: &str) -> bool {
    if let Ok(dirt_type) = serde_json::from_str::<DirtType>(type_str) {
        dirt_type.type_id == 10
    } else {
        false
    }
}

/// Select the best prediction that is not category 10
/// Returns the first prediction that is not T:10, or None if all are T:10
fn select_best_non_category_10_prediction(predictions: &[Prediction]) -> Option<Prediction> {
    predictions
        .iter()
        .find(|pred| !is_type_category_10(pred.pred()))
        .cloned()
}

/// Resolve a type to its underlying struct/union if it's a pointer
/// Returns all possible struct definitions for the type
fn resolve_to_structs(
    type_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> Vec<DirtType> {
    let Ok(dirt_type) = serde_json::from_str::<DirtType>(type_str) else {
        return Vec::new();
    };

    if dirt_type.is_struct() || dirt_type.is_union() {
        // direct struct/union
        vec![dirt_type]
    } else if dirt_type.is_pointer() {
        // pointer to struct/union - resolve via type name
        if let Some(type_name) = dirt_type.get_pointed_type_name() {
            // try exact, then normalized variants without qualifiers like "const ", "mut ", etc.
            for candidate in normalize_dirt_type_name(type_name) {
                if let Some(defs) = struct_types_multi.get(&candidate) {
                    return defs.clone();
                }
            }
            Vec::new()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    }
}

/// Produce candidate names for DIRT lookup by stripping pointer qualifiers
fn normalize_dirt_type_name(name: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    let push_unique = |vec: &mut Vec<String>, s: &str| {
        if !vec.iter().any(|v| v == s) {
            vec.push(s.into());
        }
    };

    let base = name.trim();
    push_unique(&mut candidates, base);

    // strip leading stars and spaces (in case the name carried pointer marker)
    let star_stripped = base.trim_start_matches('*').trim_start();
    push_unique(&mut candidates, star_stripped);

    // iteratively strip known qualifiers
    let mut q_stripped = star_stripped;
    loop {
        let mut progressed = false;
        if let Some(s) = q_stripped.strip_prefix("const ") {
            q_stripped = s.trim_start();
            progressed = true;
        }
        if let Some(s) = q_stripped.strip_prefix("mut ") {
            q_stripped = s.trim_start();
            progressed = true;
        }
        if let Some(s) = q_stripped.strip_prefix("volatile ") {
            q_stripped = s.trim_start();
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    push_unique(&mut candidates, q_stripped);

    // also try without a trailing numeric suffix like _0, _1, ...
    if let Some(pos) = q_stripped.rfind('_') {
        let suffix = &q_stripped[pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            let without_suffix = &q_stripped[..pos];
            push_unique(&mut candidates, without_suffix);
        }
    }

    candidates
}

/// Check if a type is a struct or pointer to struct (multi-definition aware)
fn is_struct_or_pointer_to_struct_multi(
    type_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> bool {
    !resolve_to_structs(type_str, struct_types_multi).is_empty()
}

fn load_dirt_vocab(
    path: impl AsRef<std::path::Path>,
) -> Result<
    (
        HashMap<String, DirtType>,
        HashMap<String, Vec<DirtType>>,
        Vocab,
    ),
    EvalError,
> {
    // load vocab using the Vocab struct
    let vocab = Vocab::load(path).map_err(EvalError::serialise)?;

    let mut struct_types_multi = HashMap::new();
    let mut all_types = HashMap::new(); // all types by JSON string
    let mut errors = 0;
    let mut duplicate_struct_names = 0;

    // iterate over vocab entries (which are the type strings)
    for type_str in vocab.entries() {
        if let Ok(dirt_type) = serde_json::from_str::<DirtType>(type_str) {
            // store all types indexed by their JSON string
            all_types.insert(type_str.clone(), dirt_type.clone());

            // store ALL structs and unions indexed by name for resolution
            if dirt_type.is_struct() || dirt_type.is_union() {
                if let Some(name) = &dirt_type.name {
                    let entry = struct_types_multi
                        .entry(name.clone())
                        .or_insert_with(Vec::new);
                    if !entry.is_empty() {
                        duplicate_struct_names += 1;
                    }
                    entry.push(dirt_type);
                }
            }
        } else {
            errors += 1;
            // println!("error parsing DIRT type: {:?}", type_str);
        }
    }

    println!("errors: {}", errors);
    if duplicate_struct_names > 0 {
        println!(
            "struct names with multiple definitions: {}",
            struct_types_multi.values().filter(|v| v.len() > 1).count()
        );
        println!("total duplicate definitions: {}", duplicate_struct_names);
    }

    // return both maps: all_types for lookup, struct_types_multi for pointer resolution
    Ok((all_types, struct_types_multi, vocab))
}

/// Extract type name from prediction string
/// For DIRT-format datasets, the prediction is the full JSON string
pub(crate) fn extract_type_name(pred_str: &str) -> String {
    // try to parse as DirtType to extract name
    if let Ok(dirt_type) = serde_json::from_str::<DirtType>(pred_str) {
        if let Some(name) = dirt_type.name {
            return name;
        }
    }
    pred_str.into()
}

/// Extract function name from DIRT type string (category 9)
fn extract_function_name(type_str: &str) -> Option<String> {
    if let Ok(header) = serde_json::from_str::<DirtTypeHeader>(type_str) {
        if header.type_id == 9 {
            return header.type_name;
        }
    }
    None
}

/// Extract function address suffix from variable/function name like "sub_100f"
fn extract_sub_address(name: &str) -> Option<String> {
    let Some(pos) = name.find("sub_") else {
        return None;
    };
    let rest = &name[pos + 4..];
    let mut addr = String::new();
    for ch in rest.chars() {
        if ch.is_ascii_hexdigit() {
            addr.push(ch);
        } else {
            break;
        }
    }
    if addr.is_empty() {
        None
    } else {
        Some(addr)
    }
}

/// Normalize struct name for fuzzy matching
fn normalize_struct_name(name: &str) -> String {
    let mut normalized = name.to_lowercase();

    // strip leading whitespace
    normalized = normalized.trim_start().into();

    // strip leading pointer/reference markers and any spaces that follow
    while normalized.starts_with('*') || normalized.starts_with('&') {
        normalized = normalized
            .trim_start_matches(['*', '&'])
            .trim_start()
            .into();
    }

    // strip common qualifiers repeatedly (e.g., "const ", "volatile ", "mut ")
    loop {
        let mut progressed = false;
        if let Some(rest) = normalized.strip_prefix("const ") {
            normalized = rest.trim_start().into();
            progressed = true;
        }
        if let Some(rest) = normalized.strip_prefix("volatile ") {
            normalized = rest.trim_start().into();
            progressed = true;
        }
        if let Some(rest) = normalized.strip_prefix("mut ") {
            normalized = rest.trim_start().into();
            progressed = true;
        }
        if !progressed {
            break;
        }
    }

    // remove common prefixes/suffixes
    normalized = normalized.trim_start_matches('_').into();
    normalized = normalized.trim_start_matches("struct_").into();
    normalized = normalized.trim_start_matches("struct").into();
    normalized = normalized.trim_end_matches("_t").into();
    normalized = normalized.trim_end_matches("_struct").into();

    // remove pointer prefixes (windows-style)
    normalized = normalized.trim_start_matches("lp").into();
    normalized = normalized.trim_start_matches("p").into();

    normalized
}

/// Check if two struct names are fuzzy matches
fn are_struct_names_similar(name1: &str, name2: &str) -> bool {
    // exact match
    if name1 == name2 {
        return true;
    }

    // normalized match
    let norm1 = normalize_struct_name(name1);
    let norm2 = normalize_struct_name(name2);

    if norm1 == norm2 {
        return true;
    }

    // check for common variations like sg_io_hdr vs sg_io_hdr_t
    if norm1.starts_with(&norm2) || norm2.starts_with(&norm1) {
        // ensure the difference is only a small suffix/prefix
        let diff_len = norm1.len().abs_diff(norm2.len());
        if diff_len <= 3 {
            return true;
        }
    }

    false
}

/// Build a canonical signature of a DIRT struct definition (ignores field names)
fn struct_definition_signature(
    t: &DirtType,
) -> Option<(Option<usize>, Vec<(u8, Option<String>, Option<usize>)>)> {
    if !t.is_struct() {
        return None;
    }
    let fields = t.fields.as_ref()?;
    let sig = fields
        .iter()
        .map(|f| (f.type_id, f.type_name.clone(), f.size))
        .collect::<Vec<_>>();
    Some((t.size, sig))
}

/// Return true if two DIRT struct definitions are exactly equal regardless of name
fn are_dirt_struct_definitions_equal(type1: &DirtType, type2: &DirtType) -> bool {
    match (
        struct_definition_signature(type1),
        struct_definition_signature(type2),
    ) {
        (Some(sig1), Some(sig2)) => sig1 == sig2,
        _ => false,
    }
}

/// Check if two DIRT structs are fuzzy matches
fn are_dirt_structs_fuzzy_equal(type1: &DirtType, type2: &DirtType) -> bool {
    // both must be structs
    if !type1.is_struct() || !type2.is_struct() {
        return false;
    }

    // accept immediately if full definitions match, regardless of name
    if are_dirt_struct_definitions_equal(type1, type2) {
        return true;
    }

    // check name similarity
    if let (Some(name1), Some(name2)) = (&type1.name, &type2.name) {
        if !are_struct_names_similar(name1, name2) {
            return false;
        }
    } else {
        return false;
    }

    // check structural similarity - allow minor differences
    if let (Some(fields1), Some(fields2)) = (&type1.fields, &type2.fields) {
        // allow small differences in field count (e.g., debug fields)
        if fields1.len().abs_diff(fields2.len()) > 2 {
            return false;
        }

        // count matching fields (by offset and approximate size)
        let offsets1 = type1.calculate_offsets().unwrap_or_default();
        let offsets2 = type2.calculate_offsets().unwrap_or_default();

        let mut matches = 0;
        let total = offsets1.len().min(offsets2.len());

        for (_field_name, offset1) in &offsets1 {
            // try to find a field at similar offset in type2
            for (_, offset2) in &offsets2 {
                if offset1 == offset2 {
                    matches += 1;
                    break;
                }
            }
        }

        // require at least 80% field match
        if total > 0 {
            let match_ratio = matches as f64 / total as f64;
            return match_ratio >= 0.8;
        }
    }

    true
}

/// Check if prediction and label are fuzzy matches for DIRT types (multi-definition aware)
fn are_dirt_predictions_fuzzy_equal_multi(
    pred_str: &str,
    label_str: &str,
    struct_types_multi: &HashMap<String, Vec<DirtType>>,
) -> bool {
    // resolve both types to structs (handling pointers) - get all possible definitions
    let pred_types = resolve_to_structs(pred_str, struct_types_multi);
    let label_types = resolve_to_structs(label_str, struct_types_multi);

    // both must resolve to at least one struct
    if pred_types.is_empty() || label_types.is_empty() {
        return false;
    }

    // check if any combination of definitions match
    for pred_struct in &pred_types {
        for label_struct in &label_types {
            if are_dirt_structs_fuzzy_equal(pred_struct, label_struct) {
                return true;
            }
        }
    }

    false
}

/// Run inference on a single corpus entry. Uses input / output format from original STRIDE for compatibility.
fn predict_one(
    entry: &mut Entry,
    predictor: &Predictor,
    use_top_n: bool,
) -> Result<
    (
        HashMap<String, Vec<Prediction>>,
        HashMap<String, serde_json::Value>,
        Vec<String>,
    ),
    PredictorError,
> {
    let tokens = entry.tokens().to_vec();
    let predictions = if use_top_n {
        predictor.predict_n(entry, 5)?
    } else {
        // convert top-1 predictions to same format as predict_n
        predictor
            .predict(entry)?
            .into_iter()
            .map(|p| (p.var().into(), vec![p]))
            .collect()
    };
    Ok((predictions, entry.meta().to_owned(), tokens))
}

/// Extract and hash function body tokens (between first { and last })
fn hash_function_body(tokens: &[String]) -> Option<u64> {
    // find first '{' and last '}'
    let start_pos = tokens.iter().position(|t| t == "{")?;
    let end_pos = tokens.iter().rposition(|t| t == "}")?;

    // ensure valid range
    if start_pos >= end_pos {
        return None;
    }

    // extract body tokens (excluding the braces themselves)
    let body_tokens = &tokens[start_pos + 1..end_pos];

    // hash the body tokens
    let mut hasher = DefaultHasher::new();
    for token in body_tokens {
        // drop labeled function tokens (stripped names vary across binaries)
        if token.starts_with("@@sub_") {
            continue;
        }
        token.hash(&mut hasher);
    }

    Some(hasher.finish())
}

/// Run inference on the specified (test) dataset and calculate accuracy against ground truth.
pub(crate) fn evaluate(args: &XtrideCommands) -> Result<(), EvalError> {
    let mut processing_speed_ms_per_func = 0.0;
    let mut raw_inference_duration = Option::<std::time::Duration>::None;
    let mut absolute_start = Option::<Instant>::None;

    if let XtrideCommands::Evaluate {
        input,
        vocab: vocab_path,
        output,
        db_dir,
        type_,
        flanking,
        strip,
        consider_train,
        aggregate_funcs,
        threshold,
        threshold_debug,
        threshold_sweep,
    } = args
    {
        println!("Loading resources...");

        let score_threshold = if *threshold >= 1.0 {
            println!("  [*] Score threshold: disabled");
            None
        } else {
            println!("  [*] Score threshold: {}", threshold);
            Some(*threshold)
        };
        let corpus = Corpus::new(input, *strip);
        let label_type = *type_;

        let fn_vocab_path = append_file_stem_suffix(vocab_path, ".fn");

        let fn_vocab_path = if fn_vocab_path.exists() {
            Some(fn_vocab_path)
        } else {
            log::warn!(
                "function vocabulary not found at {}, proceeding without it.",
                fn_vocab_path.display()
            );
            None
        };

        let mut builder = Predictor::builder()
            .vocab(vocab_path)
            .db_dir(db_dir)
            .label_type(label_type)
            .flanking(*flanking);

        // save fn_vocab_path for OOV tracking before moving into builder
        let fn_vocab_path_for_oov = fn_vocab_path.clone();

        if let Some(fn_vocab) = fn_vocab_path {
            builder = builder.fn_vocab(fn_vocab);
        }

        println!("  [*] Loading n-gram databases...");
        let load_start = Instant::now();
        let predictor = builder.build().map_err(EvalError::predictor)?;
        let load_duration = load_start.elapsed();
        println!(
            "  [*] {} n-gram databases loaded in {}",
            predictor.num_dbs(),
            humantime::format_duration(load_duration)
        );

        println!("  [*] Loading DIRT-style type vocabulary...");
        let data = Some(load_dirt_vocab(vocab_path)?);

        let (_dirt_all_types, struct_types_multi, vocab) =
            if let Some((all_types, struct_types_multi, vocab)) = data {
                (Some(all_types), Some(struct_types_multi), Some(vocab))
            } else {
                (None, None, None)
            };

        // load function vocabulary for OOV tracking
        let fn_vocab_for_oov = if let Some(ref fn_vocab_path) = fn_vocab_path_for_oov {
            match Vocab::load(fn_vocab_path) {
                Ok(v) => Some(v),
                Err(e) => {
                    log::warn!("failed to load function vocabulary for OOV tracking: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // load train corpus and build set of function body hashes if requested
        let train_functions = if let Some(train_path) = consider_train {
            println!("  [*] Loading training corpus for in/out-of-train analysis...");
            let train_corpus = Corpus::new(train_path, *strip);
            let total_train_count = train_corpus.len()?;
            println!("  [*] Loaded training corpus. len: {}", total_train_count);
            let mut func_body_hashes = HashSet::new();

            let pb = ProgressBar::new(total_train_count as u64).with_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .map_err(EvalError::io)?
                    .progress_chars("#>-")
            );
            pb.set_message("Processing training corpus");

            for entry in train_corpus.iter()? {
                if let Ok(entry) = entry {
                    // hash the function body tokens
                    if let Some(body_hash) = hash_function_body(entry.tokens()) {
                        func_body_hashes.insert(body_hash);
                    }
                }
                pb.inc(1);
            }
            pb.finish_with_message("Training corpus processed");

            println!(
                "      Found {} unique function bodies in training set",
                func_body_hashes.len()
            );
            Some(func_body_hashes)
        } else {
            None
        };

        absolute_start = Some(Instant::now());
        println!("  [*] Running predictions...");
        let total_count = corpus.len()?;

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let pb = ProgressBar::new(total_count as u64).with_style(
            ProgressStyle::default_bar()
                .template("{spinner:.bright_green} [{elapsed_precise}] Predicting {bar:40.gradient(green,bright_green)} {pos:>7}/{len:7} [{percent:>3}%]")
                    .map_err(EvalError::io)?,
        );

        let predictions = corpus
            .par_iter()?
            .filter_map(|entry| entry.ok())
            .filter_map(|mut entry| {
                // hash function body before prediction
                let body_hash = hash_function_body(entry.tokens());
                let result = predict_one(&mut entry, &predictor, true).ok()?;
                let current = counter_clone.fetch_add(1, Ordering::SeqCst);
                if current % 512 == 0 {
                    pb.set_position(current as u64);
                }
                Some((result.0, result.1, result.2, body_hash))
            })
            .collect::<Vec<(
                HashMap<String, Vec<Prediction>>,
                HashMap<String, serde_json::Value>,
                Vec<String>, // tokens
                Option<u64>, // function body hash
            )>>();

        pb.finish_with_message(format!(
            "Predictions complete in {}",
            humantime::format_duration(pb.elapsed())
        ));
        raw_inference_duration = Some(pb.elapsed());
        processing_speed_ms_per_func =
            pb.elapsed().as_millis() as f64 / counter_clone.load(Ordering::SeqCst) as f64;
        println!(
            "             :: {:.2} ms/func",
            processing_speed_ms_per_func
        );

        println!("  [+] Writing results...");
        let file = File::create(output).map_err(EvalError::io)?;
        let mut writer = BufWriter::new(file);

        let total_entries = predictions.len() as usize;

        let pb = ProgressBar::new(total_entries as u64).with_style(
            ProgressStyle::default_bar()
                .template("{spinner:.bright_green} [{elapsed_precise}] Storing results {bar:40.gradient(green,bright_green)} {pos:>7}/{len:7} [{percent:>3}%]")
                    .map_err(EvalError::io)?,
        );

        let mut type_stats = BTreeMap::new();
        let mut func_label_stats = BTreeMap::new();
        let mut written_var = 0;
        let mut written_func = 0;
        let mut correct_var = 0;
        let mut correct_func = 0;
        let mut label_is_struct = 0;
        let mut correct_if_label_is_struct = 0;
        let mut pred_is_struct = 0;
        let mut false_positive_struct = 0;
        let mut false_negative_struct = 0;
        // fuzzy matching metrics
        let mut fuzzy_correct_var = 0;
        let mut fuzzy_correct_if_label_is_struct = 0;
        let mut fuzzy_matches_detail = Vec::<(String, String, String)>::new();

        // layout analysis metrics
        let mut layout_total_struct = 0usize;
        let mut layout_match_struct = 0usize;
        let mut layout_total_ptr_struct = 0usize;
        let mut layout_match_ptr_struct = 0usize;
        let mut layout_total_union = 0usize;
        let mut layout_match_union = 0usize;
        let mut layout_total_ptr_union = 0usize;
        let mut layout_match_ptr_union = 0usize;
        let mut _layout_total_array = 0usize;
        let mut _layout_match_array = 0usize;

        // field-level layout recovery metrics
        // evaluated only when both pred and label are structs (aligns with baseline papers)
        let mut field_tp = 0usize;
        let mut field_fp = 0usize;
        let mut field_fn = 0usize;

        // type category 10 tracking
        let mut type_category_10_count = 0usize;

        let mut thresh_pre_var = 0usize;
        let mut thresh_pre_func = 0usize;
        let mut thresh_drop_var = 0usize;
        let mut thresh_drop_func = 0usize;
        let mut thresh_var_min = f64::INFINITY;
        let mut thresh_var_max = f64::NEG_INFINITY;
        let mut thresh_func_min = f64::INFINITY;
        let mut thresh_func_max = f64::NEG_INFINITY;

        #[derive(Clone, Copy, Debug, Default)]
        struct SweepStats {
            pre_var: usize,
            pre_func: usize,
            kept_var: usize,
            kept_func: usize,
            correct_var: usize,
            correct_func: usize,
            struct_act_total: usize,
            struct_act_tp: usize,
        }

        let sweep_thresholds = [
            (None, "none"),
            (Some(0.4), "0.40"),
            (Some(0.65), "0.65"),
            (Some(0.9), "0.90"),
            (Some(0.95), "0.95"),
        ];
        let mut sweep_stats = [SweepStats::default(); 5];

        // composite identification counters
        let mut comp_totals = HashMap::<&'static str, usize>::new();
        let mut comp_tp = HashMap::<&'static str, usize>::new();
        let mut comp_fp = HashMap::<&'static str, usize>::new();
        let mut comp_fn = HashMap::<&'static str, usize>::new();

        // in/out-of-train metrics
        let mut in_train_total = 0;
        let mut in_train_correct = 0;
        let mut out_train_total = 0;
        let mut out_train_correct = 0;

        // struct prediction quality metrics
        let mut incorrect_struct_predictions = 0;
        let mut struct_to_struct_errors = 0;
        let mut non_struct_to_struct_errors = 0;
        // OOD detection metricsa
        let mut incorrect_predictions_total = 0;
        let mut ood_errors = 0; // ground truth label not in vocabulary
        let mut missing_label_counts = HashMap::<String, usize>::new();

        // struct-level OOD detection
        let mut struct_ood_errors = 0;
        let mut struct_missing_label_counts = HashMap::<String, usize>::new();

        // HAL_* function metrics
        let mut hal_total_labels = 0usize;
        let mut hal_correct_exact = 0usize;
        let mut hal_pred_total = 0usize;
        let mut hal_true_positive = 0usize;
        let mut hal_false_positive = 0usize;
        let mut hal_false_negative = 0usize;
        let mut hal_oov_errors = 0usize;
        let mut hal_missing_label_counts = HashMap::<String, usize>::new();

        // function aggregation structures
        // per-address aggregation: predicted name -> {sum_score, count}
        let mut agg_counts = HashMap::<String, HashMap<String, (f64, usize)>>::new();
        // ground-truth function label per address (independent of threshold)
        let mut agg_labels = HashMap::<String, String>::new();

        for (preds_by_var, meta, _tokens, body_hash) in predictions {
            // check if function body hash is in train set
            let is_in_train =
                if let (Some(ref train_hashes), Some(hash)) = (&train_functions, body_hash) {
                    train_hashes.contains(&hash)
                } else {
                    false
                };
            // flatten predictions to single list (take first of each, skipping T:10 for DIRT format)
            let preds = preds_by_var
                .into_values()
                .map(|v| select_best_non_category_10_prediction(&v))
                .collect::<Vec<_>>();

            // flatten and combine with meta data
            for pred in preds {
                let Some(pred) = pred else {
                    continue;
                };
                if pred.kind() == &VarKind::Variable {
                    thresh_pre_var += 1;
                    thresh_var_min = thresh_var_min.min(pred.score());
                    thresh_var_max = thresh_var_max.max(pred.score());
                } else if pred.kind() == &VarKind::Function {
                    thresh_pre_func += 1;
                    thresh_func_min = thresh_func_min.min(pred.score());
                    thresh_func_max = thresh_func_max.max(pred.score());
                }

                // capture ground-truth function label per address before thresholding
                if *aggregate_funcs && pred.kind() == &VarKind::Function {
                    if let Some(addr) = extract_sub_address(pred.var()) {
                        let label_name = extract_function_name(pred.label())
                            .unwrap_or_else(|| pred.label().into());
                        agg_labels.entry(addr).or_insert(label_name);
                    }
                }

                if *threshold_sweep {
                    // for sweep reporting, treat DIRT category-10 labels as non-actionable (always skipped)
                    let is_type_cat_10 = is_type_category_10(pred.label());
                    if !is_type_cat_10 {
                        match pred.kind() {
                            VarKind::Variable => {
                                sweep_stats.iter_mut().for_each(|s| s.pre_var += 1)
                            }
                            VarKind::Function => {
                                sweep_stats.iter_mut().for_each(|s| s.pre_func += 1)
                            }
                        }

                        let is_exact = pred.pred() == pred.label();

                        let (label_is_struct, pred_is_struct, layouts_match_struct) =
                            if let Some(ref struct_types_multi) = struct_types_multi {
                                fn normalize_cat(cat: &str) -> &'static str {
                                    match cat {
                                        "struct" | "ptr-struct" => "struct",
                                        "union" | "ptr-union" => "union",
                                        "array" => "array",
                                        _ => "other",
                                    }
                                }

                                let label_cat = label_category(pred.label(), struct_types_multi)
                                    .map(normalize_cat);
                                let pred_cat = pred_category(pred.pred(), struct_types_multi)
                                    .map(normalize_cat);
                                let label_is_struct = label_cat == Some("struct");
                                let pred_is_struct = pred_cat == Some("struct");
                                let layouts_match = if label_is_struct && pred_is_struct {
                                    let pred_defs =
                                        resolve_to_structs(pred.pred(), struct_types_multi);
                                    let label_defs =
                                        resolve_to_structs(pred.label(), struct_types_multi);
                                    pred_defs.iter().any(|pd| {
                                        collect_struct_field_sizes(pd)
                                            .map(|ps| {
                                                label_defs.iter().any(|ld| {
                                                    collect_struct_field_sizes(ld)
                                                        .map(|ls| ls == ps)
                                                        .unwrap_or(false)
                                                })
                                            })
                                            .unwrap_or(false)
                                    })
                                } else {
                                    false
                                };

                                (label_is_struct, pred_is_struct, layouts_match)
                            } else {
                                (false, false, false)
                            };

                        for (idx, (thr, _name)) in sweep_thresholds.iter().enumerate() {
                            let keep = thr.map(|t| pred.score() >= t).unwrap_or(true);
                            if !keep {
                                continue;
                            }

                            match pred.kind() {
                                VarKind::Variable => {
                                    sweep_stats[idx].kept_var += 1;
                                    if is_exact {
                                        sweep_stats[idx].correct_var += 1;
                                    }

                                    if pred_is_struct {
                                        sweep_stats[idx].struct_act_total += 1;
                                        if label_is_struct && layouts_match_struct {
                                            sweep_stats[idx].struct_act_tp += 1;
                                        }
                                    }
                                }
                                VarKind::Function => {
                                    sweep_stats[idx].kept_func += 1;
                                    if is_exact {
                                        sweep_stats[idx].correct_func += 1;
                                    }
                                }
                            }
                        }
                    }
                }

                // discard low-confidence predictions from evaluation
                if let Some(score_threshold) = score_threshold {
                    if pred.score() < score_threshold {
                        if pred.kind() == &VarKind::Variable {
                            thresh_drop_var += 1;
                        } else if pred.kind() == &VarKind::Function {
                            thresh_drop_func += 1;
                        }
                        if pred.kind() == &VarKind::Function {
                            let label_name = extract_function_name(pred.label())
                                .unwrap_or_else(|| pred.label().into());
                            let label_is_hal = label_name.starts_with("HAL_");
                            if label_is_hal {
                                hal_total_labels += 1;
                            }
                        }
                        continue;
                    }
                }

                // skip predictions with type category 10 (missing ground truth)
                if is_type_category_10(pred.label()) {
                    type_category_10_count += 1;
                    continue;
                }

                // aggregate function predictions by address (sub_...)
                if *aggregate_funcs && pred.kind() == &VarKind::Function {
                    if let Some(addr) = extract_sub_address(pred.var()) {
                        // count predicted function names (parsed if possible)
                        let pred_name = extract_function_name(pred.pred())
                            .unwrap_or_else(|| pred.pred().into());
                        let entry = agg_counts.entry(addr).or_insert_with(HashMap::new);
                        let stats = entry.entry(pred_name).or_insert((0.0, 0));
                        stats.0 += pred.score();
                        stats.1 += 1;
                    }
                }

                // HAL_* function tracking
                if pred.kind() == &VarKind::Function {
                    let label_name =
                        extract_function_name(pred.label()).unwrap_or_else(|| pred.label().into());
                    let pred_name =
                        extract_function_name(pred.pred()).unwrap_or_else(|| pred.pred().into());
                    let label_is_hal = label_name.starts_with("HAL_");
                    let pred_is_hal = pred_name.starts_with("HAL_");
                    let is_exact = pred_name == label_name;

                    if label_is_hal {
                        hal_total_labels += 1;
                        if is_exact {
                            hal_correct_exact += 1;
                        } else {
                            hal_false_negative += 1;

                            // check if HAL label is out-of-vocabulary
                            if let Some(ref fn_vocab) = fn_vocab_for_oov {
                                if fn_vocab.lookup(pred.label()).is_none() {
                                    hal_oov_errors += 1;
                                    *hal_missing_label_counts
                                        .entry(label_name.clone())
                                        .or_insert(0) += 1;
                                    log::debug!(
                                        "HAL function label not in vocabulary: {} (var: {})",
                                        label_name,
                                        pred.var()
                                    );
                                }
                            }
                        }
                    }

                    if pred_is_hal {
                        hal_pred_total += 1;
                        if label_is_hal && is_exact {
                            hal_true_positive += 1;
                        } else {
                            hal_false_positive += 1;
                        }
                    }
                }

                // track struct labels and predictions
                let label_is_struct_bool = if let Some(ref struct_types_multi) = struct_types_multi
                {
                    is_struct_or_pointer_to_struct_multi(pred.label(), struct_types_multi)
                } else {
                    is_struct(pred.label())
                };

                let pred_is_struct_bool = if let Some(ref struct_types_multi) = struct_types_multi {
                    is_struct_or_pointer_to_struct_multi(pred.pred(), struct_types_multi)
                } else {
                    is_struct(pred.pred())
                };

                if label_is_struct_bool {
                    label_is_struct += 1;
                }

                if pred_is_struct_bool {
                    pred_is_struct += 1;

                    // track incorrect struct predictions (predicting wrong struct is always bad)
                    // exclude fuzzy-correct predictions from incorrect counts/breakdowns
                    let is_fuzzy_match_here =
                        if label_is_struct_bool && pred.kind() == &VarKind::Variable {
                            if let Some(ref struct_types_multi) = struct_types_multi {
                                are_dirt_predictions_fuzzy_equal_multi(
                                    pred.pred(),
                                    pred.label(),
                                    struct_types_multi,
                                )
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                    if pred.pred() != pred.label()
                        && pred.kind() == &VarKind::Variable
                        && !is_fuzzy_match_here
                    {
                        incorrect_struct_predictions += 1;

                        // struct-level OOD: check if ground truth label is in vocabulary
                        let is_struct_ood = if let Some(ref vocab) = vocab {
                            vocab.lookup(pred.label()).is_none()
                        } else {
                            false
                        };

                        if is_struct_ood {
                            struct_ood_errors += 1;
                            *struct_missing_label_counts
                                .entry(extract_type_name(pred.label()))
                                .or_insert(0) += 1;
                        }

                        if label_is_struct_bool {
                            struct_to_struct_errors += 1; // wrong struct chosen
                        } else {
                            non_struct_to_struct_errors += 1; // false positive
                            false_positive_struct += 1;
                        }
                    } else if !label_is_struct_bool && pred.kind() == &VarKind::Variable {
                        // still track original false positive metric
                        false_positive_struct += 1;
                    }
                } else {
                    // check for false negative: label is struct but predicted as non-struct
                    if label_is_struct_bool && pred.kind() == &VarKind::Variable {
                        false_negative_struct += 1;
                    }
                }

                // check for exact match with whitespace removed
                let strip_whitespace =
                    |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
                let pred_stripped = strip_whitespace(pred.pred());
                let label_stripped = strip_whitespace(pred.label());
                let is_exact_match = pred_stripped == label_stripped;
                // if !is_exact_match {
                //     println!("--------------------------------");
                //     println!("pred.pred():  {}", pred.pred());
                //     println!("pred.label(): {}", pred.label());
                // }

                // check for fuzzy match (only for structs)
                let is_fuzzy_match =
                    if !is_exact_match && label_is_struct_bool && pred.kind() == &VarKind::Variable
                    {
                        if let Some(ref struct_types_multi) = struct_types_multi {
                            are_dirt_predictions_fuzzy_equal_multi(
                                pred.pred(),
                                pred.label(),
                                struct_types_multi,
                            )
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                if is_exact_match {
                    match pred.kind() {
                        VarKind::Function => {
                            correct_func += 1;
                            *func_label_stats
                                .entry(pred.label().to_owned())
                                .or_insert(0usize) += 1;
                        }
                        VarKind::Variable => {
                            if label_is_struct_bool {
                                correct_if_label_is_struct += 1;
                            }
                            correct_var += 1;
                            *type_stats.entry(pred.pred().to_owned()).or_insert(0usize) += 1;

                            // track in/out-of-train metrics
                            if train_functions.is_some() {
                                if is_in_train {
                                    in_train_correct += 1;
                                } else {
                                    out_train_correct += 1;
                                }
                            }
                        }
                    }
                } else if is_fuzzy_match {
                    // track fuzzy matches separately
                    fuzzy_correct_var += 1;
                    if label_is_struct_bool {
                        fuzzy_correct_if_label_is_struct += 1;
                    }
                    // collect details for reporting
                    fuzzy_matches_detail.push((
                        pred.var().into(),
                        extract_type_name(pred.pred()),
                        extract_type_name(pred.label()),
                    ));
                } else {
                    // this is an incorrect prediction - check OOD only for variable-type labels
                    if pred.kind() == &VarKind::Variable {
                        incorrect_predictions_total += 1;

                        // Check if ground truth label is in vocabulary
                        let is_ood = if let Some(ref vocab) = vocab {
                            vocab.lookup(pred.label()).is_none()
                        } else {
                            false
                        };

                        if is_ood {
                            ood_errors += 1;
                            *missing_label_counts
                                .entry(extract_type_name(pred.label()))
                                .or_insert(0) += 1;
                        }
                    }
                }

                match pred.kind() {
                    VarKind::Function => written_func += 1,
                    VarKind::Variable => {
                        written_var += 1;

                        // track in/out-of-train totals
                        if train_functions.is_some() {
                            if is_in_train {
                                in_train_total += 1;
                            } else {
                                out_train_total += 1;
                            }
                        }
                    }
                }

                // augment meta with per-record layout analysis (DIRT only) and update composite metrics
                let record_meta = if let Some(ref struct_types_multi) = struct_types_multi {
                    if pred.kind() == &VarKind::Variable {
                        // composite identification metrics (multi-label problem per category)
                        // struct/ptr-struct and union/ptr-union are considered equivalent if underlying layouts match
                        fn normalize_cat(cat: &str) -> &'static str {
                            match cat {
                                "struct" | "ptr-struct" => "struct",
                                "union" | "ptr-union" => "union",
                                "array" => "array",
                                _ => "other",
                            }
                        }

                        if let Some(label_cat) = label_category(pred.label(), struct_types_multi) {
                            let norm_label = normalize_cat(label_cat);
                            *comp_totals.entry(norm_label).or_insert(0) += 1;
                        }
                        if let Some(pred_cat) = pred_category(pred.pred(), struct_types_multi) {
                            let norm_pred = normalize_cat(pred_cat);
                            if let Some(label_cat) =
                                label_category(pred.label(), struct_types_multi)
                            {
                                let norm_label = normalize_cat(label_cat);
                                if norm_pred == norm_label {
                                    // same base category - check if layouts match
                                    let layouts_match = match norm_pred {
                                        "struct" => {
                                            // compare struct layouts (handles both direct and pointer)
                                            let pred_defs =
                                                resolve_to_structs(pred.pred(), struct_types_multi);
                                            let label_defs = resolve_to_structs(
                                                pred.label(),
                                                struct_types_multi,
                                            );
                                            pred_defs.iter().any(|pd| {
                                                collect_struct_field_sizes(pd)
                                                    .map(|ps| {
                                                        label_defs.iter().any(|ld| {
                                                            collect_struct_field_sizes(ld)
                                                                .map(|ls| ls == ps)
                                                                .unwrap_or(false)
                                                        })
                                                    })
                                                    .unwrap_or(false)
                                            })
                                        }
                                        "union" => {
                                            let pred_defs =
                                                resolve_to_unions(pred.pred(), struct_types_multi);
                                            let label_defs =
                                                resolve_to_unions(pred.label(), struct_types_multi);
                                            pred_defs.iter().any(|pd| {
                                                collect_union_field_sizes(pd)
                                                    .map(|mut ps| {
                                                        ps.sort_unstable();
                                                        label_defs.iter().any(|ld| {
                                                            collect_union_field_sizes(ld)
                                                                .map(|mut ls| {
                                                                    ls.sort_unstable();
                                                                    ls == ps
                                                                })
                                                                .unwrap_or(false)
                                                        })
                                                    })
                                                    .unwrap_or(false)
                                            })
                                        }
                                        _ => pred_cat == label_cat,
                                    };
                                    if layouts_match {
                                        *comp_tp.entry(norm_pred).or_insert(0) += 1;
                                    } else {
                                        *comp_fp.entry(norm_pred).or_insert(0) += 1;
                                        *comp_fn.entry(norm_label).or_insert(0) += 1;
                                    }
                                } else {
                                    *comp_fp.entry(norm_pred).or_insert(0) += 1;
                                    *comp_fn.entry(norm_label).or_insert(0) += 1;
                                }
                            } else {
                                // predicted composite but label is not composite
                                *comp_fp.entry(norm_pred).or_insert(0) += 1;
                            }
                        } else if let Some(label_cat) =
                            label_category(pred.label(), struct_types_multi)
                        {
                            // label has composite category but pred is not composite
                            let norm_label = normalize_cat(label_cat);
                            *comp_fn.entry(norm_label).or_insert(0) += 1;
                        }

                        if let Some((category, is_match)) =
                            layout_compare(pred.pred(), pred.label(), struct_types_multi)
                        {
                            match category {
                                "struct" | "ptr-struct" => {
                                    if category == "struct" {
                                        layout_total_struct += 1;
                                        if is_match {
                                            layout_match_struct += 1;
                                        }
                                    } else {
                                        layout_total_ptr_struct += 1;
                                        if is_match {
                                            layout_match_ptr_struct += 1;
                                        }
                                    }
                                    // field-level matching for structs (direct or via pointer)
                                    if let Some(field_result) = field_level_compare(
                                        pred.pred(),
                                        pred.label(),
                                        struct_types_multi,
                                    ) {
                                        field_tp += field_result.tp;
                                        field_fp += field_result.fp;
                                        field_fn += field_result.fn_;
                                    }
                                }
                                "union" => {
                                    layout_total_union += 1;
                                    if is_match {
                                        layout_match_union += 1;
                                    }
                                }
                                "ptr-union" => {
                                    layout_total_ptr_union += 1;
                                    if is_match {
                                        layout_match_ptr_union += 1;
                                    }
                                }
                                "array" => {
                                    _layout_total_array += 1;
                                    if is_match {
                                        _layout_match_array += 1;
                                    }
                                }
                                _ => {}
                            }

                            let mut m = meta.clone();
                            m.insert(
                                "layout_category".into(),
                                serde_json::Value::String(category.into()),
                            );
                            m.insert("layout_match".into(), serde_json::Value::Bool(is_match));
                            Some(m)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let record = PredictionRecord {
                    var: pred.var().to_owned(),
                    pred: pred.pred().to_owned(),
                    label: pred.label().to_owned(),
                    count: pred.count().to_owned(),
                    meta: record_meta.unwrap_or_else(|| meta.to_owned()),
                };

                serde_json::to_writer_pretty(&mut writer, &record).map_err(EvalError::serialise)?;
                writeln!(writer).map_err(EvalError::io)?;

                pb.inc(1);
            }
        }
        writer.flush().map_err(EvalError::io)?;

        pb.finish_with_message(format!(
            "Wrote {} predictions in {}",
            written_var + written_func,
            humantime::format_duration(pb.elapsed())
        ));

        println!(
            "\n  [*] Wrote {} predictions to {} (from {} entries)",
            written_var + written_func,
            output.display(),
            total_entries
        );

        let accuracy = (correct_var as f64 / written_var as f64) * 100.0;
        println!(
            "      Variable Type Accuracy :: {:.2}% [{}/{}]",
            accuracy, correct_var, written_var
        );
        if thresh_pre_var > 0 {
            let coverage = written_var as f64 / thresh_pre_var as f64;
            println!(
                "      Variable Coverage      :: {:.2}% [{}/{}]",
                coverage * 100.0,
                written_var,
                thresh_pre_var
            );
        }
        if written_var > 0 {
            let selective_risk_exact = 1.0 - (correct_var as f64 / written_var as f64);
            println!(
                "      Variable Selective Risk (exact) :: {:.2}% [{}/{}]",
                selective_risk_exact * 100.0,
                written_var - correct_var,
                written_var
            );
        }
        let accuracy = (correct_func as f64 / written_func as f64) * 100.0;
        println!(
            "      Function Type Accuracy :: {:.2}% [{}/{}]",
            accuracy, correct_func, written_func
        );
        if thresh_pre_func > 0 {
            let coverage = written_func as f64 / thresh_pre_func as f64;
            println!(
                "      Function Coverage      :: {:.2}% [{}/{}]",
                coverage * 100.0,
                written_func,
                thresh_pre_func
            );
        }
        if written_func > 0 {
            let selective_risk = 1.0 - (correct_func as f64 / written_func as f64);
            println!(
                "      Function Selective Risk :: {:.2}% [{}/{}]",
                selective_risk * 100.0,
                written_func - correct_func,
                written_func
            );
        }

        if *threshold_debug {
            println!("\n  [*] Threshold filtering stats:");
            println!(
                "      pre-threshold :: var={} func={}",
                thresh_pre_var, thresh_pre_func
            );
            if let Some(score_threshold) = score_threshold {
                println!(
                    "      dropped (<{:.4}) :: var={} func={}",
                    score_threshold, thresh_drop_var, thresh_drop_func
                );
            } else {
                println!("      dropped           :: disabled");
            }
            println!(
                "      post-filter   :: var={} func={}",
                written_var, written_func
            );

            if thresh_pre_var > 0 {
                let overall_var_acc = (correct_var as f64 / thresh_pre_var as f64) * 100.0;
                println!(
                    "      var accuracy (overall) :: {:.2}% [{}/{}]",
                    overall_var_acc, correct_var, thresh_pre_var
                );
            }
            if thresh_pre_func > 0 {
                let overall_func_acc = (correct_func as f64 / thresh_pre_func as f64) * 100.0;
                println!(
                    "      func accuracy (overall) :: {:.2}% [{}/{}]",
                    overall_func_acc, correct_func, thresh_pre_func
                );
            }
            if thresh_pre_var > 0 {
                println!(
                    "      var score range :: [{:.4}, {:.4}]",
                    thresh_var_min, thresh_var_max
                );
            }
            if thresh_pre_func > 0 {
                println!(
                    "      func score range :: [{:.4}, {:.4}]",
                    thresh_func_min, thresh_func_max
                );
            }
        }

        if *threshold_sweep {
            println!("\n  [*] Threshold sweep (abstention-aware):");
            println!(
                "      {:>7} | {:>9} | {:>14} | {:>10} | {:>17} | {:>10} | {:>11}",
                "cutoff",
                "var_cov",
                "var_risk_exact",
                "func_cov",
                "func_risk_exact",
                "struct_act",
                "struct_risk"
            );

            for (idx, (_thr, name)) in sweep_thresholds.iter().enumerate() {
                let s = sweep_stats[idx];
                let var_cov = if s.pre_var > 0 {
                    s.kept_var as f64 / s.pre_var as f64
                } else {
                    0.0
                };
                let func_cov = if s.pre_func > 0 {
                    s.kept_func as f64 / s.pre_func as f64
                } else {
                    0.0
                };
                let var_risk_exact = if s.kept_var > 0 {
                    1.0 - (s.correct_var as f64 / s.kept_var as f64)
                } else {
                    0.0
                };
                let func_risk_exact = if s.kept_func > 0 {
                    1.0 - (s.correct_func as f64 / s.kept_func as f64)
                } else {
                    0.0
                };

                let struct_act = if s.pre_var > 0 {
                    s.struct_act_total as f64 / s.pre_var as f64
                } else {
                    0.0
                };
                let struct_risk = if s.struct_act_total > 0 {
                    1.0 - (s.struct_act_tp as f64 / s.struct_act_total as f64)
                } else {
                    0.0
                };

                println!(
                    "      {:>7} | {:>8.2}% | {:>13.2}% | {:>9.2}% | {:>16.2}% | {:>9.2}% | {:>10.2}%",
                    name,
                    var_cov * 100.0,
                    var_risk_exact * 100.0,
                    func_cov * 100.0,
                    func_risk_exact * 100.0,
                    struct_act * 100.0,
                    struct_risk * 100.0
                );
            }
        }
        // OOD detection metrics
        if incorrect_predictions_total > 0 {
            let ood_percentage = (ood_errors as f64 / incorrect_predictions_total as f64) * 100.0;
            println!(
                "      OOD Detection Analysis :: {}/{} ({:.2}%) incorrect variable predictions had ground truth labels missing from vocabulary",
                ood_errors, incorrect_predictions_total, ood_percentage
            );
            if ood_errors > 0 {
                let mut counts = missing_label_counts.iter().collect::<Vec<_>>();
                counts.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                if !counts.is_empty() {
                    println!("        top 3 missing labels:");
                    for (label, count) in counts.into_iter().take(3) {
                        println!("          - {} ({})", label, count);
                    }
                }
            }
        }

        // aggregated function predictions by address
        if *aggregate_funcs {
            println!("\naggregated function predictions (by address, score-weighted):");
            let mut agg_correct = 0usize;
            let mut agg_total = 0usize;

            for (addr, counts) in agg_counts.iter() {
                let Some(label_name) = agg_labels.get(addr) else {
                    continue;
                };
                // select mode prediction
                // choose highest sum_score; break ties by higher count
                let best_pair = counts
                    .iter()
                    .max_by(|(_, (sum_a, cnt_a)), (_, (sum_b, cnt_b))| {
                        sum_a
                            .partial_cmp(sum_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(cnt_a.cmp(cnt_b))
                    })
                    .map(|(k, (sum, cnt))| (k.clone(), *sum, *cnt))
                    .unwrap_or((String::new(), 0.0, 0));
                let (best_pred, _best_sum, best_count) = best_pair;

                if best_count > 0 {
                    agg_total += 1;
                    if &best_pred == label_name {
                        agg_correct += 1;
                    }
                }
            }

            if agg_total > 0 {
                let agg_accuracy = (agg_correct as f64 / agg_total as f64) * 100.0;
                println!(
                    "      Aggregated Function Accuracy :: {:.2}% [{}/{}]",
                    agg_accuracy, agg_correct, agg_total
                );
            } else {
                println!("      no aggregated function addresses detected");
            }

            // HAL global (per-function) aggregated report
            // ground-truth = all HAL_* functions in binary, regardless of threshold/prediction presence
            let mut hal_gt_total = 0usize;
            let mut hal_tp = 0usize;
            let mut hal_fp = 0usize;
            let mut hal_fn = 0usize;
            let mut hal_pred_total = 0usize;

            for (addr, label_name) in agg_labels.iter() {
                let label_is_hal = label_name.starts_with("HAL_");

                let best_pred = agg_counts.get(addr).and_then(|counts| {
                    counts
                        .iter()
                        .max_by(|(_, (sum_a, cnt_a)), (_, (sum_b, cnt_b))| {
                            sum_a
                                .partial_cmp(sum_b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then(cnt_a.cmp(cnt_b))
                        })
                        .map(|(k, _)| k.as_str())
                });

                let pred_is_hal = best_pred.is_some_and(|p| p.starts_with("HAL_"));
                let is_exact = best_pred.is_some_and(|p| p == label_name);

                if label_is_hal {
                    hal_gt_total += 1;
                    if is_exact {
                        hal_tp += 1;
                    } else {
                        hal_fn += 1;
                    }
                }

                if pred_is_hal {
                    hal_pred_total += 1;
                    if !(label_is_hal && is_exact) {
                        hal_fp += 1;
                    }
                }
            }

            if hal_gt_total > 0 || hal_pred_total > 0 {
                if hal_gt_total > 0 {
                    let hal_acc = (hal_tp as f64 / hal_gt_total as f64) * 100.0;
                    println!(
                        "      HAL (global, aggregated) accuracy        :: {:.2}% [{}/{}]",
                        hal_acc, hal_tp, hal_gt_total
                    );
                }

                let p = if hal_pred_total > 0 {
                    hal_tp as f64 / hal_pred_total as f64
                } else {
                    0.0
                };
                let r = if hal_gt_total > 0 {
                    hal_tp as f64 / hal_gt_total as f64
                } else {
                    0.0
                };
                let f1 = if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                };

                println!(
                    "      HAL (global, aggregated) precision/recall/F1 :: P={:.2}% R={:.2}% F1={:.2}%",
                    p * 100.0,
                    r * 100.0,
                    f1 * 100.0
                );
                println!(
                    "      HAL (global, aggregated) counts         :: tp={} fp={} fn={} predicted={} ground-truth={}",
                    hal_tp, hal_fp, hal_fn, hal_pred_total, hal_gt_total
                );
            }
        }

        // HAL_* function statistics
        if hal_total_labels > 0 || hal_pred_total > 0 {
            println!("\n  [*] HAL function statistics:");
            if hal_total_labels > 0 {
                let hal_accuracy = (hal_correct_exact as f64 / hal_total_labels as f64) * 100.0;
                println!(
                    "      HAL accuracy        :: {:.2}% [{}/{}]",
                    hal_accuracy, hal_correct_exact, hal_total_labels
                );
            }
            let precision = if hal_pred_total > 0 {
                hal_true_positive as f64 / hal_pred_total as f64
            } else {
                0.0
            };
            let recall = if hal_total_labels > 0 {
                hal_true_positive as f64 / hal_total_labels as f64
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            println!(
                "      HAL precision/recall/F1 :: P={:.2}% R={:.2}% F1={:.2}%",
                precision * 100.0,
                recall * 100.0,
                f1 * 100.0
            );
            println!(
                "      HAL counts         :: tp={} fp={} fn={} predicted={} ground-truth={}",
                hal_true_positive,
                hal_false_positive,
                hal_false_negative,
                hal_pred_total,
                hal_total_labels
            );

            // report HAL OOV statistics
            if hal_oov_errors > 0 {
                let hal_fn_non_zero = hal_false_negative.max(1);
                let oov_percentage = (hal_oov_errors as f64 / hal_fn_non_zero as f64) * 100.0;
                println!(
                    "      HAL OOV errors     :: {} ({:.2}% of false negatives)",
                    hal_oov_errors, oov_percentage
                );

                // show top missing HAL functions
                if !hal_missing_label_counts.is_empty() {
                    let mut sorted_missing = hal_missing_label_counts.iter().collect::<Vec<_>>();
                    sorted_missing.sort_by_key(|(_, count)| Reverse(**count));

                    println!("      Top missing HAL functions (not in vocabulary):");
                    for (name, count) in sorted_missing.iter().take(10) {
                        println!("        - {} (count: {})", name, count);
                    }
                }
            } else if hal_false_negative > 0 {
                println!(
                    "      HAL OOV errors     :: 0 (all false negatives have labels in vocabulary)"
                );
            }
        }

        println!("\n  [*] Struct-specific statistics:");

        // log type category 10 occurrences for DIRT types
        if type_category_10_count > 0 {
            println!(
                "      Type category 10 skipped        :: {} (variables missing from ground truth)",
                type_category_10_count
            );
        }
        let struct_accuracy = if label_is_struct > 0 {
            (correct_if_label_is_struct as f64 / label_is_struct as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "      Struct Type Accuracy (exact)    :: {:.2}% [{}/{}]",
            struct_accuracy, correct_if_label_is_struct, label_is_struct
        );

        // fuzzy matching accuracy
        let fuzzy_struct_accuracy = if label_is_struct > 0 {
            ((correct_if_label_is_struct + fuzzy_correct_if_label_is_struct) as f64
                / label_is_struct as f64)
                * 100.0
        } else {
            0.0
        };
        println!(
            "      Struct Type Accuracy (fuzzy)    :: {:.2}% [{}/{}]",
            fuzzy_struct_accuracy,
            correct_if_label_is_struct + fuzzy_correct_if_label_is_struct,
            label_is_struct
        );

        // calculate false positive rate including discarded non-struct labels in denominator
        let non_struct_labels_kept = written_var - label_is_struct;
        let non_struct_labels_all = non_struct_labels_kept;
        let false_positive_rate = if non_struct_labels_all > 0 {
            (false_positive_struct as f64 / non_struct_labels_all as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "      Struct False Positive Rate :: {:.2}% [{}/{}]",
            false_positive_rate, false_positive_struct, non_struct_labels_all
        );

        // calculate false negative rate (recall complement)
        let false_negative_rate = if label_is_struct > 0 {
            (false_negative_struct as f64 / label_is_struct as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "      Struct False Negative Rate :: {:.2}% [{}/{}]",
            false_negative_rate, false_negative_struct, label_is_struct
        );

        println!(
            "      Total Struct Predictions :: {} (out of {} variables)",
            pred_is_struct, written_var
        );

        println!(
            "      Incorrect struct predictions :: {}",
            incorrect_struct_predictions
        );
        if incorrect_struct_predictions > 0 {
            println!("      Breakdown by error type:");
            println!(
                "        - Wrong struct type     :: {} (struct→wrong struct)",
                struct_to_struct_errors
            );
            println!(
                "        - False positives       :: {} (non-struct→struct)",
                non_struct_to_struct_errors
            );
            println!(
                "        - Struct to non-struct  :: {} (struct→non-struct)",
                false_negative_struct
            );

            // struct-level OOD detection analysis
            let struct_ood_pct =
                (struct_ood_errors as f64 / incorrect_struct_predictions as f64) * 100.0;
            println!(
                "      Struct OOD Analysis :: {}/{} ({:.2}%) incorrect struct predictions had ground truth labels missing from vocabulary",
                struct_ood_errors, incorrect_struct_predictions, struct_ood_pct
            );
            if struct_ood_errors > 0 {
                let mut counts = struct_missing_label_counts.iter().collect::<Vec<_>>();
                counts.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                if !counts.is_empty() {
                    println!("        top 3 missing struct labels:");
                    for (label, count) in counts.into_iter().take(3) {
                        println!("          - {} ({})", label, count);
                    }
                }
            }
        }

        // print fuzzy matching statistics
        if fuzzy_correct_var > 0 {
            println!("\n  [*] Fuzzy matching statistics:");
            println!(
                "      Fuzzy matches found     :: {} (additional correct matches)",
                fuzzy_correct_var
            );
            let fuzzy_var_accuracy =
                ((correct_var + fuzzy_correct_var) as f64 / written_var as f64) * 100.0;
            println!(
                "      Variable Accuracy (with fuzzy) :: {:.2}% [{}/{}]",
                fuzzy_var_accuracy,
                correct_var + fuzzy_correct_var,
                written_var
            );
        }

        println!("\n  [*] Layout analysis (DIRT):");
        if layout_total_struct > 0 {
            println!(
                "      struct layout match        :: {:.2}% [{}/{}]",
                (layout_match_struct as f64 / layout_total_struct as f64) * 100.0,
                layout_match_struct,
                layout_total_struct
            );
        }
        if layout_total_ptr_struct > 0 {
            println!(
                "      ptr->struct layout match   :: {:.2}% [{}/{}]",
                (layout_match_ptr_struct as f64 / layout_total_ptr_struct as f64) * 100.0,
                layout_match_ptr_struct,
                layout_total_ptr_struct
            );
        }
        if layout_total_union > 0 {
            println!(
                "      union layout match         :: {:.2}% [{}/{}]",
                (layout_match_union as f64 / layout_total_union as f64) * 100.0,
                layout_match_union,
                layout_total_union
            );
        }
        if layout_total_ptr_union > 0 {
            println!(
                "      ptr->union layout match    :: {:.2}% [{}/{}]",
                (layout_match_ptr_union as f64 / layout_total_ptr_union as f64) * 100.0,
                layout_match_ptr_union,
                layout_total_ptr_union
            );
        }
        let total_layout_cases = layout_total_struct
            + layout_total_ptr_struct
            + layout_total_union
            + layout_total_ptr_union;
        if total_layout_cases > 0 {
            let total_layout_matches = layout_match_struct
                + layout_match_ptr_struct
                + layout_match_union
                + layout_match_ptr_union;
            println!(
                "      total layout match         :: {:.2}% [{}/{}]",
                (total_layout_matches as f64 / total_layout_cases as f64) * 100.0,
                total_layout_matches,
                total_layout_cases
            );
        }

        // field-level layout recovery P/R/F1 (only when both pred and label are structs)
        if field_tp + field_fp + field_fn > 0 {
            let tp = field_tp as f64;
            let fp = field_fp as f64;
            let fn_ = field_fn as f64;
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            println!(
                "      field-level recovery       :: P={:.2}% R={:.2}% F1={:.2}%  [tp={}, fp={}, fn={}]",
                precision * 100.0,
                recall * 100.0,
                f1 * 100.0,
                field_tp,
                field_fp,
                field_fn
            );
        }

        // composite identification metrics (precision/recall/F1 per category and macro avg)
        // note: struct/ptr-struct combined into "struct", union/ptr-union combined into "union"
        println!("\n  [*] Composite Type Identification:");
        let categories = ["struct", "union"];
        let mut counted = 0.0;
        for &cat in &categories {
            let tp = *comp_tp.get(cat).unwrap_or(&0) as f64;
            let fp = *comp_fp.get(cat).unwrap_or(&0) as f64;
            let fn_ = *comp_fn.get(cat).unwrap_or(&0) as f64;
            let tot = *comp_totals.get(cat).unwrap_or(&0) as usize;
            if tp + fp + fn_ == 0.0 && tot == 0 {
                continue;
            }
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            println!(
                "      {:11} :: P={:.2}% R={:.2}% F1={:.2}%  [tp={}, fp={}, fn={}]",
                cat,
                precision * 100.0,
                recall * 100.0,
                f1 * 100.0,
                tp as usize,
                fp as usize,
                fn_ as usize
            );
            counted += 1.0;
        }
        if counted > 0.0 {
            // calculate weighted macro average based on category prevalence
            let mut weighted_sum_p = 0.0;
            let mut weighted_sum_r = 0.0;
            let mut weighted_sum_f1 = 0.0;
            let mut total_instances = 0.0;

            for &cat in &categories {
                let tp = *comp_tp.get(cat).unwrap_or(&0) as f64;
                let fp = *comp_fp.get(cat).unwrap_or(&0) as f64;
                let fn_ = *comp_fn.get(cat).unwrap_or(&0) as f64;
                let tot = *comp_totals.get(cat).unwrap_or(&0) as f64;

                if tp + fp + fn_ == 0.0 && tot == 0.0 {
                    continue;
                }

                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                let weight = tot;
                weighted_sum_p += precision * weight;
                weighted_sum_r += recall * weight;
                weighted_sum_f1 += f1 * weight;
                total_instances += weight;
            }

            if total_instances > 0.0 {
                println!(
                    "      weighted-avg :: P={:.2}% R={:.2}% F1={:.2}%",
                    (weighted_sum_p / total_instances) * 100.0,
                    (weighted_sum_r / total_instances) * 100.0,
                    (weighted_sum_f1 / total_instances) * 100.0
                );
            }
        }

        println!("\n  [*] Top 20 correctly inferred function labels:");
        if correct_func > 0 {
            let mut func_stats = func_label_stats.into_iter().collect::<Vec<_>>();
            func_stats.sort_by_key(|&(_, count)| Reverse(count));
            for (label, count) in func_stats.iter().take(20) {
                let percentage = (*count as f64 / correct_func as f64) * 100.0;
                println!("      {:<40} :: {:>5} ({:>6.2}%)", label, count, percentage);
            }
        } else {
            println!("      no correct function predictions (function prediction likely disabled)");
        }

        println!("\n  [*] Top 10 correctly inferred types:");
        let mut stats = type_stats.into_iter().collect::<Vec<_>>();
        stats.sort_by_key(|&(_, count)| Reverse(count));
        for (type_name, count) in stats.iter().take(10) {
            let percentage = (*count as f64 / correct_var as f64) * 100.0;
            println!(
                "      {:<40} :: {:>5} ({:>6.2}%)",
                type_name, count, percentage
            );
        }

        if train_functions.is_some() {
            println!("\n  [*] Train/Test Split Analysis (body hash comparison):");

            let in_train_accuracy = if in_train_total > 0 {
                (in_train_correct as f64 / in_train_total as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "      In-train accuracy      :: {:.2}% [{}/{}]",
                in_train_accuracy, in_train_correct, in_train_total
            );

            let out_train_accuracy = if out_train_total > 0 {
                (out_train_correct as f64 / out_train_total as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "      Out-of-train accuracy  :: {:.2}% [{}/{}]",
                out_train_accuracy, out_train_correct, out_train_total
            );

            let coverage = if written_var > 0 {
                (in_train_total as f64 / written_var as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "      Train set coverage     :: {:.2}% of test functions (body content match)",
                coverage
            );
        }
    }

    println!(
        "\n  [*] Processing speed         :: {:.2} ms/func ({:.1} functions/sec)",
        processing_speed_ms_per_func,
        1000.0 / processing_speed_ms_per_func
    );
    if let Some(d) = raw_inference_duration {
        println!(
            "      Total runtime (inference) :: {}",
            humantime::format_duration(d)
        );
    }
    if let Some(start) = absolute_start {
        println!(
            "      Total runtime (absolute)  :: {}",
            humantime::format_duration(start.elapsed())
        );
    }

    Ok(())
}
