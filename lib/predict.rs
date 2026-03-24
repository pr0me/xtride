use std::cmp::Reverse;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "retyper")]
use bias_core::ir::{Term, Type};
use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use serde::Serialize;
use thiserror::Error;

use crate::db::MappedNGramDB;
use crate::entry::{Entry, Label, LabelType, Labels};
use crate::ngram::{LocalVar, NgramIndex, VarKind};
use crate::vocab::Vocab;

#[derive(Debug, Default)]
pub struct PredictorBuilder {
    vocab_path: Option<PathBuf>,
    fn_vocab_path: Option<PathBuf>,
    db_dir: Option<PathBuf>,
    label_type: Option<LabelType>,
    flanking: bool,
}

impl PredictorBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn vocab(mut self, path: impl AsRef<Path>) -> Self {
        self.vocab_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn fn_vocab(mut self, path: impl AsRef<Path>) -> Self {
        self.fn_vocab_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn db_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.db_dir = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn label_type(mut self, label_type: LabelType) -> Self {
        self.label_type = Some(label_type);
        self
    }

    pub fn flanking(mut self, flanking: bool) -> Self {
        self.flanking = flanking;
        self
    }

    pub fn build(self) -> Result<Predictor, PredictorError> {
        let vocab_path = self
            .vocab_path
            .ok_or_else(|| PredictorError::configuration_with("vocabulary path not provided"))?;
        let db_dir = self
            .db_dir
            .ok_or_else(|| PredictorError::configuration_with("database directory not provided"))?;
        let label_type = self
            .label_type
            .ok_or_else(|| PredictorError::configuration_with("label type not provided"))?;

        let vocab = Vocab::load(vocab_path).map_err(PredictorError::vocab_load_with)?;
        let fn_vocab = self
            .fn_vocab_path
            .map(|path| Vocab::load(path).map_err(PredictorError::vocab_load_with))
            .transpose()?;

        // collect and load all .rkyv files from directory
        let db_paths = fs::read_dir(db_dir)
            .map_err(|e| {
                log::error!("failed to read database directory: {}", e);
                PredictorError::db_load_with(format!("failed to read database directory: {}", e))
            })?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("rkyv"))
            .collect::<Vec<_>>();

        let mut dbs = db_paths
            .iter()
            .map(|path| MappedNGramDB::load(path).map(Arc::new))
            .collect::<Result<Vec<Arc<_>>, _>>()
            .map_err(|e| {
                log::error!("failed to load n-gram databases: {}", e);
                PredictorError::DbLoad(e.into())
            })?;

        for db in &dbs {
            if db.size().is_empty() {
                return Err(PredictorError::db_load_with(
                    "database has corrupted size information",
                ));
            }
        }

        // ngram dbs must be sorted and queried large to small
        dbs.sort_by_key(|db| Reverse(db.size()[0]));

        Ok(Predictor {
            vocab,
            fn_vocab,
            dbs,
            label_type,
            flanking: self.flanking,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Prediction {
    var_name: String,
    kind: VarKind,
    pred: String,
    #[cfg(feature = "retyper")]
    type_pred: Option<Term<Type>>,
    label: String,
    count: usize,
    score: f64,
}

impl Prediction {
    pub fn var(&self) -> &str {
        &self.var_name
    }

    pub fn pred(&self) -> &str {
        &self.pred
    }

    #[cfg(feature = "retyper")]
    pub fn type_pred(&self) -> Option<&Term<Type>> {
        self.type_pred.as_ref()
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn score(&self) -> f64 {
        self.score
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

#[derive(Error, Debug)]
pub enum PredictorError {
    #[error("predictor configuration error: {0}")]
    Configuration(anyhow::Error),
    #[error("failed to load ngram database: {0}")]
    DbLoad(anyhow::Error),
    #[error("failed to load vocabulary: {0}")]
    VocabLoad(anyhow::Error),
    #[error("failed to generate and aggregate n-grams")]
    Aggregator,
}

impl PredictorError {
    pub fn configuration_with<M>(msg: M) -> Self
    where
        M: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        Self::Configuration(anyhow::Error::msg(msg))
    }

    pub fn db_load_with<M>(msg: M) -> Self
    where
        M: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        Self::DbLoad(anyhow::Error::msg(msg))
    }

    pub fn vocab_load_with<M>(msg: M) -> Self
    where
        M: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        Self::VocabLoad(anyhow::Error::msg(msg))
    }
}

#[derive(Debug, Clone)]
pub struct Observation<'a> {
    name: &'a str,
    freq: usize,
}

impl<'a> Observation<'a> {
    pub fn new(name: &'a str, freq: usize) -> Self {
        Self { name, freq }
    }

    pub fn name(&self) -> &str {
        self.name
    }

    pub fn freq(&self) -> usize {
        self.freq
    }
}

#[derive(Debug, Clone)]
pub struct DbResult<'a> {
    n_gram: NgramIndex,
    n: usize,
    global_occurrences: usize,
    observations: Vec<Observation<'a>>,
}

impl<'a> DbResult<'a> {
    pub fn new(
        n_gram: NgramIndex,
        n: usize,
        global_occurrences: usize,
        observations: Vec<Observation<'a>>,
    ) -> Self {
        Self {
            n_gram,
            n,
            global_occurrences,
            observations,
        }
    }

    pub fn n_gram(&self) -> &NgramIndex {
        &self.n_gram
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn global_occurrences(&self) -> usize {
        self.global_occurrences
    }

    pub fn observations(&self) -> &[Observation<'a>] {
        &self.observations
    }
}

#[derive(Debug)]
pub struct AggregatedScores<'b> {
    /// Aggregated scores: var(String) -> (pred_name -> score)
    scores: HashMap<LocalVar<'b>, HashMap<String, f64>>,
    /// Number of n-gram contexts per Variable: var -> count
    context_counts: HashMap<String, usize>,
    /// Variable total counts: var -> count
    total_counts: HashMap<&'b str, usize>,
    /// Number of contexts for which a prediction was in the top-k DB results: ((var, pred) -> count)
    context_hits: HashMap<(String, String), usize>,
    /// Labels: var -> Label
    labels: Option<&'b Labels>,
}

impl<'b> AggregatedScores<'b> {
    pub fn new(
        scores: HashMap<LocalVar<'b>, HashMap<String, f64>>,
        context_counts: HashMap<String, usize>,
        total_counts: HashMap<&'b str, usize>,
        context_hits: HashMap<(String, String), usize>,
        labels: Option<&'b Labels>,
    ) -> Self {
        Self {
            scores,
            context_counts,
            total_counts,
            context_hits,
            labels,
        }
    }

    pub fn scores(&self) -> &HashMap<LocalVar<'b>, HashMap<String, f64>> {
        &self.scores
    }

    pub fn context_counts(&self) -> &HashMap<String, usize> {
        &self.context_counts
    }

    pub fn total_counts(&self) -> &HashMap<&'b str, usize> {
        &self.total_counts
    }

    pub fn context_hits(&self) -> &HashMap<(String, String), usize> {
        &self.context_hits
    }

    pub fn labels(&self) -> Option<&'b Labels> {
        self.labels
    }
}

#[derive(Debug, Clone)]
pub struct Predictor {
    vocab: Vocab,
    fn_vocab: Option<Vocab>,
    dbs: Vec<Arc<MappedNGramDB>>,
    label_type: LabelType,
    flanking: bool,
}

/// Primary interface for online inference from n-gram databases.
impl Predictor {
    pub fn builder() -> PredictorBuilder {
        PredictorBuilder::new()
    }

    pub fn num_dbs(&self) -> usize {
        self.dbs.len()
    }

    /// Lookup an entry's n-grams across databases.
    fn lookup_ngrams<'a, 'e>(
        &'a self,
        entry: &'e mut Entry,
        preds: &mut HashMap<NgramIndex, Option<DbResult<'a>>>,
    ) -> Result<(), PredictorError> {
        for db in &self.dbs {
            for curr_ngram in entry
                .iter_ngrams(db.size()[0], self.flanking)
                .ok_or(PredictorError::Aggregator)?
            {
                if preds.get(curr_ngram.idx()).map_or(true, |p| p.is_some()) {
                    continue;
                }

                let Some(lookup_result) = db.lookup(curr_ngram.hash()) else {
                    continue;
                };

                let mut entries = Vec::<Observation>::new();
                let target_vocab = if curr_ngram.var().is_function() {
                    self.fn_vocab.as_ref().unwrap_or(&self.vocab)
                } else {
                    &self.vocab
                };

                for db_entry in lookup_result
                    .targets()
                    .iter()
                    .filter(|entry| entry.context_count() > 0)
                {
                    if let Some(name) = target_vocab.reverse(db_entry.vocab_idx()) {
                        entries.push(Observation {
                            name,
                            freq: db_entry.context_count(),
                        });
                    }
                }

                if !entries.is_empty() {
                    preds.insert(
                        curr_ngram.idx().to_owned(),
                        Some(DbResult::new(
                            curr_ngram.idx().to_owned(),
                            db.size()[0],
                            lookup_result.global_count(),
                            entries,
                        )),
                    );
                }
            }
        }

        Ok(())
    }

    /// Perform common n-gram lookup and score aggregation.
    /// Returns maps keyed by variable name.
    fn aggregate_scores<'e>(
        &self,
        entry: &'e mut Entry,
    ) -> Result<AggregatedScores<'e>, PredictorError> {
        let mut locs = HashMap::<String, (VarKind, Vec<NgramIndex>)>::new();
        let mut preds = HashMap::<NgramIndex, Option<DbResult>>::new();

        // gather context locations for each variable
        for curr_ngram in entry
            .iter_ngrams(1, self.flanking)
            .ok_or(PredictorError::Aggregator)?
        {
            locs.entry(curr_ngram.var().name().to_owned())
                .or_insert_with(|| (curr_ngram.var().kind().to_owned(), Vec::new()))
                .1
                .push(curr_ngram.idx().to_owned());
            preds.insert(curr_ngram.idx().to_owned(), None);
        }

        self.lookup_ngrams(entry, &mut preds)?;

        let counts = entry.var_counts().ok_or(PredictorError::Aggregator)?;
        let labels_struct = entry.labels(self.label_type);

        // aggregate predictions by var
        let mut agg = HashMap::<LocalVar<'e>, HashMap<String, f64>>::new();
        let mut context_counts = HashMap::<String, usize>::new();
        let mut hit_counts = HashMap::<(String, String), usize>::new();

        for (var_name, (kind, indices)) in &locs {
            // find actual &str reference from entry to reconstruct LocalVar with proper lifetime
            let var_ref_opt = counts.keys().find(|&k| k == var_name).copied();
            let Some(var_ref) = var_ref_opt else {
                continue;
            };
            let curr_var = LocalVar::new(var_ref, kind.to_owned());

            let var_scores = agg.entry(curr_var).or_default();
            *context_counts.entry(var_name.to_owned()).or_default() = indices.len();

            for idx in indices {
                let Some(db_res) = preds.get(idx).and_then(|opt| opt.as_ref()) else {
                    continue;
                };

                let entry_total = db_res
                    .observations()
                    .iter()
                    .map(|obs| obs.freq())
                    .sum::<usize>();

                if entry_total == 0 {
                    continue;
                }

                for curr_obs in db_res.observations() {
                    let pred_rel = (var_name.to_owned(), curr_obs.name().into());
                    *hit_counts.entry(pred_rel).or_default() += 1;
                    let score = (curr_obs.freq() as f64 / entry_total as f64) * 0.5 + 0.5;
                    *var_scores.entry(curr_obs.name().into()).or_default() += score;
                }
            }
        }

        Ok(AggregatedScores::new(
            agg,
            context_counts,
            counts,
            hit_counts,
            labels_struct,
        ))
    }

    /// Inference function (Top-1).
    /// Based on original Python implementation.
    pub fn predict(&self, entry: &mut Entry) -> Result<Vec<Prediction>, PredictorError> {
        let agg_scores = self.aggregate_scores(entry)?;

        let mut final_predictions = Vec::with_capacity(agg_scores.scores().len());

        if let Some(labels_struct) = agg_scores.labels() {
            for (var, scores) in agg_scores.scores() {
                let Some(label_info) = labels_struct.get(var.name()) else {
                    continue;
                };
                if !label_info.is_human() {
                    continue;
                };

                let mut candidates = scores
                    .iter()
                    .map(|(name, score)| (name.to_owned(), *score))
                    .collect::<Vec<_>>();

                if candidates.is_empty() {
                    continue;
                }

                candidates.sort_by_key(|k| Reverse(OrderedFloat(k.1)));

                if let Some((best_name, best_score)) = candidates.first() {
                    let max_possible_score = agg_scores
                        .context_hits()
                        .get(&(var.name().to_owned(), best_name.to_owned()))
                        .copied()
                        .unwrap_or(1) as f64;
                    let base_score = 0.5 * max_possible_score as f64;
                    let normalised_score =
                        if max_possible_score > base_score && *best_score > base_score {
                            (best_score - base_score) / (max_possible_score - base_score)
                        } else {
                            0.0
                        };

                    final_predictions.push(Prediction {
                        var_name: var.name().to_owned(),
                        pred: best_name.to_owned(),
                        #[cfg(feature = "retyper")]
                        type_pred: None,
                        label: label_info.get_label().into(),
                        count: agg_scores
                            .total_counts()
                            .get(var.name())
                            .copied()
                            .unwrap_or(0),
                        score: normalised_score,
                        kind: var.kind().to_owned(),
                    });
                }
            }
        }

        Ok(final_predictions)
    }

    /// Predicts the top `n` types for each variable, grouped by variable.
    /// The top-n returned contain one non-struct type _at most_.
    /// Returns a `HashMap` where keys are variable names and values are `Vec<Prediction>` with size `n`.
    pub fn predict_n(
        &self,
        entry: &mut Entry,
        n: usize,
    ) -> Result<HashMap<String, Vec<Prediction>>, PredictorError> {
        let agg_scores = self.aggregate_scores(entry)?;

        let labels_struct = agg_scores.labels().ok_or(PredictorError::Aggregator)?;

        let grouped_predictions = agg_scores
            .scores()
            .iter()
            .filter_map(|(var, scores)| {
                let label_info = labels_struct.get(var.name())?;
                if !label_info.is_human() {
                    return None;
                }

                let mut candidates = scores
                    .iter()
                    .map(|(name, score)| (name.to_owned(), *score))
                    .collect::<Vec<_>>();

                if candidates.is_empty() {
                    return None;
                }

                candidates.sort_by_key(|k| Reverse(OrderedFloat(k.1)));

                let var_preds =
                    get_top_n_predictions_for_var(var, candidates, n, &agg_scores, label_info);

                if var_preds.is_empty() {
                    None
                } else {
                    Some((var.name().to_owned(), var_preds))
                }
            })
            .collect();

        Ok(grouped_predictions)
    }
}

/// Determines if a prediction should be added based on its kind and whether it is a struct (DIRT format).
/// This enforces the "at most one primitive" rule for variables.
fn should_add_prediction_dirt(
    kind: &VarKind,
    curr_type: &String,
    primitive_added: &mut bool,
) -> bool {
    match kind {
        VarKind::Function => true,
        VarKind::Variable => {
            // check for struct or pointer type
            let is_struct = curr_type.contains(": 6,") || curr_type.contains(": 3,");

            if is_struct {
                true
            } else if !*primitive_added {
                *primitive_added = true;
                true
            } else {
                false
            }
        }
    }
}

#[cfg(feature = "retyper")]
fn should_add_prediction_bias(
    kind: &VarKind,
    curr_type: &Option<Term<Type>>,
    primitive_added: &mut bool,
) -> bool {
    match kind {
        VarKind::Function => true,
        VarKind::Variable => {
            let is_struct = curr_type
                .as_ref()
                .map_or(false, |t| t.is_pointer_kind(Type::is_struct));

            if is_struct {
                true
            } else if !*primitive_added {
                *primitive_added = true;
                true
            } else {
                false
            }
        }
    }
}

fn get_top_n_predictions_for_var<'a, 'b>(
    var: &'b LocalVar,
    candidates: Vec<(String, f64)>,
    n: usize,
    agg_scores: &'a AggregatedScores<'b>,
    label_info: &'b Label,
) -> Vec<Prediction> {
    let mut var_preds = Vec::with_capacity(n);
    let mut primitive_added = false;

    // iterate through sorted candidates and apply struct constraint
    for (name, score) in candidates.into_iter() {
        if var_preds.len() >= n {
            break;
        }

        #[cfg(feature = "retyper")]
        let curr_type = serde_json::from_str::<Term<Type>>(&name).ok();

        #[cfg(feature = "retyper")]
        if !should_add_prediction_bias(var.kind(), &curr_type, &mut primitive_added)
            && !should_add_prediction_dirt(var.kind(), &name, &mut primitive_added)
        {
            continue;
        }

        #[cfg(not(feature = "retyper"))]
        if !should_add_prediction_dirt(var.kind(), &name, &mut primitive_added) {
            continue;
        }

        let max_possible_score = agg_scores
            .context_hits()
            .get(&(var.name().to_owned(), name.to_owned()))
            .copied()
            .unwrap_or(1) as f64;
        let base_score = 0.5 * max_possible_score as f64;

        let normalised_score = if max_possible_score > base_score && score > base_score {
            (score - base_score) / (max_possible_score - base_score)
        } else {
            0.0
        };

        var_preds.push(Prediction {
            var_name: var.name().to_owned(),
            pred: name,
            #[cfg(feature = "retyper")]
            type_pred: curr_type,
            label: label_info.get_label().into(),
            count: agg_scores
                .total_counts()
                .get(var.name())
                .copied()
                .unwrap_or(0),
            score: normalised_score,
            kind: var.kind().to_owned(),
        });
    }

    var_preds
}
