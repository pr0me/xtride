use std::path::Path;
use std::time::Duration;

use bias_core::cio::TypeInfoDB;
use bias_core::decompiler::{Decompiler, DecompilerAst};
use bias_core::ir::{Term, Type};
use bias_core::prelude::{Function, FunctionId, Identifiable, Project};
use bias_core::symbols::SymbolTable;
use hashbrown::HashMap;
use thiserror::Error;

use crate::entry::{Entry, LabelType};
use crate::predict::{Prediction, Predictor};
use crate::types::{is_primitive, TypeFactory};

#[derive(Debug, Error)]
pub enum RetyperError {
    #[error("type database `{path}` could not be loaded: {source}")]
    DatabaseLoad { path: String, source: anyhow::Error },
    #[error("platform directorymisformed: {0}")]
    InvalidPlatformDirectory(anyhow::Error),
    #[error("invalid address size specified for `arch_bits`: {0}")]
    InvalidArchBits(u32),
    #[error(transparent)]
    Predictor(anyhow::Error),
    #[error(transparent)]
    Decompiler(anyhow::Error),
}

impl RetyperError {
    pub fn database_load_with<M>(path: impl Into<String>, msg: M) -> Self
    where
        M: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        Self::DatabaseLoad {
            path: path.into(),
            source: anyhow::Error::msg(msg),
        }
    }

    pub fn predictor<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Predictor(anyhow::Error::new(e))
    }

    pub fn decompiler<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Decompiler(anyhow::Error::new(e))
    }
}

/// Configuration of the type application process
#[derive(Debug, Clone, Copy)]
pub struct RetyperConfig {
    /// Current architecture's word size, used for type construction
    arch_bits: u32,
    /// Confidence Score Cutoff. 0 signals no threshold.
    threshold: f64,
    /// Should common primitive types (uint{N}, int{N}, char (but not char pointers)) be skipped
    skip_primitives: bool,
    /// Whether to use flanking tokens in n-gram prediction
    flanking: bool,
    /// Enables n-gram-based prediction of function types (experimental)
    func_pred: bool,
}

impl RetyperConfig {
    /// Creates a new configuration with default values.
    /// By default, primitive types are skipped, flanking is enabled, and function prediction is disabled.
    pub fn new() -> Self {
        Self {
            arch_bits: 0,
            threshold: 0.0,
            skip_primitives: true,
            flanking: true,
            func_pred: false,
        }
    }

    pub fn arch_bits(&self) -> u32 {
        self.arch_bits
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn skip_primitives(&self) -> bool {
        self.skip_primitives
    }

    pub fn flanking(&self) -> bool {
        self.flanking
    }

    pub fn func_pred(&self) -> bool {
        self.func_pred
    }

    pub fn with_arch_bits(mut self, arch_bits: u32) -> Self {
        self.arch_bits = arch_bits;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets whether primitive types should be skipped.
    pub fn with_skip_primitives(mut self, skip: bool) -> Self {
        self.skip_primitives = skip;
        self
    }

    /// Sets whether to use flanking tokens in n-gram prediction.
    pub fn with_flanking(mut self, flanking: bool) -> Self {
        self.flanking = flanking;
        self
    }

    pub fn with_func_pred(mut self, func_pred: bool) -> Self {
        self.func_pred = func_pred;
        self
    }

    pub fn set_arch_bits(&mut self, arch_bits: u32) {
        self.arch_bits = arch_bits;
    }
}

#[derive(Debug, Clone)]
pub struct RetypedVar {
    name: String,
    new_type: Term<Type>,
}

impl RetypedVar {
    pub fn new(name: impl Into<String>, new_type: Term<Type>) -> Self {
        Self {
            name: name.into(),
            new_type,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn new_type(&self) -> &Term<Type> {
        &self.new_type
    }
}

#[derive(Debug, Clone)]
pub struct TypedDecompilation {
    before: DecompilerAst,
    after: Option<DecompilerAst>,
    types_applied: Option<Vec<RetypedVar>>,
}

impl TypedDecompilation {
    pub(crate) fn new(
        before: DecompilerAst,
        after: Option<DecompilerAst>,
        types_applied: Option<Vec<RetypedVar>>,
    ) -> Self {
        Self {
            before,
            after,
            types_applied,
        }
    }

    pub fn before(&self) -> &DecompilerAst {
        &self.before
    }

    pub fn after(&self) -> Option<&DecompilerAst> {
        self.after.as_ref()
    }

    pub fn types_applied(&self) -> Option<&[RetypedVar]> {
        self.types_applied.as_deref()
    }
}

/// Necessary resources for Type Inference with XTRIDE.
/// Globally initialising this once is sufficient, a reference can be then used by
/// `RetyperProjectContext` as inference only requires unmutable access.
#[derive(Debug, Clone)]
pub struct RetyperResources {
    config: RetyperConfig,
    predictor32: Option<Predictor>,
    predictor64: Option<Predictor>,
    type_factory: TypeFactory,
}

impl RetyperResources {
    const XTRIDE_THRESHOLD: f64 = 0.3;

    fn create_base(type_db: TypeInfoDB) -> Self {
        Self {
            config: RetyperConfig::new(),
            predictor32: None,
            predictor64: None,
            type_factory: TypeFactory::new(type_db),
        }
    }

    pub fn new(predictor32: Predictor, predictor64: Predictor, type_db: TypeInfoDB) -> Self {
        let mut retyper = Self::create_base(type_db);
        retyper.predictor32 = Some(predictor32);
        retyper.predictor64 = Some(predictor64);
        retyper
    }

    pub fn new_32(predictor: Predictor, type_db: TypeInfoDB) -> Self {
        let mut retyper = Self::create_base(type_db);
        retyper.predictor32 = Some(predictor);
        retyper
    }

    pub fn new_64(predictor: Predictor, type_db: TypeInfoDB) -> Self {
        let mut retyper = Self::create_base(type_db);
        retyper.predictor64 = Some(predictor);
        retyper
    }

    pub fn new_with_config(
        predictor: Predictor,
        type_db: TypeInfoDB,
        config: RetyperConfig,
    ) -> Result<Self, RetyperError> {
        let mut retyper = Self::create_base(type_db);
        retyper.config = config;

        match config.arch_bits {
            32 => retyper.predictor32 = Some(predictor),
            64 => retyper.predictor64 = Some(predictor),
            bits => return Err(RetyperError::InvalidArchBits(bits)),
        }

        Ok(retyper)
    }

    fn try_load_predictor(
        platform_dbs_root: impl AsRef<Path>,
        arch_bits: u32,
        flanking: bool,
    ) -> Result<Predictor, RetyperError> {
        let vocab_base_name = format!("types_{}", arch_bits);
        let vocab_path = platform_dbs_root
            .as_ref()
            .join(&vocab_base_name)
            .with_extension("vocab");
        let fn_vocab_path = platform_dbs_root
            .as_ref()
            .join(format!("{}.fn.vocab", vocab_base_name));
        let db_dir = platform_dbs_root
            .as_ref()
            .join("ngram-dbs")
            .join(arch_bits.to_string());

        if !vocab_path.exists() || !db_dir.exists() {
            return Err(RetyperError::InvalidPlatformDirectory(anyhow::anyhow!(
                "{}-bit predictor resources not found or incomplete (vocab path: {}, db_dir: {})",
                arch_bits,
                vocab_path.display(),
                db_dir.display()
            )));
        }

        let mut builder = Predictor::builder()
            .vocab(vocab_path)
            .db_dir(db_dir)
            .label_type(LabelType::Type)
            .flanking(flanking);

        if fn_vocab_path.exists() {
            builder = builder.fn_vocab(fn_vocab_path);
        } else {
            log::warn!(
                "{}-bit function vocabulary not found at {}, proceeding without it.",
                arch_bits,
                fn_vocab_path.display()
            );
        };

        builder.build().map_err(RetyperError::predictor)
    }

    /// Creates new RetyperResources from a platform directory.
    /// The directory is expected to hold n-gram databases and vocabularies of
    /// possibly inferred types for both 32- and 64-bit contexts as well as a
    /// TypeDB with definitions for all possible outcome types.
    pub fn from_platform_dir(platform_dir: impl AsRef<Path>) -> Result<Self, RetyperError> {
        let typedb_path = platform_dir.as_ref().join("typedb").with_extension("bin");
        let typeinfo_db = TypeInfoDB::from_file(&typedb_path).map_err(|e| {
            RetyperError::database_load_with(
                typedb_path
                    .to_str()
                    .unwrap_or("path has invalid unicode")
                    .to_owned(),
                e,
            )
        })?;

        let base_config = RetyperConfig::new();

        let mut retyper = Self::create_base(typeinfo_db);
        retyper.config = base_config;

        // load 32-bit predictor
        match Self::try_load_predictor(&platform_dir, 32, base_config.flanking) {
            Ok(predictor) => retyper.predictor32 = Some(predictor),
            Err(e) => log::warn!("failed to load 32-bit predictor: {}.", e),
        }

        // load 64-bit predictor
        match Self::try_load_predictor(&platform_dir, 64, base_config.flanking) {
            Ok(predictor) => retyper.predictor64 = Some(predictor),
            Err(e) => log::warn!("failed to load 64-bit predictor: {}.", e),
        }

        if retyper.predictor32.is_none() && retyper.predictor64.is_none() {
            return Err(RetyperError::Predictor(anyhow::anyhow!(
                "no predictors could be loaded from platform directory: {}",
                platform_dir.as_ref().display()
            )));
        }

        Ok(retyper)
    }

    pub fn from_files(
        vocab_path: impl AsRef<Path>,
        fn_vocab_path: Option<impl AsRef<Path>>,
        db_dir: impl AsRef<Path>,
        typedb_path: impl AsRef<Path>,
        arch_bits: u32,
    ) -> Result<Self, RetyperError> {
        let config = RetyperConfig::new()
            .with_arch_bits(arch_bits)
            .with_threshold(Self::XTRIDE_THRESHOLD);
        Self::from_files_with_config(vocab_path, fn_vocab_path, db_dir, typedb_path, config)
    }

    pub fn from_files_with_config(
        vocab_path: impl AsRef<Path>,
        fn_vocab_path: Option<impl AsRef<Path>>,
        db_dir: impl AsRef<Path>,
        type_db_path: impl AsRef<Path>,
        config: RetyperConfig,
    ) -> Result<Self, RetyperError> {
        let mut builder = Predictor::builder()
            .vocab(vocab_path)
            .db_dir(db_dir)
            .label_type(LabelType::Type)
            .flanking(config.flanking);

        if let Some(fn_vocab) = fn_vocab_path {
            builder = builder.fn_vocab(fn_vocab);
        }

        let predictor = builder.build().map_err(RetyperError::predictor)?;

        let typeinfo_db = TypeInfoDB::from_file(&type_db_path).map_err(|e| {
            RetyperError::database_load_with(
                type_db_path
                    .as_ref()
                    .to_str()
                    .unwrap_or("path has invalid unicode"),
                e,
            )
        })?;

        let mut retyper = Self::create_base(typeinfo_db);
        retyper.config = config;

        match config.arch_bits {
            32 => retyper.predictor32 = Some(predictor),
            64 => retyper.predictor64 = Some(predictor),
            bits => {
                return Err(RetyperError::Predictor(anyhow::anyhow!(
                    "invalid architecture bit width in config: {}. cannot assign predictor.",
                    bits
                )))
            }
        }

        Ok(retyper)
    }
}

pub struct RetyperProjectContext<'a> {
    config: RetyperConfig,
    resources: &'a RetyperResources,
    symbol_table: SymbolTable,
    funcs_processed: HashMap<FunctionId, TypedDecompilation>,
}

impl<'a> RetyperProjectContext<'a> {
    pub fn new(project: &mut Project, resources: &'a RetyperResources) -> Self {
        // merge backing TypeDB into project
        project
            .type_db_mut()
            .import_new_types(resources.type_factory.type_info());

        // create project-specific config
        let mut project_config = resources.config.to_owned();
        project_config.set_arch_bits(project.lifter().address_bits());

        RetyperProjectContext {
            config: project_config,
            resources,
            symbol_table: project.symbols().to_owned(),
            funcs_processed: HashMap::new(),
        }
    }

    pub fn enable_func_pred(&mut self) {
        self.config.func_pred = true;
    }

    /// Applies type predictions to the decompiler for a specific function.
    /// Applies heuristics to reduce false positives.
    /// Returns the result of the decompilation before and after inferring types.
    pub fn try_get_conservative_before_after<'b>(
        &mut self,
        decompiler: &mut Decompiler<'b>,
        func: &'b Function,
        timeout: impl Into<Option<Duration>>,
    ) -> Result<TypedDecompilation, RetyperError> {
        let timeout = timeout.into();

        // check if we already processed this function
        if let Some(prev_result) = self.funcs_processed.get(&func.id()) {
            return Ok(prev_result.to_owned());
        }

        // get the initial AST
        let before = decompiler
            .try_decompile_ast(func, timeout)
            .map_err(RetyperError::decompiler)?;

        // create an entry from the AST for prediction
        let mut entry = Entry::from(&before);

        let current_predictor = match self.config.arch_bits {
            32 => self.resources.predictor32.as_ref(),
            64 => self.resources.predictor64.as_ref(),
            bits => {
                return Err(RetyperError::InvalidArchBits(bits));
            }
        };

        let current_predictor = match current_predictor {
            Some(p) => p,
            None => {
                return Err(RetyperError::Predictor(anyhow::anyhow!(
                    "predictor for {}-bit architecture not available/loaded",
                    self.config.arch_bits
                )));
            }
        };

        let predictions = current_predictor
            .predict_n(&mut entry, 5)
            .unwrap_or_else(|e| {
                log::warn!(
                    "prediction failed, continuing with empty predictions: {}",
                    e
                );
                HashMap::new()
            });

        let (variable_applications, function_applications) =
            self.get_filtered_predictions(predictions);

        let mut types_applied = Vec::new();
        self.try_apply_variable_predictions(
            decompiler,
            func,
            variable_applications,
            &mut types_applied,
        );

        if self.config.func_pred {
            self.try_apply_function_predictions(
                decompiler,
                function_applications,
                &mut types_applied,
            );
        }

        if types_applied.is_empty() {
            return Ok(TypedDecompilation::new(before, None, Some(types_applied)));
        }

        // decompile again to get the updated AST
        let after = decompiler
            .try_decompile_ast(func, timeout)
            .map_err(RetyperError::decompiler)?;

        let mut heuristic_triggered = None;
        if before
            .source()
            .replace(char::is_whitespace, "")
            .eq(&after.source().replace(char::is_whitespace, ""))
        {
            heuristic_triggered = Some("no changes");
        } else if after.source().contains("unaff") {
            heuristic_triggered = Some("unaffected registers");
        } else if after.source().contains("/* WARNING") && !before.source().contains("/* WARNING") {
            heuristic_triggered = Some("new warnings");
        }

        if let Some(reason) = heuristic_triggered {
            log::warn!(
                "second decompilation result discarded; heuristic triggered: {}",
                reason
            );

            let res = TypedDecompilation::new(before, None, Some(types_applied));

            self.funcs_processed.insert(func.id(), res.to_owned());
            return Ok(res);
        }

        let res = TypedDecompilation::new(before, Some(after), Some(types_applied));
        self.funcs_processed.insert(func.id(), res.to_owned());

        Ok(res)
    }

    pub fn get_conservative_before_after<'b>(
        &mut self,
        decompiler: &mut Decompiler<'b>,
        func: &'b Function,
    ) -> Result<TypedDecompilation, RetyperError> {
        self.try_get_conservative_before_after(decompiler, func, None)
    }

    fn get_filtered_predictions(
        &self,
        predictions: HashMap<String, Vec<Prediction>>,
    ) -> (Vec<Prediction>, Vec<Prediction>) {
        let (function_preds, variable_preds) = predictions.into_iter().partition::<HashMap<
            String,
            Vec<Prediction>,
        >, _>(
            |(_, preds)| preds.get(0).map_or(false, |p| p.is_function()),
        );

        let function_applications = function_preds
            .into_values()
            .filter_map(|preds| preds.into_iter().next())
            .filter(|pred| pred.count() > 1 && pred.score() > 0.6)
            .collect::<Vec<_>>();

        let variable_applications = variable_preds
            .into_iter()
            .filter_map(|(curr_var_name, top_preds)| {
                // discard if var has non-generic name (expressive names on vars are an indicator that these already got types assigned)
                if !has_generic_name(&curr_var_name) {
                    return None;
                }

                let chosen_pred = top_preds.get(0)?;
                let curr_type_pred = chosen_pred.type_pred()?;

                // discard if best choice is below threshold
                if self.config.threshold > 0.0 && chosen_pred.score() < self.config.threshold {
                    return None;
                }

                // skip primitives if configured
                if self.config.skip_primitives && is_primitive(curr_type_pred) {
                    return None;
                }

                Some(chosen_pred.to_owned())
            })
            .collect::<Vec<_>>();

        (variable_applications, function_applications)
    }

    fn try_apply_variable_predictions<'b>(
        &self,
        decompiler: &mut Decompiler<'b>,
        func: &'b Function,
        variable_applications: Vec<Prediction>,
        types_applied: &mut Vec<RetypedVar>,
    ) {
        for pred in variable_applications {
            let Some(curr_type) = pred.type_pred() else {
                continue;
            };

            if let Err(e) = decompiler.set_function_variable_type_at(
                u64::from(func.address()),
                pred.var(),
                curr_type,
            ) {
                log::warn!(
                    "failed to set type for variable {} to {}: {}",
                    pred.var(),
                    curr_type,
                    e
                );
            } else {
                log::info!("applied new type to {} :: {}", pred.var(), curr_type);
                types_applied.push(RetypedVar::new(pred.var().to_owned(), curr_type.to_owned()));
            }
        }
    }

    /// Inserts new function symbols into the decompiler context without modifying the `Project`.
    /// This fails if:
    /// - the function is a valid symbol already
    /// - the function signature is invalid
    /// - the part behind 'sub_' cannot be converted into a valid address
    fn try_apply_function_predictions(
        &self,
        decompiler: &mut Decompiler,
        function_applications: Vec<Prediction>,
        types_applied: &mut Vec<RetypedVar>,
    ) {
        for pred in function_applications {
            if self.symbol_table.contains(pred.var()) {
                continue;
            }

            let Some(addr) = u64::from_str_radix(pred.var().get(4..).unwrap_or(""), 16).ok() else {
                log::warn!(
                    "failed to parse address from function var name: {} for prediction: {}",
                    pred.var(),
                    pred.pred()
                );
                continue;
            };

            let Some(curr_type) = self
                .resources
                .type_factory
                .type_db()
                .get_type_for(pred.pred(), self.config.arch_bits())
            else {
                continue;
            };

            if let Err(e) = decompiler.add_function_symbol_with(
                pred.pred(),
                addr,
                &curr_type,
                curr_type.is_variadic_function(),
            ) {
                log::error!(
                    "failed to set signature for function {} (0x{:x}) to {}: {}",
                    pred.var(),
                    addr,
                    curr_type,
                    e
                );
            } else {
                log::info!(
                    "set function signature for `{}` (0x{:x}) -> {} [score::{}]",
                    pred.var(),
                    addr,
                    pred.pred(),
                    pred.score()
                );

                types_applied.push(RetypedVar::new(pred.var().to_owned(), curr_type.to_owned()));
            }
        }
    }
}

fn has_generic_name(var: &str) -> bool {
    var.starts_with("var") || var.starts_with("param") || var.starts_with("stack")
}

/// Public Wrapper to agnostically attempt decompilation.
/// If no Retyping instance is available, the function performs decompilation as usual.
pub fn try_get_decompilation<'a>(
    decompiler: &mut Decompiler<'a>,
    retyper: &mut Option<RetyperProjectContext>,
    func: &'a Function,
    timeout: impl Into<Option<Duration>>,
) -> Result<TypedDecompilation, RetyperError> {
    if let Some(retyper) = retyper {
        retyper.try_get_conservative_before_after(decompiler, func, timeout)
    } else {
        let default = decompiler
            .try_decompile_ast(func, timeout)
            .map_err(RetyperError::decompiler)?;
        let res = TypedDecompilation::new(default, None, None);
        Ok(res)
    }
}
