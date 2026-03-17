use std::any::Any;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use xtride::entry::LabelType;
use xtride::vocab::Vocab;

mod corpus;
mod db_creation;
mod eval;
use corpus::Corpus;
use eval::evaluate;

pub enum XtrideCommands {
    /// Build vocabulary from corpus
    BuildVocab {
        /// Path to input.jsonl (STRIDE format)
        input: PathBuf,
        /// Path to output.vocab
        output: PathBuf,
        /// Which type of vocab to build
        type_: LabelType,
    },

    /// Build n-gram database
    BuildDb {
        /// Path to input.jsonl (STRIDE format)
        input: PathBuf,
        /// Path to input.vocab (regular types)
        vocab: PathBuf,
        /// Optional path to input.fn.vocab (function signatures)
        fn_vocab: Option<PathBuf>,
        /// Path to output.db
        output: PathBuf,
        /// Which type of db to build
        type_: LabelType,
        /// Ngram size
        size: usize,
        /// Number of top-k targets to store
        topk: usize,
        /// Use flanking ngrams
        flanking: bool,
        /// Enable full strip mode (legacy)
        strip: bool,
    },

    /// Build all n-gram databases of different sizes
    BuildAllDbs {
        /// Path to input.jsonl (STRIDE format)
        input: PathBuf,
        /// Path to input.vocab
        vocab: PathBuf,
        /// Output directory for the databases
        output_dir: PathBuf,
        /// Which type of db to build
        type_: LabelType,
        /// Number of top-k targets to store
        topk: usize,
        /// Use flanking ngrams
        flanking: bool,
        /// Enable full strip mode (legacy)
        strip: bool,
    },

    Evaluate {
        /// Path to input.jsonl (STRIDE format)
        input: PathBuf,
        /// Path to input.vocab
        vocab: PathBuf,
        /// Path to output.csv
        output: PathBuf,
        /// Directory containing ngram databases
        db_dir: PathBuf,
        /// Which type of labels to use
        type_: LabelType,
        /// Use flanking ngrams
        flanking: bool,
        /// Enable full strip mode (legacy)
        strip: bool,
        /// Path to training corpus for in/out-of-train analysis
        consider_train: Option<PathBuf>,
        /// Aggregate function predictions by address across the run
        aggregate_funcs: bool,
        /// Score threshold for predictions (discard below); set to 1.0 to disable dropping
        threshold: f64,
        /// Print threshold-filtering stats (counts + score ranges)
        threshold_debug: bool,
        /// Print a threshold sweep summary table (coverage + selective risk) for common cutoffs
        threshold_sweep: bool,
    },

    /// Create a unified dataset from multiple JSONL files
    CreateDataset {
        /// Directory containing input JSONL files
        input: PathBuf,
        /// Output directory for train/test splits
        output: PathBuf,
    },

    /// Convert a directory of HDF5 (paper-interoperability) databases to rkyv format
    ConvertDbsToRkyv {
        /// Directory containing input HDF5 database files
        input: PathBuf,
        /// Output directory for rkyv databases
        output: PathBuf,
    },

    /// Print database hashes for integrity checking (to be pasted into lib/db.rs)
    PrintDbHashes {
        /// Directory containing rkyv database files
        input: PathBuf,
    },
}

pub(crate) fn append_file_stem_suffix(path: impl AsRef<Path>, suffix: &str) -> PathBuf {
    let path = path.as_ref();
    let Some(file_name) = path.file_name() else {
        return path.to_path_buf();
    };

    let mut new_file_name = OsString::from(path.file_stem().unwrap_or(file_name));
    new_file_name.push(suffix);

    if let Some(extension) = path.extension() {
        new_file_name.push(".");
        new_file_name.push(extension);
    }

    let mut output = path.to_path_buf();
    output.set_file_name(new_file_name);
    output
}

fn required_arg<T>(matches: &ArgMatches, name: &str) -> Result<T>
where
    T: Any + Clone + Send + Sync + 'static,
{
    matches
        .get_one::<T>(name)
        .cloned()
        .ok_or_else(|| anyhow!("missing required argument: {name}"))
}

fn main() -> Result<()> {
    let cmd = Command::new("xtride-cli")
        .version(env!("CARGO_PKG_VERSION"))
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("build-vocab")
                .arg(
                    Arg::new("input")
                        .help("Path to input.jsonl (STRIDE format)")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Path to output.vocab")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("type_")
                        .help("Which type of vocab to build")
                        .short('t')
                        .long("type")
                        .default_value("type")
                        .value_parser(value_parser!(LabelType)),
                ),
        )
        .subcommand(
            Command::new("build-db")
                .arg(
                    Arg::new("input")
                        .help("Path to input.jsonl (STRIDE format)")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("vocab")
                        .help("Path to input.vocab (regular types)")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("fn_vocab")
                        .help("Optional path to input.fn.vocab (function signatures)")
                        .long("fn-vocab")
                        .required(false)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Path to output.db")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("type_")
                        .help("Which type of db to build")
                        .short('t')
                        .long("type")
                        .default_value("type")
                        .value_parser(value_parser!(LabelType)),
                )
                .arg(
                    Arg::new("size")
                        .help("Ngram size")
                        .short('s')
                        .long("size")
                        .default_value("10")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("topk")
                        .help("Number of top-k targets to store")
                        .short('k')
                        .long("topk")
                        .default_value("5")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("flanking")
                        .help("Use flanking ngrams")
                        .short('f')
                        .long("flanking")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("strip")
                        .help("Enable full strip mode (legacy)")
                        .long("strip")
                        .action(ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("build-all-dbs")
                .arg(
                    Arg::new("input")
                        .help("Path to input.jsonl (STRIDE format)")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("vocab")
                        .help("Path to input.vocab")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("output_dir")
                        .help("Output directory for the databases")
                        .short('o')
                        .long("output-dir")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("type_")
                        .help("Which type of db to build")
                        .short('t')
                        .long("type")
                        .default_value("type")
                        .value_parser(value_parser!(LabelType)),
                )
                .arg(
                    Arg::new("topk")
                        .help("Number of top-k targets to store")
                        .short('k')
                        .long("topk")
                        .default_value("5")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("flanking")
                        .help("Use flanking ngrams")
                        .short('f')
                        .long("flanking")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("strip")
                        .help("Enable full strip mode (legacy)")
                        .long("strip")
                        .action(ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("evaluate")
                .arg(
                    Arg::new("input")
                        .help("Path to input.jsonl (STRIDE format)")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("vocab")
                        .help("Path to input.vocab")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Path to output.csv")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("db_dir")
                        .help("Directory containing ngram databases")
                        .short('d')
                        .long("db-dir")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("type_")
                        .help("Which type of labels to use")
                        .short('t')
                        .long("type")
                        .default_value("type")
                        .value_parser(value_parser!(LabelType)),
                )
                .arg(
                    Arg::new("flanking")
                        .help("Use flanking ngrams")
                        .short('f')
                        .long("flanking")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("strip")
                        .help("Enable full strip mode (legacy)")
                        .long("strip")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("consider_train")
                        .help("Path to training corpus for in/out-of-train analysis")
                        .long("consider-train")
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("aggregate_funcs")
                        .help("Aggregate function predictions by address across the run")
                        .long("aggregate-funcs")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("threshold")
                        .help("Score threshold for predictions (discard below); set to 1.0 to disable dropping")
                        .long("threshold")
                        .default_value("1.0")
                        .value_parser(value_parser!(f64)),
                )
                .arg(
                    Arg::new("threshold_debug")
                        .help("Print threshold-filtering stats (counts + score ranges)")
                        .long("threshold-debug")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("threshold_sweep")
                        .help("Print a threshold sweep summary table (coverage + selective risk) for common cutoffs")
                        .long("threshold-sweep")
                        .action(ArgAction::SetTrue),
                ),
        )
        .subcommand(
            Command::new("create-dataset")
                .arg(
                    Arg::new("input")
                        .help("Directory containing input JSONL files")
                        .short('i')
                        .long("input")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Output directory for train/test splits")
                        .short('o')
                        .long("output")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                ),
        )
        .subcommand(
            Command::new("convert-dbs-to-rkyv")
                .about("Convert a directory of HDF5 databases to rkyv format")
                .arg(
                    Arg::new("input")
                        .help("Directory containing input HDF5 database files")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("output")
                        .help("Output directory for rkyv databases")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                ),
        )
        .subcommand(
            Command::new("print-db-hashes")
                .about("Print database hashes for integrity checking")
                .arg(
                    Arg::new("input")
                        .help("Directory containing rkyv database files")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                ),
        );

    let matches = cmd.get_matches();

    let command_instance = match matches.subcommand() {
        Some(("build-vocab", sub_matches)) => XtrideCommands::BuildVocab {
            input: required_arg(sub_matches, "input")?,
            output: required_arg(sub_matches, "output")?,
            type_: required_arg(sub_matches, "type_")?,
        },
        Some(("build-db", sub_matches)) => XtrideCommands::BuildDb {
            input: required_arg(sub_matches, "input")?,
            vocab: required_arg(sub_matches, "vocab")?,
            fn_vocab: sub_matches.get_one::<PathBuf>("fn_vocab").cloned(),
            output: required_arg(sub_matches, "output")?,
            type_: required_arg(sub_matches, "type_")?,
            size: required_arg(sub_matches, "size")?,
            topk: required_arg(sub_matches, "topk")?,
            flanking: sub_matches.get_flag("flanking"),
            strip: sub_matches.get_flag("strip"),
        },
        Some(("build-all-dbs", sub_matches)) => XtrideCommands::BuildAllDbs {
            input: required_arg(sub_matches, "input")?,
            vocab: required_arg(sub_matches, "vocab")?,
            output_dir: required_arg(sub_matches, "output_dir")?,
            type_: required_arg(sub_matches, "type_")?,
            topk: required_arg(sub_matches, "topk")?,
            flanking: sub_matches.get_flag("flanking"),
            strip: sub_matches.get_flag("strip"),
        },
        Some(("evaluate", sub_matches)) => XtrideCommands::Evaluate {
            input: required_arg(sub_matches, "input")?,
            vocab: required_arg(sub_matches, "vocab")?,
            output: required_arg(sub_matches, "output")?,
            db_dir: required_arg(sub_matches, "db_dir")?,
            type_: required_arg(sub_matches, "type_")?,
            flanking: sub_matches.get_flag("flanking"),
            strip: sub_matches.get_flag("strip"),
            consider_train: sub_matches.get_one::<PathBuf>("consider_train").cloned(),
            aggregate_funcs: sub_matches.get_flag("aggregate_funcs"),
            threshold: required_arg(sub_matches, "threshold")?,
            threshold_debug: sub_matches.get_flag("threshold_debug"),
            threshold_sweep: sub_matches.get_flag("threshold_sweep"),
        },
        Some(("create-dataset", sub_matches)) => XtrideCommands::CreateDataset {
            input: required_arg(sub_matches, "input")?,
            output: required_arg(sub_matches, "output")?,
        },
        Some(("convert-dbs-to-rkyv", sub_matches)) => XtrideCommands::ConvertDbsToRkyv {
            input: required_arg(sub_matches, "input")?,
            output: required_arg(sub_matches, "output")?,
        },
        Some(("print-db-hashes", sub_matches)) => XtrideCommands::PrintDbHashes {
            input: required_arg(sub_matches, "input")?,
        },
        _ => return Err(anyhow!("missing subcommand")),
    };

    match command_instance {
        XtrideCommands::BuildVocab {
            input,
            output,
            type_,
        } => {
            println!("[*] Building vocabulary from: {}", input.display());
            let corpus = Corpus::new(input, false);
            let (vocab, fn_vocab) = db_creation::build_vocab(&corpus, type_)?;

            println!("    -> Saving regular vocab to: {}", output.display());
            vocab.save(&output)?;

            let fn_output = append_file_stem_suffix(&output, ".fn");

            println!("    -> Saving function vocab to: {}", fn_output.display());
            fn_vocab.save(&fn_output)?;
            println!("[*] Vocabulary building complete.");
        }

        XtrideCommands::BuildDb {
            input,
            vocab: vocab_path,
            fn_vocab: fn_vocab_opt_path,
            output,
            type_,
            size,
            topk,
            flanking,
            strip,
        } => {
            println!("[*] Building n-gram database (size: {})", size);
            println!("    Corpus: {}", input.display());
            println!("    Regular Vocab: {}", vocab_path.display());

            let corpus = Corpus::new(&input, strip);
            let vocab = Vocab::load(&vocab_path)?;

            // load fn_vocab if path is provided, otherwise use regular vocab
            let fn_vocab = match fn_vocab_opt_path {
                Some(path) => {
                    println!("    Function Vocab: {}", path.display());
                    Vocab::load(path)?
                }
                None => {
                    println!("    Function Vocab: Not provided, using regular vocabulary.");
                    Vocab::load(vocab_path)?
                }
            };

            let db = db_creation::build_ngram_db_multi(
                &corpus, &vocab, &fn_vocab, type_, size, topk, flanking,
            )?;

            println!("    -> Saving DB to: {}", output.display());
            db.save_rkyv(output)?;
            println!("[*] Database building complete.");
        }

        XtrideCommands::BuildAllDbs { .. } => db_creation::build_all_dbs(&command_instance)?,

        XtrideCommands::Evaluate { .. } => {
            evaluate(&command_instance)?;
        }

        XtrideCommands::CreateDataset { input, output } => {
            println!("[*] Creating dataset from {}", input.display());
            db_creation::create_dataset(&input, &output)?;
            println!("[+] Dataset created successfully.");
        }

        XtrideCommands::ConvertDbsToRkyv { input, output } => {
            println!(
                "[*] Converting HDF5 databases in {} to rkyv format in {}",
                input.display(),
                output.display()
            );
            db_creation::convert_dir_to_rkyv(input, output)?;
        }

        XtrideCommands::PrintDbHashes { input } => {
            db_creation::print_db_hashes(&input)?;
        }
    }

    Ok(())
}
