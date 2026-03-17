use std::cmp::Reverse;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use hashbrown::HashMap;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use memmap2::Mmap;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde_json::json;
use thiserror::Error;
use twox_hash::XxHash64;
use xtride::db::{DbEntry, DbError, NGramDBMulti};
use xtride::entry::{Entry, LabelType};
use xtride::ngram::{LabelCountMap, Processor, ProcessorOutput};
use xtride::predict::Observation;
use xtride::vocab::Vocab;

use crate::corpus::{Corpus, CorpusError};
use crate::{append_file_stem_suffix, XtrideCommands};

/// N-gram span sizes for all databases used during inference.
const NGRAM_DB_SIZES: [usize; 5] = [48, 12, 8, 4, 2];
// const NGRAM_DB_SIZES: [usize; 16] = [64, 32, 16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2];

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    DbCreation(DbError),
    #[error("could not perform IO operation: {0}")]
    Io(#[from] std::io::Error),
    #[error("error encountered while serialising/deserialising JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("resource error: {0}")]
    Resource(String),
    #[error("error while serialising/deserialising Rkyv: {0}")]
    Rkyv(String),
    #[error("corpus error: {0}")]
    Corpus(#[from] CorpusError),
}

type DBEntryTuple = ([u8; 12], usize, Vec<DbEntry>);

fn create_vocab_from_counts(counts_map: HashMap<String, usize>) -> Vocab {
    let mut pairs = counts_map.into_iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(_, count)| Reverse(*count));

    let mut entries = Vec::with_capacity(pairs.len());
    let mut counts = Vec::with_capacity(pairs.len());

    for (k, v) in pairs {
        entries.push(k);
        counts.push(v);
    }

    Vocab::new(entries, counts)
}

pub fn build_vocab(corpus: &Corpus, label_type: LabelType) -> Result<(Vocab, Vocab), Error> {
    let mut regular_counts = HashMap::<String, usize>::new();
    let mut function_counts = HashMap::<String, usize>::new();

    let pb = ProgressBar::new(corpus.len()? as u64);
    for func_result in corpus.iter()?.progress_with(pb) {
        let func = func_result?;
        let counts = func
            .var_counts()
            .ok_or(Error::Resource("failed to get var counts".into()))?;

        let Some(curr_labels) = func.labels(label_type) else {
            continue;
        };

        for (var, count) in counts {
            if let Some(label_info) = curr_labels.get(&var) {
                if !label_info.is_human() {
                    continue;
                }

                let target_counts = if var.starts_with("sub_") {
                    &mut function_counts
                } else {
                    &mut regular_counts
                };

                *target_counts
                    .entry(label_info.get_label().into())
                    .or_insert(0) += count;
            }
        }
    }

    let regular_vocab = create_vocab_from_counts(regular_counts);
    let function_vocab = create_vocab_from_counts(function_counts);

    Ok((regular_vocab, function_vocab))
}

/// Builds Databases for all n-gram sizes required for inference and persists them to disk
pub fn build_all_dbs(args: &XtrideCommands) -> Result<(), Error> {
    if let XtrideCommands::BuildAllDbs {
        input,
        vocab: vocab_path,
        output_dir,
        type_,
        topk,
        flanking,
        strip,
    } = args
    {
        fs::create_dir_all(output_dir)?;

        let total_dbs = NGRAM_DB_SIZES.len();

        println!("[*] Loading corpus from: {}", input.display());
        let corpus = Corpus::new(input, *strip);

        println!("[*] Loading regular vocab from: {}", vocab_path.display());
        let vocab = Vocab::load(vocab_path)
            .map_err(|e| Error::Resource(format!("failed to load regular vocab: {}", e)))?;

        let fn_vocab_path = append_file_stem_suffix(vocab_path, ".fn");

        // try to load function vocab, fall back to regular vocab if not found
        let fn_vocab = match Vocab::load(&fn_vocab_path) {
            Ok(v) => {
                println!(
                    "[*] Loading function vocab from: {}",
                    fn_vocab_path.display()
                );
                v
            }
            Err(e) => {
                println!(
                    "[!] Function vocabulary not found at {} (error: {}), using regular vocabulary for function types.",
                    fn_vocab_path.display(), e
                );
                Vocab::load(vocab_path)
                    .map_err(|e| Error::Resource(format!("failed to load function vocab: {}", e)))?
            }
        };

        let label_type = *type_;

        println!(
            "  [*] Building {} databases with sizes: {:?}",
            total_dbs, NGRAM_DB_SIZES
        );

        let pb = ProgressBar::new(total_dbs as u64).with_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] Building DBs {bar:40.gradient(green,bright_green)} {pos}/{len} :: Size: {msg}")
                .map_err(|e| Error::Resource(format!("failed to create progress bar style: {}", e)))?,
        );

        for &size in &NGRAM_DB_SIZES {
            pb.set_message(format!("{}", size));

            let output_path = output_dir.join(format!("ngram_{}.rkyv", size));
            println!(
                "\n  [+] Building database with size {} -> {}",
                size,
                output_path.display()
            );

            let db = build_ngram_db_multi(
                &corpus, &vocab, &fn_vocab, label_type, size, *topk, *flanking,
            )?;
            db.save_rkyv(&output_path)
                .map_err(|e| Error::Rkyv(e.to_string()))?;
            pb.inc(1);
        }

        pb.finish_with_message(format!(
            "All databases built in {}",
            humantime::format_duration(pb.elapsed())
        ));
    }

    Ok(())
}

const HASHFUNC_SEED: u64 = 0x1337;

pub fn print_db_hashes(input_dir: impl AsRef<Path>) -> Result<(), Error> {
    let mut entries = fs::read_dir(input_dir.as_ref())?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| ext == "rkyv")
        })
        .collect::<Vec<_>>();

    entries.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    for entry in entries {
        let path = entry.path();
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let file_hash = XxHash64::oneshot(HASHFUNC_SEED, mmap.as_ref());

        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // print in format to directly copy/paste into lib/db.rs for prod use
            println!("m.insert(\"{}\", {});", filename, file_hash);
        }
    }

    Ok(())
}

/// Creates a test/train dataset split from a directory of JSONL files (in the original DIRT format) and persists it to disk
pub fn create_dataset(
    input_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<(), Error> {
    let input_dir = input_dir.as_ref();
    let output_dir = output_dir.as_ref();

    fs::create_dir_all(output_dir)?;

    println!(
        "[*] Reading JSONL files from directory: {}",
        input_dir.display()
    );

    // group all entries with the same tokens
    let mut token_groups = HashMap::<String, Vec<(String, Entry)>>::new();
    let mut total_entries = 0;

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            let file = fs::File::open(&path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                let entry = serde_json::from_str::<Entry>(&line)?;
                let tokens = entry.tokens().join(" ");

                token_groups.entry(tokens).or_default().push((line, entry));
                total_entries += 1;
            }
        }
    }

    println!("\n[i] Dataset Statistics:");
    println!("    Total entries processed: {}", total_entries);
    println!("    Unique token sequences: {}", token_groups.len());
    println!(
        "    Duplicates found: {}",
        total_entries - token_groups.len()
    );

    let mut final_entries = Vec::new();
    let mut type_conflicts = 0;

    for (_tokens, group) in token_groups.iter() {
        let (first_line, first_entry) = &group[0];

        if group.len() == 1 {
            final_entries.push(first_line.to_owned());
            continue;
        }

        // get majority vote for each variable
        let mut var_type_counts = HashMap::<String, HashMap<String, usize>>::new();

        for (_, entry) in group {
            let Some(type_labels) = entry.labels(LabelType::Type) else {
                continue;
            };

            for (var_name, label) in type_labels.vars() {
                var_type_counts
                    .entry(var_name.to_owned())
                    .or_default()
                    .entry(label.get_label().to_owned())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        // determine winning type for each variable
        let mut winning_types = HashMap::<String, String>::new();
        let mut has_conflicts = false;

        for (var_name, type_counts) in &var_type_counts {
            let mut type_votes = type_counts
                .iter()
                .map(|(t, &c)| Observation::new(t, c))
                .collect::<Vec<Observation>>();

            type_votes.sort_by_key(|o| Reverse(o.freq()));

            // diffs for anonymous enum types are expected
            if type_votes.len() > 1 && !type_votes[0].name().contains("__anon_ty") {
                has_conflicts = true;
            }

            // take the type with most votes
            let Some(first_type) = type_votes.first() else {
                continue;
            };

            let mut winning_type = first_type.name();

            // tie break
            if type_votes.len() > 1 && type_votes[0].freq() == type_votes[1].freq() {
                if type_votes[0].name().len() > 45 && type_votes[1].name().len() > 45 {
                    // compare first 45 bytes (Pointer->Struct->Start of Name)
                    let prefix_0 = &type_votes[0].name()[..45];
                    let prefix_1 = &type_votes[1].name()[..45];

                    // type representations containing anon types (enums, unions) are expected
                    // to disagree but will resolve to the same overall type when rebuilding
                    if prefix_0 == prefix_1 && !type_votes[0].name().contains("__anon_ty") {
                        // typedefs might be missing correct sizes so ultimately we choose the
                        // longer representation (e.g., `128`.len() > `0`.len())
                        if type_votes[0].name().len() >= type_votes[1].name().len() {
                            winning_type = type_votes[0].name();
                        } else {
                            winning_type = type_votes[1].name();
                        }
                    }
                }
            }

            winning_types.insert(var_name.to_owned(), winning_type.to_owned());
        }

        if has_conflicts {
            type_conflicts += 1;
        }

        // create new entry with majority-voted types
        let mut entry_map = serde_json::Map::new();

        entry_map.insert("tokens".into(), json!(first_entry.tokens()));
        if !first_entry.meta().is_empty() {
            entry_map.insert("meta".into(), json!(first_entry.meta()));
        }

        // type labels come from original STRIDE implementation and the DIRT type inference benchmarks
        // they hold information about the ground truth labels in evaluation datasets
        let mut type_labels = serde_json::Map::new();
        for (var_name, type_str) in winning_types {
            type_labels.insert(
                var_name,
                json!({
                    "human": true,
                    "label": type_str
                }),
            );
        }

        let mut labels_map = serde_json::Map::new();
        labels_map.insert("type".into(), json!(type_labels));
        entry_map.insert("labels".into(), json!(labels_map));

        // serialise and add to final entries
        let new_entry = json!(entry_map);
        final_entries.push(serde_json::to_string(&new_entry)?);
    }

    println!("\n[i] Found {} groups with type conflicts", type_conflicts);

    // shuffle and split
    let mut rng = rand::rng();
    final_entries.shuffle(&mut rng);

    let split_point = (final_entries.len() as f64 * 0.8) as usize;
    let (train_data, test_data) = final_entries.split_at(split_point);

    println!("\n[i] Final Dataset Statistics:");
    println!("    Train set size: {}", train_data.len());
    println!("    Test set size: {}", test_data.len());

    let train_path = output_dir.join("converted_train.jsonl");
    let test_path = output_dir.join("converted_test.jsonl");

    let mut train_file = fs::File::create(train_path)?;
    let mut test_file = fs::File::create(test_path)?;

    for entry in train_data {
        writeln!(train_file, "{}", entry)?;
    }

    for entry in test_data {
        writeln!(test_file, "{}", entry)?;
    }

    println!("\ndataset created successfully");
    println!(
        "train data saved to {}",
        output_dir.join("converted_train.jsonl").display()
    );
    println!(
        "test data saved to {}",
        output_dir.join("converted_test.jsonl").display()
    );

    Ok(())
}

pub fn build_ngram_db_multi(
    corpus: &Corpus,
    vocab: &Vocab,
    fn_vocab: &Vocab,
    label: LabelType,
    size: usize,
    topk: usize,
    flanking: bool,
) -> Result<NGramDBMulti, Error> {
    let processor = Processor::new(label, size, flanking);

    let total_count = corpus.len()? as u64;
    let pb_style = ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] {msg:^20} {bar:40.cyan/blue} {pos}/{len} [{percent}%] :: ETA {eta}",
        )
        .map_err(|e| Error::Resource(format!("failed to create progress bar style: {}", e)))?;

    let pb = ProgressBar::new(total_count).with_style(pb_style.to_owned());
    pb.set_message("Processing entries");

    let hmaps = corpus
        .par_iter()?
        .progress_with(pb)
        .filter_map(|entry| entry.ok())
        .filter_map(|mut entry| processor.process(&mut entry))
        .collect::<Vec<ProcessorOutput>>();

    let pb = ProgressBar::new(hmaps.len() as u64).with_style(pb_style.to_owned());
    pb.set_message("Merging results");

    let mut merged_hmap = ProcessorOutput::new();
    for sub in hmaps {
        for (h, (is_function_sub, sub_counts)) in sub {
            let (is_function_merged, counts) = merged_hmap
                .entry(h)
                .or_insert_with(|| (is_function_sub, LabelCountMap::new()));

            // sanity check: a hash should consistently belong to either a function or not
            assert_eq!(
                *is_function_merged, is_function_sub,
                "hash origin mismatch during merge"
            );

            for (target, count) in sub_counts {
                *counts.entry(target).or_default() += count;
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("Merging complete");

    let pb = ProgressBar::new(merged_hmap.len() as u64).with_style(pb_style);
    pb.set_message("Building DB entries");

    // transform merged n-gram counts into final database entries.
    let mut entries = merged_hmap
        .into_iter()
        .map(|(h, (is_function, counts))| {
            // select the vocab based on flag
            let target_vocab = if is_function { fn_vocab } else { vocab };

            // get associated type/name counts to get topk
            let mut top = counts
                .into_iter()
                .filter_map(|(name, count)| {
                    target_vocab.lookup(&name).map(|id| DbEntry::new(id, count))
                })
                .collect::<Vec<_>>();

            top.sort_by_key(|db_entry| Reverse(db_entry.context_count()));
            top.truncate(topk);

            // padding
            while top.len() < topk {
                top.push(DbEntry::new(0, 0));
            }

            let total = top.iter().map(|db_entry| db_entry.context_count()).sum();

            pb.inc(1);
            (h, total, top)
        })
        .collect::<Vec<DBEntryTuple>>();

    pb.finish_with_message("DB entry building complete");

    entries.sort_by_key(|&(h, _, _)| h);

    // final format
    let hsh = entries
        .iter()
        .map(|&(h, _, _)| h)
        .collect::<Vec<[u8; 12]>>();

    let total = entries.iter().map(|&(_, t, _)| t).collect::<Vec<usize>>();

    let (typ, counts): (Vec<usize>, Vec<usize>) = entries
        .iter()
        .flat_map(|&(_, _, ref pairs)| {
            pairs
                .iter()
                .map(|db_entry| (db_entry.vocab_idx(), db_entry.context_count()))
        })
        .unzip();

    log::info!("done building ngram database");
    NGramDBMulti::new(vec![size], hsh, total, typ, counts, Some(topk)).map_err(Error::DbCreation)
}

/// Convert all HDF5 databases in a directory to rkyv format.
/// HDF5 was used in the original STRIDE paper implementation, rkyv is recommended for production
pub fn convert_dir_to_rkyv(
    input_dir: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<(), DbError> {
    fs::create_dir_all(&output_dir).map_err(DbError::io)?;

    let db_paths = fs::read_dir(&input_dir)
        .map_err(DbError::io)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("db"))
        .collect::<Vec<_>>();

    for path in db_paths {
        let db = load_hdf5(&path)?;

        let out_path = output_dir.as_ref().join(
            path.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.replace(".db", ".rkyv"))
                .unwrap_or_else(|| "unknown.rkyv".into()),
        );

        db.save_rkyv(&out_path)?;
    }

    Ok(())
}

fn load_hdf5(fpath: impl AsRef<Path>) -> Result<NGramDBMulti, DbError> {
    let file = hdf5::File::open(fpath).map_err(DbError::io)?;

    let size = file
        .dataset("size")
        .map_err(DbError::hdf5)?
        .read_raw::<usize>()
        .map_err(DbError::hdf5)?;

    let hsh_ds = file.dataset("hsh").map_err(DbError::hdf5)?;
    let flat_hsh = hsh_ds.read_raw::<u8>().map_err(DbError::hdf5)?;
    let hsh = flat_hsh
        .chunks(12)
        .map(|chunk| {
            let mut arr = [0u8; 12];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect::<Vec<[u8; 12]>>();

    let total = file
        .dataset("total")
        .map_err(DbError::hdf5)?
        .read_raw::<usize>()
        .map_err(DbError::hdf5)?;
    let typ = file
        .dataset("typ")
        .map_err(DbError::hdf5)?
        .read_raw::<usize>()
        .map_err(DbError::hdf5)?;
    let counts = file
        .dataset("counts")
        .map_err(DbError::hdf5)?
        .read_raw::<usize>()
        .map_err(DbError::hdf5)?;
    let topk = typ.len() / hsh.len();

    NGramDBMulti::new(size, hsh, total, typ, counts, Some(topk))
}
