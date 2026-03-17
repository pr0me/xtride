use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use tar::Archive;
use walkdir::WalkDir;
use xz2::read::XzDecoder;

/// Extract DIRT dataset archives.
pub struct Extractor {
    base_dir: PathBuf,
    output_dir: PathBuf,
    num_threads: usize,
    multi_progress: MultiProgress,
}

impl Extractor {
    pub fn new(
        base_dir: impl AsRef<Path>,
        output_dir: impl AsRef<Path>,
        num_threads: usize,
    ) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        let output_dir = output_dir.as_ref().to_path_buf();

        fs::create_dir_all(&output_dir).with_context(|| {
            format!(
                "failed to create output directory: {}",
                output_dir.display()
            )
        })?;

        Ok(Self {
            base_dir,
            output_dir,
            num_threads,
            multi_progress: MultiProgress::new(),
        })
    }

    /// Extract all matching archive entries.
    pub fn extract_all(
        &self,
        skip_extraction: bool,
        hash_file: Option<&Path>,
        additional_samples: Option<usize>,
    ) -> Result<ExtractionStats> {
        let start_time = Instant::now();

        println!("starting DIRT dataset extraction");
        println!("base directory: {}", self.base_dir.display());
        println!("output directory: {}", self.output_dir.display());
        println!("using {} threads", self.num_threads);

        let hash_filter = if let Some(hash_file_path) = hash_file {
            let hashes = self.read_hash_file(hash_file_path)?;
            println!(
                "filtering for {} specific hashes from {}",
                hashes.len(),
                hash_file_path.display()
            );

            if let Some(count) = additional_samples {
                println!("sampling {} additional random hashes", count);
            }

            Some(hashes)
        } else if let Some(count) = additional_samples {
            println!("randomly sampling {} hashes from dataset", count);
            Some(HashSet::new())
        } else {
            println!("extracting all files");
            None
        };

        let extraction_stats = if skip_extraction {
            println!("skipping tar extraction");
            ExtractionStats {
                archives_extracted: 0,
                files_decompressed: 0,
                total_bytes_processed: 0,
                total_time: std::time::Duration::new(0, 0),
            }
        } else {
            let tar_files = self.find_tar_xz_files()?;
            println!("found {} tar.xz archives to extract", tar_files.len());

            self.extract_archives_parallel(&tar_files, hash_filter.as_ref(), additional_samples)?
        };

        let total_time = start_time.elapsed();

        let final_stats = ExtractionStats {
            archives_extracted: extraction_stats.archives_extracted,
            files_decompressed: extraction_stats.files_decompressed,
            total_bytes_processed: extraction_stats.total_bytes_processed,
            total_time,
        };

        self.print_final_stats(&final_stats)?;

        Ok(final_stats)
    }

    fn find_tar_xz_files(&self) -> Result<Vec<PathBuf>> {
        let mut tar_files = Vec::new();

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "xz") {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        if name_str.ends_with(".tar.xz") {
                            tar_files.push(path);
                        }
                    }
                }
            }
        }

        tar_files.sort();
        Ok(tar_files)
    }

    fn extract_archives_parallel(
        &self,
        tar_files: &[PathBuf],
        hash_filter: Option<&HashSet<String>>,
        additional_samples: Option<usize>,
    ) -> Result<ExtractionStats> {
        let start_time = Instant::now();

        if let Some(sample_count) = additional_samples {
            if let Some(hashes) = hash_filter {
                return self.extract_with_sampling(tar_files, hashes, sample_count);
            }
        }

        self.extract_standard(tar_files, hash_filter, start_time)
    }

    fn extract_standard(
        &self,
        tar_files: &[PathBuf],
        hash_filter: Option<&HashSet<String>>,
        start_time: Instant,
    ) -> Result<ExtractionStats> {
        let archives_completed = Arc::new(AtomicU64::new(0));
        let extracted_hashes = Arc::new(Mutex::new(HashSet::new()));

        let pb = self
            .multi_progress
            .add(ProgressBar::new(tar_files.len() as u64));
        pb.set_style(Self::progress_style(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} archives ({per_sec}) | {msg}",
        )?);
        pb.set_message("extracting archives...");

        let results: Result<Vec<_>> = tar_files
            .par_iter()
            .map(|tar_path| {
                let result = self.extract_single_archive_fast(
                    tar_path,
                    hash_filter,
                    Some(&extracted_hashes),
                );

                if let Ok((_, files)) = &result {
                    let completed = archives_completed.fetch_add(1, Ordering::Relaxed) + 1;
                    pb.set_position(completed);

                    if let Some(name) = tar_path.file_name() {
                        pb.set_message(format!(
                            "extracted {} files from {}",
                            files,
                            name.to_string_lossy()
                        ));
                    }
                }

                result
            })
            .collect();

        pb.finish_with_message("all archives extracted");

        let extraction_results = results?;
        let total_bytes_processed = extraction_results.iter().map(|(bytes, _)| *bytes).sum();
        let total_files_extracted = extraction_results.iter().map(|(_, files)| *files).sum();

        let all_extracted_hashes = Arc::try_unwrap(extracted_hashes)
            .map_err(|_| anyhow::anyhow!("failed to unwrap extracted hashes"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("failed to unwrap extracted hashes mutex"))?;

        self.write_hashes_output(&all_extracted_hashes)?;

        Ok(ExtractionStats {
            archives_extracted: tar_files.len(),
            files_decompressed: total_files_extracted,
            total_bytes_processed,
            total_time: start_time.elapsed(),
        })
    }

    fn extract_with_sampling(
        &self,
        tar_files: &[PathBuf],
        hash_set: &HashSet<String>,
        sample_count: usize,
    ) -> Result<ExtractionStats> {
        let start_time = Instant::now();

        println!("extracting files with reservoir sampling");
        println!(
            "target: {} specified hashes + {} random hash samples",
            hash_set.len(),
            sample_count
        );
        println!(
            "each hash corresponds to 2 files (bins + types) = {} + {} total files",
            hash_set.len() * 2,
            sample_count * 2
        );

        let hash_reservoir = Arc::new(Mutex::new(Vec::with_capacity(sample_count)));
        let unique_hashes_seen = Arc::new(Mutex::new(HashSet::new()));
        let extracted_hashes = Arc::new(Mutex::new(HashSet::new()));
        let hash_matches = Arc::new(AtomicU64::new(0));
        let total_bytes = Arc::new(AtomicU64::new(0));
        let total_extracted = Arc::new(AtomicU64::new(0));

        let pb = self
            .multi_progress
            .add(ProgressBar::new(tar_files.len() as u64));
        pb.set_style(Self::progress_style(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} archives | {msg}",
        )?);
        pb.set_message("processing archives...");

        tar_files
            .par_iter()
            .try_for_each(|tar_path| -> Result<()> {
                let file = File::open(tar_path)?;
                let file_size = file.metadata()?.len();
                let decoder = XzDecoder::new(file);
                let mut archive = Archive::new(decoder);

                let mut local_extracted = 0;

                for entry in archive.entries()? {
                    let mut entry = entry?;
                    let path = entry.path()?.to_string_lossy().into_owned();

                    if let Some(hash) = self.extract_hash_from_full_path(&path) {
                        if hash_set.contains(&hash) {
                            let output_path = self.create_flattened_output_path(&path)?;
                            if let Some(parent) = output_path.parent() {
                                fs::create_dir_all(parent)?;
                            }
                            entry.unpack(&output_path)?;
                            local_extracted += 1;
                            hash_matches.fetch_add(1, Ordering::Relaxed);

                            extracted_hashes
                                .lock()
                                .map_err(|_| anyhow::anyhow!("failed to lock extracted hashes"))?
                                .insert(hash.clone());
                        } else if sample_count > 0 && self.is_bins_file(&path) {
                            let mut unique_hashes = unique_hashes_seen
                                .lock()
                                .map_err(|_| anyhow::anyhow!("failed to lock seen hashes"))?;
                            if !unique_hashes.contains(&hash) && !hash_set.contains(&hash) {
                                unique_hashes.insert(hash.clone());
                                let current_count = unique_hashes.len() - 1;

                                let mut reservoir_guard = hash_reservoir.lock().map_err(|_| {
                                    anyhow::anyhow!("failed to lock hash reservoir")
                                })?;
                                if reservoir_guard.len() < sample_count {
                                    reservoir_guard.push(hash);
                                } else {
                                    let mut rng = thread_rng();
                                    let rand_idx = rng.gen_range(0..=current_count);
                                    if rand_idx < sample_count {
                                        reservoir_guard[rand_idx] = hash;
                                    }
                                }
                            }
                        }
                    }
                }

                total_bytes.fetch_add(file_size, Ordering::Relaxed);
                total_extracted.fetch_add(local_extracted as u64, Ordering::Relaxed);
                pb.inc(1);

                if local_extracted > 0 {
                    pb.set_message(format!("extracted {} hash matches", local_extracted));
                }

                Ok(())
            })?;

        pb.finish_with_message("initial extraction complete");

        if sample_count > 0 {
            let sampled_hashes = Arc::try_unwrap(hash_reservoir)
                .map_err(|_| anyhow::anyhow!("failed to unwrap hash reservoir"))?
                .into_inner()
                .map_err(|_| anyhow::anyhow!("failed to unwrap hash reservoir mutex"))?;

            if !sampled_hashes.is_empty() {
                println!(
                    "extracting {} randomly sampled hashes ({}x2 = {} files)",
                    sampled_hashes.len(),
                    sampled_hashes.len(),
                    sampled_hashes.len() * 2
                );

                let sample_hash_set = sampled_hashes.iter().collect::<HashSet<_>>();

                let pb2 = self
                    .multi_progress
                    .add(ProgressBar::new(tar_files.len() as u64));
                pb2.set_style(Self::progress_style(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} archives | {msg}",
                )?);
                pb2.set_message("extracting sample pairs...");

                let sample_extracted = Arc::new(AtomicU64::new(0));

                tar_files
                    .par_iter()
                    .try_for_each(|tar_path| -> Result<()> {
                        let file = File::open(tar_path)?;
                        let decoder = XzDecoder::new(file);
                        let mut archive = Archive::new(decoder);

                        let mut local_sample_extracted = 0;

                        for entry in archive.entries()? {
                            let mut entry = entry?;
                            let path = entry.path()?.to_string_lossy().into_owned();

                            if let Some(hash) = self.extract_hash_from_full_path(&path) {
                                if sample_hash_set.contains(&hash) {
                                    let output_path = self.create_flattened_output_path(&path)?;
                                    if let Some(parent) = output_path.parent() {
                                        fs::create_dir_all(parent)?;
                                    }
                                    entry.unpack(&output_path)?;
                                    local_sample_extracted += 1;

                                    extracted_hashes
                                        .lock()
                                        .map_err(|_| {
                                            anyhow::anyhow!("failed to lock extracted hashes")
                                        })?
                                        .insert(hash);
                                }
                            }
                        }

                        sample_extracted
                            .fetch_add(local_sample_extracted as u64, Ordering::Relaxed);
                        pb2.inc(1);
                        Ok(())
                    })?;

                pb2.finish_with_message("sample extraction complete");

                let sample_count_extracted = sample_extracted.load(Ordering::Relaxed);
                total_extracted.fetch_add(sample_count_extracted, Ordering::Relaxed);
            }
        }

        let final_extracted = total_extracted.load(Ordering::Relaxed) as usize;
        let final_hash_matches = hash_matches.load(Ordering::Relaxed) as usize;
        let final_samples = final_extracted.saturating_sub(final_hash_matches);
        let sampled_hash_count = final_samples / 2;

        println!("extraction summary:");
        println!(
            "   hash matches extracted: {} files ({} hashes)",
            final_hash_matches,
            final_hash_matches / 2
        );
        println!(
            "   random samples extracted: {} files ({} hashes)",
            final_samples, sampled_hash_count
        );
        println!("   total files extracted: {}", final_extracted);

        let all_extracted_hashes = Arc::try_unwrap(extracted_hashes)
            .map_err(|_| anyhow::anyhow!("failed to unwrap extracted hashes"))?
            .into_inner()
            .map_err(|_| anyhow::anyhow!("failed to unwrap extracted hashes mutex"))?;

        self.write_hashes_output(&all_extracted_hashes)?;

        Ok(ExtractionStats {
            archives_extracted: tar_files.len(),
            files_decompressed: final_extracted,
            total_bytes_processed: total_bytes.load(Ordering::Relaxed),
            total_time: start_time.elapsed(),
        })
    }

    fn write_hashes_output(&self, hashes: &HashSet<String>) -> Result<()> {
        let output_file = self.output_dir.join("hashes_out.txt");

        let mut sorted_hashes = hashes.iter().collect::<Vec<_>>();
        sorted_hashes.sort();

        println!(
            "writing {} extracted hashes to {}",
            sorted_hashes.len(),
            output_file.display()
        );

        let content = sorted_hashes
            .iter()
            .map(|h| h.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let content_with_newline = if content.is_empty() {
            String::new()
        } else {
            format!("{}\n", content)
        };

        fs::write(&output_file, content_with_newline).with_context(|| {
            format!(
                "failed to write hashes output file: {}",
                output_file.display()
            )
        })?;

        Ok(())
    }

    fn is_bins_file(&self, path: &str) -> bool {
        let normalized = path.strip_prefix("./").unwrap_or(path);
        normalized.starts_with("bins/")
    }

    fn extract_hash_from_full_path(&self, path: &str) -> Option<String> {
        let normalized = path.strip_prefix("./").unwrap_or(path);

        if let Some(remaining) = normalized.strip_prefix("bins/") {
            let parts = remaining.split('/').collect::<Vec<_>>();
            if parts.len() == 2 && parts[1].ends_with(".jsonl.gz") {
                return parts[1].strip_suffix(".jsonl.gz").map(|s| s.into());
            }
        }

        if let Some(remaining) = normalized.strip_prefix("types/") {
            let parts = remaining.split('/').collect::<Vec<_>>();
            if parts.len() == 2 && parts[1].ends_with(".json.gz") {
                return parts[1].strip_suffix(".json.gz").map(|s| s.into());
            }
        }

        None
    }

    fn extract_single_archive_fast(
        &self,
        tar_path: &Path,
        hash_filter: Option<&HashSet<String>>,
        extracted_hashes: Option<&Arc<Mutex<HashSet<String>>>>,
    ) -> Result<(u64, usize)> {
        let file = File::open(tar_path)
            .with_context(|| format!("failed to open archive: {}", tar_path.display()))?;

        let file_size = file.metadata()?.len();
        let decoder = XzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        let mut extracted_count = 0;

        if let Some(hashes) = hash_filter {
            for entry in archive.entries()? {
                let mut entry = entry?;

                let path_owned = entry.path()?.to_string_lossy().into_owned();
                let should_extract = self.should_extract_file(&path_owned, hashes);

                if should_extract {
                    let output_path = self.create_flattened_output_path(&path_owned)?;

                    if let Some(parent) = output_path.parent() {
                        fs::create_dir_all(parent)?;
                    }

                    entry
                        .unpack(&output_path)
                        .with_context(|| format!("failed to extract file: {}", path_owned))?;

                    extracted_count += 1;

                    if let (Some(hash), Some(hashes_set)) = (
                        self.extract_hash_from_full_path(&path_owned),
                        extracted_hashes,
                    ) {
                        hashes_set
                            .lock()
                            .map_err(|_| anyhow::anyhow!("failed to lock extracted hashes"))?
                            .insert(hash);
                    }
                }
            }

            if extracted_count > 0 {
                if let Some(name) = tar_path.file_name() {
                    println!(
                        "extracted {} files from {}",
                        extracted_count,
                        name.to_string_lossy()
                    );
                }
            }
        } else {
            for entry in archive.entries()? {
                let mut entry = entry?;
                let path_owned = entry.path()?.to_string_lossy().into_owned();

                let output_path = self.create_flattened_output_path(&path_owned)?;

                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }

                entry
                    .unpack(&output_path)
                    .with_context(|| format!("failed to extract file: {}", path_owned))?;

                extracted_count += 1;

                if let (Some(hash), Some(hashes_set)) = (
                    self.extract_hash_from_full_path(&path_owned),
                    extracted_hashes,
                ) {
                    hashes_set
                        .lock()
                        .map_err(|_| anyhow::anyhow!("failed to lock extracted hashes"))?
                        .insert(hash);
                }
            }
        }

        Ok((file_size, extracted_count))
    }

    fn print_final_stats(&self, stats: &ExtractionStats) -> Result<()> {
        println!("\nextraction completed successfully");
        println!("archives extracted: {}", stats.archives_extracted);
        println!("files extracted: {}", stats.files_decompressed);
        println!(
            "total data processed: {}",
            humansize::format_size(stats.total_bytes_processed, humansize::DECIMAL)
        );
        println!("total time: {:.2?}", stats.total_time);

        if stats.total_time.as_secs() > 0 {
            let throughput = stats.total_bytes_processed as f64 / stats.total_time.as_secs_f64();
            println!(
                "throughput: {}/s",
                humansize::format_size(throughput as u64, humansize::DECIMAL)
            );
        }

        let gz_files = self.count_files_with_extension("gz")?;

        println!("final result: {} .gz files extracted", gz_files);
        println!("files extracted to flattened structure in ./bins/ and ./types/");

        Ok(())
    }

    fn read_hash_file(&self, hash_file_path: &Path) -> Result<HashSet<String>> {
        let file = File::open(hash_file_path)
            .with_context(|| format!("failed to open hash file: {}", hash_file_path.display()))?;

        let reader = BufReader::new(file);
        let mut hashes = HashSet::new();

        for line in reader.lines() {
            let line = line?;
            let hash = line.trim();
            if !hash.is_empty() {
                hashes.insert(hash.into());
            }
        }

        if hashes.is_empty() {
            anyhow::bail!(
                "hash file is empty or contains no valid hashes: {}",
                hash_file_path.display()
            );
        }

        Ok(hashes)
    }

    fn should_extract_file(&self, path: &str, hashes: &HashSet<String>) -> bool {
        let normalized_path = path.strip_prefix("./").unwrap_or(path);

        if let Some(remaining) = normalized_path.strip_prefix("bins/") {
            return self.extract_hash_from_path(remaining, hashes);
        }

        if let Some(remaining) = normalized_path.strip_prefix("types/") {
            return self.extract_hash_from_path(remaining, hashes);
        }

        false
    }

    fn extract_hash_from_path(&self, path: &str, hashes: &HashSet<String>) -> bool {
        let parts = path.split('/').collect::<Vec<_>>();
        if parts.len() != 2 {
            return false;
        }

        let filename = parts[1];

        let hash = if filename.ends_with(".jsonl.gz") {
            filename.strip_suffix(".jsonl.gz").unwrap_or("")
        } else if filename.ends_with(".json.gz") {
            filename.strip_suffix(".json.gz").unwrap_or("")
        } else {
            return false;
        };

        hashes.contains(hash)
    }

    fn create_flattened_output_path(&self, original_path: &str) -> Result<PathBuf> {
        let normalized_path = original_path.strip_prefix("./").unwrap_or(original_path);

        if let Some(remaining) = normalized_path.strip_prefix("bins/") {
            let parts = remaining.split('/').collect::<Vec<_>>();
            if parts.len() == 2 {
                let filename = parts[1];
                return Ok(self.output_dir.join("bins").join(filename));
            }
        }

        if let Some(remaining) = normalized_path.strip_prefix("types/") {
            let parts = remaining.split('/').collect::<Vec<_>>();
            if parts.len() == 2 {
                let filename = parts[1];
                return Ok(self.output_dir.join("types").join(filename));
            }
        }

        Ok(self.output_dir.join(normalized_path))
    }

    fn progress_style(template: &str) -> Result<ProgressStyle> {
        let style = ProgressStyle::default_bar()
            .template(template)
            .context("failed to build progress bar style")?;

        Ok(style.progress_chars("#>-"))
    }

    fn count_files_with_extension(&self, extension: &str) -> Result<usize> {
        let count = WalkDir::new(&self.output_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|entry| {
                entry.path().is_file()
                    && entry.path().extension().is_some_and(|ext| ext == extension)
            })
            .count();

        Ok(count)
    }
}

#[derive(Debug)]
pub struct ExtractionStats {
    pub archives_extracted: usize,
    pub files_decompressed: usize,
    pub total_bytes_processed: u64,
    pub total_time: std::time::Duration,
}
