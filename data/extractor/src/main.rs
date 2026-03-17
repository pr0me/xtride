use anyhow::Result;
use clap::Parser;
use extract::Extractor;
use std::path::PathBuf;
use std::process;

/// Extract DIRT dataset archives.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(name = "extract")]
struct Args {
    /// directory containing the .tar.xz files to extract
    #[arg(short, long, default_value = ".")]
    directory: PathBuf,

    /// target directory for extracted files
    #[arg(short = 'o', long, default_value = "extracted")]
    output: PathBuf,

    /// number of threads to use for parallel processing
    #[arg(short, long)]
    threads: Option<usize>,

    /// skip tar extraction and only process gz files
    #[arg(short, long)]
    skip_extraction: bool,

    /// show verbose output
    #[arg(short, long)]
    verbose: bool,

    /// file containing list of hashes to extract (one per line)
    #[arg(short = 'f', long)]
    hash_file: Option<PathBuf>,

    /// number of additional files to randomly sample beyond those in hash_file
    #[arg(short, long)]
    add: Option<usize>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);

        for cause in e.chain().skip(1) {
            eprintln!("   caused by: {}", cause);
        }

        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("verbose mode enabled");
    }

    if !args.directory.exists() {
        anyhow::bail!("directory does not exist: {}", args.directory.display());
    }

    if !args.directory.is_dir() {
        anyhow::bail!("path is not a directory: {}", args.directory.display());
    }

    let num_threads = args.threads.unwrap_or_else(num_cpus::get);

    if args.verbose {
        println!("using {} threads", num_threads);
    }

    let extractor = Extractor::new(&args.directory, &args.output, num_threads)?;
    let stats = extractor.extract_all(args.skip_extraction, args.hash_file.as_deref(), args.add)?;

    if args.verbose {
        println!("\ndetailed statistics:");
        println!("   archives: {}", stats.archives_extracted);
        println!("   files processed: {}", stats.files_decompressed);
        println!("   data: {} bytes", stats.total_bytes_processed);
        println!("   time: {:.3}s", stats.total_time.as_secs_f64());
    }

    println!("\nextraction completed successfully");

    Ok(())
}
