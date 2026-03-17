use std::fs::File;
use std::io::{BufRead, BufReader, Error};
use std::path::PathBuf;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use xtride::entry::Entry;

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error(transparent)]
    Io(#[from] Error),
    #[error(transparent)]
    Entry(anyhow::Error),
}

impl CorpusError {
    pub fn entry<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Entry(anyhow::Error::new(e))
    }
}

pub struct Corpus {
    path: PathBuf,
    full_strip: bool,
}

impl Corpus {
    pub fn new(path: impl Into<PathBuf>, full_strip: bool) -> Self {
        Self {
            path: path.into(),
            full_strip,
        }
    }

    pub fn len(&self) -> Result<usize, CorpusError> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().count())
    }

    pub fn iter(&self) -> Result<CorpusIterator, CorpusError> {
        let file = File::open(&self.path)?;
        Ok(CorpusIterator {
            reader: BufReader::new(file),
            full_strip: self.full_strip,
        })
    }

    pub fn par_iter(&self) -> Result<ParCorpusIterator, CorpusError> {
        let file = File::open(&self.path)?;
        // pre-load all lines from the file once
        let lines = BufReader::new(file)
            .lines()
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ParCorpusIterator {
            lines,
            full_strip: self.full_strip,
        })
    }
}

/// Custom iterator to hold the file reader
pub struct CorpusIterator {
    reader: BufReader<File>,
    full_strip: bool,
}

impl Iterator for CorpusIterator {
    type Item = Result<Entry, CorpusError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => {
                let entry = serde_json::from_str(&line)
                    .map_err(CorpusError::entry)
                    .and_then(|raw| Entry::new(raw, self.full_strip).map_err(CorpusError::entry));
                Some(entry)
            }
            Err(e) => Some(Err(e.into())),
        }
    }
}

pub struct ParCorpusIterator {
    lines: Vec<String>,
    full_strip: bool,
}

impl ParallelIterator for ParCorpusIterator {
    type Item = Result<Entry, CorpusError>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        self.lines
            .into_par_iter()
            .map(|line| {
                serde_json::from_str(&line)
                    .map_err(CorpusError::entry)
                    .and_then(|raw| Entry::new(raw, self.full_strip).map_err(CorpusError::entry))
            })
            .drive_unindexed(consumer)
    }
}
