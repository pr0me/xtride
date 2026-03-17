use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

use hashbrown::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum VocabError {
    #[error(transparent)]
    Io(anyhow::Error),
    #[error(transparent)]
    Serialisation(anyhow::Error),
    #[error("missing {0} in vocab file")]
    MissingField(&'static str),
}

impl VocabError {
    pub fn io<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Io(anyhow::Error::new(e))
    }

    pub fn serialisation<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Serialisation(anyhow::Error::new(e))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    entries: Vec<String>,
    counts: Vec<usize>,
    map: HashMap<String, usize>,
}

impl Vocab {
    pub fn new(entries: Vec<String>, counts: Vec<usize>) -> Self {
        let map = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.to_owned(), i))
            .collect::<HashMap<String, usize>>();

        Self {
            entries,
            counts,
            map,
        }
    }

    pub fn save(&self, fpath: impl AsRef<Path>) -> Result<(), VocabError> {
        let file = File::create(fpath).map_err(VocabError::io)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer(writer, self).map_err(VocabError::serialisation)?;
        Ok(())
    }

    pub fn load(fpath: impl AsRef<Path>) -> Result<Self, VocabError> {
        let file = File::open(fpath).map_err(VocabError::io)?;
        let reader = BufReader::new(file);

        let vocab = serde_json::from_reader(reader).map_err(VocabError::serialisation)?;
        Ok(vocab)
    }

    /// Manually load Vocab file in legacy format for inter-operability with original STRIDE dataset.
    #[deprecated]
    pub fn load_stride_fmt(fpath: impl AsRef<Path>) -> Result<Self, VocabError> {
        let file = File::open(fpath).map_err(VocabError::io)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut counts = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(VocabError::io)?;
            let mut parts = line.split('\t');

            let entry = parts
                .next()
                .ok_or(VocabError::MissingField("entry"))?
                .into();

            let count = parts
                .next()
                .ok_or(VocabError::MissingField("count"))?
                .parse::<usize>()
                .map_err(VocabError::serialisation)?;

            entries.push(entry);
            counts.push(count);
        }

        Ok(Self::new(entries, counts))
    }

    pub fn lookup(&self, key: &str) -> Option<usize> {
        self.map.get(key).copied()
    }

    pub fn reverse(&self, id: usize) -> Option<&String> {
        self.entries.get(id)
    }

    pub fn count_by_id(&self, id: usize) -> Option<usize> {
        self.counts.get(id).copied()
    }

    pub fn entries(&self) -> &[String] {
        &self.entries
    }
}
