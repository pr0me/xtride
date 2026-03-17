use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use bytecheck::CheckBytes;
use memmap2::Mmap;
use ouroboros::self_referencing;
use rkyv::rancor::Error;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error(transparent)]
    Io(anyhow::Error),
    #[error(transparent)]
    Hdf5(anyhow::Error),
    #[error("unexpected n-gram database found: {0}")]
    UnexpectedDatabase(anyhow::Error),
    #[error("error during database creation: {0}")]
    Creation(anyhow::Error),
    #[error("lookup operation failed")]
    Lookup(anyhow::Error),
}

impl DbError {
    pub fn io<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Io(anyhow::Error::new(e))
    }

    pub fn hdf5<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Hdf5(anyhow::Error::new(e))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbEntry {
    vocab_idx: usize,
    context_count: usize,
}

impl DbEntry {
    pub fn new(vocab_idx: usize, context_count: usize) -> Self {
        Self {
            vocab_idx,
            context_count,
        }
    }

    pub fn vocab_idx(&self) -> usize {
        self.vocab_idx
    }

    pub fn context_count(&self) -> usize {
        self.context_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupResult {
    global_count: usize,
    targets: Vec<DbEntry>,
}

impl LookupResult {
    pub fn new(total_count: usize, targets: Vec<DbEntry>) -> Self {
        Self {
            global_count: total_count,
            targets,
        }
    }

    pub fn global_count(&self) -> usize {
        self.global_count
    }

    pub fn targets(&self) -> &[DbEntry] {
        &self.targets
    }
}

#[derive(Archive, Deserialize, Serialize, CheckBytes, Debug)]
pub struct NGramPrediction {
    typ: u32,
    counts: u32,
}

#[derive(Archive, Deserialize, Serialize, CheckBytes, Debug)]
pub struct NGramEntry {
    hsh: [u8; 12],
    total: u32,
    predictions: Vec<NGramPrediction>,
}

fn ngram_entry_with(
    i: usize,
    current_hsh: [u8; 12],
    total: &[usize],
    typ: &[usize],
    counts: &[usize],
    k: usize,
) -> Option<NGramEntry> {
    let total_val = *total.get(i)?;
    let typ_slice = typ.get(i * k..(i + 1) * k)?;
    let counts_slice = counts.get(i * k..(i + 1) * k)?;

    let predictions = typ_slice
        .iter()
        .zip(counts_slice.iter())
        .map(|(&t, &c)| {
            Some(NGramPrediction {
                typ: u32::try_from(t).ok()?,
                counts: u32::try_from(c).ok()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    Some(NGramEntry {
        hsh: current_hsh,
        total: u32::try_from(total_val).ok()?,
        predictions,
    })
}

/// Database holding n-gram hashes and their data.
#[derive(Archive, Deserialize, Serialize, CheckBytes, Debug)]
pub struct NGramDBMulti {
    /// The N in n-gram represents the size of the token span.
    size: Vec<u32>,
    entries: Vec<NGramEntry>,
    /// The number of top most frequent labels to keep for each n-gram.
    topk: Option<u32>,
}

impl NGramDBMulti {
    pub fn new(
        size: Vec<usize>,
        hsh: Vec<[u8; 12]>,
        total: Vec<usize>,
        typ: Vec<usize>,
        counts: Vec<usize>,
        topk: Option<usize>,
    ) -> Result<Self, DbError> {
        let k = topk.unwrap_or(1);

        let mut entries = hsh
            .into_iter()
            .enumerate()
            .map(|(i, hsh)| ngram_entry_with(i, hsh, &total, &typ, &counts, k))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                DbError::Creation(anyhow::anyhow!(
                    "index or value out of bounds during db construction"
                ))
            })?;

        // sort entries by hash for binary search
        entries.sort_unstable_by_key(|a| a.hsh);

        Ok(Self {
            size: size.into_iter().map(|x| x as u32).collect(),
            entries,
            topk: topk.and_then(|x| u32::try_from(x).ok()),
        })
    }

    pub fn save_rkyv(&self, fpath: impl AsRef<Path>) -> Result<(), DbError> {
        let bytes = rkyv::to_bytes::<Error>(self)
            .map_err(|e| DbError::io(io::Error::new(io::ErrorKind::Other, e)))?;

        let mut file = File::create(fpath).map_err(DbError::io)?;
        file.write_all(&bytes).map_err(DbError::io)
    }

    pub fn size(&self) -> &[u32] {
        &self.size
    }

    pub fn topk(&self) -> Option<u32> {
        self.topk
    }
}

#[self_referencing]
pub struct MappedNGramDB {
    mmap: Mmap,
    #[borrows(mmap)]
    db: &'this rkyv::Archived<NGramDBMulti>,
}

impl MappedNGramDB {
    pub fn load<'a>(fpath: impl AsRef<Path>) -> Result<Self, DbError> {
        let path = fpath.as_ref();
        let file = File::open(path).map_err(DbError::io)?;
        let mmap = unsafe { Mmap::map(&file).map_err(DbError::io)? };

        Ok(mmap.into())
    }

    pub fn lookup(&self, key: &[u8; 12]) -> Option<LookupResult> {
        self.with_db(|db| match db.entries.binary_search_by_key(key, |e| e.hsh) {
            Ok(idx) => {
                let entry = &db.entries.get(idx)?;
                let total = entry.total.to_native() as usize;
                let pairs = entry
                    .predictions
                    .iter()
                    .map(|p| {
                        DbEntry::new(p.typ.to_native() as usize, p.counts.to_native() as usize)
                    })
                    .collect();

                Some(LookupResult::new(total, pairs))
            }
            Err(_) => None,
        })
    }

    pub fn size(&self) -> Vec<usize> {
        self.with_db(|db| db.size.iter().map(|x| x.to_native() as usize).collect())
    }
}

impl From<Mmap> for MappedNGramDB {
    fn from(mmap: Mmap) -> Self {
        MappedNGramDBBuilder {
            mmap,
            db_builder: |mmap: &Mmap| unsafe {
                rkyv::access_unchecked::<ArchivedNGramDBMulti>(mmap)
            },
        }
        .build()
    }
}

impl std::fmt::Debug for MappedNGramDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.with_db(|db| {
            f.debug_struct("MappedNGramDB")
                .field(
                    "size_n",
                    &db.size
                        .iter()
                        .map(|x| x.to_native() as usize)
                        .collect::<Vec<_>>(),
                )
                .field("entries_count", &db.entries.len())
                .finish()
        })
    }
}
