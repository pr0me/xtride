# XTRIDE: N-gram-based Practical Type Recovery

## Introduction
<a href="https://arxiv.org/pdf/2603.08225" target="_blank"> <img title="" src="images/paper.png" alt="xtride paper" align="right" width="200"></a>
We present XTRIDE, an improved n-gram-based (cf. [STRIDE](https://arxiv.org/pdf/2407.02733)) approach on type recovery for binaries that focuses on practicality: 
highly optimized throughput and actionable confidence scores allow for deployment in automated pipelines. 
When compared to the state of the art in struct recovery, our method achieves comparable performance while being between 70 and 2300× faster.

<br />
<br />
<br />

## Build Instructions
The CLI tool in `./bin` requires a `library of version 1.8.4 or later` for hdf5 installed (as per the crate's docs).
Building with the latest version fails on MacOS, we recommend installing hdf5 v1.10, e.g., with
```
brew install hdf5@1.10
```

## How to use the CLI Tool
1. create tokenized dataset of form, see [Dataset Preparation](./data/README.md).
2. create dataset splits
    ```
    cargo run --release -- create-dataset -i ../new_dataset/ -o ./
    ```
3. build vocab
    ```
    cargo run --release -- build-vocab ./xtride_plus_train.jsonl xtride_plus.vocab -t type
    ```
4. build ngram databases for n = {2, 4, 8, 12, 48} (specify in `bin/src/db_creation.rs`).
    ```
    cargo run --release -- build-all-dbs -t type -k 5 --flanking -o xtride_plus_dbs/ ./xtride_plus_train.jsonl xtride_plus.vocab
    ```
5. Evaluate on the test set split
    ```
    cargo run --release -- evaluate --threshold-sweep ./xtride_plus_test.jsonl xtride_plus.vocab ./out_xtride.json --flanking --db-dir ./xtride_plus_dbs
    ```

## Provided Models
We include the preprocessed data to replicate the $XTRIDE_{PLUS}$ models described in our paper in the `./data` directory.
The JSONL files can be directly used to extract a vocabulary and train the model (steps 3 and onwards, choose the 16-db configuration in `bin/src/db_creation.rs`).
While the training dataset includes a large amount of data from a wide variety of binaries, we want to reiterate that the generalizability of n-gram-based approaches is limited. 
We always recommend adding domain-specific samples to the dataset, depending on where you plan to employ the model.

The provided dataset contains samples that are
- stripped
- ELF binaries
- collected from Ghidra

Trying to run inference on samples that diverge from this distribution will most likely result in unusable predictions.

## Datasets
Further information on how to extract data for new datasets or to retrain and evaluate on the DIRT dataset are included in [Dataset Preparation Docs](data/README.md).