# Dataset Preparation

## DIRT

In our paper, we evaluated against the existing DIRT dataset.
The raw, unpreprocessed data is available at: https://kilthub.cmu.edu/articles/dataset/Unpreprocessed_Dataset_for_Idiomatic_ReTyping_DIRT_/20732656

To reproduce the DIRT dataset splits as used in the original [publication by Chen et al.](https://cmustrudel.github.io/papers/ChenDIRTY2022.pdf), you can match against the hashes in `dirty_test_hashes.txt` and `dirty_train_hashes.txt` respectively.
You can use the provided [`extractor`](./extractor/) crate to extract files by hash from the DIRT corpus like this:
```
RAYON_NUM_THREADS=10 cargo run --release -- --hash-file ./dirty_train_test_concat_hashes.txt --directory /path/to/extracted/DIRT_archive/data/ --output ./dataset_DIRT_extracted --threads 10
```

To preprocess the raw files, we made use of the [utility scripts provided by DIRTY's open source version](https://github.com/CMUSTRUDEL/DIRTY/tree/main/dirty/utils):
```
uv run python utils/preprocess.py ./dataset_DIRT_extracted ./dataset_DIRT_extracted/hashes_out.txt ./dataset_DIRT_processed --test-file ./dirty_test_hashes.txt
```
The extended datasets were sampled from the same unpreprocessed source. For training and validation sets, we added additional file hashes to the input list, specified, e.g., `--max=315000` for the dataset with 300k samples (accounting for the auto-generated test split), and ensured that no file was included that is also contained in the original DIRT test split.
We include our [hash lists for the validation set used in RQ1](./hashes_validation_set_rq1.txt) as well as others for replication.

For further conversion to a raw token representation that looks like this
```
{"labels":{"type":{"param1":{"human":true,"label":"ptr<ptr<void>>"},"var1":{"human":true,"label":"ptr<ptr<struct<_EFI_PEI_SERVICES>>>"}}},"tokens":["void","sub_75b0","(","bits64_t","@@param1@@",")","{","int64_t","*","@@var1@@",";","@@var1@@","=","GetPeiServicesTablePointer","(",")",";","(","*","(","*","@@var1@@","+","0x48",")",")","(","@@var1@@",",","@@param1@@",")",";","return",";","}"]}
```
we made use of the scripts included in [STRIDE's open source version](https://github.com/hgarrereyn/STRIDE):
```
python3 -m stride.converters.dirt \
    ./dataset_DIRT_processed \
    ./dataset_DIRT_tokenized
```
The resulting JSONL format can be used for database generation and inference both with the original STRIDE implementation as well as with XTRIDE.

## coreutils
The functions and type labels from arbitrary binaries can be extracted in a format compatible with DIRTY / STRIDE / XTRIDE with the scripts provided in https://github.com/CMUSTRUDEL/DIRTY/tree/main/dataset-gen (IDA pro-based, see [IDA 9 Guide](./ida_collection/IDA_9_MIGRATION_GUIDE.md)).  
Note that the original script version does retain function names and to ensure fairness in evaluation against HyRES and TypeForge, we had to fully strip the binaries, e.g., by using [our stripping script](./strip_elf_debug.sh), and [enable support in the DIRT collection scripts](./ida_collection/dataset.diff).

For our evaluation regarding struct identification and recovery, we used a dataset of binaries in the exact version as published by HyRES: https://github.com/Sandspeare/HyRES/tree/main/binaries  
After running the data collection scripts, you can continue as described above, starting from `utils/preprocess.py`.
