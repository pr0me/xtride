# Dataset Preparation

# DIRT

In our paper, we made use of the existing DIRT dataset.
The raw, unpreprocessed data is available at: https://kilthub.cmu.edu/articles/dataset/Unpreprocessed_Dataset_for_Idiomatic_ReTyping_DIRT_/20732656

To reproduce the DIRT dataset splits as used in the original publication by Chen et al., you can match against 
the hashes in `dirty_test_hashes.txt` and `dirty_train_hashes.txt` respectively.

To preprocess the raw files, we made use of the [utility scripts provided by DIRTY's open source version](https://github.com/CMUSTRUDEL/DIRTY/tree/main/dirty/utils):
```
uv run python utils/preprocess.py ./dataset_DIRT_extracted ./dirty_train_hashes.txt ./dataset_DIRT_og --test-file ./dirty_test_hashes.txt
```
The extended datasets were sampled from the same unpreprocessed source. For training and validation sets, we added additional file hashes to the input list, specified, e.g., `--max=315000` for the dataset with 300k samples (accounting for the auto-generated test split), and ensured that no file was included that is also contained in the original DIRT test split.

For further conversion to a raw token representation that looks like this
```
{"labels":{"type":{"param1":{"human":true,"label":"ptr<ptr<void>>"},"var1":{"human":true,"label":"ptr<ptr<struct<_EFI_PEI_SERVICES>>>"}}},"tokens":["void","sub_75b0","(","bits64_t","@@param1@@",")","{","int64_t","*","@@var1@@",";","@@var1@@","=","GetPeiServicesTablePointer","(",")",";","(","*","(","*","@@var1@@","+","0x48",")",")","(","@@var1@@",",","@@param1@@",")",";","return",";","}"]}
```
we made use of the scripts included in [STRIDE's open source version](https://github.com/hgarrereyn/STRIDE):
```
python3 -m stride.converters.dirt \
    ./dataset_DIRT_og \
    ./dataset_DIRT_tokenized
```
This format can be used for database generation and inference both with the original STRIDE implementation as well as with XTRIDE.

# coreutils
For our evaluation regarding struct identification and recovery, we used a dataset of binaries in the exact version
as published by HyRES: https://github.com/Sandspeare/HyRES/tree/main/binaries