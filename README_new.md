# Preprocessing steps

## Load problem files and create initial dump
- python3 log_loader.py strat1new_better/classes_testing_compress/ strat1new_better/loop0_logs.txt > strat1new_better/classes_testing_compress/log_loading.txt
## Pre: Compress individual problems and reduce them a bit.
- python3 compressor.py mode=pre file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre > strat1new_better/classes_testing_compress/compression_pre.txt
## Split: Split problem set into trainingf and validation part.
- python3 compressor.py mode=split folder=strat1new_better/classes_testing_compress/ file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre > strat1new_better/classes_testing_compress/compression_split.txt
## Map: Create maps for identification of nodes, to move their weights to exactly one problem later. Has to be done for train/valid individually.
- python3 compressor.py mode=map file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.train out_file_1=strat1new_better/classes_testing_compress/id_map_train.pt > strat1new_better/classes_testing_compress/compression_map_train.txt
- python3 compressor.py mode=map file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.valid out_file_1=strat1new_better/classes_testing_compress/id_map_valid.pt > strat1new_better/classes_testing_compress/compression_map_valid.txt
## Adj: Adjust the individual problems with the created maps and then reduce them, where we don't need nodes anymore.
- python3 compressor.py mode=adj file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.train add_file_1=strat1new_better/classes_testing_compress/id_map_train.pt out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.train.adj > strat1new_better/classes_testing_compress/compression_adj_train.txt
- python3 compressor.py mode=adj file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.valid add_file_1=strat1new_better/classes_testing_compress/id_map_valid.pt out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.valid.adj > strat1new_better/classes_testing_compress/compression_adj_valid.txt
## Gen: Merge the problems up to threshold:
- python3 compressor.py mode=gen file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.train.adj folder=strat1new_better/classes_testing_compress/ add_mode_1=train > strat1new_better/classes_testing_compress/compression_gen_train.txt
- python3 compressor.py mode=gen file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.valid.adj folder=strat1new_better/classes_testing_compress/ add_mode_1=valid > strat1new_better/classes_testing_compress/compression_gen_valid.txt