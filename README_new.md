# Preprocessing steps
Perform the following steps in order:
## Load problem files and create initial dump
- python3 log_loader.py strat1new_better/classes_testing_compress/ strat1new_better/loop0_logs.txt > strat1new_better/classes_testing_compress/log_loading.txt
## pre. Compress individual problems and reduce them a bit.
- python3 compressor.py mode=pre file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre > strat1new_better/classes_testing_compress/compression_pre.txt
## split. Split problem set into trainingf and validation part.
- python3 compressor.py mode=split folder=strat1new_better/classes_testing_compress/ file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre > strat1new_better/classes_testing_compress/compression_split.txt
## map. Create maps for identification of nodes, to move their weights to exactly one problem later. Has to be done for train/valid individually.
- python3 compressor.py mode=map file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid> out_file_1=strat1new_better/classes_testing_compress/id_map_<train | valid>.pt > strat1new_better/classes_testing_compress/compression_map_<train | valid>.txt
## reduce. Reduce the individual problems for unnecessary nodes (Choose one of <train | valid>):
- python3 compressor.py mode=reduce file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid> add_file_1=strat1new_better/classes_testing_compress/id_map_<train | valid>.pt out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red > strat1new_better/classes_testing_compress/compression_red_<train | valid>.txt
## compress. Compress the problems up to threshold (Choose one of <train | valid>):
- python3 compressor.py mode=compress file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red.com > strat1new_better/classes_testing_compress/compression_com_<train | valid>.txt
## adjust. Re-distribute weights evenly over the training/validation instances (Choose one of <train | valid>):
- python3 compressor.py mode=adjust file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red.com out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red.com.adj > strat1new_better/classes_testing_compress/compression_adj_<train | valid>.txt
## greedy. Compute the greedy evaluation scheme for every compressed problem (Choose one of <train | valid>):
- python3 compressor.py mode=greedy file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red.com.adj out_file_1=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red.com.adj.gre > strat1new_better/classes_testing_compress/compression_gre_<train | valid>.txt
## pieces. Write out individual pieces for training and validation (Choose one of <train | valid>):
- python3 compressor.py mode=pieces file=strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.pre.<train | valid>.red.com.adj[.gre] file=strat1new_better/classes_testing_compress/ add_mode_1=<train | valid> > strat1new_better/classes_testing_compress/compression_pie_<train | valid>.txt

