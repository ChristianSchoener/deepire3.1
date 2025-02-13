# Preprocessing steps
Perform the following steps in order. Choose therefore a
- folder `b_folder`, and use 
- `b_file=raw_log_data_avF_thaxAxiomNames_useSineFalse.pt` (which is output by `log_loader`)
## 0. Load problem files and create initial dump
<pre>
python3 log_loader.py\
        folder=$b_folder\
        file=loop0_logs.txt\
      > $b_folder/log_loading.txt
</pre>
## 1. pre. Compress individual problems, reduce them a bit, and split into training and validation sets.
<pre>
python3 compressor.py\
        mode=pre\
        file=$b_folder/$b_file\
        out_file_1=$b_folder/$b_file.pre\
      > $b_folder/compression_pre.txt
</pre>
## 2. compress. Delete unnecessary nodes, compress the problems up to threshold, and assign the weights:
<pre>
python3 compressor.py\
        mode=compress\
        file=$b_folder/$b_file.pre.valid\
        out_file_1=$b_folder/$b_file.pre.valid.com\
      > $b_folder/compression_com_valid.txt
python3 compressor.py\
        mode=compress\
        file=$b_folder/$b_file.pre.train\
        out_file_1=$b_folder/$b_file.pre.train.com\
      > $b_folder/compression_com_train.txt
</pre>