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
## 1. pre. Compress individual problems and reduce them a bit.
<pre>
python3 compressor.py\
        mode=pre\
        file=$b_folder/$b_file\
        out_file_1=$b_folder/$b_file.pre\
      > $b_folder/compression_pre.txt
</pre>
## 2. split. Split problem set into training and validation part.
<pre>
python3 compressor.py\
        mode=split\
        folder=$b_folder\
        file=$b_folder/$b_file.pre\
      > $b_folder/compression_split.txt
</pre>
## 3. reduce. Create maps of the problems, then reduce them by deleting unnecessary nodes:
<pre>
python3 compressor.py\
        mode=reduce\
        file=$b_folder/$b_file.pre.valid\
        out_file_1=$b_folder/$b_file.pre.valid.red\
      > $b_folder/compression_red_valid.txt
python3 compressor.py\
        mode=reduce\
        file=$b_folder/$b_file.pre.train\
        out_file_1=$b_folder/$b_file.pre.train.red\
      > $b_folder/compression_red_valid.txt
</pre>
## 4. reduce. Reduce the individual problems for unnecessary nodes (Choose one of <train | valid>):
<pre>
python3 compressor.py\
        mode=reduce\
        file=$b_folder/$b_file.pre.valid\
        out_file_1=$b_folder/$b_file.valid.red\
      > $b_folder/compression_red_valid.txt
python3 compressor.py\
        mode=reduce\
        file=$b_folder/$b_file.pre.train\
        out_file_1=$b_folder/$b_file.train.red\
      > $b_folder/compression_red_train.txt
</pre>
## 5. compress. Compress the problems up to threshold (Choose one of <train | valid>):
<pre>
python3 compressor.py\
        mode=compress\
        file=$b_folder/$b_file.pre.valid.red\
        out_file_1=$b_folder/$b_file.pre.valid.red.com\
      > $b_folder/compression_com_valid.txt
python3 compressor.py\
        mode=compress\
        file=$b_folder/$b_file.pre.train.red\
        out_file_1=$b_folder/$b_file.pre.train.red.com\
      > $b_folder/compression_com_train.txt
</pre>
## 6. adjust. Re-distribute weights evenly over the training/validation instances (Choose one of <train | valid>):
<pre>
python3 compressor.py\
        mode=adjust\
        file=$b_folder/$b_file.pre.valid.red.com\
        out_file_1=$b_folder/$b_file.pre.valid.red.com.adj\
      > $b_folder/compression_adj_valid.txt
python3 compressor.py\
        mode=adjust\
        file=$b_folder/$b_file.pre.train.red.com\
        out_file_1=$b_folder/$b_file.pre.train.red.com.adj\
      > $b_folder/compression_adj_train.txt
</pre>
## 7. greedy. Compute the greedy evaluation scheme for every compressed problem (Choose one of <train | valid>):
<pre>
python3 compressor.py\
        mode=greedy\
        file=$b_folder/$b_file.pre.valid.red.com.adj\
        out_file_1=$b_folder/$b_file.pre.valid.red.com.adj.gre\
      > $b_folder/compression_gre_valid.txt
python3 compressor.py\
        mode=greedy\
        file=$b_folder/$b_file.pre.train.red.com.adj\
        out_file_1=$b_folder/$b_file.pre.train.red.com.adj.gre\
      > b_folder/compression_gre_train.txt
</pre>
## 8. pieces. Write out individual pieces for training and validation:
<pre>
python3 compressor.py\
        mode=pieces\
        file=$b_folder/$b_file.pre.valid.red.com.adj.gre\
        folder=$b_folder\
        add_mode_1=valid\
      > $b_folder/compression_pie_valid.txt
python3 compressor.py\
        mode=pieces\
        file=$b_folder/$b_file.pre.train.red.com.adj.gre\
        folder=$b_folder\
        add_mode_1=train\
      > $b_folder/compression_pie_train.txt
</pre>

