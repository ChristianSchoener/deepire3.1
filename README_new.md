- README_new.md: Contains this explanation.
## Loading logs, compressing, running training, validation, exporting the model 
- hyperparams.py: Configuration file. Read the comments in there! Many
parameters are at the bottom. There are also some parameters we do not
use/care about.
- all.txt: Contains all references to the problem files downloadable from
http://grid01.ciirc.cvut.cz/ mptp/7.13.01_4.181.1147/MPTP2/problems_-
small_consist.tar.gz
- results/base_s4k: The log files from unguided Vampire attempting to
solve the problems in all.txt in 10s with memory limit 8 GB.
- loop0_logs.txt: Holds references to the log files of the successfully solved
problems in all.txt with additionally enabled proof output. See 7) at
https://github.com/quickbeam123/deepire-mizar-paper-supplementary-materials
for further information.
- log_loader.py: Loads the log files referenced in loop0_logs.txt, and cre-
ates the initial raw data as specified in hyperparams.py. Run as python3
log_loader.py
- compressor.py: Performs pre-compression and the compression leading to
files used for the training and validation. Set the parameters in hyperparams.py,
then run python3 compressor.py mode=pre, and afterwards python3
compressor.py mode=compress, the latter once with the setting train and
once with valid in hyperparams.py.
- inf_common.py: Contains most of the algorithms and models. Not called
directly.
- multi_inf_parallel_files_continuous.py: The function to perform the
training. Call by python3 multi_inf_parallel_files_continuous.py.
- validator.py: Runs the validation. Adjust the parameters in hyperparams.py
to point to your project directory containing checkpoint files, and another
folder for the validation results. Execute by running python3 validator.py.
- exporter.py: Exports checkpoint files to jit-scripted models used for the
guidance of Vampire. Adjust the parameters in hyperparams.py to point
to the desired checkpoint file and model file name. Execute by running
python3 exporter.py.

## Testing
- testing/testing.py Runs the comparison between original and modified
code on the logs in testing_logs.txt. Execute: python3 testing.py.
- testing/hyperparams.py: Contains the parameters for the testing. Setting
Dropout to 0.0 ensures identical results. The compression threshold is set
quite high such that all selected logs file be compressed into a single problem
instance.
- testing/testing_logs.txt: Contains the references to the log files in
results/base_s4k, for which a comparison of the results between the orig-
inal and the modified code shall be performed, and a computation speed
comparison.
- testing/loop0_logs.txt: Contains all references to the log files in results/base_-
s4k.