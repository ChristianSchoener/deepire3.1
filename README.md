# Modifications for GPU-readiness
If you're new, probably first check out the requirements, then the links and then the testing section.

## Requirements
Other than what is listed in requirements.txt, Python 3.12.
Obligatory CUDA.

## Important links for this repo:
- The Mizar40 problem set in the TPTP format: [Mizar40 Problem-set @ cvut.cz](http://grid01.ciirc.cvut.cz/~mptp/7.13.01_4.181.1147/MPTP2/problems_small_consist.tar.gz)
- Vampire with Deepire modifications: [Deepire 3.1 commit @ github.com](https://github.com/vprover/vampire/commit/110f414207d632819dea4cf01a1ddaca86d0cca3)
- libtorch to be compiled together with Vampire: [libtorch @ pytorch.org](https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.7.0.zip)
- The original script files, containing also some information about how to  perform the guided runs with Vampire: [deepire-mizar-paper-supplementary-materials @ github.com](https://github.com/quickbeam123/deepire-mizar-paper-supplementary-materials)
- The publication on the subject: [Vampire with a brain is a good ITP hammer](https://link.springer.com/chapter/10.1007/978-3-030-86205-3_11)

## Loading logs, compressing, running training, validating, exporting the model 
- hyperparams.py: Configuration file. Read the comments in there! Many parameters are at the bottom. There are also some parameters we do not use/care about.
- all.txt: Contains all references to the problem files downloadable from [Mizar40 Problem-set @ cvut.cz](http://grid01.ciirc.cvut.cz/~mptp/7.13.01_4.181.1147/MPTP2/problems_small_consist.tar.gz)
- results/base_s4k: The log files from unguided Vampire attempting to solve the problems in all.txt in 10s with memory limit 8 GB. 
- loop0_logs.txt: Holds references to the log files of the successfully solved problems in all.txt with additionally enabled proof output. See 7) at [deepire-mizar-paper-supplementary-materials @ github.com](https://github.com/quickbeam123/deepire-mizar-paper-supplementary-materials) for further information.
- log_loader.py: Loads the log files referenced in loop0_logs.txt, and creates the initial raw data as specified in hyperparams.py. Run as python3 log_loader.py
- compressor.py: Performs pre-compression and the compression leading to files used for the training and validation. Set the parameters in hyperparams.py, then run python3 compressor.py mode=pre, and afterwards python3 compressor.py mode=compress, the latter once with the setting train and once with valid in hyperparams.py.
- inf_common.py: Contains most of the algorithms and models. Not called directly.
- multi_inf_parallel_files_continuous.py: The function to perform the training. Call by python3 multi_inf_parallel_files_continuous.py.
- validator.py: Runs the validation. Adjust the parameters in hyperparams.py to point to your project directory containing checkpoint files, and another folder for the validation results. Execute by running python3 validator.py. Watch out, this one runs on CPU, not GPU. So adjust the value for number of processes.
- exporter.py: Exports checkpoint files to jit-scripted models used for the guidance of Vampire. Adjust the parameters in hyperparams.py to point to the desired checkpoint file and model file name. Execute by running python3 exporter.py.

## Testing
Be aware that execution of the scripts consumes quite some RAM (close to 60 GB, if using 6 processes).
- testing/testing.py Runs the comparison between original and modified code on the logs in testing_logs.txt. Execute: python3 testing.py.
- testing/hyperparams.py: Contains the parameters for the testing. Setting Dropout to 0.0 ensures identical results. The compression threshold is set quite high such that all selected logs file be compressed into a single problem instance.
- testing/testing_logs.txt: Contains the references to the log files in results/base_s4k, for which a comparison of the results between the original and the modified code shall be performed, and a computation speed comparison.
- testing/loop0_logs.txt: Contains all references to the log files in results/base_s4k.

## Models
In the models folder, there are two trained models, ready for guidance.
They were both obtained from 500 revealed axioms and embedding dimension 128.
- models/l0_500_e43_original.pt is a reference model from the original code.
- models/greedy-500-e37-PerProblem_mixed.pt was generated with the greedy approach and the "PerProblem_mixed" weighting strategy, which takes into account positive and negative rating before collapsing axioms. The original code implements the "PerProblem" weigthting strategy, which doesn't do that. Furthermore, since unit resulting resolution (rule 52) is not in the training, we replaced its forward function by the sum of incoming vectors, which is cheap to compute (not like this in inf_common.py), so that guidance doesn't waste time with the evaluation of this untrained part of the model.

We observe 20632 problems solved for unguided Vampire, 20415 for Vampire guided by the mentioned model from the original code, and 25909 problems solved by the modified weighting strategy from the new code.
