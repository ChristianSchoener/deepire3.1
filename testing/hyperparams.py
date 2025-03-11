#!/usr/bin/env python3

import torch

# the folder containing the pieces. A subdirectory of your project folder
SCRATCH = "./pieces"

MAX_EPOCH = 100

# Do not modify
TreatAvatarEmpties_JUSTFINAL = 1
TreatAvatarEmpties_INCLUDEALL = 2

def TreatAvatarEmptiesName(val):
  if val == TreatAvatarEmpties_JUSTFINAL:
    return "F"
  elif val == TreatAvatarEmpties_INCLUDEALL:
    return "E"

AVATAR_EMPTIES = TreatAvatarEmpties_JUSTFINAL

ThaxSource_THAX_FEATURE = 1
ThaxSource_AXIOM_NAMES = 2

def ThaxSourceName(val):
  if val == ThaxSource_THAX_FEATURE:
    return "VampThax"
  elif val == ThaxSource_AXIOM_NAMES:
    return "AxiomNames"

THAX_SOURCE = ThaxSource_AXIOM_NAMES

SPLIT_AT_ACTIVATION = False
ONLY_GENERATING_PARENTS = False
# Do not modify END

# How many axioms to reveal? The most common this many will be represented individually, the rest collapsed to a single one, and derived clauses accordingly 
MAX_USED_AXIOM_CNT = 500

# Compress pieces, if size below threshold (may lead to pieces up to 2*(threshold-1))
# For testing, we want a single file, therefore we choose a big number here
COMPRESSION_THRESHOLD = 1000000000

# For testing, we want just training, no validation
VALID_SPLIT_RATIO = 1.0

# currently unused
# these are now ignored in multi_inf_parallel_files_continuous.py
WHAT_IS_BIG = 12000
WHAT_IS_HUGE = 120000

# currently unused
# used for both training and model export (should be kept the same)
USE_SINE = False

# currently unused
# any other value than -1 (which means "off") will get hardwired during export into the model
FAKE_CONST_SINE_LEVEL = -1

# The embedding size for the clauses and inference rules.
EMBED_SIZE = 128

NonLinKind_TANH = 1
NonLinKind_RELU = 2

def NonLinKindName(val):
  if val == NonLinKind_TANH:
    return "Tanh"
  elif val == NonLinKind_RELU:
    return "ReLU"
# nonlinear activation = ReLU
NONLIN = NonLinKind_RELU

# Increases the dimension in a first linear layer of inference rules. This has probably some good effect.
BOTTLENECK_EXPANSION_RATIO = 2 # is used halved for the eval layer (and sine layer?)

# Layer-Norm finishing every inference rule
LAYER_NORM = True

# currently unused
# These two should probably used exclusively,
# also they are only imporant when we have (NONLIN == NonLinKind_RELU && LAYER_NORM == False)
CLIP_GRAD_NORM = None # either None of the max_norm value to pass to clip_grad_norm_
CLIP_GRAD_VAL = None  # either None of the clip_value value to pass to clip_grad_value_

# Dropout for the inference rules and evaluation unit
# For testing, we want 0.0
DROPOUT = 0.0

# Number of processes to use for validation, and pre-processing. Can lead to high RAM usage. 6 ~ 50-60 GB max. @ pre-processing
NUMPROCESSES = 6

# currently unused
TestRiskRegimen_VALIDATE = 1
TestRiskRegimen_OVERFIT = 2

def TestRiskRegimenName(val):
  if val == TestRiskRegimen_VALIDATE:
    return "VALIDATE"
  elif val == TestRiskRegimen_OVERFIT:
    return "OVERFIT"

TRR = TestRiskRegimen_VALIDATE

# currently unused
SWAPOUT = 0.0

# lr
LEARN_RATE = 0.00005
# currently unused
MOMENTUM = 0.9 # only for SGD

# If True, lr will increase linearly, reach lr at epoch 10 and 4*lr at epoch 40, then decay. Otherwise constant lr.
NON_CONSTANT_10_50_250_LR = False

# currently unused
# Corresponds to L2 regularization
WEIGHT_DECAY = 0.0

# Optimizer
Optimizer_SGD = 1
Optimizer_ADAM = 2
Optimizer_ADAMW = 3

def OptimizerName(val):
  if val == Optimizer_SGD:
    return "SGD"
  elif val == Optimizer_ADAM:
    return "Adam"
  elif val == Optimizer_ADAMW:
    return "AdamW"

# currently unused - Adam hard-coded
OPTIMIZER = Optimizer_ADAM

# Additional factor to fine-tune class imbalance correction
POS_WEIGHT_EXTRA = 1.0

# currently not used 
FRACTIONAL_CHECKPOINTING = 0 # 0 means disabled, 1 does not make sense

# currently not used 
FOCAL_LOSS = False

# Train on CUDA? True/False
CUDA = True

# currently not used 
NUM_STREAMS = 1

# Available weighting strategies: "PerProblem", "PerProblem_mixed", "Simple", "Additive" 
# For testing, we want "PerProblem", the same as in the original code
WEIGHT_STRATEGY = "PerProblem"

# where to output files after log loading
# Setting for testing
LOG_FOLDER = "."
# file listing the log files to read in
# Setting for testing
# You can put lines from loop0_logs.txt in this folder here, whichever you wish!
LOG_FILES_TXT = "testing_logs.txt"

# where to output files after pre-compression and also read in some smaller one not listed here
# Setting for testing
PRE_FOLDER = "."

# name of the file output by log loading. given the default folder, this will come out:
# Setting for testing
PRE_FILE = "raw_log_data_avF_thaxAxiomNames_useSineFalse.pt"

# where to output files after compression and also read in some smaller one not listed here
# Setting for testing
COM_FOLDER = "."

# name of the file output by log loading. Will be suffixed ".train" or ".valid", depending on the value of COM_ADD_MODE_1
# Setting for testing
COM_FILE = "raw_log_data_avF_thaxAxiomNames_useSineFalse.pt"

# "train" or "valid"
# Setting for testing
COM_ADD_MODE_1 = "train"

# base folder containing data signature, training and validation index
TRAIN_BASE_FOLDER = "my_projects/project_1/"
# run folder containing check-points, logs and plots.
TRAIN_TRAIN_FOLDER = "my_projects/project_1/run"
# Restart from a checkpoint file? True/False
TRAIN_USE_CHECKPOINT = False
TRAIN_CHECKPOINT_FILE = ""

# Mutatis mutandis as for training
VALID_BASE_FOLDER = "my_projects/project_1/"
VALID_TRAIN_FOLDER = "my_projects/project_1/run"
VALID_VALID_FOLDER = "my_projects/project_1/run-validate"

# Data signature, as we have to map axiom numbers back to strings for the jit, since Vampire knows the strings 
EXP_DATA_SIGN_PREPARED = "my_projects/project_1/data_sign.pt"
# The name of the model file that shall be created
EXP_NAME = "my_projects/models/greedy-500-e37-PerProblem_mixed.pt"
# The name of the checkpoint file we want to convert into a model for guidance
EXP_CHECKPOINT_FILE = "my_projects/project_1/run/check-epoch37.pt"
