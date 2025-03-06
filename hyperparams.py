#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import torch

# multi_inf_paralels_config:

SCRATCH = "/home/chris/Dokumente/Github/Projektarbeit_Vampire/strat1new_better/classes_testing_compress/pieces" # add /raid/. for dgx
MAX_EPOCH = 100

# DATA PREPARATION PARAMS:

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

# only take the first MAX_USED_AXIOM_CNT thax values to create embeddings for (all other will join 0)
# this needs to be done before/during the compression phase
# note that log-loading already introduced the axioms in the order of decreasing estimated usefulness
# only makes sense for THAX_SOURCE = ThaxSource_AXIOM_NAMES
MAX_USED_AXIOM_CNT = 500

COMPRESSION_THRESHOLD = 20000

VALID_SPLIT_RATIO = 0.9

# these are now ignored in multi_inf_parallel_files_continuous.py
WHAT_IS_BIG = 12000
WHAT_IS_HUGE = 120000

# used for both training and model export (should be kept the same)
USE_SINE = False

# any other value than -1 (which means "off") will get hardwired during export into the model
FAKE_CONST_SINE_LEVEL = -1

# MODEL PARAMS:

# a hyper-parameter of the future model
# 2*4^3 = 128 EMBED_SIZE
EMBEDDING_SIZE = 128

NonLinKind_TANH = 1
NonLinKind_RELU = 2

def NonLinKindName(val):
  if val == NonLinKind_TANH:
    return "Tanh"
  elif val == NonLinKind_RELU:
    return "ReLU"

NONLIN = NonLinKind_RELU

BOTTLENECK_EXPANSION_RATIO = 2 # is used halved for the eval layer (and sine layer?)

LAYER_NORM = True

# These two should probably used exclusively,
# also they are only imporant when we have (NONLIN == NonLinKind_RELU && LAYER_NORM == False)
CLIP_GRAD_NORM = None # either None of the max_norm value to pass to clip_grad_norm_
CLIP_GRAD_VAL = None  # either None of the clip_value value to pass to clip_grad_value_

DROPOUT = 0.2

# LEARNING PARAMS:

NUMPROCESSES = 6

TestRiskRegimen_VALIDATE = 1
TestRiskRegimen_OVERFIT = 2

def TestRiskRegimenName(val):
  if val == TestRiskRegimen_VALIDATE:
    return "VALIDATE"
  elif val == TestRiskRegimen_OVERFIT:
    return "OVERFIT"

TRR = TestRiskRegimen_VALIDATE

SWAPOUT = 0.0
LEARN_RATE = 0.00005
MOMENTUM = 0.9 # only for SGD

NON_CONSTANT_10_50_250_LR = False

# Corresponds to L2 regularization
WEIGHT_DECAY = 0.0

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

OPTIMIZER = Optimizer_ADAM

POS_WEIGHT_EXTRA = 1.0

FRACTIONAL_CHECKPOINTING = 0 # 0 means disabled, 1 does not make sense

ALL_ONCE = True

FOCAL_LOSS = False

CUDA = True

NUM_STREAMS = 1

LOG_FOLDER = "strat1new_better/classes_testing_compress_guided/"
LOG_FILES_TXT = "loop0_guided.txt"

ZERO_FOLDER = "strat1new_better/classes_testing_compress/"
ZERO_FILE = "strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt"

PRE_FOLDER = "strat1new_better/classes_testing_compress/"
PRE_FILE = "strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt"

COM_FOLDER = "strat1new_better/classes_testing_compress/"
COM_FILE = "strat1new_better/classes_testing_compress/raw_log_data_avF_thaxAxiomNames_useSineFalse.pt.train"
COM_ADD_MODE_1 = "train"

TRAIN_BASE_FOLDER = "strat1new_better/classes_testing_compress/"
TRAIN_TRAIN_FOLDER = "strat1new_better/classes_testing_compress/run"
TRAIN_USE_CHECKPOINT = False
TRAIN_CHECKPOINT_FILE = "strat1new_better/classes_testing_compress/run/check-epoch2.pt"

VALID_BASE_FOLDER = "strat1new_better/classes_testing_compress/"
VALID_TRAIN_FOLDER = "strat1new_better/classes_testing_compress/run"
VALID_VALID_FOLDER = "strat1new_better/classes_testing_compress/run-validate"

EXP_DATA_SIGN = "strat1new_better/classes_testing_compress/data_sign_full.pt"
EXP_DATA_SIGN_PREPARED = "strat1new_better/classes_testing_compress/data_sign.pt"
EXP_FILE = "strat1new_better/classes_testing_compress/full_tree_cropped_revealed_500.pt"
EXP_NAME = "strat1new_better/models/zero-test.pt"
EXP_MODEL_FOLDER = "/media/chris/8b23eee9-a098-4f74-8836-d5f49654310e/Projektarbeit_Vampire_dump/zero-test-models"
EXP_PROBLEM_CONFIGURATIONS = "key_problemName_val_axiomNumberSet.pt"
EXP_PROBLEM_FILES_ALL = "all.txt"
EXP_PROBLEM_FILES = "all.txt"
EXP_CHECKPOINT_FILE = ""
EXP_AXIOM_NUMBER_MAPPING = "axiom2Number_number2Axiom.pt"
EXP_SELEC_GOOD_FILE = "strat1new_better/classes_testing_compress/global_selec_and_good.pt"
EXP_RESULTS_FOLDER = "results/zero-test-500"
EXP_REVEAL = True
EXP_LOAD_DERIV_ABSTRACTIONS = True
EXP_LOAD_DERIV_ABSTRACTIONS_FILE = "strat1new_better/classes_testing_compress/deriv_abstractions_file.pt"