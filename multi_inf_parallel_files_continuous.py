#!/usr/bin/env python3
import torch

import inf_common as IC
import hyperparams as HP

import torch

torch.set_printoptions(precision=16)
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float16)
# torch.backends.cuda.matmul.allow_tf32 = False

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.autograd.set_detect_anomaly(True)

# from torch import Tensor
# import multiprocessing
# import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system')

from copy import deepcopy

import time

from adan_pytorch import Adan
import torch_optimizer as optim
from hessianfree.optimizers import HessianFree

# from typing import Dict, List, Tuple, Optional

# from collections import defaultdict
# from collections import ChainMap

import sys,random,itertools,os,gc

import numpy as np

# To release claimed memory back to os; Call:   libc.malloc_trim(ctypes.c_int(0))
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def eval_and_or_learn_on_one(problem, start_time, master_parts, data, epoch):
  tot_pos = data["tot_pos"]
  tot_neg = data["tot_neg"]

  print(time.time() - start_time, "Generating Learning Model for {}".format(problem), flush=True)
  model = IC.LearningModel(*master_parts, data, HP.CUDA)
  model.train()
  print(time.time() - start_time, "Starting training for {}".format(problem), flush=True)
  (loss, posOK_sum, negOK_sum) = model()
  print(time.time() - start_time, "Finished training for {}, starting backpropagation".format(problem), flush=True)
  if torch.isnan(loss):
    print("Loss is NaN!")
    exit()
  if torch.isinf(loss):
    print("Loss is Inf!")
    exit()
  loss.backward()

  torch.cuda.synchronize()
  print(time.time() - start_time, "Finished backpropagation for {}, starting optimizer.step()".format(problem), flush=True)
  return loss.detach().cpu().item(), posOK_sum, negOK_sum, tot_pos, tot_neg

def save_checkpoint(epoch, model):
  print("checkpoint", epoch, flush=True)

  check_name = "{}/check-epoch{}.pt".format(HP.TRAIN_TRAIN_FOLDER, epoch)
  check = (epoch, model)
  torch.save(check, check_name)

def load_checkpoint(filename):
  return torch.load(filename, weights_only=False)

def weighted_std_deviation(weighted_mean,scaled_values,weights,weight_sum):
  values = np.divide(scaled_values, weights, out=np.ones_like(scaled_values), where=weights!=0.0) # whenever the respective weight is 0.0, the result should be understood as 1.0
  squares = (values - weighted_mean)**2
  std_dev = np.sqrt(np.sum(weights*squares,axis=0) / weight_sum)
  return std_dev

def do_the_assertions():
  assert hasattr(HP, "TRAIN_TRAIN_FOLDER"), "Parameter TRAIN_TRAIN_FOLDER in hyperparams.py not set. In this folder, restart files and logs of the run will be stored."
  assert isinstance(HP.TRAIN_TRAIN_FOLDER, str), "Parameter TRAIN_TRAIN_FOLDER in hyperparams.py is not a string. In this folder, restart files and logs of the run will be stored."
  assert os.path.isfile(HP.TRAIN_TRAIN_FOLDER), "Parameter TRAIN_TRAIN_FOLDER in hyperparams.py does not point to an existing directory. In this folder, restart files and logs of the run will be stored."

  assert hasattr(HP, "TRAIN_BASE_FOLDER"), "Parameter TRAIN_BASE_FOLDER in hyperparams.py not set. This is the base folder containing the data signature, ..."
  assert isinstance(HP.TRAIN_BASE_FOLDER, str), "Parameter TRAIN_BASE_FOLDER in hyperparams.py is not a string. This is the base folder containing the data signature, ..."
  assert os.path.isfile(HP.TRAIN_BASE_FOLDER), "Parameter TRAIN_BASE_FOLDER in hyperparams.py does not point to an existing directory. This is the base folder containing the data signature, ..."

  assert hasattr(HP, "CUDA"), "Parameter CUDA in hyperparams.py not set. This parameter determines, if the computation is performed on CUDA or CPU."
  assert isinstance(HP.CUDA, bool), "Parameter CUDA in hyperparams.py is not Boolean. This parameter determines, if the computation is performed on CUDA or CPU."

  assert hasattr(HP, "USE_CHECKPOINT"), "Parameter USE_CHECKPOINT in hyperparams.py not set. This parameter determines, if a restart file shall be used."
  assert isinstance(HP.USE_CHECKPOINT, bool), "Parameter USE_CHECKPOINT in hyperparams.py is not a Boolean. This parameter determines, if a restart file shall be used."

  assert hasattr(HP, "TRAIN_CHECKPOINT_FILE"), "Parameter TRAIN_CHECKPOINT_FILE in hyperparams.py not set. This is the restart file to be used for the computation."
  assert isinstance(HP.TRAIN_CHECKPOINT_FILE, str), "Parameter TRAIN_CHECKPOINT_FILE in hyperparams.py is not a string. This is the restart file to be used for the computation."
  assert os.path.isfile(HP.TRAIN_CHECKPOINT_FILE), "Parameter TRAIN_CHECKPOINT_FILE in hyperparams.py does not point to a file. This is the restart file to be used for the computation."

  assert hasattr(HP, "MAX_EPOCH"), "Parameter MAX_EPOCH in hyperparams.py not set."
  assert isinstance(HP.MAX_EPOCH, int), "Parameter MAX_EPOCH is not an integer!"

  assert hasattr(HP, "LEARN_RATE"), "Parameter LEARN_RATE in hyperparams.py not set."
  assert isinstance(HP.LEARN_RATE, float), "Parameter LEARN_RATE is not a float!"
  assert(HP.LEARN_RATE > 0.), "Parameter LEARN_RATE is below 0 but must be positive!"

if __name__ == "__main__":
  do_the_assertions()

  # multiprocessing.set_start_method('spawn', force=True)
  log = open("{}/run{}".format(HP.TRAIN_TRAIN_FOLDER, IC.name_learning_regime_suffix()), 'w')
  sys.stdout = log
  sys.stderr = log
  
  start_time = time.time()
  
  if HP.CUDA and torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"

  train_data_idx = torch.load("{}/train_index.pt".format(HP.TRAIN_BASE_FOLDER), weights_only=False)
  print("Loaded train data:",len(train_data_idx), flush=True)
  valid_data_idx = torch.load("{}/valid_index.pt".format(HP.TRAIN_BASE_FOLDER), weights_only=False)
  print("Loaded valid data:",len(valid_data_idx), flush=True)
  if HP.SWAPOUT > 0.0:
    axiom_counts = torch.load("{}/axiom_counts.pt".format(HP.TRAIN_BASE_FOLDER), weights_only=False)
    print("Loaded axiom counts for uniformly distributed SWAPOUT", flush=True)

  total_count = len(train_data_idx) + len(valid_data_idx)

  print(flush=True)
  print(time.time() - start_time, "Initialization finished", flush=True)

  epoch = 0
  if HP.TRAIN_USE_CHECKPOINT:
    epoch, master_parts = load_checkpoint(HP.TRAIN_CHECKPOINT_FILE)
    for x in master_parts.parameters():
      x = x.to(device)
    print("Loaded checkpoint", HP.TRAIN_CHECKPOINT_FILE, flush=True)

  MAX_EPOCH = HP.MAX_EPOCH

  samples_per_epoch = len(train_data_idx)

  tt = epoch*samples_per_epoch

  lr = HP.LEARN_RATE

  thax_sign, deriv_arits, thax_to_str = torch.load("{}/data_sign.pt".format(HP.TRAIN_BASE_FOLDER), weights_only=False)
    
  if not HP.TRAIN_USE_CHECKPOINT:
    master_parts = IC.get_initial_model(thax_sign, deriv_arits)
    model_name = "{}/master{}".format(HP.TRAIN_TRAIN_FOLDER, IC.name_initial_model_suffix())
    torch.save(master_parts, model_name)
    print(time.time() - start_time, "Created model parts and saved to", model_name, flush = True)
    
  times = []
  losses = []
  losses_devs = []
  posrates = []
  posrates_devs = []
  negrates = []
  negrates_devs = []

  stats = np.zeros((samples_per_epoch, 3)) # loss_sum, posOK_sum, negOK_sum
  weights = np.zeros((samples_per_epoch, 2)) # pos_weight, neg_weight

  train_data_orig = deepcopy(train_data_idx)

  torch.cuda.empty_cache()

  optimizer = torch.optim.Adam(master_parts.parameters(), lr = lr)
  # optimizer = optim.Adahessian(master_parts.parameters(), lr = 1.e-4)

  data_dict = {}
  for i, (size, name) in enumerate(train_data_idx):
    # print(i/len(train_data_idx))
    data_dict[name] = torch.load("{}/pieces/{}".format(HP.TRAIN_BASE_FOLDER, name), weights_only=False)

  print(flush=True)
  print(time.time() - start_time, "Starting loop", flush=True)

  while epoch < MAX_EPOCH:
    train_data_idx = deepcopy(train_data_orig)
    while train_data_idx:
      this_problem = random.choice(train_data_idx)
      train_data_idx.remove(this_problem)
      loss, posOK_sum, negOK_sum, tot_pos, tot_neg = eval_and_or_learn_on_one(this_problem[1], start_time, master_parts, data_dict[this_problem[1]], epoch)

      stats[tt % samples_per_epoch] = (loss, posOK_sum.cpu(), negOK_sum.cpu())
      weights[tt % samples_per_epoch] = (tot_pos.cpu(), tot_neg.cpu())
      tt += 1

      loss_sum, posOK_sum, negOK_sum = np.sum(stats, axis=0)
      tot_pos, tot_neg = np.sum(weights, axis=0)

      print("loss_sum, posOK_sum, negOK_sum", loss_sum, posOK_sum, negOK_sum, flush=True)
      print("tot_pos, tot_neg", tot_pos, tot_neg, flush=True)

      sum_stats = np.sum(stats, axis = 0)
      loss = sum_stats[0]/(tot_pos + tot_neg)
      posrate = sum_stats[1] / tot_pos
      negrate = sum_stats[2] / tot_neg

      loss_dev = weighted_std_deviation(loss, stats[:,0], np.sum(weights, axis = 1), tot_pos + tot_neg)
      posrate_dev = weighted_std_deviation(posrate, stats[:,1], weights[:,0], tot_pos)
      negrate_dev = weighted_std_deviation(negrate, stats[:,2], weights[:,1], tot_neg)

      print("Training stats:", flush=True)
      print("Loss:", loss, "+/-", loss_dev, flush=True)
      print("Posrate:", posrate, "+/-", posrate_dev, flush=True)
      print("Negrate:", negrate, "+/-", negrate_dev, flush=True)

      optimizer.step()
      for param in master_parts.parameters():
        param.grad = None
      print(time.time() - start_time, "Finished performing optimizer.step()", flush=True)

    epoch += 1

    print("Epoch", epoch, "finished at", time.time() - start_time, flush=True)
    save_checkpoint(epoch, master_parts)
    print(flush=True)

    # sum-up stats over the "samples_per_epoch" entries (retain the var name):
    loss_sum,posOK_sum,negOK_sum = np.sum(stats,axis=0)
    tot_pos,tot_neg = np.sum(weights,axis=0)
  
    print("loss_sum,posOK_sum,negOK_sum",loss_sum,posOK_sum,negOK_sum,flush=True)
    print("tot_pos,tot_neg",tot_pos,tot_neg,flush=True)

    sum_stats = np.sum(stats,axis=0)
    loss = sum_stats[0]/(tot_pos+tot_neg)
    posrate = sum_stats[1]/tot_pos
    negrate = sum_stats[2]/tot_neg

    loss_dev = weighted_std_deviation(loss,stats[:,0],np.sum(weights,axis=1),tot_pos+tot_neg)
    posrate_dev = weighted_std_deviation(posrate,stats[:,1],weights[:,0],tot_pos)
    negrate_dev = weighted_std_deviation(negrate,stats[:,2],weights[:,1],tot_neg)
    
    print("Training stats:",flush=True)
    print("Loss:",loss,"+/-",loss_dev,flush=True)
    print("Posrate:",posrate,"+/-",posrate_dev,flush=True)
    print("Negrate:",negrate,"+/-",negrate_dev,flush=True)
    print(flush=True)
  
    times.append(epoch)
    losses.append(loss)
    losses_devs.append(loss_dev)
    posrates.append(posrate)
    posrates_devs.append(posrate_dev)
    negrates.append(negrate)
    negrates_devs.append(negrate_dev)
    
    IC.plot_with_devs("{}/plot.png".format(HP.TRAIN_TRAIN_FOLDER),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)
