#!/usr/bin/env python3
import torch

import inf_common as IC
import hyperparams as HP

import torch
torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat32 for faster matmul
# # torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking
# torch.backends.cuda.preferred_linalg_library("cusolver")  # Use optimized solvers
# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch import Tensor
import multiprocessing
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from copy import deepcopy

from multiprocessing import Lock

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools,os,gc

import numpy as np

# To release claimed memory back to os; Call:   libc.malloc_trim(ctypes.c_int(0))
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def copy_grads_back_from_param(parts,parts_copies):
  for param, param_copy in zip(parts.parameters(),parts_copies.parameters()):
    param.grad = param_copy

def copy_vals_to_his_parts(masterparts,his_parts):
# Copy gradients from weights to grads and weights from master to part
  with torch.no_grad():
    master_dict = dict(masterparts.named_parameters())
    for name, param in his_parts.named_parameters():
      if "weight" in name or "bias" in name:
        param.grad = param.data
        param.data = master_dict[name].data

def copy_vals_to_master_parts(masterparts,his_parts):
# Copy just the optimized weights back
  with torch.no_grad():
    master_dict = dict(masterparts.named_parameters())
    for name, param in his_parts.named_parameters():
      if "weight" in name or "bias" in name:
        master_dict[name].data = param.data

def eval_and_or_learn_on_one(these_problems, training, master_parts, start_time):
  # torch.cuda.empty_cache()
  print(time.time() - start_time,"Inner loop starts here.", flush=True)  
  tot_pos = 0.0
  tot_neg = 0.0
  for _, probname in these_problems:
    data = torch.load("{}/pieces/{}".format(sys.argv[1], probname), weights_only=False)
    tot_pos += data["tot_pos"]
    tot_neg += data["tot_neg"]
    print(time.time() - start_time,"Piece {} loaded.".format(probname), flush=True)
  loss_dict = {}
  if training:
    print(time.time() - start_time, "Training starts here.", flush=True)
    model = IC.LearningModel(*master_parts, data, True, HP.CUDA)
    # model = IC.LearningModel(*master_parts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg, greedy_eval_scheme, False, True)
    model.train()
    print(time.time() - start_time, "Starting training for {}".format(probname), flush=True)
    # with torch.amp.autocast(device_type="cuda"):
    (loss_dict[probname], posOK_sum, negOK_sum) = model()
    print(time.time() - start_time, "Finished training for {}".format(probname), flush=True)
    losses = sum(loss_dict.values())
    print(time.time() - start_time, "Starting backward propagation for problems of combined size", sum([size for size, _ in these_problems]), flush=True)
    losses.backward()
    print(time.time() - start_time, "Finished backward propagation for problems of combined size", sum([size for size, _ in these_problems]), flush=True)
  else:
    with torch.no_grad():
      model = IC.LearningModel(*master_parts, *data, False, HP.CUDA)
      model.eval()
      with torch.no_grad():
        (losses,posOK_sum,negOK_sum) = model()

    del model # I am getting desperate!

  return (losses,posOK_sum,negOK_sum,tot_pos,tot_neg,master_parts)

          # these_parts = torch.nn.ModuleList()
      # these_parts.append(torch.nn.ModuleDict())
      # with torch.no_grad():
      #   for i in range(1,len(master_parts)):
      #     these_parts.append(deepcopy(master_parts[i]))
      #   data = torch.load("{}/pieces/{}".format(sys.argv[1],probname),weights_only=False)
      #   (init,_,_,_,_,_,_,_) = data
      #   this_init = set()
      #   for _,(thax,_) in init:
      #     this_init.add(str(thax))
      #   this_init.add("0")
      #   for thax in this_init:
      #     these_parts[0][thax] = deepcopy(master_parts[0][thax])
      #   torch.save(these_parts,parts_file)
      # torch.save(master_parts, parts_file)

def worker(these_problems,train,master_parts):
  # size,probname,training,num,master_parts = input_data
  start_time = time.time()
  (loss,posOK_sum,negOK_sum,tot_pos,tot_neg,master_parts) = eval_and_or_learn_on_one(these_problems,train,master_parts,start_time)
  return loss,posOK_sum,negOK_sum,tot_pos,tot_neg,start_time,time.time(),master_parts

def save_checkpoint(epoch_num, epoch_id, model, optimizer):
  print("checkpoint",epoch,flush=True)

  check_name = "{}/check-epoch{}.pt".format(sys.argv[2],epoch_id)
  check = (epoch_num,model,optimizer)
  torch.save(check,check_name)

def load_checkpoint(filename):
  return torch.load(filename,weights_only=False)

def weighted_std_deviation(weighted_mean,scaled_values,weights,weight_sum):
  values = np.divide(scaled_values, weights, out=np.ones_like(scaled_values), where=weights!=0.0) # whenever the respective weight is 0.0, the result should be understood as 1.0
  squares = (values - weighted_mean)**2
  std_dev = np.sqrt(np.sum(weights*squares,axis=0) / weight_sum)
  return std_dev

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', force=True)
  log = open("{}/run{}".format(sys.argv[2],IC.name_learning_regime_suffix()), 'w')
  sys.stdout = log
  sys.stderr = log
  
  start_time = time.time()
  
  if HP.CUDA and torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
    
  train_data_idx = torch.load("{}/train_index.pt".format(sys.argv[1]),weights_only=False)
  print("Loaded train data:",len(train_data_idx),flush=True)
  valid_data_idx = torch.load("{}/valid_index.pt".format(sys.argv[1]),weights_only=False)
  print("Loaded valid data:",len(valid_data_idx),flush=True)
  if HP.SWAPOUT > 0.0:
    axiom_counts = torch.load("{}/axiom_counts.pt".format(sys.argv[1]),weights_only=False)
    print("Loaded axiom counts for uniformly distributed SWAPOUT",flush=True)

  # exclude = []
  # rule_52_counts = torch.load(sys.argv[1]+"/pieces/rule_52_counts.pt")
  # for num,count in rule_52_counts.items():
  #   if count > HP.MAX_RULE_52_LENGTH:
  #     exclude.append("piece" + num + ".pt")
  # print(flush=True)
  # print("len(train_data_idx) before rule_52 exclude:",len(train_data_idx),flush=True)
  # train_data_idx = [train_data_idx[x] for x in range(len(train_data_idx)) if str(x) not in exclude]
  # print("len(train_data_idx) after rule_52 exclude:",len(train_data_idx),flush=True)
  # print(flush=True)

  # train_data_idx = train_data_idx[0:1]

  if HP.TRR == HP.TestRiskRegimen_OVERFIT:
    # merge validation data back to training set (and ditch early stopping regularization)
    train_data_idx += valid_data_idx
    valid_data_idx = []
    print("Merged valid with train; final:",len(train_data_idx),flush=True)

  # if HP.ALL_ONCE:
  #   train_data_idx = [(x,y) for (x,y) in train_data_idx if x < HP.WHAT_IS_HUGE]
  #   print("Removed HUGE data from training, remaining:",len(train_data_idx),flush=True)

  total_count = len(train_data_idx)+len(valid_data_idx)

  print(flush=True)
  print(time.time() - start_time,"Initialization finished",flush=True)

  epoch = 0
  if len(sys.argv) >= 4:
    (epoch,master_parts,optimizer) = load_checkpoint(sys.argv[3])
    for x in master_parts.parameters():
      x = x.to(device)
    print("Loaded checkpoint",sys.argv[3],flush=True)

  MAX_EPOCH = HP.MAX_EPOCH

  samples_per_epoch = len(train_data_idx)

  tt = epoch*samples_per_epoch

  lr = HP.LEARN_RATE
  
  if len(sys.argv) >= 5:
    MAX_EPOCH = int(sys.argv[4])
    thax_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]),weights_only=False)
  else:
    thax_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]),weights_only=False)
    master_parts = IC.get_initial_model(thax_sign,deriv_arits)
    # for x in master_parts.parameters():
    #   x = x.to(device)
    model_name = "{}/master{}".format(sys.argv[2],IC.name_initial_model_suffix())
    torch.save(master_parts,model_name)
    print("Created model parts and saved to",model_name,flush=True)
    
  if HP.OPTIMIZER == HP.Optimizer_SGD:
    optimizer = torch.optim.SGD(master_parts.parameters(), lr=HP.LEARN_RATE)
  elif HP.OPTIMIZER == HP.Optimizer_ADAM:
    optimizer = torch.optim.Adam(master_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)
  elif  HP.OPTIMIZER == HP.Optimizer_ADAMW:
    optimizer = torch.optim.AdamW(master_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)

  times = []
  losses = []
  losses_devs = []
  posrates = []
  posrates_devs = []
  negrates = []
  negrates_devs = []

  stats = np.zeros((samples_per_epoch,3)) # loss_sum, posOK_sum, negOK_sum
  weights = np.zeros((samples_per_epoch,2)) # pos_weight, neg_weight

  # profiler = torch.profiler.profile(
  #   activities=[torch.profiler.ProfilerActivity.CPU],
  #   record_shapes=True,
  #   profile_memory=True,  
  #   with_stack=False)

  # profiler.start()

  print(flush=True)
  print(time.time() - start_time,"Starting loop",flush=True)

  pool = torch.multiprocessing.Pool(HP.NUMPROCESSES)

  train_data_orig = deepcopy(train_data_idx)
  train_data_idx = set(train_data_idx)

  while epoch < MAX_EPOCH:
    train_data_idx = set(deepcopy(train_data_orig))
    while len(train_data_idx):
      these_problems = random.sample(list(train_data_idx), min(len(train_data_idx), HP.SAMPLES))
      train_data_idx.difference_update(these_problems)
      # for results in pool.imap_unordered(worker, [(size,probname,True,tt+i,master_parts) for i,(size,probname) in enumerate(these_problems)]):
      loss,posOK_sum,negOK_sum,tot_pos,tot_neg,time_start,time_end,master_parts = worker(these_problems,True,master_parts)

      stats[tt % samples_per_epoch] = (loss.cpu().item(),posOK_sum.cpu(),negOK_sum.cpu())
      weights[tt % samples_per_epoch] = (tot_pos,tot_neg)

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

      print(time.time() - start_time,"Performing optimizer.step()",flush=True)
      # losses = sum(loss_dict).to(device)
      # losses.backward()
      optimizer.step()
      optimizer.zero_grad()
      print(time.time() - start_time,"Finished performing optimizer.step()", flush=True)
      torch.cuda.empty_cache()

      tt += 1
      if HP.NON_CONSTANT_10_50_250_LR:
        if tt <= 40*samples_per_epoch:
          lr = HP.LEARN_RATE*tt/(10*samples_per_epoch)
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr
          print("Increasing effective LR to",lr,flush=True)
        else:
          lr = 160*samples_per_epoch/tt*HP.LEARN_RATE
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr
          print("Dropping effective LR to",lr,flush=True)
      
    epoch += 1

    print("Epoch",epoch,"finished at",time.time() - start_time,flush=True)
    save_checkpoint(epoch,epoch,master_parts,optimizer)
    print(flush=True)

    # sum-up stats over the "samples_per_epoch" entries (retain the var name):
    loss_sum,posOK_sum,negOK_sum = np.sum(stats,axis=0)
    tot_pos,tot_neg = np.sum(weights,axis=0)
  
    print("loss_sum,posOK_sum,negOK_sum",loss_sum,posOK_sum,negOK_sum,flush=True)
    print("tot_pos,tot_neg",tot_pos,tot_neg,flush=True)

    # CAREFULE: could divide by zero!
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
    
    IC.plot_with_devs("{}/plot.png".format(sys.argv[2]),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)
      
  # profiler.stop() 

# Print results
  # print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=100))
