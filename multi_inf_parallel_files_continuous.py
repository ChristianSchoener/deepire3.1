#!/usr/bin/env python3

import os

import inf_common as IC
import hyperparams as HP

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch import Tensor

from copy import deepcopy

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
    his_dict = dict(his_parts.named_parameters())
    for name, _ in his_parts.named_parameters():
      if "weight" in name:
        his_dict[name].grad = his_dict[name].data
        his_dict[name].data = master_dict[name].data

def copy_vals_to_master_parts(masterparts,his_parts):
# Copy just the optimized weights back
  with torch.no_grad():
    master_dict = dict(masterparts.named_parameters())
    his_dict = dict(his_parts.named_parameters()) 
    for name, _ in his_parts.named_parameters():
      if "weight" in name:
        master_dict[name].data = his_dict[name].data

def eval_and_or_learn_on_one(size_and_prob_list,parts_file,training,log):
  loss_sum_list = []
  posOK_sum_list = []
  negOK_sum_list = []
  tot_pos_list = []
  tot_neg_list = []

  myparts = torch.load(parts_file)
    # not sure if there is any after load -- TODO: check if necessary
  for param in myparts.parameters():
    # taken from Optmizier zero_grad, roughly
    with torch.no_grad():
      if param.grad is not None:
        param.grad.zero_()
    param.requires_grad = True

  if training:
    for _,prob in size_and_prob_list:
      data = torch.load("{}/pieces/{}".format(sys.argv[1],prob))
      (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,_) = data

      model = IC.LearningModel(*myparts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg, False, True)
      model.train()
      (loss_sum,posOK_sum,negOK_sum) = model()
    
      loss_sum.backward()
      loss_sum_list.append(loss_sum[0].detach().item())
      posOK_sum_list.append(posOK_sum)
      negOK_sum_list.append(negOK_sum)
      tot_pos_list.append(tot_pos)
      tot_neg_list.append(tot_neg)
    # put grad into actual tensor to be returned below (gradients don't go through the Queue)
    for param in myparts.parameters():
      grad = param.grad
      param.requires_grad = False # to allow the in-place operation just below
      if grad is not None:
        param.copy_(grad)
      else:
        param.zero_()
    del model
    
    torch.save(myparts,parts_file)
  
  else:
    with torch.no_grad():
      model = IC.LearningModel(*myparts,init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg, False, False)
      model.eval()
      (loss_sum,posOK_sum,negOK_sum) = model()

      loss_sum_list.append(loss_sum[0].detach().item())
      posOK_sum_list.append(posOK_sum)
      negOK_sum_list.append(negOK_sum)
      tot_pos_list.append(tot_pos)
      tot_neg_list.append(tot_neg)

    del model # I am getting desperate!

  return (loss_sum_list,posOK_sum_list,negOK_sum_list,tot_pos_list,tot_neg_list)

def worker(q_in, q_out):
  log = sys.stdout

  while True:
    (size_and_prob_list,parts_file,training) = q_in.get()
    
    start_time = time.time()
    (loss_sum_list,posOK_sum_list,negOK_sum_list,tot_pos_list,tot_neg_list) = eval_and_or_learn_on_one(size_and_prob_list,parts_file,training,log)
    q_out.put((size_and_prob_list,loss_sum_list,posOK_sum_list,negOK_sum_list,tot_pos_list,tot_neg_list,parts_file,start_time,time.time()))

    libc.malloc_trim(ctypes.c_int(0))

def save_checkpoint(epoch_num, epoch_id, model, optimizer):
  print("checkpoint",epoch)

  check_name = "{}/check-epoch{}.pt".format(sys.argv[2],epoch_id)
  check = (epoch_num,model,optimizer)
  torch.save(check,check_name)

def load_checkpoint(filename):
  return torch.load(filename)

def weighted_std_deviation(weighted_mean,scaled_values,weights,weight_sum):
  # print(weighted_mean)
  # print(scaled_values)
  # print(weights)
  # print(weight_sum,flush=True)

  values = np.divide(scaled_values, weights, out=np.ones_like(scaled_values), where=weights!=0.0) # whenever the respective weight is 0.0, the result should be understood as 1.0
  squares = (values - weighted_mean)**2
  std_dev = np.sqrt(np.sum(weights*squares,axis=0) / weight_sum)
  return std_dev

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # Learn in parallel using a Pool of processes, or something similar
  #
  # probably needs to be run with "ulimit -Sn 3000" or something large
  #
  # To be called as in: ./multi_inf_parallel.py <folder_in> <folder_out> <initial-model>
  #
  # it expects <folder_in> to contain "training_data.pt" and "validation_data.pt"
  # (and maybe also "data_sign.pt")
  #
  # if <initial-model> is not specified,
  # it creates a new one in <folder_out> using the same naming scheme as initializer.py
  #
  # The log, the plot, and intermediate models are also saved in <folder_out>
  
  # global redirect of prints to the just upen "logfile"
  log = open("{}/run{}".format(sys.argv[2],IC.name_learning_regime_suffix()), 'w')
  sys.stdout = log
  sys.stderr = log
  
  start_time = time.time()
  
  train_data_idx = torch.load("{}/training_index.pt".format(sys.argv[1]))
  print("Loaded train data:",len(train_data_idx))
  valid_data_idx = torch.load("{}/validation_index.pt".format(sys.argv[1]))
  print("Loaded valid data:",len(valid_data_idx))
  
  if HP.TRR == HP.TestRiskRegimen_OVERFIT:
    # merge validation data back to training set (and ditch early stopping regularization)
    train_data_idx += valid_data_idx
    valid_data_idx = []
    print("Merged valid with train; final:",len(train_data_idx))

  if HP.ALL_ONCE:
    train_data_idx = [(x,y) for (x,y) in train_data_idx if x < HP.WHAT_IS_HUGE]
    print("Removed HUGE data from training, remaining:",len(train_data_idx))

  print()
  print(time.time() - start_time,"Initialization finished",flush=True)

  epoch = 0

  MAX_EPOCH = HP.MAX_EPOCH
  
  if len(sys.argv) >= 4:
    (epoch,master_parts,optimizer) = load_checkpoint(sys.argv[3])
    print("Loaded checkpoint",sys.argv[3],flush=True)
  
    if len(sys.argv) >= 5:
      MAX_EPOCH = int(sys.argv[4])
  
    # update the learning rate according to hyperparams
    for param_group in optimizer.param_groups:
        param_group['lr'] = HP.LEARN_RATE
    print("Set optimizer's (nominal) learning rate to",HP.LEARN_RATE)
  else:
    thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load("{}/data_sign.pt".format(sys.argv[1]))
    master_parts = IC.get_initial_model(thax_sign,sine_sign,deriv_arits)
    model_name = "{}/initial{}".format(sys.argv[2],IC.name_initial_model_suffix())
    torch.save(master_parts,model_name)
    print("Created model parts and saved to",model_name,flush=True)
    
    if HP.OPTIMIZER == HP.Optimizer_SGD: # could also play with momentum and its dampening here!
      optimizer = torch.optim.SGD(master_parts.parameters(), lr=HP.LEARN_RATE)
    elif HP.OPTIMIZER == HP.Optimizer_ADAM:
      optimizer = torch.optim.Adam(master_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)
    elif  HP.OPTIMIZER == HP.Optimizer_ADAMW:
      optimizer = torch.optim.AdamW(master_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(HP.NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  times = []
  losses = []
  losses_devs = []
  posrates = []
  posrates_devs = []
  negrates = []
  negrates_devs = []

  id = 0 # id synchronizes writes to the worker pipe

  # train_data_idx = train_data_idx[:100]

  samples_per_epoch = len(train_data_idx)

  t = epoch*samples_per_epoch # time synchronizes writes to master_parts and the stasts

  stats = np.zeros((samples_per_epoch,3)) # loss_sum, posOK_sum, negOK_sum
  weights = np.zeros((samples_per_epoch,2)) # pos_weight, neg_weight

  MAX_ACTIVE_TASKS = HP.NUMPROCESSES
  num_active_tasks = 0

  MAX_CAPACITY = 1000000
  assert HP.NUMPROCESSES * HP.COMPRESSION_THRESHOLD * 5 // 4 < MAX_CAPACITY
  cur_allocated = 0

  if HP.ALL_ONCE:
    train_data_idx_orig = deepcopy(train_data_idx)
  while True:
    while num_active_tasks < MAX_ACTIVE_TASKS and len(train_data_idx) > 0:
      num_active_tasks += 1
      pieces_active_task = 0
      size_and_prob_list = []
      while len(train_data_idx) > 0 and (pieces_active_task < HP.NUM_PIECES_SIMULTANEOUS or sum([x for x,_ in size_and_prob_list]) < HP.WORKER_LOAD):
        (size,probname) = random.choice(train_data_idx)
        size_and_prob_list.append((size,probname))
        pieces_active_task += 1
        if HP.ALL_ONCE:
          train_data_idx  = list(set(train_data_idx).difference({(size,probname)}))
        
      print(len(size_and_prob_list),"problems of total size",sum([x for x,_ in size_and_prob_list]),"loaded.",len(train_data_idx),"pieces remaining for this epoch.")
      # id += 1
      # parts_file = "{}/parts_{}_{}.pt".format(HP.SCRATCH,os.getpid(),id)
      parts_file = sys.argv[1] + "/pieces/" + size_and_prob_list[0][1].split(".")[0] + "_parts.pt"
      these_parts = torch.nn.ModuleList()
      these_parts.append(torch.nn.ModuleDict())
      with torch.no_grad():
        for i in range(1,len(master_parts)):
          these_parts.append(deepcopy(master_parts[i]))
        data = torch.load("{}/pieces/{}".format(sys.argv[1],size_and_prob_list[0][1]))
        (init,_,_,_,_,_,_,_) = data
        this_init = set()
        for _,(thax,_) in init:
          this_init.add(str(thax))
        this_init.add("0")
        for thax in this_init:
          these_parts[0][thax] = deepcopy(master_parts[0][thax])
        torch.save(these_parts,parts_file)
      q_in.put((size_and_prob_list, parts_file, True))

    (size_and_prob_list,loss_sum_list,posOK_sum_list,negOK_sum_list,tot_pos_list,tot_neg_list,parts_file,time_start,time_end) = q_out.get() # this may block

    cur_allocated -= sum([size for size,_ in size_and_prob_list]) 

    for i in range(len(size_and_prob_list)):
      stats[t % samples_per_epoch] = (loss_sum_list[i],posOK_sum_list[i],negOK_sum_list[i])
      weights[t % samples_per_epoch] = (tot_pos_list[i],tot_neg_list[i])

      t += 1

    # sum-up stats over the "samples_per_epoch" entries (retain the var name):
    loss_sum,posOK_sum,negOK_sum = np.sum(stats,axis=0)
    tot_pos,tot_neg = np.sum(weights,axis=0)
  
    print("loss_sum,posOK_sum,negOK_sum",loss_sum,posOK_sum,negOK_sum)
    print("tot_pos,tot_neg",tot_pos,tot_neg)

    # CAREFULE: could divide by zero!
    sum_stats = np.sum(stats,axis=0)
    loss = sum_stats[0]/(tot_pos+tot_neg)
    posrate = sum_stats[1]/tot_pos
    negrate = sum_stats[2]/tot_neg

    loss_dev = weighted_std_deviation(loss,stats[:,0],np.sum(weights,axis=1),tot_pos+tot_neg)
    posrate_dev = weighted_std_deviation(posrate,stats[:,1],weights[:,0],tot_pos)
    negrate_dev = weighted_std_deviation(negrate,stats[:,2],weights[:,1],tot_neg)
    
    print("Training stats:")
    print("Loss:",loss,"+/-",loss_dev,flush=True)
    print("Posrate:",posrate,"+/-",posrate_dev,flush=True)
    print("Negrate:",negrate,"+/-",negrate_dev,flush=True)
  
    if HP.NON_CONSTANT_10_50_250_LR:
      if t <= 40*samples_per_epoch: # initial warmup: take "50 000" optimizer steps (= 50 epochs) to reach 5*HP.LEARN_RATE (in 10 epochs, HP.LEARN_RATE has been reached and then it's gradually overshot)
        lr = HP.LEARN_RATE*t/(10*samples_per_epoch)
        print("Increasing effective LR to",lr,flush=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
      else: # hyperbolic cooldown (reach HP.LEARN_RATE at "250 000" = 250 epochs) # newly reaches HP.LEARN_RATE at epoch 160
        lr = 160*samples_per_epoch/t*HP.LEARN_RATE
        print("Dropping effective LR to",lr,flush=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    num_active_tasks -= 1
    his_parts = torch.load(parts_file)
    # os.remove(parts_file)
    copy_vals_to_his_parts(master_parts,his_parts)
    this_optimizer = torch.optim.Adam(his_parts.parameters(), lr=HP.LEARN_RATE, weight_decay=HP.WEIGHT_DECAY)
    this_optimizer.step()
    copy_vals_to_master_parts(master_parts,his_parts)
    # copy_grads_back_from_param(master_parts,his_parts)
    # print(time.time() - start_time,"copy_grads_back_from_param finished")
    # if HP.CLIP_GRAD_NORM:
    #   torch.nn.utils.clip_grad_norm_(master_parts.parameters(), HP.CLIP_GRAD_NORM)
    # if HP.CLIP_GRAD_VAL:
    #   torch.nn.utils.clip_grad_value_(master_parts.parameters(), HP.CLIP_GRAD_VAL)
      
    # optimizer.step()

    print(time.time() - start_time,"optimizer.step() and copying finished", flush=True)

    modulus = t % samples_per_epoch
    print("Modulus =",modulus,flush=True)
    if modulus == 0:
      epoch += 1

      if HP.ALL_ONCE:
        train_data_idx = deepcopy(train_data_idx_orig)
    
      print("Epoch",epoch,"finished at",time.time() - start_time)
      save_checkpoint(epoch,epoch,master_parts,optimizer)
      print()

      # print("stats",stats)
      # print("weights",weights)

      # sum-up stats over the "samples_per_epoch" entries (retain the var name):
      loss_sum,posOK_sum,negOK_sum = np.sum(stats,axis=0)
      tot_pos,tot_neg = np.sum(weights,axis=0)
    
      print("loss_sum,posOK_sum,negOK_sum",loss_sum,posOK_sum,negOK_sum)
      print("tot_pos,tot_neg",tot_pos,tot_neg)

      # CAREFULE: could divide by zero!
      sum_stats = np.sum(stats,axis=0)
      loss = sum_stats[0]/(tot_pos+tot_neg)
      posrate = sum_stats[1]/tot_pos
      negrate = sum_stats[2]/tot_neg

      loss_dev = weighted_std_deviation(loss,stats[:,0],np.sum(weights,axis=1),tot_pos+tot_neg)
      posrate_dev = weighted_std_deviation(posrate,stats[:,1],weights[:,0],tot_pos)
      negrate_dev = weighted_std_deviation(negrate,stats[:,2],weights[:,1],tot_neg)
      
      print("Training stats:")
      print("Loss:",loss,"+/-",loss_dev,flush=True)
      print("Posrate:",posrate,"+/-",posrate_dev,flush=True)
      print("Negrate:",negrate,"+/-",negrate_dev,flush=True)
      print()
    
      times.append(epoch)
      losses.append(loss)
      losses_devs.append(loss_dev)
      posrates.append(posrate)
      posrates_devs.append(posrate_dev)
      negrates.append(negrate)
      negrates_devs.append(negrate_dev)
      
      IC.plot_with_devs("{}/plot.png".format(sys.argv[2]),times,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs)
      
      if epoch >= MAX_EPOCH:
        break
    else:
      # fractional checkpointing ... TODO: make only enabled later in the run?
      if HP.FRACTIONAL_CHECKPOINTING > 0:
        modulus_frac = samples_per_epoch // HP.FRACTIONAL_CHECKPOINTING
        if modulus % modulus_frac == 0:
          frac = modulus // modulus_frac
          print("Epoch frac {}.{} finished at {}".format(epoch,frac,time.time() - start_time))
          save_checkpoint(epoch+frac/HP.FRACTIONAL_CHECKPOINTING,"{}.{}".format(epoch,frac),master_parts,optimizer)
          print()
      
  # a final "cleanup"
  for p in my_processes:
    p.kill()
