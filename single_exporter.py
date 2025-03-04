#!/usr/bin/env python3

import torch
from torch import Tensor

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import multiprocessing
import time

import multiprocessing.shared_memory as shm

from concurrent.futures import ProcessPoolExecutor

from joblib import Parallel, delayed

from typing import Dict, List, Tuple, Optional

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from collections import defaultdict
from collections import ChainMap

import gc

import msgpack

import numpy as np

import inf_common as IC

import hyperparams as HP

import os
import subprocess

shm_thax = None
shm_neg = None
shm_thax_to_str = None
shm_deriv_abstractions_keys_rule = None
shm_deriv_abstractions_keys_first_par = None
shm_deriv_abstractions_keys_second_par = None
shm_deriv_abstractions_values = None

def worker(shared_data, this_data):
    
  shm_thax = shm.SharedMemory(name=shared_data['shm_thax'])
  shm_neg = shm.SharedMemory(name=shared_data['shm_neg'])
  shm_thax_to_str = shm.SharedMemory(name=shared_data['shm_thax_to_str'])
  shm_deriv_abstractions_keys_rule = shm.SharedMemory(name=shared_data['shm_deriv_abstractions_keys_rule'])
  shm_deriv_abstractions_keys_first_par = shm.SharedMemory(name=shared_data['shm_deriv_abstractions_keys_first_par'])
  shm_deriv_abstractions_keys_second_par = shm.SharedMemory(name=shared_data['shm_deriv_abstractions_keys_second_par'])
  shm_deriv_abstractions_values = shm.SharedMemory(name=shared_data['shm_deriv_abstractions_values'])
  
  thax = msgpack.unpackb(shm_thax.buf)
  neg = torch.tensor(msgpack.unpackb(shm_neg.buf), dtype=torch.int32)
  thax_to_str = msgpack.unpackb(shm_thax_to_str.buf, strict_map_key=False)
  deriv_abstractions_keys_rule = msgpack.unpackb(shm_deriv_abstractions_keys_rule.buf)
  deriv_abstractions_keys_first_par = msgpack.unpackb(shm_deriv_abstractions_keys_first_par.buf)
  deriv_abstractions_keys_second_par = msgpack.unpackb(shm_deriv_abstractions_keys_second_par.buf)
  deriv_abstractions_values = msgpack.unpackb(shm_deriv_abstractions_values.buf)

  shm_thax.close()
  shm_neg.close()
  shm_thax_to_str.close()
  shm_deriv_abstractions_keys_rule.close()
  shm_deriv_abstractions_keys_first_par.close()
  shm_deriv_abstractions_keys_second_par.close()
  shm_deriv_abstractions_values.close()

  thax_to_str[-1] = thax.index(-1)
  if HP.EXP_REVEAL:
    thax_to_str[0] = thax.index(0)

  init_abstractions = {}
  init_abstractions.update({str(thax_to_str[this_thax]): thax.index(this_thax) for this_thax in thax})
  if HP.EXP_REVEAL:
    init_abstractions.update({str(this_thax): thax.index(0) for this_thax in set(this_data["axioms"]) - set(thax_to_str.values())})

  del thax, thax_to_str

  max_id = max((*init_abstractions.values(), *deriv_abstractions_values), default=-1) + 1

  folder_file = HP.EXP_MODEL_FOLDER + "/" + this_data["short_name"] + ".model"

  IS.save_net(folder_file, 
              init_abstractions, 
              torch.tensor(deriv_abstractions_keys_rule, dtype=torch.int32), 
              torch.tensor(deriv_abstractions_keys_first_par, dtype=torch.int32), 
              torch.tensor(deriv_abstractions_keys_second_par, dtype=torch.int32), 
              torch.tensor(deriv_abstractions_values, dtype=torch.int32), 
              neg, 
              max_id)
  
  del init_abstractions, deriv_abstractions_keys_rule, deriv_abstractions_keys_first_par, deriv_abstractions_keys_second_par, deriv_abstractions_values, neg, max_id

  gc.collect()
  
  command = './vampire_Release_deepire3_4872 {} -tstat on --decode dis+1010_3:2_acc=on:afr=on:afp=1000:afq=1.2:amm=sco:bs=on:ccuc=first:fde=none:nm=0:nwc=4:urr=ec_only_100 -p off --output_axiom_names on -e4k {} -nesq on -nesqr 2,1 > {} 2>&1'.format(this_data["full_name"], folder_file, HP.EXP_RESULTS_FOLDER + "/" + this_data["short_name"] + ".log")
  # print(command)
  subprocess.Popen(command,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
  
  command2 = 'rm {}'.format(folder_file)
  subprocess.Popen(command2,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()

if __name__ == "__main__":

  thax_sign, deriv_arits, thax_to_str = torch.load(HP.EXP_DATA_SIGN_PREPARED)
  print("Loaded signature from", HP.EXP_DATA_SIGN_PREPARED)

  IC.create_saver_zero(deriv_arits)
  import inf_saver_zero as IS

  problem_configuration = torch.load(HP.EXP_PROBLEM_CONFIGURATIONS)
  axiom2Number, number2Axiom = torch.load(HP.EXP_AXIOM_NUMBER_MAPPING)
  print("Loaded problem configurations", HP.EXP_PROBLEM_CONFIGURATIONS)

  with open(HP.EXP_PROBLEM_FILES, "r") as f:
    file_names = f.readlines()

  file_names = [x.replace("\n","") for x in file_names]
  prob_list = [{"short_name": x.rsplit("/", 1)[1], "full_name" : x} for x in file_names if x.strip() != ""]

  del file_names
 
  tree = torch.load(HP.EXP_FILE, weights_only=False)
  print("Loaded tree from", HP.EXP_FILE)

  selec, good = torch.load(HP.EXP_SELEC_GOOD_FILE)
  print("Loaded selec and good from", HP.EXP_SELEC_GOOD_FILE)

  neg = set(selec.tolist()) - set(good.tolist())
  neg = list(np.where(np.isin(np.array(tree["ids"]), np.array(list(neg))))[0])
  neg = torch.tensor(neg, dtype=torch.int32)
  neg, _ = torch.sort(neg)
  neg = neg.tolist()

  del selec, good

  if HP.EXP_LOAD_DERIV_ABSTRACTIONS:
    deriv_abstractions_values, deriv_abstractions_keys_rule, deriv_abstractions_keys_first_par, deriv_abstractions_keys_second_par = torch.load(HP.EXP_LOAD_DERIV_ABSTRACTIONS_FILE)
  else:
    # Initialize empty tensors
    deriv_abstractions_keys = torch.tensor([], dtype=torch.int32)
    deriv_abstractions_values = torch.cat(tree["ind_steps"])

    for step, rule in enumerate(tree["rule_steps"]):
      print(step, tree["rule_steps"].numel(), flush=True)
      ind_step_len = tree["ind_steps"][step].numel()
      pars_step_len = tree["pars_ind_steps"][step].numel()
    
      if ind_step_len == pars_step_len:
        repeated_rule = rule.repeat(ind_step_len)
        stacked_tensors = torch.stack([repeated_rule, tree["pars_ind_steps"][step], tree["pars_ind_steps"][step]])
        deriv_abstractions_keys = torch.cat((deriv_abstractions_keys, stacked_tensors.T), dim=0)
      elif 2 * ind_step_len == pars_step_len:
        repeated_rule = rule.repeat(ind_step_len)
        pars_step = tree["pars_ind_steps"][step]
        stacked_tensors = torch.stack([repeated_rule, pars_step[::2], pars_step[1::2]], dim=0)
        deriv_abstractions_keys = torch.cat((deriv_abstractions_keys, stacked_tensors.T), dim=0)

    # Reshape and sort
    deriv_abstractions_keys = deriv_abstractions_keys.view(-1, 3)
    sorted_ind = np.lexsort(deriv_abstractions_keys.numpy()[:, ::-1].T)

    # Apply sorting
    deriv_abstractions_keys = deriv_abstractions_keys[sorted_ind]
    deriv_abstractions_values = deriv_abstractions_values[sorted_ind]

    # Split into separate tensors
    deriv_abstractions_keys_rule = deriv_abstractions_keys[:, 0].contiguous()
    deriv_abstractions_keys_first_par = deriv_abstractions_keys[:, 1].contiguous()
    deriv_abstractions_keys_second_par = deriv_abstractions_keys[:, 2].contiguous()

    torch.save((deriv_abstractions_values, deriv_abstractions_keys_rule, deriv_abstractions_keys_first_par, deriv_abstractions_keys_second_par), HP.EXP_LOAD_DERIV_ABSTRACTIONS_FILE)


  # Store data in shared memory manager
  serialized_thax = msgpack.packb(tree["thax"].tolist())
  serialized_neg = msgpack.packb(neg)
  serialized_thax_to_str = msgpack.packb(thax_to_str)
  serialized_deriv_abstractions_keys_rule = msgpack.packb(deriv_abstractions_keys_rule.tolist())
  serialized_deriv_abstractions_keys_first_par = msgpack.packb(deriv_abstractions_keys_first_par.tolist())
  serialized_deriv_abstractions_keys_second_par = msgpack.packb(deriv_abstractions_keys_second_par.tolist())
  serialized_deriv_abstractions_values = msgpack.packb(deriv_abstractions_values.tolist())

  shm_thax = shm.SharedMemory(create=True, size=len(serialized_thax))
  shm_neg = shm.SharedMemory(create=True, size=len(serialized_neg))
  shm_thax_to_str = shm.SharedMemory(create=True, size=len(serialized_thax_to_str))
  shm_deriv_abstractions_keys_rule = shm.SharedMemory(create=True, size=len(serialized_deriv_abstractions_keys_rule))  
  shm_deriv_abstractions_keys_first_par = shm.SharedMemory(create=True, size=len(serialized_deriv_abstractions_keys_first_par))
  shm_deriv_abstractions_keys_second_par = shm.SharedMemory(create=True, size=len(serialized_deriv_abstractions_keys_second_par))
  shm_deriv_abstractions_values = shm.SharedMemory(create=True, size=len(serialized_deriv_abstractions_values))

  shm_thax.buf[:len(serialized_thax)] = serialized_thax
  shm_neg.buf[:len(serialized_neg)] = serialized_neg
  shm_thax_to_str.buf[:len(serialized_thax_to_str)] = serialized_thax_to_str

  shm_deriv_abstractions_keys_rule.buf[:len(serialized_deriv_abstractions_keys_rule)] = serialized_deriv_abstractions_keys_rule  
  shm_deriv_abstractions_keys_first_par.buf[:len(serialized_deriv_abstractions_keys_first_par)] = serialized_deriv_abstractions_keys_first_par
  shm_deriv_abstractions_keys_second_par.buf[:len(serialized_deriv_abstractions_keys_second_par)] = serialized_deriv_abstractions_keys_second_par
  shm_deriv_abstractions_values.buf[:len(serialized_deriv_abstractions_values)] = serialized_deriv_abstractions_values

  shared_data = {
    'shm_thax': shm_thax.name,
    'shm_neg': shm_neg.name,
    'shm_thax_to_str': shm_thax_to_str.name,
    'shm_deriv_abstractions_keys_rule': shm_deriv_abstractions_keys_rule.name,
    'shm_deriv_abstractions_keys_first_par': shm_deriv_abstractions_keys_first_par.name,
    'shm_deriv_abstractions_keys_second_par': shm_deriv_abstractions_keys_second_par.name,
    'shm_deriv_abstractions_values': shm_deriv_abstractions_values.name
  }

  del tree
  gc.collect()

  this_data = [{"short_name": prob_list[numProblem]["short_name"], 
                "full_name": prob_list[numProblem]["full_name"], 
                "axioms": {number2Axiom[x] for x in problem_configuration[prob_list[numProblem]["short_name"]]}
               } 
                for numProblem in range(len(prob_list))
              ]
  with multiprocessing.Pool(processes=HP.NUMPROCESSES) as pool:
    results = pool.starmap(worker, [(shared_data, data) for data in this_data])

  shm_thax.unlink()
  shm_neg.unlink()
  shm_thax_to_str.unlink()
  shm_deriv_abstractions_keys_rule.unlink()
  shm_deriv_abstractions_keys_first_par.unlink()
  shm_deriv_abstractions_keys_second_par.unlink()
  shm_deriv_abstractions_values.unlink()
