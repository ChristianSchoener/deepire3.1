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

import msgpack

import numpy as np

import inf_common as IC

import hyperparams as HP

import os
import subprocess

shm_thax = None
shm_ids = None
shm_rule_steps = None
shm_ind_steps = None
shm_pars_ind_steps = None
shm_neg = None
shm_thax_to_str = None

def worker(shared_data, this_data):
    
  shm_thax = shm.SharedMemory(name=shared_data['shm_thax'])
  shm_ids = shm.SharedMemory(name=shared_data['shm_ids'])
  shm_rule_steps = shm.SharedMemory(name=shared_data['shm_rule_steps'])
  shm_ind_steps = shm.SharedMemory(name=shared_data['shm_ind_steps'])
  shm_pars_ind_steps = shm.SharedMemory(name=shared_data['shm_pars_ind_steps'])
  shm_neg = shm.SharedMemory(name=shared_data['shm_neg'])
  shm_thax_to_str = shm.SharedMemory(name=shared_data['shm_thax_to_str'])
  
  thax = msgpack.unpackb(shm_thax.buf)
  ids = msgpack.unpackb(shm_ids.buf)
  rule_steps = msgpack.unpackb(shm_rule_steps.buf)
  ind_steps = msgpack.unpackb(shm_ind_steps.buf)
  pars_ind_steps = msgpack.unpackb(shm_pars_ind_steps.buf)
  neg = set(msgpack.unpackb(shm_neg.buf))
  thax_to_str = msgpack.unpackb(shm_thax_to_str.buf, strict_map_key=False)

  if HP.EXP_REVEAL:
    this_init = {ids[i]: thax[i] for i in range(len(thax))}
  else:
    this_init = {ids[i]: thax[i] for i in range(len(thax)) if thax_to_str[thax[i]] in this_data["axioms"]}
  these_ids = set(this_init.keys())

  deriv_abstractions_keys = torch.tensor([], dtype=torch.int32)
  deriv_abstractions_values = torch.tensor([], dtype=torch.int32)

  for step, rule in enumerate(rule_steps):
    pars = pars_ind_steps[step][:]
    inds = ind_steps[step][:]

    if len(pars) == len(inds):
      for i, par in enumerate(pars):
        if par in these_ids:
          new_id = ids[inds[i]]
          these_ids.add(new_id)
          deriv_abstractions_keys = torch.cat((deriv_abstractions_keys, torch.tensor((rule, par, par), dtype=torch.int32)))
          deriv_abstractions_values = torch.cat((deriv_abstractions_values, torch.tensor([new_id], dtype=torch.int32)))

    elif 2 * len(inds) == len(pars):
      for i, ind in enumerate(inds):
        p1, p2 = pars[2 * i], pars[2 * i + 1]
        if p1 in these_ids and p2 in these_ids:
          new_id = ids[ind]
          these_ids.add(new_id)
          deriv_abstractions_keys = torch.cat((deriv_abstractions_keys, torch.tensor((rule, p1, p2), dtype=torch.int32)))
          deriv_abstractions_values = torch.cat((deriv_abstractions_values, torch.tensor([new_id], dtype=torch.int32)))

  deriv_abstractions_keys = deriv_abstractions_keys.view(-1, 3)

  sorted_ind = np.lexsort(deriv_abstractions_keys.numpy()[:, ::-1].T)
  deriv_abstractions_keys = deriv_abstractions_keys[sorted_ind]
  deriv_abstractions_values = deriv_abstractions_values[sorted_ind]

  deriv_abstractions_keys_rule = deriv_abstractions_keys[:, 0].contiguous()
  deriv_abstractions_keys_first_par = deriv_abstractions_keys[:, 1].contiguous()
  deriv_abstractions_keys_second_par = deriv_abstractions_keys[:, 2].contiguous()
  del deriv_abstractions_keys

  this_neg = torch.tensor(list(neg & these_ids), dtype=torch.int32)

  this_neg, _ = torch.sort(this_neg)

  init_abstractions = {"-1": -1} if -1 in this_init.values() else {}
  init_abstractions.update({thax_to_str[this_thax]: id for id, this_thax in this_init.items() if this_thax != -1 and this_thax != 0})
  if HP.EXP_REVEAL:
    init_abstractions.update({"0": ids[thax.index(0)]})
    init_abstractions.update({this_thax: ids[thax.index(0)] for this_thax in set(this_data["axioms"]) - set(thax_to_str.values())})

  max_id = max((*init_abstractions.values(), *deriv_abstractions_values.tolist()), default=-1) + 1

  folder_file = HP.EXP_MODEL_FOLDER + "/" + this_data["short_name"] + ".model"

  IS.save_net(folder_file, 
              init_abstractions, 
              deriv_abstractions_keys_rule, 
              deriv_abstractions_keys_first_par, 
              deriv_abstractions_keys_second_par, 
              deriv_abstractions_values, 
              this_neg, 
              max_id)
  
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

  thax_to_str[-1] = "-1"
 
  tree = torch.load(HP.EXP_FILE, weights_only=False)
  print("Loaded tree from", HP.EXP_FILE)

  selec, good = torch.load(HP.EXP_SELEC_GOOD_FILE)
  print("Loaded selec and good from", HP.EXP_SELEC_GOOD_FILE)

  neg = selec - good

  del selec

# Store data in shared memory manager
  serialized_thax = msgpack.packb(tree["thax"].tolist())
  serialized_ids = msgpack.packb(tree["ids"].tolist())
  serialized_rule_steps = msgpack.packb(tree["rule_steps"].tolist())
  serialized_ind_steps = msgpack.packb([these_inds.tolist() for these_inds in tree["ind_steps"]])
  serialized_pars_ind_steps = msgpack.packb([these_pars_inds.tolist() for these_pars_inds in tree["pars_ind_steps"]])
  serialized_neg = msgpack.packb(list(neg))
  serialized_thax_to_str = msgpack.packb(thax_to_str)

  del tree

  shm_thax = shm.SharedMemory(create=True, size=len(serialized_thax))
  shm_ids = shm.SharedMemory(create=True, size=len(serialized_ids))
  shm_rule_steps = shm.SharedMemory(create=True, size=len(serialized_rule_steps))
  shm_ind_steps = shm.SharedMemory(create=True, size=len(serialized_ind_steps))
  shm_pars_ind_steps = shm.SharedMemory(create=True, size=len(serialized_pars_ind_steps))
  shm_neg = shm.SharedMemory(create=True, size=len(serialized_neg))
  shm_thax_to_str = shm.SharedMemory(create=True, size=len(serialized_thax_to_str))

  shm_thax.buf[:len(serialized_thax)] = serialized_thax
  shm_ids.buf[:len(serialized_ids)] = serialized_ids
  shm_rule_steps.buf[:len(serialized_rule_steps)] = serialized_rule_steps
  shm_ind_steps.buf[:len(serialized_ind_steps)] = serialized_ind_steps
  shm_pars_ind_steps.buf[:len(serialized_pars_ind_steps)] = serialized_pars_ind_steps
  shm_neg.buf[:len(serialized_neg)] = serialized_neg
  shm_thax_to_str.buf[:len(serialized_thax_to_str)] = serialized_thax_to_str

  shared_data = {
    'shm_thax': shm_thax.name,
    'shm_ids': shm_ids.name,
    'shm_rule_steps': shm_rule_steps.name,
    'shm_ind_steps': shm_ind_steps.name,
    'shm_pars_ind_steps': shm_pars_ind_steps.name,
    'shm_neg': shm_neg.name,
    'shm_thax_to_str': shm_thax_to_str.name
  }


  this_data = [{"short_name": prob_list[numProblem]["short_name"], 
                "full_name": prob_list[numProblem]["full_name"], 
                "axioms": {number2Axiom[x] for x in problem_configuration[prob_list[numProblem]["short_name"]]}
               } 
                for numProblem in range(len(prob_list))
              ]
  with multiprocessing.Pool(processes=HP.NUMPROCESSES) as pool:
    results = pool.starmap(worker, [(shared_data, data) for data in this_data])

  shm_thax.close()
  shm_ids.close()
  shm_rule_steps.close()
  shm_ind_steps.close()
  shm_pars_ind_steps.close()
  shm_neg.close()
  shm_thax_to_str.close()

  shm_thax.unlink()
  shm_ids.unlink()
  shm_rule_steps.unlink()
  shm_ind_steps.unlink()
  shm_pars_ind_steps.unlink()
  shm_neg.unlink()
  shm_thax_to_str.unlink()