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

import inf_common as IC

import hyperparams as HP

import os
import subprocess

shm_thax = None
shm_ids = None
shm_rule_steps = None
shm_ind_steps = None
shm_pars_ind_steps = None
shm_good = None
shm_neg = None
shm_thax_to_str = None

def worker(shared_data, this_data):
    
  shm_thax = shm.SharedMemory(name=shared_data['shm_thax'])
  shm_ids = shm.SharedMemory(name=shared_data['shm_ids'])
  shm_rule_steps = shm.SharedMemory(name=shared_data['shm_rule_steps'])
  shm_ind_steps = shm.SharedMemory(name=shared_data['shm_ind_steps'])
  shm_pars_ind_steps = shm.SharedMemory(name=shared_data['shm_pars_ind_steps'])
  shm_good = shm.SharedMemory(name=shared_data['shm_good'])
  shm_neg = shm.SharedMemory(name=shared_data['shm_neg'])
  shm_thax_to_str = shm.SharedMemory(name=shared_data['shm_thax_to_str'])
  
  thax = msgpack.unpackb(shm_thax.buf)
  ids = msgpack.unpackb(shm_ids.buf)
  rule_steps = msgpack.unpackb(shm_rule_steps.buf)
  ind_steps = msgpack.unpackb(shm_ind_steps.buf)
  pars_ind_steps = msgpack.unpackb(shm_pars_ind_steps.buf)
  good = set(msgpack.unpackb(shm_good.buf))
  neg = set(msgpack.unpackb(shm_neg.buf))
  thax_to_str = msgpack.unpackb(shm_thax_to_str.buf, strict_map_key=False)

  this_init = {ids[i]: thax[i] for i in range(len(thax)) if thax_to_str[thax[i]] in this_data["axioms"]}
  these_ids = set(this_init.keys())

  this_deriv = []
  these_pars = {}

  for step, rule in enumerate(rule_steps):
    pars = pars_ind_steps[step]
    inds = ind_steps[step]

    if len(pars) == len(inds):  # Single parent case
      for i, ind in enumerate(pars):
        if ind in these_ids:
          new_id = inds[i]
          these_ids.add(new_id)
          this_deriv.append((new_id, rule))
          these_pars[new_id] = [ind]

    elif 2 * len(inds) == len(pars):  # Two-parent case
      for i, ind in enumerate(inds):
        p1, p2 = pars[2 * i], pars[2 * i + 1]
        if p1 in these_ids and p2 in these_ids:
          these_ids.add(ind)
          this_deriv.append((ind, rule))
          these_pars[ind] = [p1, p2]

  this_good = good & these_ids
  this_neg = neg & these_ids 

  init_abstractions = {"-1": -1} if -1 in this_init.values() else {}
  init_abstractions.update({thax_to_str[thax]: id for id, thax in this_init.items() if thax != -1})
  deriv_abstractions = {
    ",".join([str(rule)] + list(map(str, these_pars[id]))): id for id, rule in this_deriv
  }

  eval_store = {id: 1.0 for id in this_good}
  eval_store.update({id: 0.0 for id in this_neg})

  max_id = max((*init_abstractions.values(), *deriv_abstractions.values()), default=-1) + 1

  folder_file = HP.EXP_MODEL_FOLDER + "/" + this_data["short_name"] + ".model"
  IS.save_net(folder_file, init_abstractions, deriv_abstractions, eval_store, max_id)
  command = './vampire_Release_deepire3_4872 {} -tstat on --decode dis+1010_3:2_acc=on:afr=on:afp=1000:afq=1.2:amm=sco:bs=on:ccuc=first:fde=none:nm=0:nwc=4:urr=ec_only_100 -p off --output_axiom_names on -e4k {} -nesq on -nesqr 2,1 > {} 2>&1'.format(this_data["full_name"], folder_file, "results"+"/"+ "zero-test" + "/" + this_data["short_name"] + ".log")
  # print(command)
  subprocess.Popen(command,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
  
  command2 = 'rm {}'.format(folder_file)
  subprocess.Popen(command2,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()

if __name__ == "__main__":

  thax_sign, deriv_arits, thax_to_str = torch.load(HP.EXP_DATA_SIGN)
  print("Loaded signature from", HP.EXP_DATA_SIGN)

  IC.create_saver_zero(deriv_arits)
  import inf_saver_zero as IS

  problem_configuration = torch.load(HP.EXP_PROBLEM_CONFIGURATIONS)
  axiom2Number, number2Axiom = torch.load(HP.EXP_AXIOM_NUMBER_MAPPING)
  print("Loaded problem configurations", HP.EXP_PROBLEM_CONFIGURATIONS)

  with open(HP.EXP_PROBLEM_FILES, "r") as f:
    file_names = f.readlines()

  file_names = [x.replace("\n","") for x in file_names]
  prob_list = [{"short_name": x.rsplit("/", 1)[1], "full_name" : x} for x in file_names]

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
  serialized_good = msgpack.packb(list(good))
  serialized_neg = msgpack.packb(list(neg))
  serialized_thax_to_str = msgpack.packb(thax_to_str)

  del tree

  shm_thax = shm.SharedMemory(create=True, size=len(serialized_thax))
  shm_ids = shm.SharedMemory(create=True, size=len(serialized_ids))
  shm_rule_steps = shm.SharedMemory(create=True, size=len(serialized_rule_steps))
  shm_ind_steps = shm.SharedMemory(create=True, size=len(serialized_ind_steps))
  shm_pars_ind_steps = shm.SharedMemory(create=True, size=len(serialized_pars_ind_steps))
  shm_good = shm.SharedMemory(create=True, size=len(serialized_good))
  shm_neg = shm.SharedMemory(create=True, size=len(serialized_neg))
  shm_thax_to_str = shm.SharedMemory(create=True, size=len(serialized_thax_to_str))

  shm_thax.buf[:len(serialized_thax)] = serialized_thax
  shm_ids.buf[:len(serialized_ids)] = serialized_ids
  shm_rule_steps.buf[:len(serialized_rule_steps)] = serialized_rule_steps
  shm_ind_steps.buf[:len(serialized_ind_steps)] = serialized_ind_steps
  shm_pars_ind_steps.buf[:len(serialized_pars_ind_steps)] = serialized_pars_ind_steps
  shm_good.buf[:len(serialized_good)] = serialized_good
  shm_neg.buf[:len(serialized_neg)] = serialized_neg
  shm_thax_to_str.buf[:len(serialized_thax_to_str)] = serialized_thax_to_str

  shared_data = {
    'shm_thax': shm_thax.name,
    'shm_ids': shm_ids.name,
    'shm_rule_steps': shm_rule_steps.name,
    'shm_ind_steps': shm_ind_steps.name,
    'shm_pars_ind_steps': shm_pars_ind_steps.name,
    'shm_good': shm_good.name,
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
  shm_pos.close()
  shm_neg.close()
  shm_thax_to_str.close()

  shm_thax.unlink()
  shm_ids.unlink()
  shm_rule_steps.unlink()
  shm_ind_steps.unlink()
  shm_pars_ind_steps.unlink()
  shm_pos.unlink()
  shm_neg.unlink()
  shm_thax_to_str.unlink()