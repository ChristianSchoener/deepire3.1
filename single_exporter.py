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

shm_init = None
shm_deriv = None
shm_pars = None
shm_good = None
shm_neg = None
shm_thax_to_str = None

def chunk_large_data(data, chunk_size=100000):
  chunks = []
  items = list(data.items())
  
  for i in range(0, len(items), chunk_size):
    chunk_dict = dict(items[i:i + chunk_size])  # Create a dictionary for each chunk
    chunks.append(chunk_dict)
  
  return chunks

def worker(shared_data, this_data):
    
  shm_init = shm.SharedMemory(name=shared_data['shm_init'])
  shm_deriv = shm.SharedMemory(name=shared_data['shm_deriv'])
  shm_pars = shm.SharedMemory(name=shared_data['shm_pars'])
  shm_good = shm.SharedMemory(name=shared_data['shm_good'])
  shm_neg = shm.SharedMemory(name=shared_data['shm_neg'])
  shm_thax_to_str = shm.SharedMemory(name=shared_data['shm_thax_to_str'])
  
  init = msgpack.unpackb(shm_init.buf)
  deriv = msgpack.unpackb(shm_deriv.buf)
  pars = msgpack.unpackb(shm_pars.buf, strict_map_key=False)
  good = set(msgpack.unpackb(shm_good.buf))
  neg = set(msgpack.unpackb(shm_neg.buf))
  thax_to_str = msgpack.unpackb(shm_thax_to_str.buf, strict_map_key=False)

  this_init = [(id, thax) for id, thax in init if thax_to_str[thax] in this_data["axioms"]]
  these_ids = {x for x, _ in this_init}
  this_deriv = []
  these_pars = {}
  for id, rule in deriv:
    if set(pars[id]) <= these_ids:
      this_deriv.append((id, rule))
      these_ids.add(id)
      these_pars[id] = pars[id]
  this_good = good & these_ids
  this_neg = neg & these_ids 

  init_abstractions = {}
  for id, thax_nr in this_init:
    if thax_nr == -1:
      init_abstractions["-1"] = -1
    else:
      init_abstractions[thax_to_str[thax_nr]] = id
  deriv_abstractions = {}
  for id, rule in this_deriv:
    abskey = ",".join([str(rule)]+[str(par) for par in these_pars[id]])
    deriv_abstractions[abskey] = id
  eval_store = {}
  for id in this_good:
    eval_store[id] = 1.0
  for id in this_neg:
    eval_store[id] = 0.0

  max_id = max(set(init_abstractions.values()) | set(deriv_abstractions.values())) + 1

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
  print("Loaded problem configuration", HP.EXP_PROBLEM_CONFIGURATIONS)

  with open(HP.EXP_PROBLEM_FILES, "r") as f:
    file_names = f.readlines()

  file_names = [x.replace("\n","") for x in file_names]
  prob_list = [{"short_name": x.rsplit("/", 1)[1], "full_name" : x} for x in file_names]

  del file_names

  thax_to_str[-1] = "-1"
 
  tree = torch.load(HP.EXP_FILE, weights_only=False)
  print("Loaded tree from", HP.EXP_FILE)

# Store data in shared memory manager
  serialized_init = msgpack.packb(tree[0][1][0])
  serialized_deriv = msgpack.packb(tree[0][1][1])
  serialized_pars = msgpack.packb(tree[0][1][2])
  serialized_good = msgpack.packb(list(tree[0][1][4]))
  serialized_neg = msgpack.packb(list(tree[0][1][3] - tree[0][1][4]))
  serialized_thax_to_str = msgpack.packb(thax_to_str)

  del tree

  shm_init = shm.SharedMemory(create=True, size=len(serialized_init))
  shm_deriv = shm.SharedMemory(create=True, size=len(serialized_deriv))
  shm_pars = shm.SharedMemory(create=True, size=len(serialized_pars))
  shm_good = shm.SharedMemory(create=True, size=len(serialized_good))
  shm_neg = shm.SharedMemory(create=True, size=len(serialized_neg))
  shm_thax_to_str = shm.SharedMemory(create=True, size=len(serialized_thax_to_str))

  shm_init.buf[:len(serialized_init)] = serialized_init
  shm_deriv.buf[:len(serialized_deriv)] = serialized_deriv
  shm_pars.buf[:len(serialized_pars)] = serialized_pars
  shm_good.buf[:len(serialized_good)] = serialized_good
  shm_neg.buf[:len(serialized_neg)] = serialized_neg
  shm_thax_to_str.buf[:len(serialized_thax_to_str)] = serialized_thax_to_str

  shared_data = {
    'shm_init': shm_init.name,
    'shm_deriv': shm_deriv.name,
    'shm_pars': shm_pars.name,
    'shm_good': shm_good.name,
    'shm_neg': shm_neg.name,
    'shm_thax_to_str': shm_thax_to_str.name
  }


  this_data = [{"short_name": prob_list[numProblem]["short_name"], 
                "full_name": prob_list[numProblem]["full_name"], 
                "axioms": problem_configuration[prob_list[numProblem]["short_name"]]
               } 
                for numProblem in range(len(prob_list))
              ]
  with multiprocessing.Pool(processes=HP.NUMPROCESSES) as pool:
      results = pool.starmap(worker, [(shared_data, data) for data in this_data])

  shm_init.close()
  shm_deriv.close()
  shm_pars.close()
  shm_good.close()
  shm_neg.close()
  shm_thax_to_str.close()

  shm_init.unlink()
  shm_deriv.unlink()
  shm_pars.unlink()
  shm_good.unlink()
  shm_neg.unlink()
  shm_thax_to_str.unlink()