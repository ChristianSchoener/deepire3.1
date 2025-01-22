#!/usr/bin/env python3

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools,gc

import inf_common as IC

import hyperparams as HP

import os
import subprocess

import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def worker(q_in, q_out):
  log = sys.stdout

  while True:
# q_in.put(this_data,(init_embeds,init_embed_mods,deriv_mlps,eval_net,thax_to_str),sys.argv[4])
    (this_data,init_embeds,init_embed_mods,deriv_mlps,eval_net,thax_to_str,folder) = q_in.get()
    # this_data = [file problem, folder/file problem, axiom numbers] 
    initEmbeds = {}
    temp = torch.zeros(len(init_embeds["0"].weight))
    for id in this_data[1]:
      temp += init_embeds[str(id)].weight/len(this_data[0])
    for id in this_data[1]:
      initEmbeds[thax_to_str[id]] = torch.mv(init_embed_mods[str(id)].weight,temp)

    folder_file = folder+"/"+this_data[0][0]+".model"
    ISM.save_net_matrix(folder,this_data[0][0],initEmbeds,deriv_mlps,eval_net)
    command = './vampire_Release_deepire3_4872 {} -tstat on --decode dis+1010_3:2_acc=on:afr=on:afp=1000:afq=1.2:amm=sco:bs=on:ccuc=first:fde=none:nm=0:nwc=4:urr=ec_only_100 -p off --output_axiom_names on -e4k {} -nesq on -nesqr 2,1 > {} 2>&1'.format(this_data[0][1],folder_file,"results"+"/"+"matrix_64" + "/"+this_data[0][1].replace("/","_")+".log")
    # print(command)
    subprocess.Popen(command,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
    
    command2 = 'rm {}'.format(folder_file)
    subprocess.Popen(command2,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
    q_out.put((1))
    libc.malloc_trim(ctypes.c_int(0))

if __name__ == "__main__":

  thax_sign,deriv_arits,thax_to_str = torch.load(sys.argv[1])
  print("Loaded signature from",sys.argv[1])

  IC.create_saver(deriv_arits)
  import inf_saver_matrix as ISM

  problem_configuration = torch.load(sys.argv[2])
  print("Loaded problem configuration",sys.argv[2])

  (_,parts,_) = torch.load(sys.argv[3])
  (_,parts_copies,_) = torch.load(sys.argv[3])

  print("Loaded model from",sys.argv[3])

  with open("all.txt","r") as f:
    file_names = f.readlines()

  file_names = [x.replace("\n","") for x in file_names]
  prob_map = { x.split("__")[1] : x for x in file_names}

  del file_names

  for part,part_copy in zip(parts,parts_copies):
    part_copy.load_state_dict(part.state_dict())
  
  # eval mode and no gradient
  part_copy.eval()
  for param in part_copy.parameters():
    param.requires_grad = False

  # from here on only use the updated copies
  (init_embeds,init_embed_mods,deriv_mlps,eval_net) = parts_copies

  thax_to_str[0] = "0"
  thax_to_str[-1] = "-1"

  q_in = torch.multiprocessing.Queue()
  q_out = torch.multiprocessing.Queue()
  my_processes = []
  for i in range(HP.NUMPROCESSES):
    p = torch.multiprocessing.Process(target=worker, args=(q_in,q_out))
    p.start()
    my_processes.append(p)

  MAX_ACTIVE_TASKS = HP.NUMPROCESSES
  num_active_tasks = 0
  numProblem = 0
  while True:
    while num_active_tasks < MAX_ACTIVE_TASKS and numProblem < len(prob_map):
      num_active_tasks += 1
      this_prob = list(prob_map.items())[numProblem]
      this_data = [this_prob, problem_configuration[this_prob[0]]]
      numProblem += 1
      q_in.put((this_data,init_embeds,init_embed_mods,deriv_mlps,eval_net,thax_to_str,sys.argv[4]))
    result = q_out.get()
    num_active_tasks -= 1

    if numProblem == len(problem_configuration):
      break
      
  print("Exported to",sys.argv[4])

  for p in my_processes:
    p.kill()

