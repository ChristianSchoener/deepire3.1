#!/usr/bin/env python3

import inf_common as IC

import hyperparams as HP

import torch
from torch import Tensor

import time

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import argparse

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_one(task):
  i, logname = task
  
  print(i)
  start_time = time.time()
  result = IC.load_one(logname) # ,max_size=15000)
  print("Took", time.time()-start_time, flush=True)
  if result:
    probdata,time_elapsed = result
    # probdata = IC.setup_pos_vals_neg_vals(probdata)
    # probdata = IC.replace_axioms(probdata)
    # probdata = IC.compress_prob_data([probdata])
    return (logname,time_elapsed),probdata
  else:
    None

# def parse_args():

#   parser = argparse.ArgumentParser(description="Process command-line arguments with key=value format.")
#   parser.add_argument("arguments", nargs="+", help="Arguments in key=value format (e.g., mode=pre folder=/path file=file.txt).")

#   args_ = parser.parse_args()

#   args = {}
#   for arg in args_.arguments:
#     if "=" not in arg:
#       parser.error(f"Invalid argument format '{arg}'. Use key=value.")
#     key, value = arg.split("=", 1)  # Split only on the first '='
#     args[key] = value
  
#   return args

if __name__ == "__main__":

  assert(HP.LOG_FOLDER)
  assert(HP.LOG_FILES_TXT)
  args = {}
  args["folder"] = HP.LOG_FOLDER
  args["file"] = HP.LOG_FILES_TXT

  prob_data_list = [] # [(logname,(init,deriv,pars,selec,good)]

  tasks = []
  with open(args["file"], "r") as f:
    for i,line in enumerate(f):
      logname = line[:-1]
      tasks.append((i, logname))
    
  pool = Pool(processes = HP.NUMPROCESSES) # number of cores to use
  results = pool.map(load_one, tasks, chunksize = 100)
  pool.close()
  pool.join()
  del pool
  prob_data_list = list(filter(None, results))

  print(len(prob_data_list),"problems loaded!")

  # assign weights to problems, especially if prob_easiness file has been provided
  times = []
  sizes = []
  easies = []

  for i,((logname,time_elapsed),((_,_,size),rest)) in enumerate(prob_data_list):
    probname = IC.logname_to_probname(logname)

    probweight = 1.0    
    
    prob_data_list[i] = (probname,probweight,size),rest
    
    times.append(time_elapsed)
    sizes.append(size)


  thax_sign, deriv_arits, axiom_hist = IC.prepare_signature(prob_data_list)

  thax_sign, prob_data_list, thax_to_str = IC.axiom_names_instead_of_thax(thax_sign, axiom_hist, prob_data_list)
  
  print("thax_sign", thax_sign)
  print("deriv_arits", deriv_arits)
  print("thax_to_str", thax_to_str)

  filename = "{}/data_sign_full.pt".format(args["folder"])
  print("Saving signature to", filename)
  torch.save((thax_sign, deriv_arits, thax_to_str), filename)
  print()

  filename = "{}/raw_log_data{}".format(args["folder"], IC.name_raw_data_suffix())
  print("Saving raw data to", filename)
  torch.save(prob_data_list, filename)
  print()

  # print(prob_data_list[0])

  print("Done")

