#!/usr/bin/env python3

import inf_common as IC

import hyperparams as HP

import torch

import time

import gc

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def load_one(task):
  i, logname = task
  
  print(i)
  start_time = time.time()
  result = IC.load_one(logname) # ,max_size=15000)
  print("Took", time.time()-start_time, flush=True)
  if result:
    probdata,time_elapsed = result
    return (logname,time_elapsed),probdata
  else:
    None

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
      print(logname)

  with ProcessPoolExecutor(max_workers=HP.NUMPROCESSES) as executor:
    prob_data_list = list(filter(None, executor.map(load_one, tasks, chunksize=1000)))

  print(len(prob_data_list),"problems loaded!")

  gc.collect()

  # assign weights to problems, especially if prob_easiness file has been provided
  times = []
  sizes = []
  easies = []

  print(prob_data_list[0])

  for i, ((logname, time_elapsed), ((_, _, size), rest)) in enumerate(prob_data_list):
    probname = IC.logname_to_probname(logname)

    probweight = 1.0    
    
    prob_data_list[i] = (probname, probweight, size), rest
    
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

