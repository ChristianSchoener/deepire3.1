#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import numpy as np

import torch
torch.set_printoptions(precision=16)
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float16)
# from torch import Tensor

# import heapq

from functools import partial

import argparse

# import operator

from itertools import combinations
from bitarray import bitarray

import time,bisect,random,math,os,errno

# from typing import Dict, List, Tuple, Optional

# from multiprocessing import shared_memory

from collections import defaultdict
# from collections import ChainMap

import sys,random,itertools

import multiprocessing
# import asyncio

import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

# import multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from copy import deepcopy

def parse_args():

  parser = argparse.ArgumentParser(description="Process command-line arguments with key=value format.")
  parser.add_argument("mode", help="Determines the mode. Can be zero-test, pre, or compress.")
  
  # This allows handling of unknown key=value arguments
  args_, unknown_args = parser.parse_known_args()

  # If -h or --help is provided, argparse will automatically print help and exit
  if "-h" in sys.argv or "--help" in sys.argv:
    parser.print_help()
    sys.exit()  # Ensure script exits

  args = {}
  for arg in unknown_args:  # Process additional key=value arguments
    if "=" not in arg:
        parser.error(f"Invalid argument format '{arg}'. Use key=value.")
    key, value = arg.split("=", 1)  # Split only on the first '='
    args[key] = value

  return args

class RuleWorker(threading.Thread):
# class RuleWorker(mp.Process):
  """ Persistent worker process that removes `id_pool` elements from `shared_pars[rule]`. """
  def __init__(self, rule, shared_pars, task_queue, result_queue):
    super().__init__()
    self.rule = rule
    self.shared_pars = {key: val for key, val in shared_pars.items()}  # Rule-specific shared data
    self.task_queue = task_queue
    self.result_queue = result_queue
    self.num_empty_keys = 0
    self.empty_keys = set()
    self.rev_shared_pars = dict()
    for key, vals in self.shared_pars.items():
      for val in vals:
        self.rev_shared_pars[val] = set()
    for key, vals in self.shared_pars.items():
      for val in vals:
        self.rev_shared_pars[val].add(key)

  def run(self):
    # print(f"Worker {self.rule} started with {len(self.shared_pars)} entries.", flush=True)

    while True:
      task = self.task_queue.get()

      if task is None:
        break

      task_type, data = task

      if task_type == "delete":
        result = self._handle_delete(data)

      if task_type == "clean_biggest":
        result = self._handle_clean_biggest(data)

      self.result_queue.put(result) 

  def _handle_delete(self, data):
    for id in set(data) & set(self.rev_shared_pars.keys()):
      for val in self.rev_shared_pars[id]:
        self.shared_pars[val].discard(id)
      del self.rev_shared_pars[id]
    self.empty_keys.update({key for key, val in self.shared_pars.items() if not val})
    self.shared_pars = {key: val for key, val in self.shared_pars.items() if val}
    result = (self.rule, len(self.empty_keys), self.empty_keys)
    return result

  def _handle_clean_biggest(self, data):
    if data == self.rule:
      self.num_empty_keys = 0
      self.empty_keys = set()
    result = 1
    return result

def greedy(data, global_selec, global_good, stop_early=0):
  init, deriv, pars = data[1][:3] 

  depth_dict = IC.get_depth_dict(init, deriv, pars)
  ids = [id for id, _ in init]
  # print(ids)
  id_to_ind = {ids[i]: i for i in range(len(ids))}
  rules = list(set(rule for _, rule in deriv))
  rule_ids = dict()
  for rule in rules:
    rule_ids[rule] = set()

  for id, rule in deriv:
    rule_ids[rule].add(id)

  shared_pars = {rule: {id: set(vals) for id, vals in pars.items() if id in rule_ids[rule]} for rule in rules}

  # pars_len = {id: len(pars[id]) for id in pars}
  # task_queues = {rule: mp.Queue() for rule in rules}
  # result_queue = mp.Queue()

  task_queues = {rule: queue.Queue() for rule in rules}
  result_queue = queue.Queue()

  workers = {rule: RuleWorker(rule, shared_pars[rule], task_queues[rule], result_queue) for rule in rules}
  for worker in workers.values():
    worker.start()

  thax = torch.tensor([thax for _, thax in init], dtype=torch.int32)
  remaining_count = sum(map(len, rule_ids.values()))
  rule_steps, ind_steps, pars_ind_steps, rule_52_limits = [], [], [], {}

  id_pool = ids

  while remaining_count > 0:
    if stop_early > 0 and (len(rule_steps) == stop_early):
      for rule in rules:
        task_queues[rule].put(None)
      for p in workers.values():
        p.join()
      return (0, 0, [None] * (stop_early + 1))

    print(remaining_count, end =" ", flush=True)

    for rule in rules:
      task_queues[rule].put(("delete", id_pool))

    # shared_pars = {rule: {key : val.difference(id_pool) for key, val in shared_pars[rule].items()} for rule in shared_pars}
    # for rule in empty_keys:
    #   empty_keys[rule].update([key for key, val in shared_pars.get(rule, {}).items() if not val])
    # empties = {rule: len(val) for rule, val in empty_keys.items()}
    # shared_pars = {rule: {key: val for key, val in shared_pars[rule].items() if val} for rule in rules}
    
    gain = {}
    empty_keys = {}

    results_ = []
    while len(results_) < len(rules):
      time.sleep(0.0001)
      # print(r,len(results_), end=" ", flush = True)
      if not result_queue.empty():
        results_.append(result_queue.get_nowait())
        if results_[-1] is not None:
          rule_, empty_count_, empty_keys_ = results_[-1]
          gain[rule_] = empty_count_
          empty_keys[rule_] = empty_keys_

    # best_rule = max(empties, key=empties.get)
    best_rule = max(gain, key=gain.get)
    # print(best_rule, gain, flush=True)

    task_queues[best_rule].put(("clean_biggest", best_rule))
    results_ = []
    while len(results_) < 1:
      time.sleep(0.0001)
      if not result_queue.empty():
        results_.append(result_queue.get_nowait())

    id_pool = list(empty_keys[best_rule])
    if best_rule != 52:
      id_to_ind.update({id: len(ids) + i for i, id in enumerate(id_pool)})
      ids.extend(id_pool)
      rule_steps.append(best_rule)
      ind_steps.append(torch.tensor([id_to_ind[id] for id in id_pool], dtype=torch.int32))
      pars_ind_steps.append(torch.tensor([id_to_ind[this_id] for id in id_pool for _, this_id in enumerate(pars[id])], dtype=torch.int32))

      # if best_rule == 52:
      #   rule_52_limits[len(ind_steps)-1] = torch.tensor([0] + list(np.cumsum([pars_len[id] for id in id_pool])), dtype=torch.int32)

    rule_ids[best_rule] -= set(id_pool)

    # for id in id_pool:
    #   del pars[id]

    remaining_count = sum(map(len, rule_ids.values()))
    # empty_keys[best_rule] = set()
    # empties[best_rule] = 0

  for rule in rules:
    task_queues[rule].put(None)
  for p in workers.values():
    p.join()

  this_good = global_good & set(ids)
  this_neg = (global_selec - global_good) & set(ids)

  persistent_good = IC.get_subtree(this_good, deriv, pars)
  ok = persistent_good - this_good

  pos = []
  neg = []
  for _, id in enumerate(ids):
    if id in this_good:
      pos.append(1)
      neg.append(0)
    elif id in this_neg:
      pos.append(0)
      neg.append(1)
    elif id in ok:
      pos.append(0)
      neg.append(0)
    else:
      pos.append(0)
      neg.append(0)

  pos = torch.tensor(pos, dtype=torch.float64)
  neg = torch.tensor(neg, dtype=torch.float64)

  tot_pos = sum(pos)
  tot_neg = sum(neg)

  target = pos / (pos + neg)

  # mask = torch.tensor([id in cropped_keys for _, id in enumerate(ids)], dtype=torch.bool)

  print()

  return {"thax": thax, "ids": torch.tensor(ids, dtype=torch.int32), "rule_steps": torch.tensor(rule_steps, dtype=torch.int32), "ind_steps": ind_steps, "pars_ind_steps": pars_ind_steps, "pos": pos, "neg": neg, "tot_pos": tot_pos, "tot_neg": tot_neg, "target": target}

if __name__ == "__main__":

  args = parse_args()

  assert("mode" in args), "A command line argument mode=... must be passed. \n" + \
                          "Options:\n" + \
                          "1. mode=zero-test, preparing the files for the analytic test.\n" + \
                          "2. mode=pre, preparing the raw data for compression by excluding large cases, setting all axioms to zero which aren't among the chosen number of most common ones, and cleaning up data a bit, as well as splitting into training and validation sets.\n" + \
                          "3. mode=compress, compressing the prepared files and splitting them up into individual training and validation instances."

  if args["mode"] == "zero-test":
    assert hasattr(HP, "ZERO_FILE"), "Parameter ZERO_FILE in hyperparams.py not set. This file is read in for producing the one big multi-tree."
    assert isinstance(HP.ZERO_FILE, str), "Parameter ZERO_FILE in hyperparams.py is not a string. This file is read in for producing the one big multi-tree."
    assert os.path.isfile(HP.ZERO_FILE), "Parameter ZERO_FILE in hyperparams.py does not point to an existing file. This file is read in for producing the one big multi-tree."

    assert hasattr(HP, "ZERO_FOLDER"), "Parameter ZERO_FOLDER in hyperparams.py not set. This folder shall contain the raw data for producing the one big multi-tree."
    assert isinstance(HP.ZERO_FOLDER, str), "Parameter ZERO_FOLDER in hyperparams.py is not a string. This folder shall contain the raw data for producing the one big multi-tree."
    assert os.path.isfile(HP.ZERO_FOLDER), "Parameter ZERO_FOLDER in hyperparams.py does not point to an existing folder. This folder shall contain the raw data for producing the one big multi-tree."

    print("Loading problem file.", flush=True)
    prob_data_list = torch.load(HP.ZERO_FILE, weights_only=False)

    print("Dropping axiom information, not needed anymore.")
    prob_data_list = [(metainfo, (init, deriv, pars, selec, good)) for (metainfo, (init, deriv, pars, selec, good, _)) in prob_data_list]

    print("Dropping large proofs (init+deriv > 30000).")
    prob_data_list = [(metainfo, (init, deriv, pars, selec, good)) for (metainfo, (init, deriv, pars, selec, good)) in prob_data_list if len(init) + len(deriv) < 30000]

    print("Generating one multi-tree.", flush=True)
    prob, old2new, selec, good = IC.compress_prob_data(prob_data_list, "long", True)
    torch.save((selec, good), "{}/global_selec_and_good.pt".format(HP.ZERO_FOLDER))

    print("Cropping multitree to what is induced by the selection.", flush=True)
    prob_data_list = IC.crop(prob)

    print("Saving.", flush=True)
    torch.save(prob_data_list, "{}/full_tree_cropped.pt".format(HP.ZERO_FOLDER))

    print("Done.", flush=True)
    exit()

  if args["mode"] == "pre":
    assert hasattr(HP, "PRE_FILE"), "Parameter PRE_FILE in hyperparams.py not set. This file contains the raw data from log_loading."
    assert isinstance(HP.PRE_FILE, str), "Parameter PRE_FILE in hyperparams.py is not a string. This file contains the raw data from log_loading."
    assert os.path.isfile(HP.PRE_FILE), "Parameter PRE_FILE in hyperparams.py does not point to an existing file. This file contains the raw data from log_loading."

    assert hasattr(HP, "PRE_FOLDER"), "Parameter PRE_FOLDER in hyperparams.py not set. This folder shall contains the raw data produced by log_loading."
    assert isinstance(HP.PRE_FOLDER, str), "Parameter PRE_FOLDER in hyperparams.py is not a string. This folder shall contain the raw data produced by log_loading."
    assert os.path.isfile(HP.PRE_FOLDER), "Parameter PRE_FOLDER in hyperparams.py does not point to an existing folder. This folder shall contain the raw data produced by log_loading."

    print("Loading problem file.", flush=True)
    prob_data_list = torch.load(HP.PRE_FILE, weights_only=False)

    print("Dropping axiom information, not needed anymore.")
    prob_data_list = [(metainfo, (init, deriv, pars, selec, good)) for (metainfo, (init, deriv, pars, selec, good, _)) in prob_data_list]

    print("Dropping large proofs (init+deriv > 30000).")
    prob_data_list = [(metainfo, (init, deriv, pars, selec, good)) for (metainfo, (init, deriv, pars, selec, good)) in prob_data_list if len(init) + len(deriv) < 30000]

    print("Setting axiom numbers to 0, if above MAX_USED_AXIOM_CNT, and updating thax number to name mapping.")
    thax_sign, deriv_arits, thax_to_str = torch.load("{}/data_sign_full.pt".format(HP.PRE_FOLDER), weights_only=False)
    prob_data_list, thax_sign, thax_to_str = IC.set_zero(prob_data_list, thax_to_str)

    print("Updating data signature.", flush=True)
    torch.save((thax_sign, deriv_arits, thax_to_str), "{}/data_sign.pt".format(HP.PRE_FOLDER))

    print("Compressing problems to get rid of several identical axioms and derivations form setting to zero.", flush=True)
    prob_data_list = [IC.compress_prob_data([prob], "long") for prob in prob_data_list]
    print("Generated old2new, selec, good from the modified data.", flush=True)
    _, old2new, selec, good = IC.compress_prob_data(prob_data_list, "long", True)
    torch.save((selec, good), "{}/global_selec_and_good.pt".format(HP.PRE_FOLDER))
    print("Assigning new ids to individual problems and cropping.", flush=True)
    with ThreadPoolExecutor(max_workers=HP.NUMPROCESSES) as executor:
      prob_data_list = list(executor.map(lambda j, prob: IC.adjust_ids_and_crop(prob, old2new[j], selec), 
                                         range(len(prob_data_list)), prob_data_list))

    print("Extracting axiom-to-problem  and problem-to.axiom mappings to ensure, that every axiom is in the training. (And also having the data for swapping out uniformly, currently not implemented).", flush=True)
    ax_to_prob = dict()
    for i,(_, (init, _, _, _, _)) in enumerate(prob_data_list):
      for _, thax in init:
        if thax not in ax_to_prob:
          ax_to_prob[thax] = {i}
        else:
          ax_to_prob[thax].add(i)
    torch.save(ax_to_prob, "{}/axiom_counts.pt".format(HP.PRE_FOLDER))
    print("Saved axiom counts for uniformly distributed expectation SWAPOUT")
    rev_ax_to_prob = dict()
    for key, vals in ax_to_prob.items():
      for val in vals:
        if not val in rev_ax_to_prob:
          rev_ax_to_prob[val] = {key}
        else:
          rev_ax_to_prob[val].add(key)
    gather = set()
    train_data_list = []
    to_delete = []
    train_length = math.ceil(len(prob_data_list) * HP.VALID_SPLIT_RATIO)
    while len(set(ax_to_prob.keys() - gather)) > 0:
      this_ax = list(set(ax_to_prob.keys() - gather))[0]
      to_delete.append(list(ax_to_prob[this_ax])[0])
      gather = gather | rev_ax_to_prob[list(ax_to_prob[this_ax])[0]]
    train_data_list = [prob_data_list[i] for i in range(len(prob_data_list)) if i in to_delete] 
    prob_data_list = [prob_data_list[i] for i in range(len(prob_data_list)) if not i in to_delete]
    train_length -= len(train_data_list)
    train_length = max(train_length, 0)
    random.shuffle(prob_data_list)
    train_data_list += prob_data_list[:train_length]
    valid_data_list = prob_data_list[train_length:]
    del prob_data_list
    print("Split problems into {} training instances and {} validation instances. Saving.".format(len(train_data_list), len(valid_data_list)), flush=True)
    torch.save(train_data_list, "{}.train".format(HP.PRE_FILE))
    torch.save(valid_data_list, "{}.valid".format(HP.PRE_FILE))
    print("Done.", flush=True)
    exit()

  if args["mode"] == "compress":
    assert hasattr(HP, "COM_FILE"), "Parameter COM_FILE in hyperparams.py not set. This file contains the prepared data of the preparation step, either for training or validation."
    assert isinstance(HP.COM_FILE, str), "Parameter COM_FILE in hyperparams.py is not a string. This file contains the prepared data of the preparation step, either for training or validation."
    assert os.path.isfile(HP.COM_FILE), "Parameter COM_FILE in hyperparams.py does not point to an existing file. This file contains the prepared data of the preparation step, either for training or validation."

    assert hasattr(HP, "COM_FOLDER"), "Parameter COM_FOLDER in hyperparams.py not set. This folder is the base folder of the project."
    assert isinstance(HP.COM_FOLDER, str), "Parameter COM_FOLDER in hyperparams.py is not a string. This folder is the base folder of the project."
    assert os.path.isfile(HP.COM_FOLDER), "Parameter COM_FOLDER in hyperparams.py does not point to an existing directory. This folder is the base folder of the project."

    assert hasattr(HP, "COM_ADD_MODE_1"), "Parameter COM_ADD_MODE_1 in hyperparams.py not set. This parameter takes values either train or valid an indicates, which of the data sets shall be compressed and further transformed to perform the computations."
    assert(HP.COM_ADD_MODE_1 == "train" or HP.COM_ADD_MODE_1 == "valid"), "Parameter COM_ADD_MODE_1 in hyperparams.py is not train or valid. This parameter indicates, which of the data sets shall be compressed and further transformed to perform the computations."

    print("Loading problem file and selec and good.", flush=True)
    prob_data_list = torch.load(HP.COM_FILE, weights_only=False)
    selec, good = torch.load("{}/global_selec_and_good.pt".format(HP.COM_FOLDER))

    print("Compressing.", flush=True)
    compressed = []
    while len(prob_data_list) > 1:
      print(len(prob_data_list), flush=True)
      prob_data_list.sort(key=lambda t: -(len(t[1][0]) + len(t[1][1])))
      to_compress = [prob_data_list.pop()]
      size = len(to_compress[-1][1][0]) + len(to_compress[-1][1][1])
      while size < HP.COMPRESSION_THRESHOLD and prob_data_list:
        to_compress.append(prob_data_list.pop())
        size += len(to_compress[-1][1][0]) + len(to_compress[-1][1][1])
      to_compress = [IC.compress_prob_data_with_fixed_ids(to_compress)]
      size = len(to_compress[-1][1][0]) + len(to_compress[-1][1][1])
      if size >= HP.COMPRESSION_THRESHOLD:
        compressed.extend(to_compress)
      else:
        prob_data_list.extend(to_compress)
    print("Compressed to {} many instances.".format(len(compressed)), flush=True)
    print("Done.", flush=True)
    prob_data_list.extend(compressed)
    del compressed

    print("Computing Greedy Evaluation Schemes and setting pos vals and neg vals.", flush=True)
    greedy_partial = partial(greedy, global_selec=selec, global_good=good)
    with ProcessPoolExecutor(max_workers=HP.NUMPROCESSES) as executor:
      prob_data_list = list(executor.map(greedy_partial, prob_data_list))
    print("Done. Saving.", flush=True)

    print("Saving Pieces.", flush=True)
    dir = "{}/pieces".format(HP.COM_FOLDER)
    os.makedirs(dir, exist_ok=True)
    for i, rest in enumerate(prob_data_list):
      piece_name = "piece{}.pt.{}".format(i, HP.COM_ADD_MODE_1)
      print("Saving {}".format(piece_name), flush=True)
      torch.save(rest, "{}/pieces/{}".format(HP.COM_FOLDER, piece_name))
    if "ids" in prob_data_list[0]:
      prob_data_list = [(len(prob_data_list[i]["ids"]), "piece{}.pt.{}".format(i, HP.COM_ADD_MODE_1)) for i in range(len(prob_data_list))]
    else:
      prob_data_list = [(len(prob_data_list[i][1][0]) + len(prob_data_list[i][1][1]), "piece{}.pt.{}".format(i, HP.COM_ADD_MODE_1)) for i in range(len(prob_data_list))]
    filename = "{}/{}_index.pt".format(HP.COM_FOLDER, HP.COM_ADD_MODE_1)

    print("Saving {} part to".format(HP.COM_ADD_MODE_1), filename)
    torch.save(prob_data_list, filename)

    print("Done.", flush=True)
    exit()