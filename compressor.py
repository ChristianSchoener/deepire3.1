#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import numpy as np

import torch
from torch import Tensor

import heapq

import argparse

import operator

from itertools import combinations
from bitarray import bitarray

import time,bisect,random,math,os,errno

from typing import Dict, List, Tuple, Optional

from multiprocessing import shared_memory

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

import asyncio

import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

import multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from copy import deepcopy

def parse_args():

  parser = argparse.ArgumentParser(description="Process command-line arguments with key=value format.")
  parser.add_argument("arguments", nargs="+", help="Arguments in key=value format (e.g., mode=pre folder=/path file=file.txt).")

  args_ = parser.parse_args()

  args = {}
  for arg in args_.arguments:
    if "=" not in arg:
      parser.error(f"Invalid argument format '{arg}'. Use key=value.")
    key, value = arg.split("=", 1)  # Split only on the first '='
    args[key] = value
  
  return args

def compress_by_probname(prob_data_list):
  by_probname = defaultdict(list)

  for metainfo,rest in prob_data_list:
    probname = metainfo[0]
    by_probname[probname].append((metainfo,rest))

  compressed = []

  for probname,bucket in by_probname.items():
    print("Compressing bucket of size",len(bucket),"for",probname,"of total weight",sum(metainfo[1] for metainfo,_ in bucket),"and sizes",[len(rest[0])+len(rest[1]) for _,rest in bucket],"and posval sizes",[len(rest[3]) for _,rest in bucket])

    metainfo,rest = IC.compress_prob_data(bucket)

    # print(metainfo)

    print("Final size",len(rest[0])+len(rest[1]))

    compressed.append((metainfo,rest))

  return compressed

def build_id_dict(num, p, id_dict):
  union_set = set(x for x, _ in p[1][0]) | set(x for x, _ in p[1][1])
  id_dict[num] = {"num": num, "length": len(union_set), "ids": union_set}

class MergeWorker(threading.Thread):
# class MergeWorker(mp.Process):
  """ Persistent worker process that merges """
  def __init__(self, id, nums, p_dicts, id_dicts, task_queue, result_queue):
    super().__init__()
    self.id = id
    self.num_initial = nums
    self.p_dict = p_dicts
    self.id_dict = id_dicts
    self.task_queue = task_queue
    self.result_queue = result_queue
    del id, nums, p_dicts, id_dicts

  def run(self):
    while True:
      # while not self.task_queue.empty():
        task = self.task_queue.get()
        # print(f"Worker {self.id} received task: {task}", flush=True)  # Debugging statement
            
        if task is None:
          # print(f"Worker {self.id} terminating.", flush=True)  # Debugging statement
          break
        
        task_type, data = task

        if task_type == "get_min":
          result = self._handle_get_min(data)

        if task_type == "compare":
          result = self._handle_compare(data)

        if task_type == "delete":
          result = self._handle_delete(data)

        if task_type == "update":
          result = self._handle_update(data)

        if task_type == "query":
          result = self._handle_query(data)

        if task_type == "finalize":
          result = self._handle_finalize(data)

        self.result_queue.put(result)
 
  def _handle_get_min(self, data):
    if self.id_dict:
      this_min = min((val for x, val in self.id_dict.items()), key=lambda t: t["length"])
      result = (this_min["num"], this_min, self.p_dict[this_min["num"]])
    else:
      result = None
    return result

  def _handle_query(self, data):
    num = data
    if num in self.id_dict:
      result = (num, self.id_dict[num], self.p_dict[num])
    else:
      result = None
    return result
    
  def _handle_compare(self, data):
    if self.id_dict:
      mini1 = data
      diffs = {num: mini1["ids"] - self.id_dict[num]["ids"] for num in self.id_dict}
      lengths = {num: len(diffs[num]) + self.id_dict[num]["length"] for num in self.id_dict}
      good = set(num for num in self.id_dict if lengths[num] < HP.COMPRESSION_THRESHOLD)
      if good:
        mini_num = min((num for num in good), key=lambda t: diffs[t])
        mini2 = {"num": mini_num, "length": lengths[mini_num], "ids": self.id_dict[mini_num]["ids"], "diff": len(diffs[mini_num])}
        result = (mini_num, mini2, self.p_dict[mini_num])
      else:
        result = None
    else:
      result = None
    return result

  def _handle_delete(self, data):
    num = data
    if num in self.num_initial:
      del self.p_dict[num]
      del self.id_dict[num]
    result = 1
    return result

  def _handle_update(self, data):
    num, mini, p = data
    if num in self.num_initial:
      self.p_dict[num] = p
      self.id_dict[num] = mini
    result = 1
    return result

  def _handle_finalize(self, data):
    result = self.p_dict
    return result

def initialize_workers(num_threads, id_dict, p_dict):
  """ Create workers and distribute tasks among threads. """
  # task_queues = {i: mp.Queue() for i in range(num_threads)}
  # result_queue = mp.Queue()

  task_queues = {i: queue.Queue() for i in range(num_threads)}
  result_queue = queue.Queue()
  
  split_points = np.linspace(0, len(id_dict), num_threads + 1, dtype=int)
  ranges = {i: set(np.arange(split_points[i], split_points[i+1])) for i in range(num_threads)}
  nums = {i: set(list(id_dict.keys())[x] for x in ranges[i]) for i in range(num_threads)}
  p_dicts = {i: {x: p_dict[x] for x in nums[i]} for i in range(num_threads)}
  id_dicts = {i: {x: id_dict[x] for x in nums[i]} for i in range(num_threads)}
  
  workers = {i: MergeWorker(i, nums[i], p_dicts[i], id_dicts[i], task_queues[i], result_queue) for i in range(num_threads)}
  
  for worker in workers.values():
    worker.start()
  
  return workers, task_queues, result_queue

def get_minimum_element(num_threads, task_queues, result_queue):
  """ Retrieve minimum elements from all workers. """
  for i in range(num_threads):
    task_queues[i].put(("get_min", None))

  results_, mins, ps = [], {}, {}
  while len(results_) < num_threads:
    time.sleep(0.0001)
    if not result_queue.empty():
      results_.append(result_queue.get_nowait())
      if results_[-1] is not None:
        num_, mini_, p_ = results_[-1]
        mins[num_] = mini_
        ps[num_] = p_

  # print("Mins:", [(x, y["length"], ps[x][0][0]) for x, y in mins.items()],flush=True)
  mini = min((num for num in mins.values()), key=lambda t: t["length"])
  p = ps[mini["num"]]
  
  return mini, p

def query_element(num_threads, task_queues, result_queue, num):
  """ Query element. """
  for i in range(num_threads):
    task_queues[i].put(("query", num))

  results_, mins, ps = [], {}, {}
  while len(results_) < num_threads:
    time.sleep(0.0001)
    if not result_queue.empty():
      results_.append(result_queue.get_nowait())
      if results_[-1] is not None:
        num_, mini_, p_ = results_[-1]
        mins[num_] = mini_
        ps[num_] = p_

  # print([(x, y["length"]) for x, y in mins.items()],flush=True)
  mini = min((num for num in mins.values()), key=lambda t: t["length"])
  p = ps[mini["num"]]
  
  return num, mini, p

def get_compared_element(num_threads, task_queues, result_queue, mini):
  """ Compare with mini. """
  for i in range(num_threads):
    task_queues[i].put(("compare", mini))

  results_, mins, ps = [], {}, {}
  while len(results_) < num_threads:
    time.sleep(0.0001)
    if not result_queue.empty():
      results_.append(result_queue.get_nowait())
      if results_[-1] is not None:
        num_, mini_, p_ = results_[-1]
        mins[num_] = mini_
        ps[num_] = p_

  # print("Comp:", [(x, y["length"], y["diff"], ps[x][0][0]) for x, y in mins.items()], flush=True)
  mini = min((num for num in mins.values()), key=lambda t: (t["diff"], t["length"]))
  p = ps[mini["num"]]
  
  return mini, p

def delete_element(num_threads, task_queues, result_queue, num):
  """ Delete the current minimum element from all workers. """
  for i in range(num_threads):
    task_queues[i].put(("delete", num))

  results_ = []
  while len(results_) < num_threads:
    time.sleep(0.0001)
    if not result_queue.empty():
      results_.append(result_queue.get_nowait())


def update_all_workers(num_threads, task_queues, result_queue, mini, p):
  """ Update all workers with new merged data. """
  for i in range(num_threads):
      task_queues[i].put(("update", (mini["num"], mini, p)))

  results_ = []
  while len(results_) < num_threads:
    time.sleep(0.0001)
    if not result_queue.empty():
      results_.append(result_queue.get_nowait())

def finalize(num_threads, task_queues, result_queue):
  for i in range(num_threads):
    task_queues[i].put(("finalize", None))

  results_ = []
  while len(results_) < num_threads:
    time.sleep(0.0001)
    if not result_queue.empty():
      results_.append(result_queue.get_nowait())

  prob_data_list = []
  for result in results_:
    if result:
      prob_data_list.extend(result.values())

  return prob_data_list

def threaded_smallest_min_overlap_compression(prob_data_list):
# Initially: ~20442 problems
# After removing non-first occurence: ~16450
# After root reduction: 16003
# Just pick as many smallest problems to size strictly < 20000: 804
# Pick smallest problem, merge with problem such that minimal additional nodes, and among those one with minimal size, to size strictly < 20000: 741
# Huffmann-encoding (mergest 2 smallest problems) to size strictly < 20000: 986
  num_threads = 12
  # prob_data_list = torch.load(sys.argv[1],weights_only=False)
  p_dict = {num: p for num, p in enumerate(prob_data_list) if p[1][0]}
  del prob_data_list
  print("Problems read in. Performing Huffman-alike merge to threshold.",flush=True)
  id_dict = {}
  for num, p in p_dict.items():
    this_set = set([x for x, _ in p[1][0]]) | set([x for x, _ in p[1][1]]) 
    id_dict[num] = {"num": num, "length": len(this_set), "ids": this_set}

  workers, task_queues, result_queue = initialize_workers(num_threads, id_dict, p_dict)

  mini1, p1 = get_minimum_element(num_threads, task_queues, result_queue)
  delete_element(num_threads, task_queues, result_queue, mini1["num"])
  mini2, p2 = get_minimum_element(num_threads, task_queues, result_queue)

  while mini1["length"] + mini2["length"] < HP.COMPRESSION_THRESHOLD:
    del mini2, p2

    mini2, p2 = get_compared_element(num_threads, task_queues, result_queue, mini1)
    delete_element(num_threads, task_queues, result_queue, mini2["num"])

    p = IC.compress_prob_data_with_fixed_ids([p1, p2])
    new_set = mini1["ids"] | mini2["ids"]
    mini = {"num": mini1["num"], "length": len(new_set), "ids": new_set}

    update_all_workers(num_threads, task_queues, result_queue, mini, p)
    num_, mini_, p_ = query_element(num_threads, task_queues, result_queue, mini1["num"])

    del mini, mini1, p1, mini2, p2, p, new_set

    mini1, p1 = get_minimum_element(num_threads, task_queues, result_queue)
    delete_element(num_threads, task_queues, result_queue, mini1["num"])
    print("Min length:",mini1["length"],flush=True)
    mini2, p2 = get_minimum_element(num_threads, task_queues, result_queue)

  prob_data_list = finalize(num_threads, task_queues, result_queue)

  for i in range(num_threads):
    task_queues[i].put(None)
  for p in workers.values():
    p.join()

  print()
  print("Compressed to", len(prob_data_list), "merged problems")
  return prob_data_list
  # torch.save(prob_data_list,sys.argv[1]+".final_single_correct")

class RuleWorker(threading.Thread):
  """ Persistent worker process that removes `id_pool` elements from `shared_pars[rule]`. """
  def __init__(self, rule, shared_pars, task_queue, result_queue):
    super().__init__()
    self.rule = rule
    self.shared_pars = {key: val for key, val in shared_pars.items()}  # Rule-specific shared data
    self.task_queue = task_queue
    self.result_queue = result_queue
    self.num_empty_keys = 0
    self.empty_keys = set()

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
    self.shared_pars = {key : val.difference(data) for key, val in self.shared_pars.items()}
    self.empty_keys.update([key for key, val in self.shared_pars.items() if not val])
    self.shared_pars = {key: val for key, val in self.shared_pars.items() if val}
    result = (self.rule, len(self.empty_keys), self.empty_keys)
    return result

  def _handle_clean_biggest(self, data):
    if data == self.rule:
      self.num_empty_keys = 0
      self.empty_keys = set()
    result = 1
    return result

def greedy(data, stop_early=0):
  init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg = data[:7] 
  
  num_threads = 12
  ids = [id for id, _ in init]
  id_to_ind = {ids[i]: i for i in range(len(ids))}
  rules = list(set(rule for _, rule in deriv))
  rule_ids = dict()
  for rule in rules:
    rule_ids[rule] = set()

  for id, rule in deriv:
    rule_ids[rule].add(id)

  shared_pars = {rule: {id: set(vals) for id, vals in pars.items() if id in rule_ids[rule]} for rule in rules}

  pars_len = {id: len(pars[id]) for id in pars}
 
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
    result_queue.get()

    id_pool = list(empty_keys[best_rule])
    id_to_ind.update({id: len(ids) + i for i, id in enumerate(id_pool)})
    ids.extend(id_pool)
    rule_steps.append(best_rule)
    ind_steps.append(torch.tensor([id_to_ind[id] for id in id_pool], dtype=torch.int32))
    pars_ind_steps.append(torch.tensor([id_to_ind[this_id] for id in id_pool for this_id in pars[id]], dtype=torch.int32))

    if best_rule == 52:
      rule_52_limits[len(ind_steps)-1] = torch.tensor([0] + list(np.cumsum([pars_len[id] for id in id_pool])), dtype=torch.int32)

    rule_ids[best_rule] -= set(id_pool)

    for id in id_pool:
      del pars[id]

    remaining_count = sum(map(len, rule_ids.values()))
    # empty_keys[best_rule] = set()
    # empties[best_rule] = 0

  for rule in rules:
    task_queues[rule].put(None)
  for p in workers.values():
    p.join()


  sorted_keys = sorted(set(pos_vals.keys()) | set(neg_vals.keys()))  # Union of keys sorted

  sorted_pos_values = []
  sorted_neg_values = []
  for k in sorted_keys:
      sorted_pos_values.append(pos_vals.get(k, 0.0))  # Use .get() to default to 0.0
      sorted_neg_values.append(neg_vals.get(k, 0.0))  # Use .get() to default to 0.0

  pos = torch.tensor(sorted_pos_values, dtype=torch.float)
  neg = torch.tensor(sorted_neg_values, dtype=torch.float)

  target = pos / (pos + neg)

  mask = torch.tensor([id in sorted_keys for id in ids], dtype=torch.bool)

  print()

  return (thax, torch.tensor(ids, dtype=torch.int32), torch.tensor(rule_steps, dtype=torch.int32), ind_steps, pars_ind_steps, rule_52_limits, pos, neg, torch.tensor(tot_pos, dtype=torch.float), torch.tensor(tot_neg, dtype=torch.float), mask, target)

def compress_to_threshold(prob_data_list,threshold):
  
  # size_hist = defaultdict(int)
  
  # sizes = []
  # times = []
  
  # size_and_prob = []
  
  # for i,(metainfo,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms)) in enumerate(prob_data_list):
  #   print(metainfo)

  #   size = len(init)+len(deriv)
  #   if size > 0:
  #     size_and_prob.append((size,(metainfo,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms))))
  #     size_hist[len(init)+len(deriv)] += 1

  # print("size_hist")
  # tot = 0
  # sum = 0
  # small = 0
  # big = 0
  # for val,cnt in sorted(size_hist.items()):
  #   sum += val*cnt
  #   tot += cnt
  #   # print(val,cnt)
  #   if val > threshold:
  #     big += cnt
  #   else:
  #     small += cnt
  # print("Average",sum/tot)
  # print("Big",big)
  # print("Small",small)

  print("Compressing for threshold",threshold)
  prob_data_list.sort(key=lambda x : x[0][2])
  
  compressed = []
  
  while prob_data_list:
    (a,b,size), my_rest = prob_data_list.pop(0)
    my_friends = [((a,b,size), my_rest)]

# We add up sizes before compression, which we only do up to size = size 
    while size < threshold and prob_data_list:
# When having compressed, we are sometimes below size = size again, hence we repeat the conditional loop
      while size < threshold and prob_data_list:
    
        (a,b,friend_size), my_friend = prob_data_list.pop(0)
        size += friend_size
        my_friends.append(((a, b, friend_size), my_friend))

        print("friend_size", friend_size, "total size", size, flush=True)

      (a, b, size), my_rest = IC.compress_prob_data_with_fixed_ids(my_friends)
      my_friends = [((a, b, size), my_rest)]

    compressed.append(my_friends)

  print()
  print("Compressed to",len(compressed),"merged problems")
  return compressed

if __name__ == "__main__":
  # Experiments with pytorch and torch script
  # what can be learned from a super-simple TreeNN
  # which distinguishes:
  # 1) conj, user_ax, theory_ax_kind in the leaves
  # 2) what inference leads to this in the tree nodes
  #
  # To be called as in: ./compressor.py <folder> raw_log_data_*.pt data_sign.pt
  #
  # raw_log_data is compressed via abstraction (and a smoothed representation is created)
  #
  # data_sign.pt is updated (thax might be getting culled using MAX_USED_AXIOM_CNT and stored to <folder>)
  #
  # optionally, multiple problems can be grouped together (also using the compression code)
  #
  # finally, a split on the shuffled list is performed (according to HP.VALID_SPLIT_RATIO) and training_data.pt validation_data.pt are saved to folder


  args = parse_args()

  if args["mode"] == "pre":
    prob_data_list = torch.load(args["file"], weights_only=False)

    print("Making smooth compression discreet again (and forcing weight back to 1.0!)")
    for i, ((probname, probweight, size),(init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg, axioms)) in enumerate(prob_data_list):
      if True:
        probweight = 1.0
    
      print(probname,probweight)
      print(tot_pos,tot_neg)
      
      tot_pos = 0.0
      tot_neg = 0.0
              
      for id,val in neg_vals.items():
        if id in pos_vals and pos_vals[id] > 0.0: # pos has priority
          neg_vals[id] = 0.0
        elif val > 0.0:
          neg_vals[id] = 1.0 # neg counts as one
          tot_neg += 1.0

      for id, val in pos_vals.items():
        if val > 0.0:
          pos_vals[id] = 1.0 # pos counts as one too
          tot_pos += 1.0

      # new stuff -- normalize so that each abstracted clause in a problem has so much "voice" that the whole problem has a sum of probweight
      factor = probweight / (tot_pos + tot_neg)
      for id, val in pos_vals.items():
        pos_vals[id] *= factor
      for id, val in neg_vals.items():
        neg_vals[id] *= factor
      tot_pos *= factor
      tot_neg *= factor

      print(tot_pos, tot_neg)

      prob_data_list[i] = ((probname, probweight, size),(init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg, axioms))

    print("Compressing every individual problem.")
    prob_data_list = [IC.compress_prob_data([prob_data_list[i]]) for i in range(len(prob_data_list))]
    torch.save(prob_data_list, args["file"])
    print("Done.")
    exit()
  
  if args["mode"] == "map":
    prob_data_list = torch.load(args["file"], weights_only=False)
    print("Extracting mapping between id in individual problem and id in one big problem, and according pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals of first occurence.", flush=True)
    _, old2new, pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals = IC.compress_prob_data(prob_data_list, True)
    print("Generated old2new and num_to_pos/neg_vals. Aligning and saving.", flush=True)
    combined_keys = set(list(num_to_pos_vals.keys()) + list(num_to_neg_vals.keys()))
    full_range = set([x for x in range(len(prob_data_list))])
    missing = set(full_range - combined_keys)
    num_to_pos_vals_ = dict()
    num_to_neg_vals_ = dict()
    old2new_ = dict()
    new_prob_data_list = [None] * (len(full_range)-len(missing))
    for key in full_range:
      if key not in num_to_pos_vals:
        num_to_pos_vals[key] = set()
    for key in full_range:
      if key not in num_to_neg_vals:
        num_to_neg_vals[key] = set()
    for i in range(len(combined_keys)):
      num_to_pos_vals_[i] = num_to_pos_vals[list(combined_keys)[i]]
      num_to_neg_vals_[i] = num_to_neg_vals[list(combined_keys)[i]]
      old2new_[i] = old2new[list(combined_keys)[i]]
      new_prob_data_list[i] = prob_data_list[list(combined_keys)[i]]
    torch.save((old2new_, pos_vals, neg_vals, num_to_pos_vals_, num_to_neg_vals_), args["out_file_1"])
    torch.save(new_prob_data_list, args["file"])
    del prob_data_list
    del new_prob_data_list
    print("Done.",flush=True)
    exit()

  if args["mode"] == "adj":
    prob_data_list = torch.load(args["file"], weights_only=False)
    print("Reading in mapping between id in individual problem and id in one big problem, and according pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals of first occurence.", flush=True)
    old2new, pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals = torch.load(args["add_file_1"], weights_only=False)
    print("Done.", flush=True)
    print("Assigning new ids and weights to individual problems.", flush=True)
    prob_data_list = IC.adjust_ids_and_pos_neg_vals(prob_data_list, old2new, pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals)
    print("Reducing problems a bit.", flush=True)
    prob_data_list = IC.reduce_problems(prob_data_list)
    torch.save(prob_data_list, args["out_file_1"])
    print("Done.", flush=True)
    exit()

  if args["mode"] == "match":
## ULTIMATE heuristic: Root-reduction
#
# We gather the pos_val/neg_val roots with weights for every problem, i.e., those without children.
# Then we investigate if they are in another problem, but not as root.
# We can then move the weight and delete the root and possibly parents.
    prob_data_list = torch.load(args["file"], weights_only=False)
    old2new, pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals = torch.load(args["add_file_1"], weights_only=False)
    print("Problems read in. Performing root-reduction.", flush=True)
    ids = {}
    num_2_roots_with_weight = {}
    for i in range(len(prob_data_list)):
      ids[i] = set()
      ids[i] = set([x for x,_ in prob_data_list[i][1][0]] + [x for x,_ in prob_data_list[i][1][1]])
      num_2_roots_with_weight[i] = set()
    print("Initial sum of nodes: {}".format(sum([len(ids[x]) for x in range(len(ids))])), flush=True)
    for i in num_to_pos_vals.keys():
      if len(num_to_pos_vals[i]) == 0:
        num_to_pos_vals[i] = set()
    for i in num_to_neg_vals.keys():
      if len(num_to_neg_vals[i]) == 0:
        num_to_neg_vals[i] = set()

    for i in range(len(prob_data_list)):
      more_roots = True
      while more_roots:
        num_2_roots_with_weight[i] = set(prob_data_list[i][1][3].keys()).union(set(prob_data_list[i][1][4].keys())).difference(set().union(*prob_data_list[i][1][2].values()))
        print(i,len(prob_data_list),i/len(prob_data_list),flush=True)
        these_roots = num_2_roots_with_weight[i]
        if len(these_roots):
          a,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,ax) = prob_data_list[i]
          other_probs = [x for x in ids.keys() if x != i]
          these_intersections = {}
          for ind in other_probs:
            these_intersections[ind] = these_roots & ids[ind]
            these_roots.difference_update(these_intersections[ind])
            if len(these_roots) == 0:
              break
          for key in these_intersections.keys():
            a_,(init_,deriv_,pars_,pos_vals_,neg_vals_,tot_pos_,tot_neg_,ax_) = prob_data_list[key]
            for id in these_intersections[key]:
              if id in pos_vals:
                pos_vals_[id] = pos_vals[id]
                del pos_vals[id]
              if id in neg_vals:
                neg_vals_[id] = neg_vals[id]
                del neg_vals[id]
            prob_data_list[key] = (a_,(init_,deriv_,pars_,pos_vals_,neg_vals_,tot_pos_,tot_neg_,ax_))
          prob_data_list[i] = (a,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,ax))
          prob_data_list[i] = IC.reduce_problems([prob_data_list[i]])[0]
        more_roots = num_2_roots_with_weight[i] != set(prob_data_list[i][1][3].keys()).union(set(prob_data_list[i][1][4].keys())).difference(set().union(*prob_data_list[i][1][2].values()))

    for i in range(len(prob_data_list)):
      ids[i] = set()
      ids[i] = set([x for x,_ in prob_data_list[i][1][0]] + [x for x,_ in prob_data_list[i][1][1]])
    print("Root-reduced sum of nodes: {}".format(sum([len(ids[x]) for x in range(len(ids))])),flush=True)
    torch.save(prob_data_list, "{}.root_reduced".format(args["file"]))
    exit()
 
  if args["mode"] == "split":
    prob_data_list = torch.load(args["file"], weights_only=False)
    ax_to_prob = dict()
    for i,(_,(init,_,_,_,_,_,_,_)) in enumerate(prob_data_list):
      for _, thax in init:
        if thax not in ax_to_prob:
          ax_to_prob[thax] = {i}
        else:
          ax_to_prob[thax].add(i)
    torch.save(ax_to_prob, "{}/axiom_counts.pt".format(args["folder"]))
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
    print("Split problems into {} training instances and {} validation instances".format(len(train_data_list), len(valid_data_list)), flush=True)
    torch.save(train_data_list, "{}.train".format(args["file"]))
    torch.save(valid_data_list, "{}.valid".format(args["file"]))
    print("Done.", flush=True)
    exit()

  if args["mode"] == "gen":
    prob_data_list = torch.load(args["file"], weights_only=False)
    prob_data_list = threaded_smallest_min_overlap_compression(prob_data_list)
    dir = "{}/pieces".format(args["folder"])
    os.makedirs(dir, exist_ok=True)
    for i, (metainfo, rest) in enumerate(prob_data_list):
      piece_name = "piece{}.pt.{}".format(i, args["add_mode_1"])
      print("Calculating Greedy Evaluation Scheme for {}".format(piece_name), flush=True)
      torch.save(greedy(rest), "{}/pieces/{}".format(args["folder"], piece_name))
    prob_data_list = [(prob_data_list[i][0][2], "piece{}.pt.{}".format(i, args["add_mode_1"])) for i in range(len(prob_data_list))]
    filename = "{}/{}_index.pt".format(args["folder"], args["add_mode_1"])
    print("Saving {} part to".format(args["add_mode_1"]), filename)
    torch.save(prob_data_list, filename)
    print("Done.", flush=True)