#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import numpy as np

import torch
torch.set_printoptions(precision=16)
torch.set_default_dtype(torch.float64)

import argparse

import time,random,math,os

import sys,random

import gc

from collections import Counter

import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():

  parser = argparse.ArgumentParser(description="Process command-line arguments with key=value format.")
  # parser.add_argument("mode", help="Determines the mode. Can be zero-test, pre, or compress.")
  
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

def greedy(prob_data):
  init, deriv, pars = prob_data[1][:3]
  name = prob_data[0][0]

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
      # print(r, len(results_), end=" ", flush = True)
      if not result_queue.empty():
        results_.append(result_queue.get_nowait())
        if results_[-1] is not None:
          rule_, empty_count_, empty_keys_ = results_[-1]
          gain[rule_] = empty_count_
          empty_keys[rule_] = empty_keys_

    # print("\n {", end=" ",flush=True)
    # for rule in rules:
    #   print("{}: {}".format(rule, gain[rule]), end="    ", flush=True)
    # print("} ", flush=True)

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
    # if best_rule != 52:
    id_to_ind.update({id: len(ids) + i for i, id in enumerate(id_pool)})
    ids.extend(id_pool)
    rule_steps.append(best_rule)
    ind_steps.append(torch.tensor([id_to_ind[id] for id in id_pool], dtype=torch.int32))
    pars_ind_steps.append(torch.tensor([id_to_ind[this_id] for id in id_pool for _, this_id in enumerate(pars[id])], dtype=torch.int32))

    if best_rule == 52:
      rule_52_limits[len(ind_steps)-1] = torch.tensor([0] + list(np.cumsum([pars_len[id] for id in id_pool])), dtype=torch.int32)

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

  return {"name": name, "thax": thax, "ids": torch.tensor(ids, dtype=torch.int32), "rule_steps": torch.tensor(rule_steps, dtype=torch.int32), "ind_steps": ind_steps, "pars_ind_steps": pars_ind_steps, "rule_52_limits": rule_52_limits}

def setup_weights(prob, selecs, goods, negs):

  pos = [0.] * len(prob["ids"])
  neg = [0.] * len(prob["ids"])

  # print(selecs)
  # print()
  # print(goods)

  if HP.WEIGHT_STRATEGY == "Simple":
    this_good = set(k for inner_dict in goods.values() for k in inner_dict.keys())
    this_neg = set.union(*selecs.values()) - this_good 
    for num, id in enumerate(prob["ids"]):
      if id.item() in this_good:
        pos[num] = 1.
      elif id in this_neg:
        neg[num] = 1.
# positive preferred over negative problem-wise, but not between merged problems
  elif HP.WEIGHT_STRATEGY == "PerProblem":
    for i, vals in selecs.items():
      these_selec = vals
      these_good = set(goods[i].keys())
      these_neg = these_selec - these_good
      factor = 1./len(these_selec)
      for num, id in enumerate(prob["ids"]):
        if id.item() in these_good:
          pos[num] += factor
        elif id.item() in these_neg:
          neg[num] += factor
# positive and negative equal problem-wise and between problems
  elif HP.WEIGHT_STRATEGY == "PerProblem_mixed":
    for i, _ in selecs.items():
      these_good = set(goods[i].keys())
      these_neg = set(negs[i].keys())
      factor = 1. / len(these_good | these_neg)
      for num, id in enumerate(prob["ids"]):
        if id.item() in these_good:
          pos[num] += factor * goods[i][id.item()] / (goods[i][id.item()] + negs[i].get(id.item(), 0))
        if id.item() in these_neg:
          neg[num] += factor * negs[i][id.item()] / (goods[i].get(id.item(), 0) + negs[i][id.item()])
  elif HP.WEIGHT_STRATEGY == "Additive":
    for i, vals in selecs.items():
      these_selec = vals
      these_good = goods[i]
      these_neg = these_selec - these_good
      factor = 1./len(these_selec)
      for num, id in enumerate(prob["ids"]):
        if id.item() in these_good:
          pos[num] += factor
        elif id.item() in these_neg:
          neg[num] += factor

  pos = torch.tensor(pos, dtype=torch.float64)
  neg = torch.tensor(neg, dtype=torch.float64)

  tot_pos = sum(pos)
  tot_neg = sum(neg)

  target = pos / (pos + neg)

  mask = (pos > 0) | (neg > 0)
  
  prob["pos"] = pos
  prob["neg"] = neg
  prob["tot_pos"] = tot_pos
  prob["tot_neg"] = tot_neg
  prob["target"] = target
  prob["mask"] = mask

  return prob


if __name__ == "__main__":

  args = parse_args()

  assert("mode" in args), "A command line argument mode=... must be passed. \n" + \
                          "Options:\n" + \
                          "1. mode=zero-test, preparing the files for the analytic test.\n" + \
                          "2. mode=pre, preparing the raw data for compression by excluding large cases, setting all axioms to zero which aren't among the chosen number of most common ones, and cleaning up data a bit, as well as splitting into training and validation sets.\n" + \
                          "3. mode=compress, compressing the prepared files and splitting them up into individual training and validation instances."

  # if args["mode"] == "zero-test":
  #   assert hasattr(HP, "ZERO_FILE"), "Parameter ZERO_FILE in hyperparams.py not set. This file is read in for producing the one big multi-tree."
  #   assert isinstance(HP.ZERO_FILE, str), "Parameter ZERO_FILE in hyperparams.py is not a string. This file is read in for producing the one big multi-tree."
  #   assert os.path.isfile(HP.ZERO_FILE), "Parameter ZERO_FILE in hyperparams.py does not point to an existing file. This file is read in for producing the one big multi-tree."

  #   assert hasattr(HP, "ZERO_FOLDER"), "Parameter ZERO_FOLDER in hyperparams.py not set. This folder shall contain the raw data for producing the one big multi-tree."
  #   assert isinstance(HP.ZERO_FOLDER, str), "Parameter ZERO_FOLDER in hyperparams.py is not a string. This folder shall contain the raw data for producing the one big multi-tree."
  #   assert os.path.isdir(HP.ZERO_FOLDER), "Parameter ZERO_FOLDER in hyperparams.py does not point to an existing folder. This folder shall contain the raw data for producing the one big multi-tree."

  #   print("Loading problem file.", flush=True)
  #   prob_data_list = torch.load(HP.ZERO_FILE, weights_only=False)

  #   print("Dropping axiom information, not needed anymore.")
  #   prob_data_list = [(metainfo, (init, deriv, pars, selec, good)) for (metainfo, (init, deriv, pars, selec, good, _)) in prob_data_list]

  #   print("Dropping large proofs (init+deriv > 30000).")
  #   prob_data_list = [(metainfo, (init, deriv, pars, selec, good)) for (metainfo, (init, deriv, pars, selec, good)) in prob_data_list if len(init) + len(deriv) < 30000]

  #   print("Setting axiom numbers to 0, if above MAX_USED_AXIOM_CNT, and updating thax number to name mapping.")
  #   thax_sign, deriv_arits, thax_to_str = torch.load("{}/data_sign_full.pt".format(HP.ZERO_FOLDER), weights_only=False)
  #   prob_data_list, thax_sign, thax_to_str = IC.set_zero(prob_data_list, thax_to_str)

  #   print("Updating data signature.", flush=True)
  #   torch.save((thax_sign, deriv_arits, thax_to_str), "{}/data_sign.pt".format(HP.ZERO_FOLDER))

  #   print("Generating one multi-tree.", flush=True)
  #   prob, old2new, selec, good, prob_names = IC.compress_prob_data(prob_data_list, True)
  #   torch.save((torch.tensor(list(selec), dtype=torch.int32), torch.tensor(list(good), dtype=torch.int32)), "{}/global_selec_and_good.pt".format(HP.ZERO_FOLDER))
  #   torch.save((old2new, prob_names), "{}/old2new_and_prob_names.pt".format(HP.PRE_FOLDER))

  #   # print("Cropping multitree to what is induced by the selection.", flush=True)
  #   # prob_data_list = IC.crop(prob)

  #   print("Computing Greedy Evaluation Scheme for fast processing on more efficient data structure.", flush=True)
  #   tree_dict = greedy(prob_data_list[0], selec, good, old2new, prob_names)

  #   print("Saving.", flush=True)
  #   torch.save(tree_dict, "{}/full_tree_cropped_revealed_{}.pt".format(HP.ZERO_FOLDER, HP.MAX_USED_AXIOM_CNT))

  #   print("Done.", flush=True)
  #   exit()

  if args["mode"] == "pre":
    assert hasattr(HP, "PRE_FILE"), "Parameter PRE_FILE in hyperparams.py not set. This file contains the raw data from log_loading."
    assert isinstance(HP.PRE_FILE, str), "Parameter PRE_FILE in hyperparams.py is not a string. This file contains the raw data from log_loading."
    assert os.path.isfile(HP.PRE_FILE), "Parameter PRE_FILE in hyperparams.py does not point to an existing file. This file contains the raw data from log_loading."

    assert hasattr(HP, "PRE_FOLDER"), "Parameter PRE_FOLDER in hyperparams.py not set. This folder shall contains the raw data produced by log_loading."
    assert isinstance(HP.PRE_FOLDER, str), "Parameter PRE_FOLDER in hyperparams.py is not a string. This folder shall contain the raw data produced by log_loading."
    assert os.path.isdir(HP.PRE_FOLDER), "Parameter PRE_FOLDER in hyperparams.py does not point to an existing folder. This folder shall contain the raw data produced by log_loading."

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

    print("Generate old2new, selec, good and prob_names from the modified data.", flush=True)
    _, old2new, selec, good, neg, prob_names = IC.compress_prob_data(prob_data_list, "long", True)
    torch.save((selec, good, neg), "{}/global_selec_and_good.pt".format(HP.PRE_FOLDER))
    torch.save(prob_names, "{}/prob_names.pt".format(HP.PRE_FOLDER))

    print("Assigning new ids to individual problems and cropping.", flush=True)
    def adjust_ids_and_crop_wrapper(args):
      prob_index, prob_data = args
      return IC.adjust_ids_and_crop(prob_data, old2new[prob_index])
    with ThreadPoolExecutor() as executor:
      prob_data_list = list(executor.map(adjust_ids_and_crop_wrapper, enumerate(prob_data_list)))

    print("Extracting axiom-to-problem  and problem-to-axiom mappings to ensure, that every axiom is in the training. (And also having the data for swapping out uniformly, currently not implemented).", flush=True)
    ax_to_prob = dict()
    for i,(_, (init, _, _)) in enumerate(prob_data_list):
      for _, thax in init:
        if thax not in ax_to_prob:
          ax_to_prob[thax] = {i}
        else:
          ax_to_prob[thax].add(i)
    torch.save(ax_to_prob, "{}/axiom_counts.pt".format(HP.PRE_FOLDER))
    print("Saved axiom counts for uniformly distributed expectation SWAPOUT", flush=True)
    rev_ax_to_prob = dict()
    for key, vals in ax_to_prob.items():
      for val in vals:
        if not val in rev_ax_to_prob:
          rev_ax_to_prob[val] = {key}
        else:
          rev_ax_to_prob[val].add(key)
    print("Selecting problems for training to ensure that every axiom is in training.", flush=True)
    gather = set()
    train_data_list = []
    to_delete = set()
    train_length = math.ceil(len(prob_data_list) * HP.VALID_SPLIT_RATIO)
    while len(set(ax_to_prob.keys()) - gather) > 0:
      this_ax = random.choice(list(set(ax_to_prob.keys()) - gather))
      this_prob = random.choice(list(ax_to_prob[this_ax] - to_delete))
      to_delete.add(this_prob)
      gather = gather | rev_ax_to_prob[this_prob]
    train_data_list = [prob_data_list[i] for i in range(len(prob_data_list)) if i in to_delete] 
    prob_data_list = [prob_data_list[i] for i in range(len(prob_data_list)) if not i in to_delete]
    print("Splitting the rest of the problems according to the split ratio.", flush=True)
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
    assert hasattr(HP, "COM_FOLDER"), "Parameter COM_FOLDER in hyperparams.py not set. This folder is the base folder of the project."
    assert isinstance(HP.COM_FOLDER, str), "Parameter COM_FOLDER in hyperparams.py is not a string. This folder is the base folder of the project."
    assert os.path.isdir(HP.COM_FOLDER), "Parameter COM_FOLDER in hyperparams.py does not point to an existing directory. This folder is the base folder of the project."

    assert hasattr(HP, "COM_ADD_MODE_1"), "Parameter COM_ADD_MODE_1 in hyperparams.py not set. This parameter takes values either train or valid an indicates, which of the data sets shall be compressed and further transformed to perform the computations."
    assert(HP.COM_ADD_MODE_1 == "train" or HP.COM_ADD_MODE_1 == "valid"), "Parameter COM_ADD_MODE_1 in hyperparams.py is not train or valid. This parameter indicates, which of the data sets shall be compressed and further transformed to perform the computations."

    assert hasattr(HP, "COM_FILE"), "Parameter COM_FILE in hyperparams.py not set. This filename is the one created by log_loader and is suffixed with .valid or .train (done automatically)."
    assert isinstance(HP.COM_FILE, str), "Parameter COM_FILE in hyperparams.py is not a string. This filename is the one created by log_loader and is suffixed with .valid or .train (done automatically)."
    assert os.path.isfile(HP.COM_FILE + "." + HP.COM_ADD_MODE_1), "The string COM_FILE + \".\" + COM_ADD_MODE_1 does not point to an existing file. This file contains the prepared data of the preparation step, either for training or validation."

    print("Loading problem file, selec and good, and old2new and prob_names.", flush=True)
    prob_data_list = torch.load(HP.COM_FILE + "." + HP.COM_ADD_MODE_1, weights_only=False)

    selec, good, neg = torch.load("{}/global_selec_and_good.pt".format(HP.COM_FOLDER))
    prob_names = torch.load("{}/prob_names.pt".format(HP.PRE_FOLDER), weights_only=False)

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
    print("Compressed to {} instances.".format(len(compressed)), flush=True)
    print("Done.", flush=True)
    prob_data_list.extend(compressed)
    del compressed

    print("Computing Greedy Evaluation Schemes.", flush=True)
    with ProcessPoolExecutor(max_workers=HP.NUMPROCESSES) as executor:
      prob_data_list = list(executor.map(greedy, prob_data_list))

    print()
    print("Setting up weights.", flush=True)
    inv_prob_names = {val: key for key, val in prob_names.items()}
    # old2news = [{x: set(y.values()) for x, y in old2new.items() if inv_prob_names[x] in prob_data_list[i]["name"]} for i in range(len(prob_data_list))]
    selecs = [{x: y for x, y in selec.items() if inv_prob_names[x] in prob_data_list[i]["name"]} for i in range(len(prob_data_list))]
    goods = [{x: y for x, y in good.items() if inv_prob_names[x] in prob_data_list[i]["name"]} for i in range(len(prob_data_list))]
    negs = [{x: y for x, y in neg.items() if inv_prob_names[x] in prob_data_list[i]["name"]} for i in range(len(prob_data_list))]
    
    def setup_weights_wrapper(args):
      prob_index, prob_data = args
      return setup_weights(prob_data, selecs[prob_index], goods[prob_index], negs[prob_index])
    with ProcessPoolExecutor(max_workers=HP.NUMPROCESSES) as executor:
      prob_data_list = list(executor.map(setup_weights_wrapper, enumerate(prob_data_list)))
    del selec, good, neg
    print("Done.", flush=True)

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