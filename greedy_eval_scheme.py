import torch
import numpy as np

import sys
from collections import defaultdict

import itertools

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

def doit(data):
  num = data[0]
  file = data[1]
  folder_path = data[2]
  this_piece = torch.load(folder_path + "/pieces/" + file,weights_only=False)
  init = this_piece[0]
  deriv = this_piece[1]
  pars = this_piece[2]

  pos_vals = this_piece[3]
  neg_vals = this_piece[4]

  # sum_vals = set([x for x,_ in pos_vals.items()]).union(set([x for x,_ in neg_vals.items()]))

  # test_ids  = set([id for id,_ in init] + [id for id,_ in deriv])
  # test_par_ids = set(itertools.chain(*pars.values()))
  # print("no parents:",len(test_ids),len(test_ids.difference(test_par_ids)),len(test_ids.difference(test_par_ids).difference(sum_vals)),flush=True)

  ids = [id for id,_ in init]
  id_to_ind = {id: i for i, (id,_) in enumerate(init)}

  rules = list({rule for _,rule in deriv})
  rule_ids = defaultdict(set)
  for id, rule in deriv:
    rule_ids[rule].add(id)
  rule_num = 0

  remaining_count = sum(map(len, rule_ids.values()))

  rule_52_count = 0

  rule_steps = []
  ind_steps = []
  pars_ind_steps = []

  counter = 0
 
  while remaining_count > 0:
    gain = dict()
    old_ids = set(ids)
    for rule_num in range(len(rules)):
      this_rule = rules[rule_num]
      if not rule_ids[this_rule]:
        gain[rule_num] = 0
        rule_num = (rule_num + 1) % len(rules)
        continue

      gain[rule_num] = len([id for id in rule_ids[this_rule] if set(pars[id]).issubset(old_ids)])

    rule_num = max(gain, key=lambda x: gain[x])
    this_rule = rules[rule_num]
    id_pool = [id for id in rule_ids[this_rule] if set(pars[id]).issubset(old_ids)]
    id_to_ind.update([(id_pool[i], len(ids)+i) for i in range(len(id_pool))])

    if this_rule == 52:
      for id in id_pool:
        rule_52_count += 1
        rule_steps.append(rules[rule_num])
        ids.append(id)
        ind_steps.append(id_to_ind[id])
        pars_ind_steps.append([id_to_ind[this_id] for this_id in pars[id]])
        # if ((id in pos_vals) and not torch.all(torch.tensor([x in pos_vals for x in pars[id]]))):
        #   counter += 1
          # print("rule",rules[rule_num],id, id in pos_vals,[x in pos_vals and x not in neg_vals for x in pars[id]],flush=True)
        rule_ids[this_rule].discard(id)
    else:
      rule_steps.append(rules[rule_num])
      ids.extend(id_pool)
      # for id in id_pool:
        # if ((id in pos_vals) and torch.any(torch.tensor([x in neg_vals and not x in pos_vals for x in pars[id]]))):
        #   counter += 1
          # print("rule",rules[rule_num],id, id in pos_vals,[x in neg_vals and x not in pos_vals for x in pars[id]],flush=True)
      ind_steps.append([id_to_ind[id] for id in id_pool])
      pars_ind_steps.append([id_to_ind[this_id] for id in id_pool for this_id in pars[id]])
      rule_ids[this_rule].difference_update(id_pool)

    remaining_count = sum(map(len, rule_ids.values()))

  torch.save((ids, rule_steps, ind_steps, pars_ind_steps), folder_path + "/pieces/" + "greedy_eval_" + file)

  # print(counter,flush=True)
  # print(num, len(ids), len(sum_vals)-len(ids), flush=True)

  return num,rule_52_count

if __name__ == "__main__":

  folder_path=sys.argv[1]
  training_index = torch.load(folder_path+"training_index.pt",weights_only=False)
  validation_index = torch.load(folder_path+"validation_index.pt",weights_only=False)

  pool = torch.multiprocessing.Pool(12)

  rule_52_counts = dict()

  for num,count in pool.imap_unordered(doit, [(num,file,folder_path) for num,file in training_index+validation_index]):
    rule_52_counts[num] = count

  torch.save(rule_52_counts,folder_path + "/pieces/" + "rule_52_counts.pt")
