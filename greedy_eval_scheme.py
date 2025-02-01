import torch

import os
import sys
from collections import defaultdict

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
  ids = [id for id,_ in init]
  id_to_ind = {id: i for i, (id,_) in enumerate(init)}

  rules = list({rule for _,rule in deriv})
  rule_ids = defaultdict(set)
  for id, rule in deriv:
    rule_ids[rule].add(id)
  rule_num = 0

  remaining_count = sum(map(len, rule_ids.values()))

  rule_steps = []
  ind_steps = []
  pars_ind_steps = []

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
    rule_steps.append(rules[rule_num])

    this_rule = rules[rule_num]
    id_pool = [id for id in rule_ids[this_rule] if set(pars[id]).issubset(old_ids)]
    id_to_ind.update([(id_pool[i], len(ids)+i) for i in range(len(id_pool))])
    ids.extend(id_pool)
    ind_steps.append(torch.tensor([id_to_ind[id] for id in id_pool], dtype=torch.int))
    if this_rule == 52:
      pars_ind_steps.append([torch.tensor([id_to_ind[this_id] for this_id in pars[id]], dtype=torch.int) for id in id_pool])
    else:
      pars_ind_steps.append(torch.tensor([id_to_ind[this_id] for id in id_pool for this_id in pars[id]], dtype=torch.int))

    rule_ids[this_rule].difference_update(id_pool)

    remaining_count = sum(map(len, rule_ids.values()))

  ids = torch.tensor(ids, dtype=torch.int)
  torch.save((ids, rule_steps, pars_ind_steps, ind_steps), folder_path + "/pieces/" + "greedy_eval_" + file)

  print(num, len(rule_steps), flush=True)

  return(num, len(rule_steps))

if __name__ == "__main__":

  folder_path=sys.argv[1]
  training_index = torch.load(folder_path+"training_index.pt",weights_only=False)
  validation_index = torch.load(folder_path+"validation_index.pt",weights_only=False)

  pool = torch.multiprocessing.Pool(12)

  for result in pool.map(doit, [(num,file,folder_path) for num,file in training_index+validation_index]):
    1
