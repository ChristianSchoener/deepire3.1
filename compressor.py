#!/usr/bin/env python3

import inf_common as IC
import hyperparams as HP

import torch
from torch import Tensor

import time,bisect,random,math,os,errno

from typing import Dict, List, Tuple, Optional

from collections import defaultdict
from collections import ChainMap

import sys,random,itertools

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


def compress_to_treshold(prob_data_list,treshold):
  
  size_hist = defaultdict(int)
  
  sizes = []
  times = []
  
  size_and_prob = []
  
  for i,(metainfo,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms)) in enumerate(prob_data_list):
    print(metainfo)

    size = len(init)+len(deriv)
    
    size_and_prob.append((size,(metainfo,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms))))
    
    size_hist[len(init)+len(deriv)] += 1

  print("size_hist")
  tot = 0
  sum = 0
  small = 0
  big = 0
  for val,cnt in sorted(size_hist.items()):
    sum += val*cnt
    tot += cnt
    # print(val,cnt)
    if val > treshold:
      big += cnt
    else:
      small += cnt
  print("Average",sum/tot)
  print("Big",big)
  print("Small",small)

  print("Compressing for treshold",treshold)

# ## Huffmann-encoding
#   if len(prob_data_list) > 1:
#     prob_data_list.sort(key=lambda x : x[0][2])
#     (name,weight,size), my_rest = prob_data_list[0]
#     (friend_name,friend_weight,friend_size), friend_rest = prob_data_list[1]
#     # print(size,friend_size,len(prob_data_list))
#     while size + friend_size < treshold:

#       rest = IC.compress_prob_data([((name,weight,size), my_rest),((friend_name,friend_weight,friend_size), friend_rest)])
#       prob_data_list.pop(0)
#       prob_data_list.pop(0)
#       prob_data_list.append(rest)

#       if len(prob_data_list) == 1:
#         break
#       prob_data_list.sort(key=lambda x : x[0][2])
#       (name,weight,size), my_rest = prob_data_list[0]
#       (friend_name,friend_weight,friend_size), friend_rest = prob_data_list[1]

  prob_data_list.sort(key=lambda x : x[0][2])
  
  compressed = []
  
  while size_and_prob:
    size, my_rest = size_and_prob.pop()

    # print("Popped guy of size",size)

    while size < treshold and size_and_prob:
      # print("Looking for a friend")
      likes_sizes = int((treshold-size)*1.2)
      idx_upper = bisect.bisect_right(size_and_prob,(likes_sizes, my_rest))

      if not idx_upper:
        idx_upper = 1

      idx = random.randrange(idx_upper)
    
      # print("Idxupper",idx_upper,"idx",idx)

      _, friend_rest = size_and_prob[idx]
      del size_and_prob[idx]

      # print("friend_size",friend_size)

      my_rest = IC.compress_prob_data([my_rest,friend_rest])
      meta, _ = my_rest
      size = meta[2]
    
      # print("aftermerge",size)

    # print("Storing a guy of size",size,"weight",metainfo[-1])
    compressed.append(my_rest)

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

  prob_data_list = torch.load(sys.argv[2])

  thax_sign,sine_sign,deriv_arits,thax_to_str = torch.load(sys.argv[3])

  print("Loaded raw prob_data_list of len:",len(prob_data_list))

  # torch.save((thax_sign,deriv_arits,thax_to_str), "{}/data_sign.pt".format(sys.argv[1]))
  # print(f"Done; data_sign updated (and saved to {sys.argv[1]})")

  # # prob_data_list = compress_by_probname(prob_data_list)

  if True:
    print("Making smooth compression discreet again (and forcing weight back to 1.0!)")
    for i, ((probname,probweight,size),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms)) in enumerate(prob_data_list):
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

      for id,val in pos_vals.items():
        if val > 0.0:
          pos_vals[id] = 1.0 # pos counts as one too
          tot_pos += 1.0

      # new stuff -- normalize so that each abstracted clause in a problem has so much "voice" that the whole problem has a sum of probweight
      factor = probweight/(tot_pos+tot_neg)
      for id,val in pos_vals.items():
        pos_vals[id] *= factor
      for id,val in neg_vals.items():
        neg_vals[id] *= factor
      tot_pos *= factor
      tot_neg *= factor

      print(tot_pos,tot_neg)

      prob_data_list[i] = ((probname,probweight,size),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms))

  prob_data_list = compress_to_treshold(prob_data_list,treshold = HP.COMPRESSION_THRESHOLD)

  SAVE_PIECES = True

  if SAVE_PIECES:
    print("Saving pieces")
    dir = "{}/pieces".format(sys.argv[1])
    try:
      os.mkdir(dir)
    except OSError as exc:
      if exc.errno != errno.EEXIST:
          raise
      pass
    for i,(metainfo,rest) in enumerate(prob_data_list):
      piece_name = "piece{}.pt".format(i)
      torch.save(rest, "{}/{}".format(dir,piece_name))
    print("Done")

  all_in_train = False
  while not all_in_train:
    random.shuffle(prob_data_list)
    spl = math.ceil(len(prob_data_list) * HP.VALID_SPLIT_RATIO)
    train_thax = {}
    valid_thax = {}
    counter = 0
    for _,(init,_,_,_,_,_,_,_) in prob_data_list[spl:]:
      for _,(thax,_) in init:
        if not thax in valid_thax:
          valid_thax[thax] = 1
        else:
          valid_thax[thax] += 1

    counter = 0
    for _,(init,_,_,_,_,_,_,_) in prob_data_list[:spl]:
      for _,(thax,_) in init:
        if not thax in train_thax:
          train_thax[thax] = 1
        else:
          train_thax[thax] += 1
    all_in_train = len(set(valid_thax.keys()).difference(set(train_thax.keys()))) == 0
  print("shuffled and split at idx",spl,"out of",len(prob_data_list),"and ensured all revealed thax are in a training problem.")
  print()

  for i in range(len(prob_data_list)):
    prob_data_list[i] = (prob_data_list[i][0][2],"piece{}.pt".format(i))

  if SAVE_PIECES:
    # save just names:
    filename = "{}/training_index.pt".format(sys.argv[1])
    print("Saving training part to",filename)
    torch.save(prob_data_list[:spl], filename)
    filename = "{}/validation_index.pt".format(sys.argv[1])
    print("Saving validation part to",filename)
    torch.save(prob_data_list[spl:], filename)
  else:
    filename = "{}/training_data.pt".format(sys.argv[1])
    print("Saving training part to",filename)
    torch.save(prob_data_list[:spl], filename)
    filename = "{}/validation_data.pt".format(sys.argv[1])
    print("Saving validation part to",filename)
    torch.save(prob_data_list[spl:], filename)
