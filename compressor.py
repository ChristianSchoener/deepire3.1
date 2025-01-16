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
   
  print("Compressing for treshold",treshold)

## Huffmann-encoding
  if len(prob_data_list) > 1:
    prob_data_list.sort(key=lambda x : x[0][2])
    (name,weight,size), my_rest = prob_data_list[0]
    (friend_name,friend_weight,friend_size), friend_rest = prob_data_list[1]
    # print(size,friend_size,len(prob_data_list))
    while size + friend_size < treshold:

      rest = IC.compress_prob_data([((name,weight,size), my_rest),((friend_name,friend_weight,friend_size), friend_rest)])
      prob_data_list.pop(0)
      prob_data_list.pop(0)
      prob_data_list.append(rest)

      if len(prob_data_list) == 1:
        break
      prob_data_list.sort(key=lambda x : x[0][2])
      (name,weight,size), my_rest = prob_data_list[0]
      (friend_name,friend_weight,friend_size), friend_rest = prob_data_list[1]

    print()
    print("Compressed to",len(prob_data_list),"merged problems")
    prob_data_list.sort(key=lambda x : -x[0][2])
    sizes = dict()
    for i,((_,_,size),_) in enumerate(prob_data_list):
      sizes["piece "+ str(i)] = size
    print("Sizes:",sizes)
    return prob_data_list

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

  thax_sign,deriv_arits,thax_to_str = torch.load(sys.argv[3])

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
      prob_data_list[i] = (metainfo[2],piece_name)
    print("Done")

  random.shuffle(prob_data_list)
  spl = math.ceil(len(prob_data_list) * HP.VALID_SPLIT_RATIO)
  print("shuffled and split at idx",spl,"out of",len(prob_data_list))
  print()

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
