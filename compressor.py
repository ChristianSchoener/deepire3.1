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
  prob_data_list = [((name,weight,len(init)+len(deriv)),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,this_map)) for (name,weight),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,this_map) in prob_data_list]

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
      # print(size,friend_size,rest[0][2],len(prob_data_list))

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

def align_additional_axioms_intersection_free(prob_data_list, thax_number_mapping):
  iList = []
  iExtra = set()
  for (_,(init,_,_,_,_,_)) in prob_data_list:
    iList.append([])
    for _, (thax,sine) in init:
      iList[-1].append((thax,sine))
      if thax > HP.MAX_USED_AXIOM_CNT:
        iExtra.add((thax,sine))
  iExtra = list(iExtra)
  if len(iExtra) > 0:
    iDict = defaultdict(lambda: set(), dict())
    for init in iList:
      temp = list(set(init))
      for i in range(len(temp)):
        for j in range(len(temp)):
          iDict[temp[i]].add(temp[j])
    newspots = defaultdict(lambda:set(), dict())
    while len(iExtra)>0:
      while len(iExtra)>0 and not iExtra[0] in iDict:
        iExtra.pop(0)
      if len(iExtra) == 0:
        break
      mini = 100000000
      found = False
      if len(newspots) > 0:
        for i in range(HP.MAX_USED_AXIOM_CNT+1,max(list(newspots.keys()))+1):
          if (not newspots[i].intersection(iDict[iExtra[0]])) and (list(newspots[i])[0][1] == iExtra[0][1]) and (len(newspots[i]) < mini):
            pos = i
            mini = len(newspots[i])
            found = True
        if found:
          newspots[pos].add(iExtra[0])
          del iDict[iExtra.pop(0)]
        else:
          newspots[max(list(newspots.keys()))+1] = {iExtra[0]}
          del iDict[iExtra.pop(0)]
      else:
        newspots[HP.MAX_USED_AXIOM_CNT+1] = {iExtra[0]}
        del iDict[iExtra.pop(0)]
    reverse_newspots = {val: key for key, vals in newspots.items() for val in vals}
    for i in range(len(iList)):
      for j in range(len(iList[i])):
        if iList[i][j][0] > HP.MAX_ADDITIONAL_AXIOMS:
          thax_number_mapping[iList[i][j][0]] = reverse_newspots[iList[i][j]]
    for i,(info,(init,deriv,pars,selec,good,this_map)) in enumerate(prob_data_list):
      for _,(ax,_) in init:
        this_map[ax] = thax_number_mapping[ax]
      prob_data_list[i]  = (info,(init,deriv,pars,selec,good,this_map))
    print("Additional axioms for embedding:",len(list(newspots.keys())))
  return prob_data_list,thax_number_mapping

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

  thax_sign,sine_sign,deriv_arits,thax_to_str,thax_number_mapping = torch.load(sys.argv[3])

  # prob_data_list = prob_data_list[:10]
  
  print("Loaded raw prob_data_list of len:",len(prob_data_list))

  print("Dropping axiom information, not needed anymore")
  prob_data_list = [(metainfo,(init,deriv,pars,selec,good,this_map)) for (metainfo,(init,deriv,pars,selec,good,_,this_map)) in prob_data_list]
  print("Done")

  prob_data_list = [(info,(init,deriv,pars,selec,good,this_map)) for (info,(init,deriv,pars,selec,good,this_map)) in prob_data_list if len(set([init[i][1] for i in range(len(init))])) <= HP.MAX_ADDITIONAL_AXIOMS and len(init)+len(deriv) <= HP.WHAT_IS_HUGE]
  print("Removed problems with more than",HP.MAX_ADDITIONAL_AXIOMS,"axioms and those with more than",HP.WHAT_IS_HUGE,"combined length of inits + derivs" )

  if HP.ALIGN_INTERSECTION_FREE:
    print("Arranging additional axioms intersection-free.")
    prob_data_list, thax_number_mapping = align_additional_axioms_intersection_free(prob_data_list, thax_number_mapping)

    print("Arranged additional axioms intersection-free.")


  for i, ((probname,probweight),(init,deriv,pars,selec,good,this_map)) in enumerate(prob_data_list):
    print(probname,len(init),len(deriv),len(pars),len(selec),len(good),len(this_map))
  
    pos_vals = defaultdict(float)
    neg_vals = defaultdict(float)
    tot_pos = 0.0
    tot_neg = 0.0

    # Longer proofs have correspondly less weight per clause (we are fair on the per problem level)
    one_clause_weigth = probweight/len(selec)

    for id in selec:
      if id in good:
        pos_vals[id] = one_clause_weigth
        tot_pos += one_clause_weigth
      else:
        neg_vals[id] = one_clause_weigth
        tot_neg += one_clause_weigth

    prob_data_list[i] = ((probname,probweight),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,this_map))

## Add thax->0 mapping values for missing values
  for i in thax_to_str:
    if i not in thax_number_mapping:
      thax_number_mapping[i] = 0

  torch.save((thax_sign,sine_sign,deriv_arits,thax_to_str,thax_number_mapping), "{}/data_sign.pt".format(sys.argv[1]))
  print(f"Done; data_sign updated (and saved to {sys.argv[1]})")

  # prob_data_list = compress_by_probname(prob_data_list)

  if True:
    print("Making smooth compression discreet again (and forcing weight back to 1.0!)")
    for i, ((probname,probweight),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,this_map)) in enumerate(prob_data_list):
      print()
    
      if True:
        probweight = 1.0
    
      print(probname,probweight)
      print(tot_pos,tot_neg)
      
      tot_pos = 0.0
      tot_neg = 0.0
              
      for id,val in neg_vals.items():
        if id in pos_vals and pos_vals[id] > 0.0: # pos has priority
          '''
          if val != 1.0:
            print("negval goes from",val,"to 0.0 for posval",pos_vals[id])
          '''
          neg_vals[id] = 0.0
        elif val > 0.0:
          '''
          if val != 1.0:
            print("negval goes from",val,"to 1.0")
          '''
          neg_vals[id] = 1.0 # neg counts as one
          tot_neg += 1.0

      for id,val in pos_vals.items():
        if val > 0.0:
          '''
          if val != 1.0:
            print("posval goes from",val,"to 1.0")
          '''
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

      prob_data_list[i] = ((probname,probweight),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,this_map))

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
