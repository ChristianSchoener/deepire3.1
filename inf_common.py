#!/usr/bin/env python3

# a module of concepts common to the inference based model development

import os

import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

import operator

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
from torch import Tensor

torch.set_num_threads(1)

from typing import Dict, List, Tuple, Optional

import numpy as np

from copy import deepcopy

from bitarray import bitarray

import math

try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

import itertools
from collections import defaultdict
import sys,random

import hyperparams as HP

import multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# A hacky, hardcoded log name normalizer!
def logname_to_probname(logname):
  logname = logname.split("/")[-1]
  assert(".log" == logname[-4:])
  if logname.startswith("small_np_"):
    assert("small_np_" == logname[:9])
    return "small_np/"+logname[9:-4]
  elif logname.startswith("_nfs_jakubja5_atp_benchmarks_mizar50_train_premsel_cek2_knn_preds__64_"):
    return "small_np/"+logname[70:-4]
  elif logname.startswith("_nfs_jakubja5_atp_benchmarks_mizar50_train_premsel_pepa1_lgb_preds__0.1_"):
    return "small_np/"+logname[72:-4]
  elif logname.startswith("_nfs_jakubja5_atp_benchmarks_mizar50_train_premsel_mirek1_gnn_preds__-1_"):
    return "small_np/"+logname[72:-4]
  elif logname.endswith(".smt2.log"):
    return logname[:-4]
  else: # jinja
    assert(logname.endswith(".log"))
    spl = logname[:-4].split("_")
    assert(spl[-1].startswith("m"))
    return "_".join(spl[:-1]) # drop the m<something> part altogether, because why not?

class Embed(torch.nn.Module):
  weight: Tensor
  
  def __init__(self, dim : int):
    super().__init__()
    
    self.weight = torch.nn.parameter.Parameter(torch.Tensor(dim))
    self.reset_parameters()
  
  def reset_parameters(self):
    torch.nn.init.normal_(self.weight)

  def forward(self) -> Tensor:
    return self.weight

class CatAndNonLinearBinary(torch.nn.Module):
  def __init__(self, dim : int, arit: int):
    super().__init__()
    
    if HP.DROPOUT > 0.0:
      self.prolog = torch.nn.Dropout(HP.DROPOUT)
    else:
      self.prolog = torch.nn.Identity(arit*dim)
    
    if HP.NONLIN == HP.NonLinKind_TANH:
      self.nonlin = torch.nn.Tanh()
    else:
      self.nonlin = torch.nn.ReLU()
    
    self.arit = arit
    
    self.first = torch.nn.Linear(arit*dim,dim*2)
    self.second = torch.nn.Linear(dim*2,dim)
    
    if HP.LAYER_NORM:
      self.epilog = torch.nn.LayerNorm(dim)
    else:
      self.epilog = torch.nn.Identity(dim) 

  def forward_impl_stack(self, args : Tensor) -> Tensor:
    if self.arit == 2:
      return self.epilog(self.second(self.nonlin(self.first(self.prolog(args.view(args.shape[0] // 2, -1))))))
      # return self.epilog(self.second(self.nonlin(self.first(self.prolog(args.view(1, -1))))))
    else:
      return self.epilog(self.second(self.nonlin(self.first(self.prolog(args)))))

  def forward(self, args : Tensor) -> Tensor:
    return self.forward_impl_stack(args)

class CatAndNonLinearMultiary(torch.nn.Module):
  def __init__(self, dim : int, arit: int):
    super().__init__()
  
    if HP.DROPOUT > 0.0:
      self.prolog = torch.nn.Dropout(HP.DROPOUT)
    else:
      self.prolog = torch.nn.Identity(arit*dim)
    
    if HP.NONLIN == HP.NonLinKind_TANH:
      self.nonlin = torch.nn.Tanh()
    else:
      self.nonlin = torch.nn.ReLU()
    
    self.arit = arit
    
    self.first = torch.nn.Linear(arit*dim,dim*2)
    self.second = torch.nn.Linear(dim*2,dim)
    
    if HP.LAYER_NORM:
      self.epilog = torch.nn.LayerNorm(dim)
    else:
      self.epilog = torch.nn.Identity(dim) 

  def forward_impl_list(self, args : Tensor) -> Tensor:
    return self.epilog(self.second(self.nonlin(self.first(self.prolog(args)))))
  
  # def forward(self, args : Tensor) -> Tensor:
  #   x = args
  #   length = x.size(0)
  #   limit = torch.tensor([0,length])

  #   select_length = 2 * (length // 2)
  #   fill_length = select_length // 2
  #   fill_limit = torch.tensor([0,fill_length])

  #   end_ind = select_length

  #   while length > 1:
  #     if limit[1] > end_ind:
  #       # print(torch.cat((x[end_ind], self.forward_impl_list(x[:select_length].view(select_length // 2, -1))[torch.arange(fill_limit[0],fill_limit[1])].ravel())).view(fill_length+1,-1),flush=True)
  #       x[:fill_length+1] = torch.cat((x[end_ind], self.forward_impl_list(x[:select_length].view(select_length // 2, -1))[torch.arange(fill_limit[0],fill_limit[1])].ravel())).view(fill_length+1,-1)
  #     else:
  #       x[:fill_length] = self.forward_impl_list(x[:select_length].view(select_length // 2, -1))[torch.arange(fill_limit[0],fill_limit[1])]

  #     length = (length + 1) // 2
  #     limit = torch.tensor([0, length])

  #     select_length = 2 * (length // 2)
  #     fill_length = select_length // 2
  #     fill_limit = torch.tensor([0, fill_length])

  #     end_ind = 2 * (length // 2)
  #     # print(x,flush=True)
     
  #   return x[0]

  # def forward(self, args : Tensor) -> Tensor: # 19s
  #   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #   length = args.shape[0]

  #   start_ind = torch.zeros(1,dtype=torch.int32)
  #   end_ind = 2 * (length // 2)
  #   fill_start_ind = length
  #   fill_end_ind = fill_start_ind + (length // 2)

  #   full_sized = torch.empty(2*length - 1, HP.EMBED_SIZE).to(device)

  #   full_sized[:length] = args

  #   while length > 1:
  #     full_sized[fill_start_ind:fill_end_ind] = self.forward_impl_list(full_sized[start_ind:end_ind].view(length // 2, -1))

  #     length = (length + 1) // 2
  #     start_ind = end_ind
  #     end_ind = start_ind + 2 * (length // 2)
  #     fill_start_ind = start_ind + length
  #     fill_end_ind = fill_start_ind + (length // 2)
     
  #   return full_sized[start_ind]

  # def forward(self, args : Tensor) -> Tensor: # 23s
  #   x = args
  #   while x.size(0) > 1:
  #     x = torch.cat((x[2:],self.forward_impl_list(x[:2].view(1,-1))))

  #   return x

  def forward(self, args : Tensor, limits : Tensor, device : str) -> Tensor:
    limits = limits.to(device)
    lengths = torch.diff(limits)
    the_len = lengths.numel()

    full_lengths = 2*lengths - 1
    start_inds = torch.cat((torch.tensor([0]).to(device), full_lengths[:-1].cumsum(dim=0)))
    end_inds = start_inds + 2 * (lengths // 2)
    fill_start_inds = start_inds + lengths
    fill_end_inds = fill_start_inds + (lengths // 2)

    return_mat = torch.zeros(the_len, HP.EMBED_SIZE)
# Looping to prevent big temporary matrix
    for i in range(the_len):
      full_sized = torch.zeros(full_lengths[i], HP.EMBED_SIZE).to(device)
      select_range = torch.arange(limits[i], limits[i+1])
      full_sized = args[select_range]

      while lengths[i] > 1:
        select_range = torch.arange(start_inds[i], end_inds[i])
        fill_range = torch.arange(fill_start_inds[i], fill_end_inds[i])

        full_sized[fill_range] = self.forward_impl_list(full_sized[select_range].view(lengths[i] // 2, -1))

        lengths[i] = (lengths[i] + 1) // 2
        start_inds[i] = end_inds[i]
        end_inds[i] = start_inds[i] + 2 * (lengths[i] // 2)
        fill_start_inds[i] = start_inds[i] + lengths[i]
        fill_end_inds[i] = fill_start_inds[i] + (lengths[i] // 2)
      
      return_mat[i] = full_sized[-1]
    return return_mat
    
  # def forward(self, args : Tensor, limits : Tensor, device : str) -> Tensor:
  #   limits = limits.to(device)
  #   lengths = torch.diff(limits)
  #   the_len = lengths.numel()

  #   full_lengths = 2*lengths - 1
  #   start_inds = torch.cat((torch.tensor([0]).to(device), full_lengths[:-1].cumsum(dim=0)))
  #   end_inds = start_inds + 2 * (lengths // 2)
  #   fill_start_inds = start_inds + lengths
  #   fill_end_inds = fill_start_inds + (lengths // 2)

  #   full_sized = torch.zeros(full_lengths.sum(), HP.EMBED_SIZE).to(device)

  #   for i in range(the_len):
  #     full_sized[torch.arange(start_inds[i], start_inds[i] + lengths[i])] = args[torch.arange(limits[i], limits[i+1])]

  #   while max(lengths) > 1:
  #     mask = torch.zeros(full_lengths.sum(), dtype=torch.bool).to(device)
  #     fill_mask = torch.zeros_like(mask).to(device)

  #     for i in range(the_len):
  #       mask[start_inds[i]:end_inds[i]] = True
  #       fill_mask[fill_start_inds[i]:fill_end_inds[i]] = True

  #     how_much = mask.sum().item()
  #     full_sized[fill_mask] = self.forward_impl_list(full_sized[mask].view(how_much // 2, -1))

  #     lengths = (lengths + 1) // 2
  #     start_inds = end_inds
  #     end_inds = start_inds + 2 * (lengths // 2)
  #     fill_start_inds = start_inds + lengths
  #     fill_end_inds = fill_start_inds + (lengths // 2)
     
  #   mask = torch.zeros(full_lengths.sum(), dtype=torch.bool).to(device)
  #   mask[start_inds] = True
  #   return full_sized[mask]

  # def forward(self, args : Tensor, limits : Tensor) -> Tensor:
  #   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #   lengths = torch.diff(limits)
  #   the_len = lengths.numel()

  #   select_lengths = 2 * (lengths // 2)
  #   fill_lengths = select_lengths // 2
  #   fill_limits = torch.cat((torch.tensor([0]).to(device), fill_lengths.cumsum(dim=0)))

  #   end_inds = limits[:-1] + select_lengths

  #   while torch.max(lengths) > 1:
  #     mask = torch.zeros(args.size(0), dtype=torch.bool)

  #     for i in torch.arange(the_len):
  #       mask[torch.arange(limits[i],end_inds[i])] = True

  #     how_much = mask.sum().item()  # Convert to Python int
  #     tmp = self.forward_impl_list(args[mask].view(how_much // 2, -1))
  #     pos = torch.tensor(0, dtype=torch.int32).to(device)
  #     for i in torch.arange(the_len):
  #       if limits[i+1] > end_inds[i]:
  #         args[pos] = args[end_inds[i]]
  #         pos += 1
  #       if fill_limits[i+1] > fill_limits[i]:
  #         args[torch.arange(pos,pos+fill_lengths[i])] = tmp[torch.arange(fill_limits[i],fill_limits[i+1])]
  #         pos += fill_lengths[i]

  #     lengths = (lengths + 1) // 2
  #     limits = torch.cat((torch.tensor([0]).to(device), lengths.cumsum(dim=0)))

  #     select_lengths = 2 * (lengths // 2)
  #     fill_lengths = select_lengths // 2
  #     fill_limits = torch.cat((torch.tensor([0]).to(device), fill_lengths.cumsum(dim=0)))

  #     end_inds = limits[:-1] + 2 * (lengths // 2)
     
  #   return args[:the_len]

def get_initial_model(thax_sign, deriv_arits):
  if HP.CUDA and torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
    
  init_embeds = torch.nn.ModuleDict()

  for i in thax_sign:
    init_embeds[str(i)] = Embed(HP.EMBED_SIZE).to(device)

  deriv_mlps = torch.nn.ModuleDict().to(device)
  for rule,arit in deriv_arits.items():
    if arit <= 2:
      deriv_mlps[str(rule)] = CatAndNonLinearBinary(HP.EMBED_SIZE, arit).to(device)
    else:
      assert(arit == 3)
      deriv_mlps[str(rule)] = CatAndNonLinearMultiary(HP.EMBED_SIZE, 2).to(device)

  eval_net = torch.nn.Sequential(
    torch.nn.Dropout(HP.DROPOUT) if HP.DROPOUT > 0.0 else torch.nn.Identity(HP.EMBED_SIZE),
    torch.nn.Linear(HP.EMBED_SIZE, HP.EMBED_SIZE * HP.BOTTLENECK_EXPANSION_RATIO // 2),
    torch.nn.Tanh() if HP.NONLIN == HP.NonLinKind_TANH else torch.nn.ReLU(),
    torch.nn.Linear(HP.EMBED_SIZE,1)).to(device)

  return torch.nn.ModuleList([init_embeds, deriv_mlps, eval_net])
  
def name_initial_model_suffix():
  return "_{}_{}_BER{}_LayerNorm{}_Dropout{}{}.pt".format(
    HP.EMBED_SIZE,
    HP.NonLinKindName(HP.NONLIN),
    HP.BOTTLENECK_EXPANSION_RATIO,
    HP.LAYER_NORM,
    HP.DROPOUT,
    "_UseSine" if HP.USE_SINE else "")

def name_learning_regime_suffix():
  return "_o{}_lr{}{}{}{}{}_wd{}_numproc{}_p{}{}_trr{}.txt".format(
    HP.OptimizerName(HP.OPTIMIZER),
    HP.LEARN_RATE,"m{}".format(HP.MOMENTUM) if HP.OPTIMIZER == HP.Optimizer_SGD else "","NonConst" if HP.NON_CONSTANT_10_50_250_LR else "",
    "clipN{}".format(HP.CLIP_GRAD_NORM) if HP.CLIP_GRAD_NORM else "",
    "clipV{}".format(HP.CLIP_GRAD_VAL) if HP.CLIP_GRAD_VAL else "",
    HP.WEIGHT_DECAY,    
    HP.NUMPROCESSES,
    HP.POS_WEIGHT_EXTRA,
    f"_swapout{HP.SWAPOUT}" if HP.SWAPOUT > 0.0 else "",
    HP.TestRiskRegimenName(HP.TRR))

def name_raw_data_suffix():
  return "_av{}_thax{}_useSine{}.pt".format(
    HP.TreatAvatarEmptiesName(HP.AVATAR_EMPTIES),
    HP.ThaxSourceName(HP.THAX_SOURCE),
    HP.USE_SINE)

bigpart1 = '''#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys,random

def save_net(name, parts, thax_to_str):
  with torch.no_grad():
    for param in parts.parameters():
      param = param.to("cpu")
  for part in parts:
    
    # eval mode and no gradient
    part.eval()
    for param in part.parameters():
      param.requires_grad = False

  # from here on only use the updated copies
  (init_embeds, deriv_mlps, eval_net) = parts
  sine_embellisher = torch.nn.Module()
  initEmbeds = {}
  for thax,embed in init_embeds.items():
    thax = int(thax)
    if thax == -1:
      st = "-1"
    elif thax in thax_to_str:
      st = thax_to_str[thax]
    else:
      assert len(thax_to_str) == 0 or thax == 0, thax
      st = str(thax)
    initEmbeds[st] = embed.weight
  
  # This is, how we envision inference:
  class InfRecNet(torch.nn.Module):
    init_abstractions : Dict[str, int]
    deriv_abstractions : Dict[str, int]
    abs_ids : Dict[int, int] # each id gets its abs_id
    embed_store : Dict[int, Tensor] # each abs_id (lazily) stores its embedding
    eval_store: Dict[int, float] # each abs_id (lazily) stores its eval

    initEmbeds : Dict[str, Tensor]
    
    def __init__(self,
        initEmbeds : Dict[str, Tensor],
        sine_embellisher : torch.nn.Module,'''

bigpart2 ='''        eval_net : torch.nn.Module):
      super().__init__()

      self.init_abstractions = {}
      self.deriv_abstractions = {}
      self.abs_ids = {}
      self.embed_store = {}
      self.eval_store = {}
      
      self.initEmbeds = initEmbeds
      self.sine_embellisher = sine_embellisher'''

sine_val_prog = "features[-1]" if HP.FAKE_CONST_SINE_LEVEL == -1 else str(HP.FAKE_CONST_SINE_LEVEL)

bigpart_no_longer_rec1 = '''
    @torch.jit.export
    def forward(self, id: int) -> float:
      abs_id = self.abs_ids[id] # must have been mentioned already
      if abs_id in self.eval_store:
        return self.eval_store[abs_id]
      else:
        val = self.eval_net(self.embed_store[abs_id]) # must have been embedded already
        self.eval_store[abs_id] = val[0].item()
        return val[0].item()

    @torch.jit.export
    def new_init(self, id: int, features : Tuple[int, int, int, int, int, int], name: str) -> None:
      # an init record is abstracted just by the name str
      abskey = name
      if abskey not in self.init_abstractions:
        abs_id = -(len(self.init_abstractions)+1) # using negative values for abstractions of init clauses
        self.init_abstractions[abskey] = abs_id
      else:
        abs_id = self.init_abstractions[abskey]

      # assumes this is called exactly once
      self.abs_ids[id] = abs_id

      if abs_id not in self.embed_store:
        if name in self.initEmbeds:
          embed = self.initEmbeds[name]
        else:
          embed = self.initEmbeds["0"]
        if {}:
          embed = self.sine_embellisher({},embed)
        self.embed_store[abs_id] = embed'''.format("+'_'+str({})".format(sine_val_prog) if HP.USE_SINE else "False",HP.USE_SINE,sine_val_prog)

bigpart_rec2='''
    @torch.jit.export
    def new_deriv{}(self, id: int, features : Tuple[int, int, int, int, int], pars : List[int]) -> None:
      rule = features[-1]
      abskey = ",".join([str(rule)]+[str(self.abs_ids[par]) for par in pars])
      
      if abskey not in self.deriv_abstractions:
        abs_id = len(self.deriv_abstractions)
        self.deriv_abstractions[abskey] = abs_id
      else:
        abs_id = self.deriv_abstractions[abskey]
      
      # assumes this is called exactly once
      self.abs_ids[id] = abs_id
      
      if abs_id not in self.embed_store:
        par_embeds = torch.stack([self.embed_store[self.abs_ids[par]].squeeze() for par in pars])
        embed = self.deriv_{}(par_embeds)
        self.embed_store[abs_id] = embed'''

bigpart_rec2_rule_52='''
    @torch.jit.export
    def new_deriv52(self, id: int, features : Tuple[int, int, int, int, int], pars : List[int]) -> None:
      rule = features[-1]
      abskey = ",".join(["52"]+[str(self.abs_ids[par]) for par in pars])
      
      if abskey not in self.deriv_abstractions:
        abs_id = len(self.deriv_abstractions)
        self.deriv_abstractions[abskey] = abs_id
      else:
        abs_id = self.deriv_abstractions[abskey]
      
      # assumes this is called exactly once
      self.abs_ids[id] = abs_id
      
      if abs_id not in self.embed_store:
        par_embeds = [self.embed_store[self.abs_ids[par]].squeeze() for par in pars]
        limits = torch.cumsum(torch.tensor([0]+[len(i) for i in par_embeds]), dim=0)
        embed = self.deriv_52(torch.stack(par_embeds), limits, "cpu")
        self.embed_store[abs_id] = embed'''

bigpart_avat = '''
    @torch.jit.export
    def new_avat(self, id: int, features : Tuple[int, int, int, int]) -> None:
      par = features[-1]
      abskey = ",".join(["666", str(self.abs_ids[par])])
      
      if abskey not in self.deriv_abstractions:
        abs_id = len(self.deriv_abstractions)
        self.deriv_abstractions[abskey] = abs_id
      else:
        abs_id = self.deriv_abstractions[abskey]
      
      # assumes this is called exactly once
      self.abs_ids[id] = abs_id
      
      if abs_id not in self.embed_store:
        par_embeds = torch.stack([self.embed_store[self.abs_ids[par]].squeeze()])
        embed = self.deriv_666(par_embeds) # special avatar code
        self.embed_store[abs_id] = embed'''

bigpart3 = '''
  module = InfRecNet(
    initEmbeds,
    sine_embellisher,'''

bigpart4 = '''    eval_net
    )
  script = torch.jit.script(module)
  script.save(name)'''

def create_saver(deriv_arits):
  with open("inf_saver.py", "w") as f:

    print(bigpart1, file=f)

    for rule in sorted(deriv_arits):
      print("        deriv_{} : torch.nn.Module,".format(rule), file=f)

    print(bigpart2,file=f)

    for rule in sorted(deriv_arits):
      print("      self.deriv_{} = deriv_{}".format(rule,rule), file=f)
    print("      self.eval_net = eval_net", file=f)

    print(bigpart_no_longer_rec1, file=f)

    for rule in sorted(deriv_arits):
      if rule not in [52, 666]: # avatar done differently in bigpart3, rul_52, too
        print(bigpart_rec2.format(str(rule), str(rule)), file=f)

    if 666 in deriv_arits:
      print(bigpart_avat, file=f)

    if 52 in deriv_arits:
      print(bigpart_rec2_rule_52, file=f)

    print(bigpart3, file=f)

    for rule in sorted(deriv_arits):
      print("    deriv_mlps['{}'],".format(rule), file=f)
    print(bigpart4, file=f)
 
# Learning model class
class LearningModel(torch.nn.Module):
  def __init__(self,
      init_embeds : torch.nn.ModuleDict,
      deriv_mlps : torch.nn.ModuleDict,
      eval_net : torch.nn.Module,
      data, training=True, use_cuda = False):
      # thax, ids, rule_steps, ind_steps, pars_ind_steps, rule_52_limits,pos,neg,tot_pos,tot_neg,mask,target, training=True, use_cuda = False):
      # init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,greedy_eval_scheme,save_logits = False,training = False):
    super(LearningModel,self).__init__()

    if use_cuda and torch.cuda.is_available():
      self.device = "cuda"
    else:
      self.device = "cpu"
    
    self.thax = data["thax"]
    self.ids = data["ids"].to(self.device)
    self.rule_steps = data["rule_steps"].to(self.device)
    self.ind_steps = data["ind_steps"]
    self.pars_ind_steps = data["pars_ind_steps"]
    self.rule_52_limits = data["rule_52_limits"]
    for i in range(len(self.rule_steps)):
      self.ind_steps[i] = self.ind_steps[i].to(self.device)
      self.pars_ind_steps[i] = self.pars_ind_steps[i].to(self.device)
    if i in self.rule_52_limits.keys():
      self.rule_52_limits[i] = self.rule_52_limits[i].to(self.device)

    self.vectors = torch.empty(len(self.ids), HP.EMBED_SIZE).to(self.device)
    self.vectors[:len(self.thax)] = torch.stack([init_embeds[str(this_thax.item())]() for this_thax in self.thax])

    self.deriv_mlps = deriv_mlps
    self.eval_net = eval_net

    self.pos = data["pos"].to(self.device)
    self.neg = data["neg"].to(self.device)
 
    self.target = data["target"].to(self.device)

    self.mask = data["mask"].to(self.device)

    self.tot_neg = data["tot_neg"].to(self.device)
    self.tot_pos = data["tot_pos"].to(self.device)
    self.pos_weight = (HP.POS_WEIGHT_EXTRA * self.tot_neg / self.tot_pos if self.tot_pos > 0 else torch.tensor(1.0)).to(self.device)  
    self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
  
  def contribute(self):
    val = self.eval_net(self.vectors[self.mask]).squeeze()

    self.posOK = (self.pos * (val >= 0)).sum()
    self.negOK = (self.neg * (val < 0)).sum()

    contrib = self.criterion(val, self.target)

    if HP.FOCAL_LOSS:
      val_sigmoid = torch.sigmoid(val).clamp(min=1.e-5, max=1. - 1.e-5)
      contrib = -self.pos_weight * self.target * (1. - val_sigmoid)**2 * torch.log(val_sigmoid) - (1. - self.target) * val_sigmoid**2 * torch.log(1. - val_sigmoid)
      # contrib = -self.pos_weight * self.target * (1. - val_sigmoid)**2 * torch.log(val_sigmoid) - (1. - self.target) * torch.log(1. - val_sigmoid)

    self.loss = ((self.pos + self.neg) * contrib).sum()

  def forward(self):
    self.loss = torch.zeros(1).to(self.device)
    self.posOK = torch.zeros(1).to(self.device)
    self.negOK = torch.zeros(1).to(self.device)

    for step in range(len(self.rule_steps)):
      if self.rule_steps[step].item() == 52:
        self.vectors[self.ind_steps[step]] = self.deriv_mlps[str(self.rule_steps[step].item())](self.vectors[self.pars_ind_steps[step]], self.rule_52_limits[step], self.device)
      else:
        self.vectors[self.ind_steps[step]] = self.deriv_mlps[str(self.rule_steps[step].item())](self.vectors[self.pars_ind_steps[step]])

    self.contribute()

    return (self.loss,self.posOK,self.negOK)

def is_generating(rule):
  if rule == 666 or rule == 777:
    return HP.SPLIT_AT_ACTIVATION
  else:
    return rule >= 40 # EVIL: hardcoding the first generating inference in the current deepire3, which is RESOLUTION

def get_ancestors(seed,pars,rules,goods_generating_parents,**kwargs):
  ancestors = kwargs.get("known_ancestors",set())
  # print("Got",len(ancestors))
  todo = [seed]
  while todo:
    cur = todo.pop()
    # print("cur",cur)
    if cur not in ancestors:
      ancestors.add(cur)
      if cur in pars:
        for par in pars[cur]:
          todo.append(par)
          # print("Adding",par,"for",cur,"because of",rules[cur])
        if is_generating(rules[cur]):
          for par in pars[cur]:
            goods_generating_parents.add(par)

  return ancestors

def abstract_initial(features):
  goal = features[-3]
  thax = -1 if goal else features[-2]
  # if HP.USE_SINE:
  #   sine = features[-1]
  # else:
  #   sine = 0
  # return (thax,sine)
  return thax

def abstract_deriv(features):
  rule = features[-1]
  return rule

def load_one(filename, max_size = None):
  print("Loading", filename, flush=True)

  init : List[Tuple[int, Tuple[int, int, int, int, int, int]]] = []
  deriv : List[Tuple[int, Tuple[int, int, int, int, int]]] = []
  pars : Dict[int, List[int]] = {}
  rules: Dict[int, int] = {} # the rule by which id has the mentioned pars
  selec = set()
  
  axioms : Dict[int, str] = {}
  
  empty = None
  good = set()
  
  goods_generating_parents = set()
  
  depths = defaultdict(int)
  max_depth = 0
  
  def update_depths(id,depths,max_depth):
    ps = pars[id]
    depth = max([depths[p] for p in ps])+1
    depths[id] = depth
    if depth > max_depth:
      max_depth = depth

  just_waiting_for_time = False
  time_elapsed = None
  activation_limit_reached = False
  time_limit_reached = False

  with open(filename, 'r') as f:
    for line in f:
      if max_size and len(init)+len(deriv) > max_size:
        return None
      
      # print(line)
      if line.startswith("% Activation limit reached!"):
        just_waiting_for_time = True
        activation_limit_reached = True
        empty = None
      
      if line.startswith("% Time limit reached!"):
        just_waiting_for_time = True
        time_limit_reached = True
        empty = None
    
      if line.startswith("% Refutation found."):
        just_waiting_for_time = True
      
      if line.startswith("% Time elapsed:"):
        time_elapsed = float(line.split()[-2])
      
      if just_waiting_for_time:
        continue
      if line.startswith("% # SZS output start Saturation."):
        print("Skipping. Is SAT.")
        return None
      spl = line.split()
      if spl[0] == "i:":
        val = eval(spl[1])
        assert(val[0] == 1)
        id = val[1]
        init.append((id,abstract_initial(val[2:])))
        
        goal = val[-3]
        
        if len(spl) > 2 and not goal: # axiom name reported and this is not a conjecture clause
          axioms[id] = spl[2]
          
      elif spl[0] == "d:":
        # d: [2,cl_id,age,weight,len,num_splits,rule,par1,par2,...]
        val = eval(spl[1])
        assert(val[0] == 2)
        deriv.append((val[1],abstract_deriv(tuple(val[2:7]))))
        id = val[1]
        pars[id] = val[7:]
        rules[id] = val[6]
        
        update_depths(id,depths,max_depth)
        
      elif spl[0] == "a:":
        # a: [3,cl_id,age,weight,len,causal_parent or -1]
        # treat it as deriv (with one parent):
        val = eval(spl[1])
        assert(val[0] == 3)
        deriv.append((val[1],abstract_deriv((val[2],val[3],val[4],1,666)))) # 1 for num_splits, 666 for rule
        id = val[1]
        pars[id] = [val[-1]]
        rules[id] = 666
      
        update_depths(id,depths,max_depth)
      
      elif spl[0] == "s:":
        selec.add(int(spl[1]))
      elif spl[0] == "r:":
        pass # ingored for now
      elif spl[0] == "e:":
        empty = int(spl[1])
        
        # THIS IS THE INCLUSIVE AVATAR STRATEGY; comment out if you only want those empties that really contributed to the final contradiction
        if HP.AVATAR_EMPTIES == HP.TreatAvatarEmpties_INCLUDEALL:
          good = good | get_ancestors(empty,pars,rules,goods_generating_parents,known_ancestors=good)
        
      elif spl[0] == "f:":
        # fake one more derived clause ("-1") into parents
        empty = -1
        pars[empty] = list(map(int,spl[1].split(",")))
        rules[empty] = 777
        
        update_depths(empty,depths,max_depth)
          
  assert (empty is not None) or activation_limit_reached or time_limit_reached, "Check "+filename

  if time_limit_reached:
    print("Warning: time limit reached for",filename)

  if empty:
    good = good | get_ancestors(empty,pars,rules,goods_generating_parents,known_ancestors=good)
    good = good & selec # proof clauses that were never selected don't count

  if HP.ONLY_GENERATING_PARENTS:
    good_before = len(good)
    print("good before",good)
    good = good & goods_generating_parents
    print("goods_generating_parents",goods_generating_parents)
    print("good after",good)
    print("ONLY_GENERATING_PARENTS reducing goods from",good_before,"to",len(good))

  # TODO: consider learning only from hard problems!
  
  # E.g., solveable by a stupid strategy (age-only), get filtered out
  if not selec:
    print("Skipping, degenerate!")
    return None

  print("init: {}, deriv: {}, select: {}, good: {}, axioms: {}, time: {}".format(len(init),len(deriv),len(selec),len(good),len(axioms),time_elapsed))

  return (("",0.0,len(init)+len(deriv)),(init,deriv,pars,selec,good,axioms)),time_elapsed

def prepare_signature(prob_data_list):
  # sine_sign = set()
  deriv_arits = {}
  axiom_hist = defaultdict(float)

  for (_,probweight,_), (_,deriv,pars,_,_,_,_,axioms) in prob_data_list:
    # for id, (_,sine) in init:
    #   sine_sign.add(sine)

    for id, features in deriv:
      rule = features
      arit = len(pars[id])

      if arit > 2:
        deriv_arits[rule] = 3 # the multi-ary way
      elif rule in deriv_arits and deriv_arits[rule] != arit:
        deriv_arits[rule] = 3 # mixing 1 and 2?
      else:
        deriv_arits[rule] = arit
  
    for id, ax in axioms.items():
      axiom_hist[ax] += probweight

  return (deriv_arits, axiom_hist)

def axiom_names_instead_of_thax(axiom_hist, prob_data_list):
  # (we didn't parse anything than 0 and -1 anyway:)
  # well, actually, in HOL/Sledgehammer we have both thax and user axioms
  # (and we treat all as user axioms (using a modified Vampire)
  
  thax_sign = set()
  ax_idx = dict()
  thax_to_str = dict() 
  good_ax_cnt = 0
  for _, (ax, _) in enumerate(sorted(axiom_hist.items(),key = lambda x : -x[1])):
    good_ax_cnt += 1
    if good_ax_cnt <= HP.MAX_USED_AXIOM_CNT:
      ax_idx[ax] = good_ax_cnt
    else:
      ax_idx[ax] = 0
    thax_to_str[good_ax_cnt] = ax

  for i,(metainfo,(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms)) in enumerate(prob_data_list):
    new_init = []
    for id, thax in init:
      if thax == 0:
        if id in axioms and axioms[id] in ax_idx:
          thax = ax_idx[axioms[id]]
      new_init.append((id,thax))
      thax_sign.add(thax)
    thax_sign.add(0)
    prob_data_list[i] = metainfo,(new_init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms)

  return thax_sign, prob_data_list, thax_to_str

def setup_pos_vals_neg_vals(prob_data):
  (probname, probweight, size), (init, deriv, pars, selec, good, axioms) = prob_data
  print(probname, len(init), len(deriv), len(pars), len(selec), len(good), len(axioms))

  pos_vals = {}
  neg_vals = {}
  tot_pos = 0.0
  tot_neg = 0.0

  # Longer proofs have correspondly less weight per clause (we are fair on the per problem level)
  # one_clause_weigth = 1.0/len(selec)

  # New strategy: Apply the recursive depth of a node as it's weight. So inititals get 1, children of depth max. n get 1/n, ... 

  # depths = {}
  # for id in [x for x, _ in init]:
  #   depths[id] = 1
  # for id in [x for x, _ in deriv]:
  #   depths[id] = max(depths[id2] for id2 in pars[id]) + 1

  for id in selec:
    if id in good:
      pos_vals[id] = 1.0
      tot_pos += 1.0
      # pos_vals[id] = one_clause_weigth
      # tot_pos += one_clause_weigth
    else:
      neg_vals[id] = 1.0
      tot_neg += 1.0
      # neg_vals[id] = one_clause_weigth
      # tot_neg += one_clause_weigth

  return ((probname,probweight,size),(init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,axioms))

def adjust_ids_and_pos_neg_vals(prob_data_list, old2new, pos_vals, neg_vals, num_to_pos_vals, num_to_neg_vals):
  for j in range(len(prob_data_list)):
    print(j, flush=True)
    (probname,probweight,_), (init,deriv,pars,_,_,_,_,_) = prob_data_list[j]
    if not (j in num_to_pos_vals.keys() or j in num_to_neg_vals.keys()):
      print("All pos_vals and neg_vals distributed. Emptying problem", j, "out of", len(prob_data_list), flush=True)
      prob_data_list[j] = (("",1.0,0.0),([],[],{},{},{},0.0,0.0,{}))
    print("Adjusting ids, init, deriv and pars for id", j, "out of", len(prob_data_list), flush=True)
    this_init = [(old2new[j][id], thax) for id, thax in init]
    this_deriv = [(old2new[j][id], rule) for id, rule in deriv]
    these_pars = {old2new[j][id]: [old2new[j][val] for val in vals] for id, vals in pars.items()}
    these_pos_vals = dict()
    these_neg_vals = dict()
    if j in set(num_to_pos_vals.keys()):
      for id in num_to_pos_vals[j]:
        these_pos_vals[id] = pos_vals[id]
    if j in set(num_to_neg_vals.keys()):
      for id in num_to_neg_vals[j]:
        these_neg_vals[id] = neg_vals[id]
    this_tot_pos = sum(these_pos_vals.values())
    this_tot_neg = sum(these_neg_vals.values())

    prob_data_list[j] = ((probname, probweight, len(this_init)+len(this_deriv)), (this_init, this_deriv, these_pars, these_pos_vals, these_neg_vals, this_tot_pos, this_tot_neg, {}))
  return prob_data_list

def reduce_problems(prob_data_list):
  for j in range(len(prob_data_list)):
    a, (init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg, _) = prob_data_list[j]
    print("Reducing problem. Lengths before: {}, {}, {}".format(len(init), len(deriv), len(init) + len(deriv)), flush=True)
    persistent = set(pos_vals.keys()) | set(neg_vals.keys())
    if len(persistent) == 0:
      prob_data_list[j] = (("", 1.0, 0.0),([], [], {}, {}, {}, 0.0, 0.0, {}))
      print("Reduced problem. Lengths after: 0.0, 0.0, 0.0", flush=True)
    else:
      pers_len = len(persistent)
      old_len = pers_len - 1
      while pers_len > old_len:
        persistent = persistent.union(set([y for x in persistent.intersection(set([z for z, _ in deriv])) for y in pars[x]]))
        old_len = pers_len
        pers_len = len(persistent)
      this_init = [(x, y) for x, y in init if x in persistent]
      this_deriv = [(x, y) for x, y in deriv if x in persistent]
      these_pars = {x: y for x, y in pars.items() if x in persistent}

      prob_data_list[j] = (a, (this_init, this_deriv, these_pars, pos_vals, neg_vals, tot_pos, tot_neg, {}))
      print("Reduced problem. Lengths after: {}, {}, {}".format(len(this_init), len(this_deriv), len(this_init) + len(this_deriv)), flush=True)
  return prob_data_list

def distribute_weights(prob_data_list):
  id_dict = {}
  print("Getting ids, their problems and pos/neg vals.", flush=True)
  for num, p in enumerate(prob_data_list):
    this_set = set([x for x, _ in p[1][0]]) | set([x for x, _ in p[1][1]])
    for id in this_set:
      if id not in id_dict:
        id_dict[id] = {}
      if "probs" not in id_dict[id]:
        id_dict[id]["probs"] = set() 
      id_dict[id]["probs"].add(num)
      if id in p[1][3]:
        id_dict[id]["pos"] = p[1][3][id]
      if id in p[1][4]:
        id_dict[id]["neg"] = p[1][4][id]

  print("Setting pos/neg vals in problems to 1. / depth and then normalize them to total of 1. If problem has positive and negative value, negative gets erased (Otherwise code doesn't work well).", flush=True)
  for i, (a, (init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg, ax)) in enumerate(prob_data_list):
    tot_pos = 0.0
    tot_neg = 0.0
    depths = {}
    for id in [x for x, _ in init]:
      depths[id] = 1
    for id in [x for x, _ in deriv]:
      depths[id] = max(depths[id2] for id2 in pars[id]) + 1
    for id in set(id_dict.keys()) & set(depths.keys()):
      if "pos" in id_dict[id]:
        pos_vals[id] = 1. / depths[id]
        tot_pos += 1. / depths[id]
        if "neg" in id_dict[id]:
          del id_dict[id]["neg"]
          if id in neg_vals:
            del neg_vals[id]
      if "neg" in id_dict[id]:
        neg_vals[id] = 1. / depths[id]
        tot_neg += 1. / depths[id]
    factor = 1. / (tot_pos + tot_neg)    
    for id in pos_vals:
      pos_vals[id] *= factor
    for id in neg_vals:
      neg_vals[id] *= factor
    prob_data_list[i] = a, (init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg, ax)

  return prob_data_list

def compress_prob_data_with_fixed_ids(some_probs):
  out_probname = ""
  out_probweight = 0.0
   
  out_init = set()
  out_deriv = set()
  out_pars = {}
  out_pos_vals = {}
  out_neg_vals = {}
  out_tot_pos = 0.0
  out_tot_neg = 0.0

  for ((probname,probweight,_), (init,deriv,pars,pos_vals,neg_vals,tot_pos,tot_neg,_)) in some_probs:
  
    just_file = probname.split("/")[-1]
    out_probname = f"{out_probname}+{just_file}" if out_probname else just_file
    out_probweight += probweight

    out_init.update(init)
    out_deriv.update(deriv)

    out_pars.update((k, v) for k, v in pars.items() if k not in out_pars)

    for k, v in pos_vals.items():
      if v > 0.0:
        out_pos_vals[k] = max(v, out_pos_vals.get(k, 0.0))
    for k, v in neg_vals.items():
      if v > 0.0:
        out_neg_vals[k] = max(v, out_neg_vals.get(k, 0.0))

    out_tot_pos += tot_pos
    out_tot_neg += tot_neg

  print("Compressed to",out_probname,len(out_init)+len(out_deriv),len(out_init),len(out_deriv),len(out_pars),len(pos_vals),len(neg_vals),out_tot_pos,out_tot_neg, flush=True)
  sys.stdout.flush()
  return (out_probname,out_probweight,len(out_init)+len(out_deriv)), (out_init,out_deriv,out_pars,out_pos_vals,out_neg_vals,out_tot_pos,out_tot_neg,{})

def compress_prob_data(some_probs, flag=False):
  id_cnt = 0
  out_probname = ""
  out_probweight = 0.0
  
  abs2new = {} # maps (thax/rule,par_new_ids) to new_id (the structurally hashed one)
  
  new_id_counter_pos = {}
  new_id_counter_neg = {}

  out_init = []
  out_deriv = []
  out_pars = {}
  out_pos_vals = {}
  out_neg_vals = {}
  out_tot_pos = 0.0
  out_tot_neg = 0.0

  out_axioms = {}

  old2new = {} # maps old_id to new_id (this is the not-necessarily-injective map)
  if flag:
    num_to_pos_vals = {}
    num_to_neg_vals = {}
    checklist_pos = set()
    checklist_neg = set()

  for i,((probname, probweight, _), (init, deriv, pars, pos_vals, neg_vals, tot_pos, tot_neg, axioms)) in enumerate(some_probs):
    # reset for evey problem in the list
    old2new[i] = {}
    just_file = probname.split("/")[-1]
    out_probname = f"{out_probname} + {just_file}" if out_probname else just_file
    out_probweight += probweight

    for old_id, features in init:
      if features not in abs2new:
        new_id = id_cnt
        id_cnt += 1
        out_init.append((new_id, features))
        abs2new[features] = new_id
      old2new[i][old_id] = abs2new[features]

    out_axioms.update(axioms)

    for old_id, features in deriv:
      new_pars = [old2new[i][par] for par in pars[old_id]]
      abskey = (features, *new_pars)
      if abskey not in abs2new:
        new_id = id_cnt
        id_cnt += 1
        out_deriv.append((new_id, features))
        out_pars[new_id] = new_pars
        abs2new[abskey] = new_id
      old2new[i][old_id] = abs2new[abskey]

    for k, v in pos_vals.items():
      if v > 0.0:
        new_id = old2new[i][k]
        if new_id not in new_id_counter_pos:
          new_id_counter_pos[new_id] = 1
        else:  
          new_id_counter_pos[new_id] += 1
        out_pos_vals[new_id] = out_pos_vals.get(new_id, 0.0) + v

    for k, v in neg_vals.items():
      if v > 0.0:
        new_id = old2new[i][k]
        if new_id not in new_id_counter_neg:
          new_id_counter_neg[new_id] = 1
        else:  
          new_id_counter_neg[new_id] += 1
        out_neg_vals[new_id] = out_neg_vals.get(new_id, 0.0) + v

    if flag:
      for k, v in pos_vals.items():
        if v > 0.0:
          new_id = old2new[i][k]
          if new_id not in checklist_pos:
            checklist_pos.add(new_id)
            num_to_pos_vals.setdefault(i, set()).add(new_id)
      for k, v in neg_vals.items():
        if v > 0.0:
          new_id = old2new[i][k]
          if new_id not in checklist_neg:
            checklist_neg.add(new_id)
            num_to_neg_vals.setdefault(i, set()).add(new_id)

  # for k in out_pos_vals:
  #   out_pos_vals[k] /= new_id_counter_pos[k]
  # for k in out_neg_vals:
  #   out_neg_vals[k] /= new_id_counter_neg[k]

  out_tot_pos = sum(out_pos_vals.values())
  out_tot_neg = sum(out_pos_vals.values())

  print("Compressed to",out_probname,len(out_init)+len(out_deriv),len(out_init),len(out_deriv),len(out_pars),len(pos_vals),len(neg_vals),out_tot_pos,out_tot_neg)
  result = (out_probname, out_probweight, len(out_init) + len(out_deriv)), (out_init, out_deriv, out_pars, out_pos_vals, out_neg_vals, out_tot_pos, out_tot_neg, out_axioms)

  if flag:
    return result, old2new, out_pos_vals, out_neg_vals, num_to_pos_vals, num_to_neg_vals
  else:
    return result 

import matplotlib.pyplot as plt

def plot_one(filename, times, train_losses, train_posrates, train_negrates, valid_losses, valid_posrates, valid_negrates):
  fig, ax1 = plt.subplots()
  
  color = 'tab:red'
  ax1.set_xlabel('time (epochs)')
  ax1.set_ylabel('loss', color=color)
  tl, = ax1.plot(times, train_losses, "--", linewidth = 1, label = "train_loss", color=color)
  vl, = ax1.plot(times, valid_losses, "-", linewidth = 1,label = "valid_loss", color=color)
  # lr, = ax1.plot(times, rates, ":", linewidth = 1,label = "learning_rate", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  # ax1.set_ylim([0.45,0.6])

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('pos/neg-rate', color=color)  # we already handled the x-label with ax1
  
  tpr, = ax2.plot(times, train_posrates, "--", label = "train_posrate", color = "blue")
  tnr, = ax2.plot(times, train_negrates, "--", label = "train_negrate", color = "cyan")
  vpr, = ax2.plot(times, valid_posrates, "-", label = "valid_posrate", color = "blue")
  vnr, = ax2.plot(times, valid_negrates, "-", label = "valid_negrate", color = "cyan")
  ax2.tick_params(axis='y', labelcolor=color)

  # For pos and neg rates, we know the meaningful range:
  ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [tl,vl,tpr,tnr,vpr,vnr], loc='lower left') # loc = 'best' is rumored to be unpredictable
  
  plt.savefig(filename,dpi=250)
  plt.close(fig)

def plot_with_devs(plotname,models_nums,losses,losses_devs,posrates,posrates_devs,negrates,negrates_devs,clip=None):
  losses = np.array(losses)
  losses_devs = np.array(losses_devs)
  posrates = np.array(posrates)
  posrates_devs = np.array(posrates_devs)
  negrates = np.array(negrates)
  negrates_devs = np.array(negrates_devs)

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('time (epochs)')
  ax1.set_ylabel('loss', color=color)
  vl, = ax1.plot(models_nums, losses, "-", linewidth = 1,label = "loss", color=color)
  ax1.fill_between(models_nums, losses-losses_devs, losses+losses_devs, facecolor=color, alpha=0.5)
  # lr, = ax1.plot(times, rates, ":", linewidth = 1,label = "learning_rate", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  if clip:
    ax1.set_ylim(clip) # [0.0,3.0]

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('pos/neg-rate', color=color)  # we already handled the x-label with ax1

  vpr, = ax2.plot(models_nums, posrates, "-", label = "posrate", color = "blue")
  ax2.fill_between(models_nums, posrates-posrates_devs, posrates+posrates_devs, facecolor="blue", alpha=0.5)
  vnr, = ax2.plot(models_nums, negrates, "-", label = "negrate", color = "cyan")
  ax2.fill_between(models_nums, negrates-negrates_devs, negrates+negrates_devs, facecolor="cyan", alpha=0.5)
  ax2.tick_params(axis='y', labelcolor=color)

  # For pos and neg rates, we know the meaningful range:
  ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  plt.legend(handles = [vl,vpr,vnr], loc='lower left') # loc = 'best' is rumored to be unpredictable

  plt.savefig(plotname,dpi=250)
  plt.close(fig)

def plot_with_devs_just_loss_and_LR(plotname,models_nums,losses,losses_devs,learning_rates,clipLoss=None):
  losses = np.array(losses)
  losses_devs = np.array(losses_devs)
  # learning_rates = 10000*np.array(learning_rates)

  fig, ax1 = plt.subplots(figsize=(3, 3))

  color = 'tab:blue'
  ax1.set_xlabel('time (epochs)')
  # ax1.set_ylabel('learning rate (x 1e-4)', color=color)  # we already handled the x-label with ax1

  # ax1.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useOffset=False)
  # vpr, = ax1.plot(models_nums, learning_rates, "-", label = "learning rate", color = color)
  
  color = 'tab:red'
  ax1.set_ylabel('training loss', color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  vl, = ax1.plot(models_nums, losses, "-", linewidth = 1,label = "training loss", color=color)
  ax1.fill_between(models_nums, losses-losses_devs, losses+losses_devs, facecolor=color, alpha=0.5)
  # lr, = ax1.plot(times, rates, ":", linewidth = 1,label = "learning_rate", color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  # if clipLoss:
  #   ax2.set_ylim(clipLoss) # [0.0,3.0]

  # For pos and neg rates, we know the meaningful range:
  # ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  # plt.legend(handles = [vpr,vl], loc='lower left') # loc = 'best' is rumored to be unpredictable

  plt.savefig(plotname,dpi=250)
  plt.close(fig)

def plot_with_devs_just_loss_and_ATPeval(plotname,models_nums,losses,losses_devs,atp_models,atp_gains,clipLoss=None):
  losses = np.array(losses)
  losses_devs = np.array(losses_devs)
  # learning_rates = np.array(learning_rates)

  fig, ax1 = plt.subplots(figsize=(3.5, 3))

  color = 'tab:red'
  ax1.set_xlabel('time (epochs)')
  # ax1.set_ylabel('loss', color=color)
  vl, = ax1.plot(models_nums, losses, "-", linewidth = 1,label = "validation loss", color=color)
  ax1.fill_between(models_nums, losses-losses_devs, losses+losses_devs, facecolor=color, alpha=0.5)
  # lr, = ax1.plot(times, rates, ":", linewidth = 1,label = "learning_rate", color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  # ax1.yaxis.set_ticklabels([])

  ax1.set_ylabel('validation loss', color=color)

  if clipLoss:
    ax1.set_ylim(clipLoss) # [0.0,3.0]

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:green'
  ax2.set_ylabel('ATP gain', color=color)  # we already handled the x-label with ax1
  ax2.tick_params(axis='y', labelcolor=color)

  vpr, = ax2.plot(atp_models, atp_gains, "-", label = "gained", color = color)


  # For pos and neg rates, we know the meaningful range:
  # ax2.set_ylim([-0.05,1.05])

  fig.tight_layout()  # otherwise the right y-label is slightly clipped

  # plt.legend(handles = [vpr,vl], loc='lower left') # loc = 'best' is rumored to be unpredictable

  plt.savefig(plotname,dpi=250)
  plt.close(fig)
