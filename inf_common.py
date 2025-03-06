#!/usr/bin/env python3

# a module of concepts common to the inference based model development

# import multiprocessing.spawn
import os

import math

from copy import deepcopy

import time
import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch

import torch.special

torch.set_printoptions(precision=16)
torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float16)

from torch import Tensor

torch.set_num_threads(1)

from typing import Dict, List, Tuple, Optional

import numpy as np

# from copy import deepcopy

# import math

import torch.utils.checkpoint as checkpoint

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

# import itertools
from collections import defaultdict
import sys,random

import hyperparams as HP

# import multiprocessing
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
  
  def __init__(self, dim: int):
    super().__init__()
    
    self.weight = torch.nn.parameter.Parameter(torch.Tensor(dim))
    self.reset_parameters()
  
  def reset_parameters(self):
    torch.nn.init.normal_(self.weight)

  def forward(self) -> Tensor:
    return self.weight
  
# Level vs. number of nodes in tree with all axioms  
# In [60]: for i in range(1, 446, 5):
#     ...:     print(" ".join(f"({(i+k):3}, {depth_count[i+k]:6})" for k in range(5)))
# (  1,  46416) (  2, 240496) (  3, 581784) (  4, 750241) (  5, 889219)
# (  6, 873045) (  7, 866634) (  8, 827017) (  9, 782900) ( 10, 753838)
# ( 11, 716596) ( 12, 685865) ( 13, 646913) ( 14, 599756) ( 15, 537930)
# ( 16, 477778) ( 17, 433063) ( 18, 390170) ( 19, 358508) ( 20, 323497)
# ( 21, 289973) ( 22, 259696) ( 23, 236985) ( 24, 215313) ( 25, 198677)
# ( 26, 180892) ( 27, 163594) ( 28, 151746) ( 29, 141299) ( 30, 129319)
# ( 31, 117906) ( 32, 106164) ( 33,  97284) ( 34,  87746) ( 35,  79146)
# ( 36,  72741) ( 37,  66970) ( 38,  61243) ( 39,  57030) ( 40,  53433)
# ( 41,  50960) ( 42,  48661) ( 43,  44976) ( 44,  42124) ( 45,  40203)
# ( 46,  37175) ( 47,  34459) ( 48,  32247) ( 49,  29450) ( 50,  27449)
# ( 51,  24972) ( 52,  23734) ( 53,  21985) ( 54,  20725) ( 55,  19106)
# ( 56,  17718) ( 57,  16871) ( 58,  15680) ( 59,  15049) ( 60,  14025)
# ( 61,  12765) ( 62,  12326) ( 63,  11349) ( 64,  10129) ( 65,   9103)
# ( 66,   8529) ( 67,   8011) ( 68,   7467) ( 69,   6888) ( 70,   6368)
# ( 71,   6149) ( 72,   5862) ( 73,   5616) ( 74,   5331) ( 75,   4927)
# ( 76,   4572) ( 77,   4484) ( 78,   4408) ( 79,   4184) ( 80,   3952)
# ( 81,   3821) ( 82,   3494) ( 83,   3308) ( 84,   3234) ( 85,   3028)
# ( 86,   2947) ( 87,   2918) ( 88,   2760) ( 89,   2530) ( 90,   2398)
# ( 91,   2464) ( 92,   2471) ( 93,   2340) ( 94,   2278) ( 95,   2254)
# ( 96,   2223) ( 97,   2116) ( 98,   1945) ( 99,   1863) (100,   1870)
# (101,   1929) (102,   1881) (103,   1772) (104,   1640) (105,   1423)
# (106,   1276) (107,   1153) (108,   1019) (109,    879) (110,    797)
# (111,    720) (112,    711) (113,    664) (114,    609) (115,    597)
# (116,    567) (117,    538) (118,    493) (119,    453) (120,    428)
# (121,    406) (122,    374) (123,    358) (124,    362) (125,    327)
# (126,    316) (127,    302) (128,    293) (129,    272) (130,    263)
# (131,    243) (132,    241) (133,    239) (134,    231) (135,    230)
# (136,    218) (137,    221) (138,    212) (139,    206) (140,    210)
# (141,    198) (142,    181) (143,    174) (144,    190) (145,    192)
# (146,    194) (147,    189) (148,    187) (149,    182) (150,    181)
# (151,    184) (152,    179) (153,    168) (154,    188) (155,    171)
# (156,    157) (157,    145) (158,    135) (159,    127) (160,    112)
# (161,    119) (162,    140) (163,    136) (164,    134) (165,    134)
# (166,    140) (167,    136) (168,    153) (169,    154) (170,    148)
# (171,    148) (172,    150) (173,    140) (174,    153) (175,    146)
# (176,    144) (177,    139) (178,    136) (179,    150) (180,    160)
# (181,    154) (182,    156) (183,    148) (184,    137) (185,    115)
# (186,    105) (187,    103) (188,    100) (189,     97) (190,     98)
# (191,     90) (192,     90) (193,     88) (194,     91) (195,     89)
# (196,     91) (197,     84) (198,     78) (199,     76) (200,     75)
# (201,     70) (202,     68) (203,     61) (204,     62) (205,     60)
# (206,     69) (207,     67) (208,     63) (209,     59) (210,     57)
# (211,     58) (212,     54) (213,     55) (214,     55) (215,     52)
# (216,     51) (217,     46) (218,     48) (219,     46) (220,     50)
# (221,     49) (222,     52) (223,     53) (224,     51) (225,     48)
# (226,     45) (227,     43) (228,     43) (229,     44) (230,     43)
# (231,     39) (232,     38) (233,     36) (234,     35) (235,     35)
# (236,     36) (237,     36) (238,     43) (239,     41) (240,     40)
# (241,     36) (242,     34) (243,     34) (244,     42) (245,     41)
# (246,     39) (247,     36) (248,     35) (249,     31) (250,     32)
# (251,     35) (252,     36) (253,     33) (254,     33) (255,     36)
# (256,     29) (257,     29) (258,     30) (259,     26) (260,     23)
# (261,     20) (262,     18) (263,     19) (264,     18) (265,     20)
# (266,     21) (267,     21) (268,     29) (269,     29) (270,     32)
# (271,     29) (272,     29) (273,     27) (274,     28) (275,     27)
# (276,     25) (277,     26) (278,     26) (279,     28) (280,     23)
# (281,     25) (282,     27) (283,     26) (284,     25) (285,     21)
# (286,     21) (287,     21) (288,     18) (289,     17) (290,     17)
# (291,     16) (292,     18) (293,     20) (294,     20) (295,     18)
# (296,     18) (297,     19) (298,     19) (299,     18) (300,     16)
# (301,     16) (302,     20) (303,     18) (304,     16) (305,     18)
# (306,     18) (307,     21) (308,     21) (309,     21) (310,     20)
# (311,     19) (312,     18) (313,     17) (314,     16) (315,     16)
# (316,     17) (317,     13) (318,     13) (319,     10) (320,      8)
# (321,      8) (322,      8) (323,      8) (324,      8) (325,      8)
# (326,      8) (327,      8) (328,      7) (329,      7) (330,      7)
# (331,      7) (332,      8) (333,      8) (334,      8) (335,      6)
# (336,      6) (337,      6) (338,      6) (339,      6) (340,      7)
# (341,      7) (342,      6) (343,      4) (344,      4) (345,      3)
# (346,      2) (347,      2) (348,      2) (349,      2) (350,      2)
# (351,      2) (352,      2) (353,      2) (354,      2) (355,      2)
# (356,      2) (357,      2) (358,      2) (359,      2) (360,      2)
# (361,      2) (362,      2) (363,      2) (364,      2) (365,      2)
# (366,      2) (367,      1) (368,      1) (369,      1) (370,      1)
# (371,      1) (372,      1) (373,      1) (374,      1) (375,      1)
# (376,      1) (377,      1) (378,      1) (379,      1) (380,      1)
# (381,      1) (382,      1) (383,      1) (384,      1) (385,      1)
# (386,      1) (387,      1) (388,      1) (389,      1) (390,      1)
# (391,      1) (392,      1) (393,      1) (394,      1) (395,      1)
# (396,      1) (397,      1) (398,      1) (399,      1) (400,      1)
# (401,      1) (402,      1) (403,      1) (404,      1) (405,      1)
# (406,      1) (407,      1) (408,      1) (409,      1) (410,      1)
# (411,      1) (412,      1) (413,      1) (414,      1) (415,      1)
# (416,      1) (417,      1) (418,      1) (419,      1) (420,      1)
# (421,      1) (422,      1) (423,      1) (424,      1) (425,      1)
# (426,      1) (427,      1) (428,      1) (429,      1) (430,      1)
# (431,      1) (432,      1) (433,      1) (434,      1) (435,      1)
# (436,      1) (437,      1) (438,      1) (439,      1) (440,      1)
# (441,      1) (442,      1) (443,      1) (444,      1) (445,      1)

# 500 revealed axioms, cone:
# In [21]: for i in range(1, 317, 6):
#     ...:     print(" ".join(f"({(i+k):3}, {data["depths"].tolist().count(i+k):6})" for k in range(6)))
# (  1,    502) (  2,   6256) (  3,  47487) (  4, 119011) (  5, 170121) (  6, 147066)
# (  7, 144312) (  8, 124135) (  9, 106812) ( 10,  97566) ( 11,  79712) ( 12,  73767)
# ( 13,  62920) ( 14,  55041) ( 15,  42455) ( 16,  39183) ( 17,  34063) ( 18,  29231)
# ( 19,  27617) ( 20,  23321) ( 21,  20629) ( 22,  19007) ( 23,  16815) ( 24,  15286)
# ( 25,  13676) ( 26,  13095) ( 27,  11839) ( 28,  11599) ( 29,  10662) ( 30,   9497)
# ( 31,   8777) ( 32,   8146) ( 33,   7342) ( 34,   6636) ( 35,   6104) ( 36,   5861)
# ( 37,   5181) ( 38,   4835) ( 39,   4443) ( 40,   4320) ( 41,   3803) ( 42,   3988)
# ( 43,   3233) ( 44,   3201) ( 45,   3048) ( 46,   2636) ( 47,   2403) ( 48,   2229)
# ( 49,   1964) ( 50,   1971) ( 51,   1627) ( 52,   1760) ( 53,   1643) ( 54,   1524)
# ( 55,   1382) ( 56,   1276) ( 57,   1445) ( 58,   1180) ( 59,   1147) ( 60,   1084)
# ( 61,    915) ( 62,    889) ( 63,    831) ( 64,    743) ( 65,    793) ( 66,    757)
# ( 67,    676) ( 68,    597) ( 69,    589) ( 70,    502) ( 71,    509) ( 72,    478)
# ( 73,    448) ( 74,    498) ( 75,    353) ( 76,    389) ( 77,    385) ( 78,    373)
# ( 79,    297) ( 80,    303) ( 81,    320) ( 82,    283) ( 83,    273) ( 84,    323)
# ( 85,    218) ( 86,    264) ( 87,    218) ( 88,    213) ( 89,    166) ( 90,    166)
# ( 91,    213) ( 92,    192) ( 93,    216) ( 94,    173) ( 95,    146) ( 96,    198)
# ( 97,    139) ( 98,    162) ( 99,    181) (100,    163) (101,    169) (102,    166)
# (103,    118) (104,    104) (105,     87) (106,     93) (107,     86) (108,     90)
# (109,     78) (110,     76) (111,     88) (112,     73) (113,     80) (114,     81)
# (115,     59) (116,     67) (117,     48) (118,     53) (119,     65) (120,     51)
# (121,     51) (122,     58) (123,     55) (124,     60) (125,     58) (126,     53)
# (127,     54) (128,     52) (129,     53) (130,     48) (131,     47) (132,     45)
# (133,     42) (134,     42) (135,     42) (136,     42) (137,     45) (138,     43)
# (139,     42) (140,     44) (141,     37) (142,     46) (143,     39) (144,     40)
# (145,     36) (146,     34) (147,     35) (148,     39) (149,     34) (150,     31)
# (151,     30) (152,     31) (153,     29) (154,     27) (155,     27) (156,     21)
# (157,     20) (158,     20) (159,     19) (160,     19) (161,     18) (162,     22)
# (163,     19) (164,     18) (165,     19) (166,     19) (167,     17) (168,     21)
# (169,     18) (170,     15) (171,     21) (172,     22) (173,     18) (174,     27)
# (175,     19) (176,     19) (177,     18) (178,     17) (179,     17) (180,     21)
# (181,     16) (182,     15) (183,     15) (184,     15) (185,     17) (186,     16)
# (187,     15) (188,     14) (189,     13) (190,     13) (191,     13) (192,     15)
# (193,     14) (194,     13) (195,     13) (196,     15) (197,     11) (198,     11)
# (199,     11) (200,     11) (201,     11) (202,     11) (203,     11) (204,     11)
# (205,     11) (206,     16) (207,     11) (208,     11) (209,     12) (210,     11)
# (211,     11) (212,     11) (213,     13) (214,      9) (215,      9) (216,      8)
# (217,     10) (218,      9) (219,      7) (220,      7) (221,      7) (222,      6)
# (223,      6) (224,      7) (225,      6) (226,      6) (227,      6) (228,      6)
# (229,      6) (230,      5) (231,      5) (232,      5) (233,      5) (234,      5)
# (235,      6) (236,      6) (237,      7) (238,     10) (239,      6) (240,      6)
# (241,      6) (242,      6) (243,      6) (244,      6) (245,      6) (246,      5)
# (247,      5) (248,      5) (249,      8) (250,      7) (251,      5) (252,      5)
# (253,      5) (254,      5) (255,      8) (256,      5) (257,      5) (258,      5)
# (259,      5) (260,      5) (261,      5) (262,      5) (263,      6) (264,      6)
# (265,      7) (266,      6) (267,      6) (268,      5) (269,      5) (270,      5)
# (271,      5) (272,      5) (273,      5) (274,      5) (275,      5) (276,      5)
# (277,      6) (278,      5) (279,      5) (280,      5) (281,      7) (282,      7)
# (283,      5) (284,      5) (285,      5) (286,      5) (287,      5) (288,      4)
# (289,      4) (290,      4) (291,      5) (292,      4) (293,      4) (294,      4)
# (295,      4) (296,      4) (297,      4) (298,      4) (299,      3) (300,      3)
# (301,      3) (302,      3) (303,      3) (304,      2) (305,      3) (306,      2)
# (307,      3) (308,      2) (309,      2) (310,      2) (311,      2) (312,      2)
# (313,      3) (314,      4) (315,      2) (316,      2) (317,      0) (318,      0)

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

  def forward(self, args : Tensor) -> Tensor:
    x = args
    if self.arit == 2:
      x = x.view(x.shape[0] // 2, -1)
    x = self.prolog(x)
    x = self.first(x)
    x = self.nonlin(x)
    x = self.second(x)
    return self.epilog(x)
  
class CatAndNonLinearMultiary(torch.nn.Module):
  def __init__(self, dim: int, arit: int):
    super().__init__()
  
    self.dim = dim

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

  def forward_impl_list(self, x : Tensor) -> Tensor:
    x = self.prolog(x)
    x = self.first(x)
    x = self.nonlin(x)
    x = self.second(x)
    return self.epilog(x)
    
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

  #   full_sized = torch.empty(2*length - 1, HP.EMBEDDING_SIZE).to(device)

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

#   def forward(self, args: Tensor, limits: Tensor, device: str) -> Tensor:
#     limits = limits.to(device)
#     lengths = torch.diff(limits)
#     the_len = lengths.numel()

#     full_lengths = 2*lengths - 1
#     start_inds = torch.zeros(the_len, dtype=torch.int32).to(device)
#     end_inds = 2 * (lengths // 2)
#     fill_start_inds = deepcopy(lengths)
#     fill_end_inds = fill_start_inds + (lengths // 2)
#     HP.EMBEDDING_SIZE = HP.EMBEDDING_FACTOR * (HP.EMBEDDING_BASE**HP.EMBEDDING_MAX_SCALE)
#     return_mat = torch.zeros(the_len, HP.EMBEDDING_SIZE).to(device)
# # Looping to prevent big temporary matrix
#     for i in range(the_len):
#       full_sized = torch.zeros(full_lengths[i], HP.EMBEDDING_SIZE).to(device)
#       select_range = torch.arange(limits[i], limits[i+1])
#       fill_range = torch.arange(0, lengths[i])
#       full_sized[fill_range] = args[select_range]
#       while lengths[i] > 1:
#         select_range = torch.arange(start_inds[i], end_inds[i])
#         fill_range = torch.arange(fill_start_inds[i], fill_end_inds[i])
#         full_sized[fill_range] = self.forward_impl_list(full_sized[select_range].view(lengths[i] // 2, -1))

#         lengths[i] = (lengths[i] + 1) // 2
#         start_inds[i] = deepcopy(end_inds[i])
#         end_inds[i] = start_inds[i] + 2 * (lengths[i] // 2)
#         fill_start_inds[i] = start_inds[i] + lengths[i]
#         fill_end_inds[i] = fill_start_inds[i] + (lengths[i] // 2)

#       return_mat[i] = full_sized[-1]
#     return return_mat
    
  def forward(self, args : Tensor, limits : Tensor, device : str) -> Tensor:
    limits = limits.to(device)
    lengths = torch.diff(limits)
    the_len = lengths.numel()

    full_lengths = 2*lengths - 1
    start_inds = torch.cat((torch.tensor([0], dtype=torch.int32).to(device), full_lengths[:-1].cumsum(dim=0)))
    end_inds = start_inds + 2 * (lengths // 2)
    fill_start_inds = start_inds + lengths
    fill_end_inds = fill_start_inds + (lengths // 2)

    full_sized = torch.zeros(full_lengths.sum(), args.size(1)).to(device)

    for i in range(the_len):
      full_sized[torch.arange(start_inds[i], start_inds[i] + lengths[i])] = args[torch.arange(limits[i], limits[i+1])]

    while max(lengths) > 1:
      mask = torch.zeros(full_lengths.sum(), dtype=torch.bool).to(device)
      fill_mask = torch.zeros_like(mask).to(device)

      for i in range(the_len):
        mask[start_inds[i]:end_inds[i]] = True
        fill_mask[fill_start_inds[i]:fill_end_inds[i]] = True

      how_much = mask.sum().item()
      full_sized[fill_mask] = self.forward_impl_list(full_sized[mask].view(how_much // 2, -1))

      lengths = (lengths + 1) // 2
      start_inds = end_inds
      end_inds = start_inds + 2 * (lengths // 2)
      fill_start_inds = start_inds + lengths
      fill_end_inds = fill_start_inds + (lengths // 2)
     
    mask = torch.zeros(full_lengths.sum(), dtype=torch.bool).to(device)
    mask[start_inds] = True
    return full_sized[mask]

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
    init_embeds[str(i)] = Embed(HP.EMBEDDING_SIZE).to(device)

  deriv_mlps = torch.nn.ModuleDict().to(device)
  for rule, arit in deriv_arits.items():
    if arit <= 2:
      deriv_mlps[str(rule)] = CatAndNonLinearBinary(HP.EMBEDDING_SIZE, arit).to(device)
    else:
      assert(arit == 3)
      deriv_mlps[str(rule)] = CatAndNonLinearMultiary(HP.EMBEDDING_SIZE, arit).to(device)

  eval_net = torch.nn.Sequential(
    torch.nn.Dropout(HP.DROPOUT) if HP.DROPOUT > 0.0 else torch.nn.Identity(HP.EMBEDDING_SIZE),
    torch.nn.Linear(HP.EMBEDDING_SIZE, HP.EMBEDDING_SIZE * HP.BOTTLENECK_EXPANSION_RATIO // 2),
    torch.nn.Tanh() if HP.NONLIN == HP.NonLinKind_TANH else torch.nn.ReLU(),
    torch.nn.Linear(HP.EMBEDDING_SIZE,1)).to(device)

  return torch.nn.ModuleList([init_embeds, deriv_mlps, eval_net])
  
def name_initial_model_suffix():
  return "_{}_{}_BER{}_LayerNorm{}_Dropout{}{}.pt".format(
    HP.EMBEDDING_SIZE,
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

bigpart1_zero = '''#!/usr/bin/env python3

import torch
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys,random

def save_net(name, init_abstractions, deriv_abstractions_keys_rule, deriv_abstractions_keys_first_par, deriv_abstractions_keys_second_par, deriv_abstractions_values, good_vals, neg_vals, max_id):
  
  # This is, how we envision inference:
  class InfRecNet(torch.nn.Module):
    init_abstractions : Dict[str, int]
    deriv_abstractions_keys_rule : Tensor
    deriv_abstractions_keys_first_par : Tensor
    deriv_abstractions_keys_second_par : Tensor
    deriv_abstractions_values : Tensor
    good_vals : Tensor
    neg_vals : Tensor
    eval_store : Dict[int, float]
    abs_ids : Dict[int, int]
    max_id : int
        
    def __init__(self,
          init_abstractions : Dict[str, int],
          deriv_abstractions_keys_rule : Tensor,
          deriv_abstractions_keys_first_par : Tensor,
          deriv_abstractions_keys_second_par : Tensor,
          deriv_abstractions_values : Tensor,
          good_vals : Tensor,          
          neg_vals : Tensor,
          max_id : int):
      super().__init__()

      self.init_abstractions = init_abstractions
      self.deriv_abstractions_keys_rule = deriv_abstractions_keys_rule
      self.deriv_abstractions_keys_first_par = deriv_abstractions_keys_first_par
      self.deriv_abstractions_keys_second_par = deriv_abstractions_keys_second_par
      self.deriv_abstractions_values = deriv_abstractions_values
      self.abs_ids = {}
      self.good_vals = good_vals
      self.neg_vals = neg_vals
      self.max_id = max_id
      self.eval_store = {}'''

bigpart_no_longer_rec1_zero = '''
    @torch.jit.export
    def forward(self, id: int) -> float:
      return -10.0
      # abs_id = self.abs_ids[id] # must have been mentioned already
      # ind_good = torch.searchsorted(self.good_vals, abs_id)
      # if ind_good < self.good_vals.numel():
      #   return 1.0
      # else:
      #   ind_neg = torch.searchsorted(self.neg_vals, abs_id)
      #   if ind_neg < self.neg_vals.numel():
      #     return 0.0
      #   else:
      #     return 0.9

    @torch.jit.export
    def new_init(self, id: int, features : Tuple[int, int, int, int, int, int], name: str) -> None:
      # an init record is abstracted just by the name str
      abskey = name
      # if abskey not in self.init_abstractions:
      #   abs_id = -(len(self.init_abstractions)+1) # using negative values for abstractions of init clauses
      #   self.init_abstractions[abskey] = abs_id
      # else:
      #   abs_id = self.init_abstractions[abskey]

      # # assumes this is called exactly once
      # self.abs_ids[id] = abs_id
      '''

bigpart_rec2_zero='''
    @torch.jit.export
    def new_deriv{}(self, id: int, features : Tuple[int, int, int, int, int], pars : List[int]) -> None:
      rule = features[-1]
      # if len(pars) == 1:
      #   abskey = torch.tensor([rule, self.abs_ids[pars[0]], self.abs_ids[pars[0]]], dtype=torch.int32)
      # else:
      #   abskey = torch.tensor([rule, self.abs_ids[pars[0]], self.abs_ids[pars[1]]], dtype=torch.int32)
      # found = True
      # rule_ind_min = torch.searchsorted(self.deriv_abstractions_keys_rule, abskey[0])
      # rule_ind_max = torch.searchsorted(self.deriv_abstractions_keys_rule, abskey[0], side="right")
      # first_par_min = torch.tensor(0, dtype=torch.int32)
      # first_par_max = torch.tensor(0, dtype=torch.int32)
      # second_par_min = torch.tensor(0, dtype=torch.int32)
      # second_par_max = torch.tensor(0, dtype=torch.int32)
      # if rule_ind_max == rule_ind_min:
      #   found = False
      # else:
      #   first_par_min = torch.searchsorted(self.deriv_abstractions_keys_first_par[rule_ind_min:rule_ind_max], abskey[1]) + rule_ind_min
      #   first_par_max = torch.searchsorted(self.deriv_abstractions_keys_first_par[rule_ind_min:rule_ind_max], abskey[1], side="right") + rule_ind_min
      #   if first_par_max == first_par_min:
      #     found = False
      #   else:
      #     second_par_min = torch.searchsorted(self.deriv_abstractions_keys_second_par[first_par_min:first_par_max], abskey[2]) + first_par_min
      #     second_par_max = torch.searchsorted(self.deriv_abstractions_keys_second_par[first_par_min:first_par_max], abskey[2], side="right") + first_par_min
      #     if second_par_max == second_par_min:
      #       found = False
  
      # if found:
      #   abs_id = self.deriv_abstractions_values[second_par_min].item()
      # else:
      #   abs_id = self.max_id
      #   self.max_id = self.max_id + 1
      
      # # assumes this is called exactly once
      # self.abs_ids[id] = abs_id
      '''

bigpart_rec2_rule_52_zero='''
    @torch.jit.export
    def new_deriv52(self, id: int, features : Tuple[int, int, int, int, int], pars : List[int]) -> None:
      1
      # abs_id = self.max_id
      # self.max_id = self.max_id + 1
           
      # # assumes this is called exactly once
      # self.abs_ids[id] = abs_id
      '''

bigpart_avat_zero = '''
    @torch.jit.export
    def new_avat(self, id: int, features : Tuple[int, int, int, int]) -> None:
      par = features[-1]
      # abskey = torch.tensor([666, self.abs_ids[par], self.abs_ids[par]], dtype=torch.int32)

      # found = True
      # rule_ind_min = torch.searchsorted(self.deriv_abstractions_keys_rule, abskey[0])
      # rule_ind_max = torch.searchsorted(self.deriv_abstractions_keys_rule, abskey[0], side="right")
      # first_par_min = torch.tensor(0, dtype=torch.int32)
      # first_par_max = torch.tensor(0, dtype=torch.int32)
      # second_par_min =torch.tensor(0, dtype=torch.int32)
      # second_par_max =torch.tensor(0, dtype=torch.int32)
      # if rule_ind_max == rule_ind_min:
      #   found = False
      # else:
      #   first_par_min = torch.searchsorted(self.deriv_abstractions_keys_first_par[rule_ind_min:rule_ind_max], abskey[1]) + rule_ind_min
      #   first_par_max = torch.searchsorted(self.deriv_abstractions_keys_first_par[rule_ind_min:rule_ind_max], abskey[1], side="right") + rule_ind_min
      #   if first_par_max == first_par_min:
      #     found = False
      #   else:
      #     second_par_min = torch.searchsorted(self.deriv_abstractions_keys_second_par[first_par_min:first_par_max], abskey[2]) + first_par_min
      #     second_par_max = torch.searchsorted(self.deriv_abstractions_keys_second_par[first_par_min:first_par_max], abskey[2], side="right") + first_par_min
      #     if second_par_max == second_par_min:
      #       found = False
  
      # if found:
      #   abs_id = self.deriv_abstractions_values[second_par_min].item()
      # else:
      #   abs_id = self.max_id
      #   self.max_id = self.max_id + 1
      
      # # assumes this is called exactly once
      # self.abs_ids[id] = abs_id
      '''

bigpart3_zero = '''
  module = InfRecNet(
    init_abstractions,
    deriv_abstractions_keys_rule,
    deriv_abstractions_keys_first_par,
    deriv_abstractions_keys_second_par,
    deriv_abstractions_values,
    good_vals,
    neg_vals,
    max_id)
  script = torch.jit.script(module)
  script.save(name)'''

def create_saver_zero(deriv_arits):
  with open("inf_saver_zero.py", "w") as f:

    print(bigpart1_zero, file=f)

    print(bigpart_no_longer_rec1_zero, file=f)

    for rule in sorted(deriv_arits):
      if rule not in [52, 666]: # avatar done differently in bigpart3, rul_52, too
        print(bigpart_rec2_zero.format(str(rule), str(rule)), file=f)

    if 666 in deriv_arits:
      print(bigpart_avat_zero, file=f)

    if 52 in deriv_arits:
      print(bigpart_rec2_rule_52_zero, file=f)

    print(bigpart3_zero, file=f)

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
      data, use_cuda = False):
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
    for i in range(len(self.rule_steps)):
      self.ind_steps[i] = self.ind_steps[i].to(self.device)
      self.pars_ind_steps[i] = self.pars_ind_steps[i].to(self.device)
    self.vectors = torch.zeros(len(self.ids), HP.EMBEDDING_SIZE).to(self.device)
    self.vectors[:len(self.thax)] = torch.stack([init_embeds[str(this_thax.item())]() for this_thax in self.thax])

    self.vals = torch.zeros(len(self.ids)).to(self.device)

    self.deriv_mlps = deriv_mlps
    self.eval_net = eval_net

    self.pos = data["pos"].to(self.device)
    self.neg = data["neg"].to(self.device)

    self.target = data["target"].to(self.device)

    self.mask = data["mask"].to(self.device)

    self.tot_neg = data["tot_neg"].to(self.device)
    self.tot_pos = data["tot_pos"].to(self.device)
    self.pos_weight = (HP.POS_WEIGHT_EXTRA * self.tot_neg / self.tot_pos if self.tot_pos > 0 else torch.tensor(1.0)).to(self.device)

  def contribute(self):
    self.vals = self.eval_net(self.vectors[self.mask]).squeeze()

    self.posOK = (self.pos[self.mask] * (self.vals >= 0.0)).sum()
    self.negOK = (self.neg[self.mask] * (self.vals < 0.0)).sum()

    val_sigmoid = torch.clamp(torch.special.expit(self.vals), min=1.e-7, max=1.-1.e-7)
    # val_sigmoid_pos = torch.clamp(val_sigmoid, max=0.5) + 0.5 - 1.e-7
    # val_sigmoid_neg = torch.clamp(val_sigmoid, min=0.5) - 0.5 + 1.e-7

    # val_sigmoid = torch.where(self.pos > 0, val_sigmoid_pos, val_sigmoid)
    # val_sigmoid = torch.where(self.neg > 0, val_sigmoid_neg, val_sigmoid)
    contrib = -self.pos_weight * self.target[self.mask] * torch.log(val_sigmoid) - (1. - self.target[self.mask]) * torch.log(1. - val_sigmoid)

    self.loss = ((self.pos[self.mask] + self.neg[self.mask]) * contrib).sum()

  def forward(self):
    self.loss = torch.zeros(1).to(self.device)
    self.posOK = torch.zeros(1).to(self.device)
    self.negOK = torch.zeros(1).to(self.device)

    for step in range(len(self.rule_steps)):
      self.vectors[self.ind_steps[step]] = self.deriv_mlps[str(self.rule_steps[step].item())](self.vectors[self.pars_ind_steps[step]])

    self.contribute()

    return (self.loss, self.posOK, self.negOK)

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
  thax_sign = set()
  # sine_sign = set()
  deriv_arits = {}
  axiom_hist = defaultdict(float)

  for (_,probweight,_), (init,deriv,pars,_,_,axioms) in prob_data_list:
    for id, thax in init:
      thax_sign.add(thax)
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

  return (thax_sign, deriv_arits, axiom_hist)
    
def axiom_names_instead_of_thax(thax_sign, axiom_hist, prob_data_list):
  # (we didn't parse anything than 0 and -1 anyway:)
  # well, actually, in HOL/Sledgehammer we have both thax and user axioms
  # (and we treat all as user axioms (using a modified Vampire)
  
  ax_idx = dict()
  thax_to_str = dict() 
  good_ax_cnt = 0
  for _, (ax, _) in enumerate(sorted(axiom_hist.items(),key = lambda x : -x[1])):
    good_ax_cnt += 1
    ax_idx[ax] = good_ax_cnt
    thax_to_str[good_ax_cnt] = ax

  for i,(metainfo,(init,deriv,pars,selec,good,axioms)) in enumerate(prob_data_list):
    new_init = []
    for id, thax in init:
      if thax == 0:
        if id in axioms and axioms[id] in ax_idx:
          thax = ax_idx[axioms[id]]
      new_init.append((id,thax))
      thax_sign.add(thax)
    thax_sign.add(0)
    prob_data_list[i] = metainfo,(new_init,deriv,pars,selec,good,axioms)

  return thax_sign, prob_data_list, thax_to_str

def get_subtree(start, match, pars):
  persistent = deepcopy(start)
  pers_len = len(persistent)
  old_len = pers_len - 1
  matches = {z for z, _ in match}
  while pers_len > old_len:
    persistent.update({y for x in persistent & matches for y in pars[x]})
    old_len = pers_len
    pers_len = len(persistent)
  return persistent

def set_zero(prob_data_list, thax_to_str):
  thax_sign = set()
  thax_to_str_out = {}
  for i, ((probname, probweight, size), (init, deriv, pars, selec, good)) in enumerate(prob_data_list):
    if HP.THAX_SOURCE == HP.ThaxSource_AXIOM_NAMES:
      new_init = []
      for id, thax in init:
        if thax > HP.MAX_USED_AXIOM_CNT:
          thax = 0
        else:
          if thax in thax_to_str:
            thax_to_str_out[thax] = thax_to_str[thax]
        
        thax_sign.add(thax)
        
        new_init.append((id, thax))
    else:
      new_init = init

    prob_data_list[i] = ((probname, probweight, size), (new_init, deriv, pars, selec, good))
  return prob_data_list, thax_sign, thax_to_str_out

def adjust_ids_and_crop(prob, old2new, global_selec):
  (probname, probweight, _), (init, deriv, pars, selec, good) = prob
  this_init = [(old2new[id], thax) for id, thax in init]
  this_deriv = [(old2new[id], rule) for id, rule in deriv]
  these_pars = {old2new[id]: [old2new[val] for val in vals] for id, vals in pars.items() if id in old2new}

  these_ids = {x for x, _ in this_init} | {x for x, _ in this_deriv}
  persistent = get_subtree(these_ids & global_selec, this_deriv, these_pars)
  this_init = [(id, thax) for id, thax in this_init if id in persistent]
  this_deriv = [(id, rule) for id, rule in this_deriv if id in persistent]
  these_pars = {id: vals for id, vals in these_pars.items() if id in persistent}
  print("Reduced. Lengths before / after: {} / {}".format(len(init) + len(deriv), len(this_init) + len(this_deriv)), flush=True)
  return ((probname, probweight, len(this_init)+len(this_deriv)), (this_init, this_deriv, these_pars, selec & persistent, good & persistent))

def crop(prob):
  (probname, probweight, _), (init, deriv, pars, selec, good) = prob
  these_ids = {x for x, _ in init} | {x for x, _ in deriv}
  persistent = get_subtree(these_ids & selec, deriv, pars)
  this_init = [(id, thax) for id, thax in init if id in persistent]
  this_deriv = [(id, rule) for id, rule in deriv if id in persistent]
  these_pars = {id: vals for id, vals in pars.items() if id in persistent}
  print("Reduced. Lengths before / after: {} / {}".format(len(init) + len(deriv), len(this_init) + len(this_deriv)), flush=True)
  return [((probname, probweight, len(this_init)+len(this_deriv)), (this_init, this_deriv, these_pars, selec & persistent, good & persistent))]

def compress_prob_data_with_fixed_ids(some_probs):
  out_probname = ""
  out_probweight = 0.0
   
  out_init = []
  out_deriv = []
  out_pars = {}
  out_selec = set()
  out_good = set()

  for ((probname, probweight, _), (init, deriv, pars, selec, good)) in some_probs:
  
    just_file = probname.split("/")[-1]
    out_probname = f"{out_probname} + {just_file}" if out_probname else just_file
    out_probweight += probweight

    out_init_set = set(out_init)
    out_init.extend([x for x in init if x not in out_init_set])
    out_deriv_set = set(out_deriv)
    out_deriv.extend([x for x in deriv if x not in out_deriv_set])
    out_pars.update(pars)
    out_selec.update(selec)
    out_good.update(good)

  print("Compressed to", out_probname, len(out_init) + len(out_deriv), len(out_init), len(out_deriv), len(out_pars), flush=True)
  sys.stdout.flush()
  return (out_probname, out_probweight, len(out_init) + len(out_deriv)), (out_init, out_deriv, out_pars, out_selec, out_good)

def compress_prob_data(some_probs, flag=False):
  id_cnt = 0
  out_probname = ""
  out_probweight = 0.0
  
  abs2new = {} # maps (thax/rule,par_new_ids) to new_id (the structurally hashed one)
  
  out_init = []
  out_deriv = []
  out_pars = {}
  out_selec = set()
  out_good = set()

  old2new = {} # maps old_id to new_id (this is the not-necessarily-injective map)

  for i, ((probname, probweight, _), specs) in enumerate(some_probs):
    init, deriv, pars, selec, good = specs

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

    for old_id in selec:
      out_selec.add(old2new[i][old_id])
    for old_id in good:
      out_good.add(old2new[i][old_id])

  print("Compressed to", out_probname, len(out_init) + len(out_deriv), len(out_init), len(out_deriv), len(out_pars), len(out_selec), len(out_good))
  result = (out_probname, out_probweight, len(out_init) + len(out_deriv)), (out_init, out_deriv, out_pars, out_selec, out_good)

  if flag:
    return result, old2new, out_selec, out_good
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
