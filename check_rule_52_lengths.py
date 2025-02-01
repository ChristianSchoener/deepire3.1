import torch
import os
import sys


folder = sys.argv[1]

files = os.listdir(folder+"/pieces/")

for file in files:
  if "greedy" in file:
    data = torch.load(folder + "/pieces/" + file)
    val = max(data[5])
    if val > 30000:
      data2 = torch.load(folder + "/pieces/" + file.split("_")[-1])
      test = sum([1 for _,y in data2[1] if y == 52])
      print(file,val,test)


