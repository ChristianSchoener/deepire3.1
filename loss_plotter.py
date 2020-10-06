#!/usr/bin/env python3

import sys
import numpy as np

import inf_common as IC

if __name__ == "__main__":
  # - open the file which multi_inf_parallel.py outputs
  # (see also the redirect in start.sh), this is argv[1],
  # and plot the development of loss into argv[2]

  times = []
  train_losses = []
  train_posrates = []
  train_negrates = []
  valid_losses = []
  valid_posrates = []
  valid_negrates = []
  
  cnt = 0
  
  reading = False
  with open(sys.argv[1],"r") as f:
    for line in f:
      if line.startswith("(Multi)-epoch") and "learning finished at" in line:
        time = int(line.split()[1])
        times.append(time)
        
      if line.startswith("Training stats:"):
        spl = line.split()
        loss = float(spl[2])
        posrate = float(spl[3])
        negrate = float(spl[4])
        
        train_losses.append(loss)
        train_posrates.append(posrate)
        train_negrates.append(negrate)
        
      if line.startswith("Validation stats:"):
        spl = line.split()
        loss = float(spl[2])
        posrate = float(spl[3])
        negrate = float(spl[4])
        
        valid_losses.append(loss)
        valid_posrates.append(posrate)
        valid_negrates.append(negrate)

    for i in range(50):
      IC.plot_one(sys.argv[2],times,train_losses,train_posrates,train_negrates,valid_losses,valid_posrates,valid_negrates)
