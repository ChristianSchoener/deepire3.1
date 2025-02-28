#!/usr/bin/env python3

import torch

import os

import inf_common as IC

import hyperparams as HP

if __name__ == "__main__":

  thax_sign, deriv_arits, thax_to_str = torch.load(HP.EXP_DATA_SIGN_PREPARED, weights_only=False)
  print("Loaded signature from", HP.EXP_DATA_SIGN_PREPARED)

  IC.create_saver(deriv_arits)
  import inf_saver as IS

  _, parts = torch.load(HP.EXP_CHECKPOINT_FILE, weights_only=False)
  with torch.no_grad():
    parts.to("cpu")

  print("Loaded model from", HP.EXP_CHECKPOINT_FILE)

  IS.save_net(HP.EXP_NAME, parts, thax_to_str)

  print("Exported to", HP.EXP_NAME)