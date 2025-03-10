#!/usr/bin/env python3

import torch

import os

import inf_common as IC

import hyperparams as HP

def dotheassertions():
  assert hasattr(HP, "EXP_DATA_SIGN_PREPARED"), "Parameter EXP_DATA_SIGN_PREPARED in hyperparams.py not set. This file contains the data signature, taking into account the number of revealed axioms."
  assert isinstance(HP.EXP_DATA_SIGN_PREPARED, str), "Parameter EXP_DATA_SIGN_PREPARED in hyperparams.py not a string. This file contains the data signature, taking into account the number of revealed axioms."
  assert os.path.isfile(HP.EXP_DATA_SIGN_PREPARED), "Parameter EXP_DATA_SIGN_PREPARED in hyperparams.py does not point to a file. This file contains the data signature, taking into account the number of revealed axioms."

  assert hasattr(HP, "EXP_CHECKPOINT_FILE"), "Parameter EXP_CHECKPOINT_FILE in hyperparams.py not set. This file contains the checkpoint data which shall be used for the scripted model."
  assert isinstance(HP.EXP_CHECKPOINT_FILE, str), "Parameter EXP_CHECKPOINT_FILE in hyperparams.py is not a string. This file contains the checkpoint data which shall be used for the scripted model."
  assert os.path.isfile(HP.EXP_CHECKPOINT_FILE), "Parameter EXP_CHECKPOINT_FILE in hyperparams.py does not point to a file. This file contains the checkpoint data which shall be used for the scripted model."

  assert hasattr(HP, "EXP_NAME"), "Parameter EXP_NAME in hyperparams.py not set. The scripted model is exported to this filename."
  assert isinstance(HP.EXP_NAME, str), "Parameter EXP_NAME in hyperparams.py is not a string. The scripted model is exported to this filename."

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