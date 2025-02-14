import torch
import inf_common as IC
import inf_common_old as IC_old
from copy import deepcopy
torch.set_printoptions(precision=16)

data_10000 = torch.load("data_10000.pt",weights_only=False)

piece0 = torch.load("pieces/piece0.pt.train",weights_only=False)

thax_sign, deriv_arits, thax_to_str = torch.load("data_sign.pt", weights_only=False)

master_parts = IC.get_initial_model(thax_sign, deriv_arits)
master_parts_old = IC_old.get_initial_model(thax_sign, {0}, deriv_arits)

for part in master_parts:
  part.to("cpu")
for part in master_parts_old:
  part.to("cpu")

with torch.no_grad():
  for i in range(len(master_parts)):
    master_parts_old[i+ int(i>0)].load_state_dict(master_parts[i].state_dict())

model = IC.LearningModel(*master_parts, piece0, 0, False)
model.train()
loss_greedy, posOK_greedy, negOK_greedy = model()
print("greedy:", loss_greedy, posOK_greedy, negOK_greedy, flush = True)

model_old = IC_old.LearningModel(*master_parts_old, *data_10000[0][1][:-1])
model_old.train()
loss_old, posOK_old, negOK_old = model_old()
print("old:", loss_old, posOK_old, negOK_old, flush = True)
