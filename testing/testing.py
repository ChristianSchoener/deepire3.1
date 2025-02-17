import torch
import sys
sys.path.append("..")
import inf_common as IC
import inf_common_old as IC_old
from copy import deepcopy
torch.set_printoptions(precision=16)

data = torch.load("depth_below_2.pt",weights_only=False)

data_greedy = torch.load("depth_below_2_greedy.pt",weights_only=False)

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

model = IC.LearningModel(*master_parts, data_greedy, 0, False)
model.train()
loss_greedy, posOK_greedy, negOK_greedy = model()
print("greedy: loss: {}, posOK: {}, negOK: {}".format(loss_greedy.item(), posOK_greedy.item(), negOK_greedy.item()), flush = True)

model_old = IC_old.LearningModel(*master_parts_old, *data[1])
model_old.train()
loss_old, posOK_old, negOK_old = model_old()
print("old: loss: {}, posOK: {}, negOK: {}".format(loss_old.item(), posOK_old, negOK_old), flush = True)

print("Abs. error |greedy - old|: loss: {}, posOK: {}, negOK: {}".format(abs(loss_greedy.item() - loss_old.item()), abs(posOK_greedy.item() - posOK_old), abs(negOK_greedy.item() - negOK_old)), flush=True)

print("Rel. error |(greedy - old) / old|: loss: {}, posOK: {}, negOK: {}".format(abs((loss_greedy.item() - loss_old.item()) / (loss_old.item())), abs((posOK_greedy.item() - posOK_old) / (posOK_old)), abs((negOK_greedy.item() - negOK_old)/ (negOK_old))), flush=True)
