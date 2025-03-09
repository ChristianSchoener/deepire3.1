import torch
import inf_common as IC
import inf_common_original as IC_original
import subprocess
from copy import deepcopy
import time
torch.set_printoptions(precision=16)

print("Don't forget to set Dropout = 0.0 in hyperparams.py.")

# command1 = "python3 log_loader.py"
# command2 = "python3 compressor.py mode=pre"
# command3 = "python3 compressor.py mode=compress"
# print("NEW CODE: Loading Log Files", flush=True)
# subprocess.Popen(command1,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
# print("NEW CODE: Pre-Compression", flush=True)
# subprocess.Popen(command2,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
# print("NEW CODE: Compression", flush=True)
# subprocess.Popen(command3,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()

# command_original1 = "python3 log_loader_original.py"
# command_original2 = "python3 compressor_original.py"
# print("ORIGINAL CODE: Loading Log Files", flush=True)
# subprocess.Popen(command_original1,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
# print("ORIGINAL CODE: Compression", flush=True)
# subprocess.Popen(command_original2,shell=True,universal_newlines=True, stdout=subprocess.PIPE).stdout.read()

print("NEW CODE: Loading Problem Instance", flush=True)
data_new = torch.load("pieces/piece0.pt.train",weights_only=False)
print("ORIGINAL CODE: Loading Problem Instance", flush=True)
data_original = torch.load("pieces/piece0.pt",weights_only=False)

print("Absolute Difference tot_pos |NEW - ORIGINAL|:", (abs(data_new["tot_pos"] - data_original[5])).item(), flush=True)
print("Absolute Difference tot_neg |NEW - ORIGINAL|:", (abs(data_new["tot_neg"] - data_original[6])).item(), flush=True)
print("Relative Difference tot_pos |NEW - ORIGINAL| / ORIGINAL:", (abs(data_new["tot_pos"] - data_original[5]) / data_original[5]).item(), flush=True)
print("Relative Difference tot_neg |NEW - ORIGINAL| / ORIGINAL:", (abs(data_new["tot_neg"] - data_original[6]) / data_original[6]).item(), flush=True)

print("Loading data signature", flush=True)
thax_sign, sine_sign, deriv_arits, thax_to_str = torch.load("data_sign.pt", weights_only=False)

print("NEW CODE: Generating Initial Model", flush=True)
master_parts_new = IC.get_initial_model(thax_sign, deriv_arits)
print("ORIGINAL CODE: Generating Initial Model", flush=True)
master_parts_original = IC_original.get_initial_model(thax_sign, {0}, deriv_arits)

for part in master_parts_new:
  part.to("cpu")
for part in master_parts_original:
  part.to("cpu")

print("Copying Model parameters from NEW Initial Model to ORIGINAL Initial Model", flush=True)
with torch.no_grad():
  for i in range(len(master_parts_new)):
    master_parts_original[i+ int(i>0)].load_state_dict(master_parts_new[i].state_dict())

print("NEW CODE: Generating Learning Model on CPU", flush=True)
model_new = IC.LearningModel(*master_parts_new, data_new, False)
model_new.eval()
print("NEW CODE: Performing Forward Pass of Learning Model on Problem Instance on CPU", flush=True)
loss_new, posOK_new, negOK_new = model_new()
print("NEW: loss: {}, posOK: {}, negOK: {}".format(loss_new.item(), posOK_new.item(), negOK_new.item()), flush = True)

print("ORIGINAL CODE: Generating Learning Model", flush=True)
model_original = IC_original.LearningModel(*master_parts_original, *data_original)
model_original.eval()
print("ORIGINAL CODE: Performing Forward Pass of Learning Model on Problem Instance on CPU", flush=True)
loss_original, posOK_original, negOK_original = model_original()
print("ORIGINAL: loss: {}, posOK: {}, negOK: {}".format(loss_original.item(), posOK_original, negOK_original), flush = True)

print("Absolute error |NEW - ORIGINAL|: loss: {}, posOK: {}, negOK: {}".format(abs(loss_new.item() - loss_original.item()), abs(posOK_new.item() - posOK_original), abs(negOK_new.item() - negOK_original)), flush=True)

if loss_original.item() > 0 and posOK_original > 0 and negOK_original > 0:
  print("Relative error |(NEW - ORIGINAL) / ORIGINAL|: loss: {}, posOK: {}, negOK: {}".format(abs((loss_new.item() - loss_original.item()) / (loss_original.item())), abs((posOK_new.item() - posOK_original) / (posOK_original)), abs((negOK_new.item() - negOK_original)/ (negOK_original))), flush=True)

print("Performing Comparison of Time Duration for Steps on CPU.", flush=True)
print("NEW CODE:", flush=True)
master_parts_new_ = deepcopy(master_parts_new)

start_time = time.time()
for _ in range(10):
    model_new = IC.LearningModel(*master_parts_new_, data_new, False)
    model_new.train()
    optimizer = torch.optim.Adam(master_parts_new_.parameters())
    optimizer.zero_grad()
    loss_new, posOK_new, negOK_new = model_new()
    loss_new.backward()
    optimizer.step()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"NEW CODE: Average time for Learning Model Generation, Frward + Backward Pass: {elapsed_time / 10:.6f} sec")

print("ORIGINAL CODE:", flush=True)
master_parts_original_ = deepcopy(master_parts_original)

start_time = time.time()

for _ in range(10):
    model_original = IC_original.LearningModel(*master_parts_original_, *data_original, False)
    model_original.train()
    optimizer = torch.optim.Adam(master_parts_original_.parameters())
    optimizer.zero_grad()
    loss_original, posOK_original, negOK_original = model_original()
    loss_original.backward()
    optimizer.step()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"ORIGINAL CODE: Average time for Learning Model Generation, Frward + Backward Pass: {elapsed_time / 10:.6f} sec")

print("Performing Comparison of Time Duration for Steps on GPU.", flush=True)
for part in master_parts_new:
  part.to("cuda")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()  # Ensure everything is ready
start_event.record()

for _ in range(10):
    model_new2 = IC.LearningModel(*master_parts_new, data_new, True)
    optimizer = torch.optim.Adam(master_parts_new.parameters())
    optimizer.zero_grad()
    loss_new2, posOK_new2, negOK_new2 = model_new2()
    loss_new2.backward()
    optimizer.step()

end_event.record()
torch.cuda.synchronize()  # Ensure computation is finished

elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
print(f"NEW CODE, GPU: Average time for Learning Model Generation, Frward + Backward Pass: {elapsed_time / 10:.3f} ms")