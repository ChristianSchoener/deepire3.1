import os
import re
from collections import defaultdict

directory = "/home/chris/Dokumente/Github/Projektarbeit_Vampire/results/"

e51_dir = "classes_testing_e51/"
old_dir = "l0_500_test_e23/"
base_dir = "base/"
new_dir = "classes_testing_SWAPOUT_0.5_e69/"
newest_dir = "Focal_Loss/"
other_dir = "classes_testing_model_more_than_50_SWAPOUT_0.25_e72/"
the_new_dir = "zero-test/"

files_e51 = os.listdir(directory+e51_dir)
files_old = os.listdir(directory+old_dir)
files_base = os.listdir(directory+base_dir)
files_new = os.listdir(directory+new_dir)
files_newest = os.listdir(directory+newest_dir)
files_other = os.listdir(directory+other_dir)

files_the_new = os.listdir(directory+the_new_dir)

files_e51 = [x for x in files_e51 if ".log" in x]
files_old = [x for x in files_old if ".log" in x]
files_base = [x for x in files_base if ".log" in x]
files_new = [x for x in files_new if ".log" in x]
files_newest = [x for x in files_newest if ".log" in x]
files_other = [x for x in files_other if ".log" in x]
files_the_new = [x for x in files_the_new if ".log" in x]

files = [x for x in files_the_new if ("small_np_" + x.split("__")[0] + "_" + x in files_new) and ("small_np_" + x.split("__")[0] + "_" + x in files_base) and ("small_np_" + x.split("__")[0] + "_" +  x in files_other) and ("small_np_" + x.split("__")[0] + "_" +  x in files_newest) and ("small_np_" + x.split("__")[0] + "_" + x in files_e51) and ("small_np_" + x.split("__")[0] + "_" + x in files_old)]
print(len(files))

e51_finds = defaultdict(lambda:set())
old_finds = defaultdict(lambda:set())
base_finds = defaultdict(lambda:set())
new_finds = defaultdict(lambda:set())
newest_finds = defaultdict(lambda:set())
other_finds = defaultdict(lambda:set())
# matrix_128_finds = defaultdict(lambda:set())
# macro_e14_finds = defaultdict(lambda:set())
the_new_finds = defaultdict(lambda:set())
for file in files:
  with open(directory+e51_dir+"small_np_" + file.split("__")[0] + "_" + file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      e51_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      e51_finds["time"].add(file)
    elif "memory" in line.lower():
      e51_finds["memory"].add(file)
  with open(directory+old_dir+"small_np_" + file.split("__")[0] + "_" + file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      old_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      old_finds["time"].add(file)
    elif "memory" in line.lower():
      old_finds["memory"].add(file)
  with open(directory+base_dir+"small_np_" + file.split("__")[0] + "_" + file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      base_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      base_finds["time"].add(file)
    elif "memory" in line.lower():
      base_finds["memory"].add(file)
  with open(directory+new_dir+"small_np_" + file.split("__")[0] + "_" + file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      new_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      new_finds["time"].add(file)
    elif "memory" in line.lower():
      new_finds["memory"].add(file)
  with open(directory+newest_dir+"small_np_" + file.split("__")[0] + "_" + file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      newest_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      newest_finds["time"].add(file)
    elif "memory" in line.lower():
      newest_finds["memory"].add(file)
  with open(directory+other_dir+"small_np_" + file.split("__")[0] + "_" + file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      other_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      other_finds["time"].add(file)
    elif "memory" in line.lower():
      other_finds["memory"].add(file)
  with open(directory+the_new_dir+file,"r") as f:
    line = f.readline()
    if "refutation found" in line.lower():
      the_new_finds["refutation"].add(file)
    elif "time limit reached" in line.lower():
      the_new_finds["time"].add(file)
    elif "memory" in line.lower():
      the_new_finds["memory"].add(file)

print("base",len(base_finds["refutation"]))
print("old",len(old_finds["refutation"]))
print("e51",len(e51_finds["refutation"]))
print("new",len(new_finds["refutation"]))
print("newest",len(newest_finds["refutation"]))
print("other",len(other_finds["refutation"]))
# print("matrix_128",len(matrix_128_finds["refutation"]))
# print("macro_e14",len(macro_e14_finds["refutation"]))
print("the_new",len(the_new_finds["refutation"]))
print("base+old",len(base_finds["refutation"].union(old_finds["refutation"])))
print("base+old+e51",len(base_finds["refutation"].union(old_finds["refutation"]).union(e51_finds["refutation"])))
print("base+old+e51+new",len(base_finds["refutation"].union(old_finds["refutation"]).union(e51_finds["refutation"].union(new_finds["refutation"]))))
print("base+old+e51+new+newest",len(base_finds["refutation"].union(old_finds["refutation"]).union(e51_finds["refutation"]).union(new_finds["refutation"]).union(newest_finds["refutation"])))
print("base+old+e51+new+newest+other",len(base_finds["refutation"].union(old_finds["refutation"]).union(e51_finds["refutation"]).union(new_finds["refutation"]).union(newest_finds["refutation"]).union(other_finds["refutation"])))
print("base+old+e51+new+newest+other+the_new",len(base_finds["refutation"].union(old_finds["refutation"]).union(e51_finds["refutation"]).union(new_finds["refutation"]).union(newest_finds["refutation"]).union(other_finds["refutation"]).union(the_new_finds["refutation"])))
print("the_new-base",len(the_new_finds["refutation"].difference(base_finds["refutation"])))
print("base-the_new",len(base_finds["refutation"].difference(the_new_finds["refutation"])))
print("the_new-old",len(the_new_finds["refutation"].difference(old_finds["refutation"])))
print("the_new-e51",len(the_new_finds["refutation"].difference(e51_finds["refutation"])))
print("the_new-new",len(the_new_finds["refutation"].difference(new_finds["refutation"])))
print("the_new-other",len(the_new_finds["refutation"].difference(other_finds["refutation"])))
print("the_new-newest",len(the_new_finds["refutation"].difference(newest_finds["refutation"])))
# print("matrix_128-base",len(matrix_128_finds["refutation"].difference(base_finds["refutation"])))
# print("matrix_128-old",len(matrix_128_finds["refutation"].difference(old_finds["refutation"])))
# print("matrix_128-e51",len(matrix_128_finds["refutation"].difference(e51_finds["refutation"])))
# print("matrix_128-new",len(matrix_128_finds["refutation"].difference(new_finds["refutation"])))
# print("matrix_128-newest",len(matrix_128_finds["refutation"].difference(newest_finds["refutation"])))
# print("matrix_128-other",len(matrix_128_finds["refutation"].difference(other_finds["refutation"])))
# print("old-matrix_128",len(old_finds["refutation"].difference(matrix_128_finds["refutation"])))

