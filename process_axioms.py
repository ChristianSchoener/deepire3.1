import torch
import os
import subprocess

from concurrent.futures import ThreadPoolExecutor

def process_folder(folder):
  """Process a single folder and return a dictionary of file names and their processed contents."""
  folder_data = {}
  if os.path.isdir(folder):
    for file in os.listdir(folder):
      file_path = os.path.join(folder, file)
      command = f'sed "/%\\|conjecture/d" {file_path} | sed "s/fof(//" | sed "s/, axiom.*//"'
      p = subprocess.Popen(command, shell=True, universal_newlines=True, stdout=subprocess.PIPE).stdout.read()
      pp = set(p.split("\n")[:-1])
      folder_data[file] = list(pp)
  return folder_data

def get_key_problemName_val_axiomSet(baseFolder):
  """Parallelizes folder processing using ThreadPoolExecutor."""
  my_list = {}
  folders = [os.path.join(baseFolder, folder) for folder in os.listdir(baseFolder) if os.path.isdir(os.path.join(baseFolder, folder))]

  with ThreadPoolExecutor() as executor:
    results = executor.map(process_folder, folders)

  # Combine results from all threads
  for folder_data in results:
    my_list.update(folder_data)

  return my_list

def get_key_axiomName_val_count(data):
  my_list = dict()
  counter = 0
  print(len(data))
  for key in data:
    print(counter/len(data))
    counter += 1
    for name in data[key]:
      if name not in my_list:
        my_list[name] = 1
      else:
        my_list[name] += 1
  return my_list

def get_sortedList_axiomName_count_ranking(data):
  sort = list(sorted(data.items(), key=lambda x:-x[1]))

  for i in range(len(sort)):
    sort[i] = (sort[i][0],sort[i][1],i)

  my_list = dict()
  for i in range(len(sort)):
    my_list[sort[i][0]] = {"counts": sort[i][1], "number": sort[i][2]}
  return my_list

def get_key_problemName_val_axiomNumberSet(data,count):
  counter = 0
  print(len(data))
  for key in data:
    print(counter/len(data))
    counter += 1
    this_set = set()
    for name in data[key]:
      this_set.add(count[name]["number"])
    data[key] = this_set
  return data

def problem2Number_and_number2Problem(data):
  problem2Number = dict()
  number2Problem = dict()
  counter = 0
  for key in data:
    problem2Number[key] = counter
    number2Problem[counter] = key
    counter += 1
  return problem2Number,number2Problem

def get_key_axiomNumber_val_problemNumberSet(data,count,problem2Number):
  my_list = dict()
  for key in count.keys():
    my_list[count[key]["number"]] = set()
  counter = 0
  print(len(data))
  for key in data:
    print(counter/len(data))
    counter += 1
    for name in data[key]:
      my_list[name].add(problem2Number[key])
  return my_list

def axiom2Number_and_number2Axiom(data):
  axiom2Number = dict()
  number2Axiom = dict()
  for key in data:
    axiom2Number[key] = data[key]["number"]
    number2Axiom[data[key]["number"]] = key
  return axiom2Number,number2Axiom

if __name__ == "__main__":
  key_problemName_val_axiomSet = get_key_problemName_val_axiomSet("/home/chris/Dokumente/Github/Projektarbeit_Vampire/small_np/")
  # torch.save(key_problemName_val_axiomSet, "key_problemName_val_axiomSet.pt")

  key_axiomName_val_count = get_key_axiomName_val_count(key_problemName_val_axiomSet)
  # torch.save(key_axiomName_val_count, "key_axiomName_val_count.pt")

  sortedList_axiomName_count_ranking = get_sortedList_axiomName_count_ranking(key_axiomName_val_count)
  # torch.save(sortedList_axiomName_count_ranking, "sortedList_axiomName_count_ranking.pt")

  axiom2Number,number2Axiom = axiom2Number_and_number2Axiom(sortedList_axiomName_count_ranking)
  torch.save((axiom2Number,number2Axiom),"axiom2Number_number2Axiom.pt")

  # key_problemName_val_axiomSet = torch.load("key_problemName_val_axiomSet.pt")

  key_problemName_val_axiomNumberSet = get_key_problemName_val_axiomNumberSet(key_problemName_val_axiomSet, sortedList_axiomName_count_ranking)
  torch.save(key_problemName_val_axiomNumberSet, "key_problemName_val_axiomNumberSet.pt")

  # problem2Number, number2Problem = problem2Number_and_number2Problem(data)
  # torch.save((problem2Number,number2Problem),"problem2Number_number2Problem")
  # my_list = torch.load("sortedList_axiomName_count_ranking")

  # my_list = get_key_axiomNumber_val_problemNumberSet(data,my_list,problem2Number)
  # torch.save(my_list,"key_axiomNumber_val_problemNumberSet")
  # data=torch.load("sortedList_axiomName_count_ranking")

