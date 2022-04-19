import subprocess
import re
import os
import csv
from io import StringIO
import argparse

NCU = "/usr/local/cuda/bin/ncu"

wd = "/home/abowman6/bfs/ggc/gen-best/bfs-wl"
binary = "test"

events = ["gpu__cycles_elapsed.sum","sm__warps_launched.sum"]
metrics = ["l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,sm__warps_launched.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"]
metrics += ["l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"]

max_service_time = 0

def run(cmd, profiler="ncu", use_file=False, random=False):
  args = f"--print-kernel-base mangled --csv --metrics {','.join(metrics+events)} "
  args += cmd 
  if not use_file:
    args = args.split()
    output = subprocess.run([profiler]+args, capture_output=True)
    output = output.stdout.decode().splitlines()
  else:
    os.system(f"{profiler} {args} > __prof_tmp")
    with open("./__prof_tmp", "r") as f:
      output = f.readlines()
    os.system("rm -f __prof_tmp")

  count = 0
  while len(output[count]) < 2 or output[count][:4] != "\"ID\"":
    count+=1
  output = output[count:]
  output = "\n".join(output)

  reader = csv.DictReader(StringIO(output))

  gpu_cycles = 0
  transactions = 0
  requests = 0
  warps = 0

  kernel_name = None

  name_map = []
  service_times = {}
  count_map = {}

  if random:
    random_scale = 64
  else:
    random_scale = 1

  with open("service_times.csv") as f:
    s_reader = csv.DictReader(f)
    for row in s_reader:
      service_times = row
    max_service_time = list(service_times.keys())[-1]

  for row in reader:
    name = row['Kernel Name']
    if not name in count_map.keys():
      count_map[name] = 0
    if row['Metric Name'] == 'gpu__cycles_elapsed.sum':
      gpu_cycles += int(row['Metric Value'].replace(",",""))
    if row['Metric Name'] == 'l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum':
      transactions += int(row['Metric Value'].replace(",", ""))
    if row['Metric Name'] == 'l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum':
      requests += int(row['Metric Value'].replace(",",""))
    if row['Metric Name'] == 'l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum':
      transactions += int(row['Metric Value'].replace(",", ""))
    if row['Metric Name'] == 'l1tex__t_requests_pipe_lsu_mem_global_op_red.sum':
      requests += int(row['Metric Value'].replace(",",""))
    if row['Metric Name'] == 'sm__warps_launched.sum':
      warps += int(row['Metric Value'].replace(",",""))

    if transactions*requests*gpu_cycles*warps != 0:
      n = str(int(transactions/requests*warps/random_scale))
      if int(n) < int(max_service_time):
        tup = eval(service_times[n])[0]
      else:
        tup = 1
      total_time = tup*transactions/random_scale
      tup = f"{name}, {count_map[name]}, {gpu_cycles}, {total_time}, %0.2f, {n}, {tup}, {transactions}"%(total_time/gpu_cycles*100)
      name_map.append(tup)
      transactions = 0 
      requests = 0 
      gpu_cycles = 0
      warps = 0 
      count_map[name] += 1

  return name_map

head = "Name, Invocation, Total Cycles, Predicted time on atomics, % on atomics, atomics in flight, service time, transactions"

if __name__ == "__main__":
  p = argparse.ArgumentParser("python3 model.py")
  p.add_argument("cmd", help="GPU program to run with arguments", nargs="+")
  p.add_argument("--use_file", dest="file", help="CUDA runs into issues with being used in a subprocess, if this is an issue set this argument to True")
  p.add_argument("--random", dest="random", help="Use random address model instead of single address model")
  args = p.parse_args()
  output = run(" ".join(args.cmd), profiler=NCU, random=args.random, use_file=args.file)
  print("\n".join(output))

