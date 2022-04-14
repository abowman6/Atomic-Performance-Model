import subprocess
import re
import os
import csv
from io import StringIO
import matplotlib.pyplot as plt

NCU = "/usr/local/cuda/bin/ncu"

wd = "/home/abowman6/bfs/ggc/gen-unoptimized/bfs-wl"
binary = "test"

events = ["gpu__cycles_elapsed.sum","sm__warps_launched.sum"]
metrics = ["l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,sm__warps_launched.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"]
metrics += ["l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"]

max_service_time = 0

def run(binary, bin_args, profiler="ncu", use_file=False, random=False):
  args = f"--print-kernel-base mangled --csv --metrics {','.join(metrics+events)} "
  args += binary + " " + bin_args
  if not use_file:
    args = args.split()
    output = subprocess.run([profiler]+args, capture_output=True)
    output = output.stdout.decode().splitlines()
  else:
    os.system(f"{profiler} {args} > __prof_tmp")
    with open("./__prof_tmp", "r") as f:
      output = f.readlines()
    os.system("rm -f __prof_tmp")

  #print(output)
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

  name_map = {}
  service_times = {}

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
    if not name in name_map.keys():
      name_map[name] = []
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

    #print(requests,transactions,gpu_cycles,warps) 
    
    if transactions*requests*gpu_cycles*warps != 0:
      n = str(int(transactions/requests*warps/random_scale))
      if int(n) < int(max_service_time):
        tup = eval(service_times[n])[0]
      else:
        tup = 1 
      tup *= transactions/random_scale
      #print(tup)
      tup = (gpu_cycles, tup, tup/gpu_cycles)
      name_map[name].append(tup)
      transactions = 0 
      requests = 0 
      gpu_cycles = 0
      warps = 0 

  return name_map

if __name__ == "__main__":
  #print(run("bmk/atomic_S", "1 32 1"))
  output = run(wd+"/"+binary,"~/bfs/inputs/random/r4-2e23.gr", use_file=True, profiler=NCU)
  #output = run("bmk/atomic_random", "640 256 1", random=True)
  #output = run("bmk/atomic_bmk", "640 256 1", random=False)
  #output = run("/home/abowman6/d/CBET_RayTracing_3D/cbet-gpu", "28", profiler=NCU)
  print(output)

