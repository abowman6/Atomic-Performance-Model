import subprocess
import re
import os
import csv
from io import StringIO
import matplotlib.pyplot as plt

wd = "./bmk"
binary = "./atomic_bmk"
profiler = "ncu"

events = ["gpu__cycles_elapsed.sum","sm__warps_launched.sum"]
metrics = ["l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum,l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,sm__warps_launched.sum,l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"]

def run(binary, bin_args, profiler="ncu"):
  args = f"--csv --metrics {','.join(metrics+events)} "
  args += binary + " " + bin_args
  args = args.split()
  output = subprocess.run([profiler]+args, capture_output=True)
  output = output.stdout.decode().splitlines()

  count = 0
  for line in output:
    if line[:3] == "\"ID":
      break
    count += 1

  output = "\n".join(output[count:])

  reader = csv.DictReader(StringIO(output))

  gpu_cycles = 0
  transactions = 0
  requests = 0
  warps = 0

  count = 1

  for row in reader:
    count = max(count, int(row['ID'])+1)
    if row['Metric Name'] == 'gpu__cycles_elapsed.sum':
      gpu_cycles += int(row['Metric Value'].replace(",",""))
    if row['Metric Name'] == 'l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum':
      transactions += int(row['Metric Value'].replace(",", ""))
    if row['Metric Name'] == 'l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum':
      requests += int(row['Metric Value'].replace(",",""))
    if row['Metric Name'] == 'sm__warps_launched.sum':
      warps += int(row['Metric Value'].replace(",",""))

  gpu_cycles /= count
  transactions /= count
  requests /count

  service_times = {}

  with open("service_times.csv") as f:
    s_reader = csv.DictReader(f)
    for row in s_reader:
      service_times = row

  n = str(int(transactions/requests*warps))
  tup = eval(service_times[n])
  return (tup,float(tup[0])*(requests/warps))


if __name__ == "__main__":
  os.chdir(wd)
  """
  X = list(range(1,257))
  Y1 = []
  Y2 = []
  Y3 = []
  for i in range(1,257):
    c = run(binary, f"1 {i} 1")
    Y1.append(c[1])
    Y2.append(c[0][1][0])
    Y3.append(c[0][1][1])

  #print(Y1)
  T1 = [f"({X[i]},{Y1[i]})" for i in range(0,len(X),4)]
  print("".join(T1))
  T2 = [f"({X[i]},{Y2[i]})" for i in range(0,len(X),4)]
  print("".join(T2))
  T3 = [f"({X[i]},{Y3[i]})" for i in range(0,len(X),4)]
  print("".join(T3))
  """
  print(run(binary, "1 256 1"))
  #plt.plot(X,Y1)
  #plt.plot(X,Y2)
  #plt.plot(X,Y3)
  #plt.show()


