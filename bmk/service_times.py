import subprocess
import csv
import statistics as stat
import math
from io import StringIO

filename = "service_times2.csv"

d = {}
with open(filename, "r") as f:
  reader = csv.DictReader(f)
  for row in reader:
    d = row

start = 257
N = 512 
step = 1
blocks = 1

M = 5

confidence = 0.9

if '0' in d.keys():
  init_cycles = float(eval(d['0'])[0])
else:
  init_cycles = 0

for i in range(start, N+1, step):
  if i != 0:
    cmd = f"ncu --csv --metrics gpu__cycles_elapsed.sum atomic_bmk {blocks} {i} 1"
  else:
    cmd = f"ncu --csv --metrics gpu__cycles_elapsed.sum atomic_bmk {blocks} 1 0"
  s = [] 
  for _ in range(M):
    output = subprocess.run(cmd.split(), capture_output=True).stdout.decode().splitlines()
    count = 0
    while len(output[count]) < 2 or output[count][:4] != "\"ID\"":
      count+=1

    output = output[count:]
    output = "\n".join(output)
  
    reader = csv.DictReader(StringIO(output))
    cycles = None
    for row in reader:
      if row['Metric Name'] == "gpu__cycles_elapsed.sum":
        cycles = row['Metric Value'].replace(",","")
    if i != 0:
      s += [(float(cycles.strip())-init_cycles)/(i*blocks)]
    else:
      s += [float(cycles.strip())]

  m = stat.mean(s)
  if i == 0:
    init_cycles = m
  sigma = stat.stdev(s)
  r = confidence*sigma/math.sqrt(M)
  d[str(i)] = str((m, (m-r,m+r))) 

with open(filename, "w") as f:
  writer = csv.DictWriter(f, fieldnames=d.keys())
  writer.writeheader()
  writer.writerow(d)


