#!/usr/bin/env python
#this scirpt will plot a single run

import sys
import matplotlib.pyplot as plt
import re
import os.path
import string
import numpy as np

if len(sys.argv) < 2:
    sys.exit("How to use: \n%s network_directory/" %sys.argv[0])

#directory
network_directory = sys.argv[1];

#read file
with  open(network_directory+"/out.txt") as f:
    lines = f.read().splitlines()

#regex
float_pattern = re.compile('\d+(\.\d+)?')

#cpu
cpu_times = []
#openmp
openmp_times = []
opencl_times = []
cuda_times = []

#get int [float(s) for  s in line.split() if s.isdigit()]

for line in lines:
    if "cpu copy time: " in line:
        cpu_times += [float(float_pattern.search(line).group())]
    elif "openmp copy time: " in line:
        openmp_times += [float(float_pattern.search(line).group())]
    elif "opencl copy time: " in line:
        opencl_times += [float(float_pattern.search(line).group())]
    elif "cuda copy time: " in line:
        cuda_times += [float(float_pattern.search(line).group())]

#print "cpu times: " + str(cpu_times)
#print "openmp times:" + str(openmp_times)
#print "opencl times:" + str(opencl_times)
#print "cuda times:" + str(cuda_times)

times = []

times += [np.mean(cpu_times)]
#print times
times += [np.mean(openmp_times)]
times += [np.mean(opencl_times)]
times += [np.mean(cuda_times)]

stds = []

stds += [np.std(cpu_times)]
stds += [np.std(openmp_times)]
stds += [np.std(opencl_times)]
stds += [np.std(cuda_times)]

nr_bars = 4;

index = np.arange(nr_bars)
width = 0.35

#global variables
y_label = "milliseconds"
title = "Time taken to reorder data from batch to single image."
plt.title(title)
#print times
#print nr_bars
plt.bar(np.arange(nr_bars), times, width, yerr=stds, color='red')
plt.xticks(np.arange(nr_bars), ("cpu", "openmp", "opencl", "cuda"))
plt.show()
