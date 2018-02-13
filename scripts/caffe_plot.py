#!/usr/bin/env python
#this scirpt will plot a single run

import sys
import matplotlib.pyplot as plt
import re
import os.path
import string
import numpy as np

if len(sys.argv) < 2:
    sys.exit("How to use: \n%s [input file] " %sys.argv[0])



#directory
data_file = sys.argv[1];
dirname = os.path.dirname(data_file)
print dirname
out_file = dirname
no_filename = False
if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
    out_file += "/out.pdf"
    no_filename = True

print out_file

#read file
with  open(data_file) as f:
    lines = f.read().splitlines()

#regex
