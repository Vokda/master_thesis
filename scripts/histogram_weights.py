import sys
import matplotlib.pyplot as plt
import math

if len(sys.argv) < 1:
    sys.exit("usage: %s [*.weights]" %sys.argv[0])

data_file = sys.argv[1]
data=[]
with open(data_file) as f:
    line = next(f)
    for x in line.split():
        data+=[float(x)]

#print data

n, bins, patches = plt.hist(data, bins=math.sqrt(len(data)))

plt.show()
