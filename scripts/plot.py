#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import sys

nlines = int(sys.argv[1])
npoints = int(sys.argv[2])
nruns = int(sys.argv[3])

labels = sys.argv[4:]

for i in range(nlines):
    x = []
    y = []
    stdev = []

    for j in range(npoints):
        x.append(int(input()) * 16*16)

        ytmp = []
        for k in range(nruns):
            ytmp.append(float(input()))

        ytmp = np.array(ytmp)
        y.append(np.mean(ytmp))
        stdev.append(np.std(ytmp))

    c = 'C'+str(i)
    #x = np.log2(x)
    logy = np.log2(y)
    m,b,r,p,s = linregress(x,y)
    #b = 2**b

    plt.loglog(x,y, 'o-', c=c, basex=2, basey=10, label=labels[i])
    #plt.plot(x,m*np.array(x) + b, '-', c=c)

    print(m, "*x +", b)
    print(r**2)

#plt.yscale('log')
plt.xlabel("N Points ($log_2$ scale)")
plt.ylabel("Time (sec, $log_{10}$ scale)")
plt.legend()
plt.grid(b=True, which='major', linestyle='-')
plt.grid(b=True, which='minor', linestyle=':')
plt.show()

