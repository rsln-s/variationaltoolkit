#!/usr/bin/env python

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import csv
import numpy as np
from collections import defaultdict
from operator import itemgetter

avgs = defaultdict(list)

with open('benchmark_mri.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        nqubits = int(row[0])
        nthreads = int(row[1])
        t = float(row[2])
        avgs[(nqubits,nthreads)].append(t)
        
toft_t = defaultdict(list) # time as a function of threads, tuples

for k, v in avgs.items():
    toft_t[k[0]].append((k[1], np.mean(v)))

toft = defaultdict(list) # time as a function of threads, tuples
for k, v in toft_t.items():
    toft[k] = [x[1] for x in sorted(v, key=itemgetter(0))]
 
for k, v in toft.items():
    plt.plot(range(1, len(v) + 1), v, '.-', label=2*k)
    # NOTE: changed `range(1, 4)` to mach actual values count
plt.legend()  # To draw legend
plt.ylabel('Time, sec')
plt.xlabel('Number of threads')
plt.show()

#header = ['nqubits', 'nthreads', 'runtime (sec)']
#with open('benchmark_mri_means.csv', 'w') as f:
#    out = csv.writer(f)
#    out.writerow(header)
#    for k, v in avgs.items():
#        out.writerow([k[0],k[1], np.mean(v)])
