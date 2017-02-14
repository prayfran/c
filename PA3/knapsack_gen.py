import os, sys
import numpy as np
import random
import math

nitems = int(sys.argv[1])
capacity = int(sys.argv[2])
filename = sys.argv[3]

ave = 2*capacity/nitems
w = np.random.normal(ave, ave*0.3, nitems)
np.random.shuffle(w)

#print ave
#print w

for i in range(nitems):
    w[i] = int(round(w[i], 0))
    if w[i] < 0 : 
        w[i] = -w[i]
        #print w[i]

percent = 0.05
seed = percent*ave
coe = 3.0

profit = coe*w + np.random.random_integers(-seed, seed, nitems)

for i in range(nitems):
    if profit[i] < 0:
        #print profit[i]
        profit[i] = -profit[i];
    
#print seed
#print profit

x = open(filename, 'w')
data = str(nitems)+" " + str(capacity) + "\n"
x.write(data)
for i in range(nitems):
    data = str(int(w[i])) + " " + str(int(profit[i])) + "\n"
    x.write(data)
