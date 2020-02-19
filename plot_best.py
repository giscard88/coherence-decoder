import os
import random
import shutil
import time
import warnings
import sys
import os


import copy


import json
import argparse
import numpy as np
from collections import defaultdict
import pylab



parser = argparse.ArgumentParser(description='learn the patterns of coherence among EEG electrodes')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed for pytorch and numpy (default: 10)')

parser.add_argument('--duration', type=int, default=500,
                    help='duration of time series of interest (default: 4000)')

parser.add_argument('--channel', type=int, default=2,
                    help='# of input channels: the half of number of frequency bands (default: 2)')

parser.add_argument('--iteration', type=int, default=100,
                    help='maximal epochs of traning (default: 100)')

parser.add_argument('--subject', type=str, default='1',
                    help=' the desired subject (default: 1 )')


args = parser.parse_args()

seed=args.seed
duration=args.duration
channel=args.channel
max_iteration=args.iteration
sid=args.subject

input_channel=int(4000/duration*channel)

cwd=os.getcwd()



var_=[500,1000,2000,4000]
plot_data=[]
plot_data_e=[]
for xin in var_:
    coll_seed=np.zeros(20)
    for r in range(1,21):
        fp=open(cwd+'/train_history/best_'+str(xin)+'-'+str(channel)+'_'+str(r)+'_'+str(sid)+'.json','r')
        temp=np.array(json.load(fp))
        fp.close()
        coll_seed[r-1]=np.amax(temp)
    plot_data.append(np.mean(coll_seed))
    plot_data_e.append(np.std(coll_seed))
pylab.figure(1)
pylab.bar([1,2,3,4],plot_data)
pylab.errorbar([1,2,3,4],plot_data,yerr=plot_data_e)



var_=range(1,15)
plot_data=[]
plot_data_e=[]
all_max=[]
for xin in var_:
    coll_seed=np.zeros(20)
    sid=xin
    for r in range(1,21):
        fp=open(cwd+'/train_history/best_'+str(duration)+'-'+str(channel)+'_'+str(r)+'_'+str(sid)+'.json','r')
        temp=np.array(json.load(fp))
        fp.close()
        coll_seed[r-1]=np.amax(temp)
        all_max.append(np.amax(temp))
    plot_data.append(np.mean(coll_seed))
    plot_data_e.append(np.std(coll_seed))

print (np.mean(np.array(all_max)))
pylab.figure(2)
pylab.bar(var_,plot_data)
pylab.errorbar(var_,plot_data,yerr=plot_data_e)
pylab.show()

pylab.show()






