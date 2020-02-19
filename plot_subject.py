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

durations_=[]

pylab.subplot(2,2,1)
duration=500
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

durations_.append((np.mean(np.array(all_max)),np.std(np.array(all_max))))

pylab.bar(var_,plot_data)
pylab.errorbar(var_,plot_data,yerr=plot_data_e)


pylab.subplot(2,2,2)
duration=1000



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

durations_.append((np.mean(np.array(all_max)),np.std(np.array(all_max))))

pylab.bar(var_,plot_data)
pylab.errorbar(var_,plot_data,yerr=plot_data_e)

pylab.subplot(2,2,3)
duration=2000

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

durations_.append((np.mean(np.array(all_max)),np.std(np.array(all_max))))

pylab.bar(var_,plot_data)
pylab.errorbar(var_,plot_data,yerr=plot_data_e)

pylab.subplot(2,2,4)
duration=4000

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
durations_.append((np.mean(np.array(all_max)),np.std(np.array(all_max))))


pylab.bar(var_,plot_data)
pylab.errorbar(var_,plot_data,yerr=plot_data_e)


pylab.savefig('best_figures/duration_dependent-'+str(channel)+'.png')
pylab.savefig('best_figures/duration_dependent-'+str(channel)+'.eps')


pylab.figure(2)
means_=[durations_[0][0],durations_[1][0],durations_[2][0],durations_[3][0]]
err_=[durations_[0][1],durations_[1][1],durations_[2][1],durations_[3][1]]
pylab.bar([500,1000,2000,4000],means_)
pylab.errorbar([500,1000,2000,4000],means_,yerr=err_)




pylab.show()






