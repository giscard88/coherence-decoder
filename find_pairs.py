import os
import random
import shutil
import time
import warnings
import sys
import os
import pylab
import torch
import json
import argparse
import numpy as np

from glob import glob
from collections import defaultdict

parser = argparse.ArgumentParser(description='learn the patterns of coherence among EEG electrodes')


parser.add_argument('--duration', type=int, default=500,
                    help='duration of time series of interest (default: 4000)')

parser.add_argument('--channel', type=int, default=2,
                    help='# of input channels: the half of number of frequency bands (default: 2)')

parser.add_argument('--iteration', type=int, default=100,
                    help='maximal epochs of traning (default: 100)')

parser.add_argument('--subject', type=str, default='1',
                    help=' the desired subject (default: 1 )')


args = parser.parse_args()


duration=args.duration
channel=args.channel
max_iteration=args.iteration
sid=args.subject

cwd=os.getcwd()

target_dir=cwd+'/LRP/'+str(sid)
os.chdir(target_dir)

num_plane=int(4000/duration)*channel


all_=defaultdict(list)

cut_off_=defaultdict(list)
cut_num=100
for xin in [0,1,2,3]: # target classes 
    target_str='H_tr'+str(duration)+'-'+str(channel)+'_10_'+str(xin)+'*.pt' # _10_ is always there and thus redundant. Need to be removed later. 
 
    file_list=glob(target_str)
    for fn in file_list:
        data=torch.load(fn)
        data=data.cpu().numpy()
        data=data.flatten()
        args_=np.argsort(data)
        args_=np.flip(args_,0)
        mean=np.mean(data)
        std=np.std(data)
        cut_off=mean+5.0
        for ii in range(cut_num):
            temp=args_[ii]
            if temp>cut_off:
                all_[str(xin)].append(temp)
        
 
final={}  
for xin in [0,1,2,3]: # target classes 
    temp=all_[str(xin)]
    count=np.zeros(num_plane*44*44)
    for yin in temp:
        count[yin]=count[yin]+1
    temp_arg=np.argsort(count)
    temp_arg=np.flip(temp_arg,0)
    
    final[str(xin)]=temp_arg[:cut_num]
    


freq=defaultdict(list)

for xin in final:
    temp=final[xin]
    for t in temp:
        freq[t].extend(xin)

print (freq)
scan=defaultdict(list)

for xin in freq:
    temp_=''
    labels=freq[xin]
    for l in labels:

        temp_=temp_+str(l)
    
    
    scan[temp_].append(xin)

print (scan)





 




