import torch
import numpy as np
import json
import os
import sys
import argparse
import glob


def lin2Array(arg,grid=44):
    row=int(arg/grid)
    col=int(arg) % int(grid)

    return (row,col)

def Array2lin(args,grid=44):
    arg=args[0]*grid+args[1]

    return arg


cwd=os.getcwd()

parser = argparse.ArgumentParser(description='summarize the results')


parser.add_argument('--duration', type=int, default=500,
                    help='duration of time series of interest (default: 4000)')

parser.add_argument('--channel', type=int, default=2,
                    help='# of input channels: the half of number of frequency bands (default: 2)')

parser.add_argument('--subject', type=int, default=1,
                    help='# select the subject (default: 1)')


args = parser.parse_args()

sid=args.subject
duration=args.duration
channel=args.channel

comm_str=str(duration)+'_'+str(channel)+'_'+str(sid)

fn=cwd+'/LRP/'+str(sid)+'/'+conn_str+'*'


files_=glob.glob(fn)
grid_=44
for fi in files:
    data=torch.load(fi)
    data=data.cpu().numpy()
    max_paris={}
    for c in range(channel):
        temp_=data[c,:,:]
        temp_=temp_.reshape(grid_*grid_)
        idx=np.argsort(temp_)
        idx_r=idx[::-1]
        for x, xin in enumerate(idx_r):
            if np.isnan(temp_[xin]):
                pass
            else:
                max_pairs[str(c)]=lin2Array(xin)


