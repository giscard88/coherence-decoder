import os
import random
import shutil
import time
import warnings
import sys
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import copy


import json
import argparse
import numpy as np
from collections import defaultdict
import pylab


#from library_lrp import *

cwd=os.getcwd()


parser = argparse.ArgumentParser(description='learn the patterns of coherence among EEG electrodes')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed for pytorch and numpy (default: 10)')

parser.add_argument('--duration', type=int, default=500,
                    help='duration of time series of interest (default: 4000)')

parser.add_argument('--channel', type=int, default=2,
                    help='# of input channels: the half of number of frequency bands (default: 2)')



args = parser.parse_args()




seed=args.seed
duration=args.duration
channel=args.channel

torch.manual_seed(seed)
np.random.seed(seed)


fn=cwd+'/best_model/best_'+str(duration)+'-'+str(channel)+'_'+str(seed)+'.pt'
param=torch.load(fn)
for xin in param:
    print (xin)









