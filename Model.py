import os
import random
import shutil
import time
import warnings
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import copy


import json
import argparse
import numpy as np
from collections import defaultdict
import pylab



parser = argparse.ArgumentParser(description='learn the patterns of coherence among EEG electrodes')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed for pytorch and numpy (default: 10)')

parser.add_argument('--duration', type=int, default=4000,
                    help='duration of time series of interest (default: 4000)')

parser.add_argument('--channel', type=int, default=2,
                    help='# of input channels: the half of number of frequency bands (default: 2)')

parser.add_argument('--iteration', type=int, default=100,
                    help='maximal epochs of traning (default: 100)')

parser.add_argument('--subject', type=int, default=1,
                    help='target subject (default: 1)')


args = parser.parse_args()



cwd=os.getcwd()
seed=args.seed
duration=args.duration
channel=args.channel
max_iteration=args.iteration
sid=str(args.subject)
torch.manual_seed(seed)
np.random.seed(seed)

input_channel=int(4000/duration*channel)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        padd_1=0
        ker_1=2
        padd_2=0
        ker_2=2
        out1=int((44-ker_1+2*padd_1)+1)
        out2=int((out1-ker_2+2*padd_2)+1)
        out2=int(out2/2) # max pooling with the kernel of 2
        final_size=int(out2*out2*input_channel*4)
        #print ('out1',out1)
        #print ('out2',out2)
        self.conv1 = nn.Conv2d(input_channel, input_channel*2, ker_1, 1)
        self.conv2 = nn.Conv2d(input_channel*2, input_channel*4, ker_2, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(final_size, 100)
        self.fc2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output





    






