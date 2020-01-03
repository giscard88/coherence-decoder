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



import json
import argparse
import numpy as np
from collections import defaultdict
import pylab

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 30, 3, 1)
        self.conv2 = nn.Conv2d(30, 60, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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

def train():
    loss_val=[]
    for tr in range(40):
        vlr=tr % 10
        if vlr==0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*1.0
        loss_total=0
        for xin in batches:
            inputs=train_inputs[xin[0]:xin[1],:,:,:]
            targets=train_labels[xin[0]:xin[1]]

            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            loss_total=loss_total+loss.item()
        print (tr,loss_total)
        loss_val.append(loss_total)
    return loss_val

def test():
    with torch.no_grad():
        ans=0
        ct=float(len(test_labels))
        
        outputs=net(test_inputs)
        pred = torch.argmax(outputs, dim=1)
        print (outputs)
        for xi, xin in enumerate(pred):
            #print (xi, xin.item(), test_labels[xi, xin].item())
            if test_labels[xi,xin].item()>0.98:
                ans=ans+1
    return float(ans)/ct

def validate():
    with torch.no_grad():
        ans=0
        ct=float(len(train_labels))
        
        outputs=net(train_inputs)
        pred = torch.argmax(outputs, dim=1)

        for xi, xin in enumerate(pred):
            #print (xi, xin.item(), test_labels[xi, xin].item(),test_labels[xi, xin.item()].item())
            if train_labels[xi,xin].item()>0.98:
                ans=ans+1
        
        
    return float(ans)/ct

            
        



parser = argparse.ArgumentParser(description='read out lfp signals from neuropixel')
parser.add_argument('--subject', type=str, default='1',
                    help=' the desired subject (default: 1 )')


args = parser.parse_args()
sid=args.subject


cwd=os.getcwd()

fn=cwd+'/converted_data/train/'+sid+'/inputtrain_'+sid+'.pt'
train_data=torch.load(fn)
fn=cwd+'/converted_data/train/'+sid+'/labeltrain_'+sid+'.pt'
train_label=torch.load(fn)


fn=cwd+'/converted_data/test/'+sid+'/inputtest_'+sid+'.pt'
test_data=torch.load(fn)
fn=cwd+'/converted_data/test/'+sid+'/labeltest_'+sid+'.pt'
test_label=torch.load(fn)

train_data=train_data.float()
train_label=train_label.float()

test_data=test_data.float()
test_label=test_label.float()

print (train_data.shape)


# end of script

