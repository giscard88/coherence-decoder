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
        self.conv1 = nn.Conv2d(5, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(8000, 128)
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
    for tr in range(10):
        vlr=tr % 10
        if vlr==0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.1
        loss_total=0
        for xin in batches:
            inputs=train_data[xin[0]:xin[1],:,:,:]
            targets=train_label[xin[0]:xin[1]]
            
            optimizer.zero_grad()
            output = net(inputs)
            loss = F.nll_loss(output, targets)
            loss.backward()
            optimizer.step()
            loss_total=loss_total+loss.item()
        print (tr,loss_total)
        loss_val.append(loss_total)
    return loss_val

def test():
    correct = 0
    with torch.no_grad():
        ans=0
        
        ct=float(test_data.shape[0])
        outputs=net(test_data)
        pred = torch.argmax(outputs, dim=1)
        correct += pred.eq(test_label.view_as(pred)).sum().item()

    return float(correct)/ct

def validate():
    correct = 0
    with torch.no_grad():
        ans=0
        
        ct=float(train_data.shape[0])
        outputs=net(train_data)
        pred = torch.argmax(outputs, dim=1)
        correct += pred.eq(train_label.view_as(pred)).sum().item()

    return float(correct)/ct

            
        



parser = argparse.ArgumentParser(description='find info from coherence patterns')
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
train_label=train_label.long()

test_data=test_data.float()
test_label=test_label.long()

print (train_data.shape)

mini_batch_size=10
total=train_data.shape[0]

net=Net()
optimizer = optim.Adadelta(net.parameters(), lr=0.01)


num=int(total/mini_batch_size)
res=int(total) % int(mini_batch_size)

batches=[]

for xin in range(num):
    st=mini_batch_size*xin
    end=mini_batch_size*(xin+1)
    batches.append((st,end))


if res==0:
    pass
else:
    batches.append((end,total))

print (batches)

train_history=train()

print (train_history)
print ('train',validate())
print ('test',test())

# end of script

