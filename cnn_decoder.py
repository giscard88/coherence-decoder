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

def train():
    loss_val=[]
    test_history=[]
    net.train()
    for tr in range(max_iteration):
        

        loss_total=0
        for xi, xin in enumerate(batches):
            inputs=train_data[xin[0]:xin[1],:,:,:]
            targets=train_label[xin[0]:xin[1]]
            #inputs=inputs.to(device)
            #targets=targets.to(device)
            
            optimizer.zero_grad()
            output = net(inputs)
            loss = F.nll_loss(output, targets)
            loss.backward()
            optimizer.step()
            loss_total=loss_total+loss.item()
        test_=test()
        test_history.append(test_)
        max_p=max(test_history)
        if max_p==test_:
            #print ('update',max_p)
            model_best=copy.deepcopy(net)
            model_best=model_best.cpu()
            #print('original',list(net.fc2.weight))
            #print('copy',list(model_best.fc2.weight))
        print (tr,loss_total)
        loss_val.append(loss_total)
        
    #del inputs, targets
    return test_history, model_best



def test():
    correct = 0
    net.eval()
    with torch.no_grad():
        ans=0
        inputs=test_data
        targets=test_label
        ct=float(test_data.shape[0])
        outputs=net(test_data)
        pred = torch.argmax(outputs, dim=1)
        correct += pred.eq(test_label.view_as(pred)).sum().item()
    del inputs, targets   
    return float(correct)/ct

def test_best():
    correct = 0
    model_best.to(device)
    model_best.eval()
    with torch.no_grad():
        ans=0
        inputs=test_data
        targets=test_label
        ct=float(test_data.shape[0])
        outputs=model_best(test_data)
        pred = torch.argmax(outputs, dim=1)
        correct += pred.eq(test_label.view_as(pred)).sum().item()
    del inputs, targets   
    return float(correct)/ct

def validate():
    correct = 0
    with torch.no_grad():
        ans=0
        inputs=train_data
        targets=train_label
        ct=float(train_data.shape[0])
        outputs=net(train_data)
        pred = torch.argmax(outputs, dim=1)
        correct += pred.eq(train_label.view_as(pred)).sum().item()
    del inputs, targets  
    return float(correct)/ct

            
       



if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  


cwd=os.getcwd()

#fn=cwd+'/converted_data/train/'+sid+'/inputtrain_'+sid+'.pt'
fn=cwd+'/p'+str(duration)+'ch'+str(channel)+'/inputtrain_'+sid+'.pt'
train_data=torch.load(fn)
fn=cwd+'/p'+str(duration)+'ch'+str(channel)+'/labeltrain_'+sid+'.pt'
train_label=torch.load(fn)



fn=cwd+'/p'+str(duration)+'ch'+str(channel)+'/inputtest_'+sid+'.pt'
test_data=torch.load(fn)
fn=cwd+'/p'+str(duration)+'ch'+str(channel)+'/labeltest_'+sid+'.pt'
test_label=torch.load(fn)

train_data=train_data.float().to(device)
train_label=train_label.long().to(device)

test_data=test_data.float().to(device)
test_label=test_label.long().to(device)


print (device)
mini_batch_size=10
total=train_data.shape[0]

net=Net().to(device)
#model_best=Net()
optimizer = optim.Adadelta(net.parameters(), lr=0.1)


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

train_history, model_best=train()

print (train_history)
print ('max',max(train_history))
#print ('train',validate())
print ('test',test())
print ('test_best',test_best())

target_dir=cwd+'/best_model'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

fn=target_dir+'/best_'+str(duration)+'-'+str(channel)+'_'+str(seed)+'_'+sid+'.pt'
torch.save(model_best.state_dict(),fn)


target_dir=cwd+'/train_history'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

fj=target_dir+'/best_'+str(duration)+'-'+str(channel)+'_'+str(seed)+'_'+sid+'.json'
fp=open(fj,'w')
json.dump(train_history,fp)
fp.close()

#pylab.plot(train_history,'-o')
#pylab.show()

del train_data, train_label, test_data, test_label
# end of script

