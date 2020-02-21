import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
from PIL import Image
from Model import *

import copy
import pylab


'''
This script is adopted from the github and modified slightly to implement variation of lrp rules

@article{li2019beyond,
  title={Beyond saliency: understanding convolutional neural networks from saliency prediction on layer-wise relevance propagation},
  author={Li, Heyi and Tian, Yunke and Mueller, Klaus and Chen, Xin},
  journal={Image and Vision Computing},
  year={2019},
  publisher={Elsevier}
}




'''


if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"  


best_model = Net()

fn=cwd+'/best_model/best_'+str(duration)+'-'+str(channel)+'_'+str(seed)+'_'+sid+'.pt'
param=torch.load(fn)

scale=int(4000/duration)


best_model.load_state_dict(param)

fn=cwd+'/p'+str(duration)+'ch'+str(channel)+'/inputtest_'+sid+'.pt'
test_data=torch.load(fn)
fn=cwd+'/p'+str(duration)+'ch'+str(channel)+'/labeltest_'+sid+'.pt'
test_label=torch.load(fn)



test_data=test_data.float().to(device)
test_label=test_label.long().to(device)

data_loader=[(test_data,test_label)]

#for i in best_model.named_parameters():
#    print (i)


# add relprop() method to each layer
########################################


class Linear(nn.Linear):
    def __init__(self, linear):
        super(nn.Linear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        print (linear.in_features,linear.out_features)
        print (linear.weight.size())
        
    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        Z = torch.mm(self.X, torch.transpose(V,0,1)) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        R = self.X * C
        return R
        
class ReLU(nn.ReLU):   
    def relprop(self, R): 
        return R


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, maxpool2d):
        kernel_size=maxpool2d.kernel_size
        super(nn.MaxPool2d, self).__init__(kernel_size)
        self.kernel_size = maxpool2d.kernel_size
        self.stride = maxpool2d.stride
        self.padding = maxpool2d.padding
        self.dilation = maxpool2d.dilation
        self.return_indices = maxpool2d.return_indices
        self.ceil_mode = maxpool2d.ceil_mode
        
    def gradprop(self, DY):
        DX = self.X * 0
        temp, indices = F.max_pool2d(self.X, self.kernel_size, self.stride, 
                                     self.padding, self.dilation, self.ceil_mode, True)
        DX = F.max_unpool2d(DY, indices, self.kernel_size, self.stride, self.padding)
        return DX

    
    def relprop(self, R):
        Z = self.Y + 1e-9

        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R

class Conv2d(nn.Conv2d):
    def __init__(self, conv2d):
        super(nn.Conv2d, self).__init__(conv2d.in_channels, 
                                        conv2d.out_channels, 
                                        conv2d.kernel_size, 
                                        conv2d.stride, 
                                        conv2d.padding, 
                                        conv2d.dilation, 
                                        conv2d.transposed, 
                                        conv2d.output_padding, 
                                        conv2d.groups,
                                        True, 
                                        conv2d.padding_mode)
        self.weight = conv2d.weight
        self.bias = conv2d.bias
        #print ('conv_w',self.weight.size(), conv2d.in_channels,conv2d.out_channels)
        
    def gradprop(self, DY):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, self.weight, stride=self.stride, 
                                  padding=self.padding, output_padding=output_padding)
        
    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R


class Reshape_net(nn.Module):
    def __init__(self):
        super(Reshape_net, self).__init__()
        
    def forward(self, x):
        return x.view(-1, channel*4*scale*21*21) #64 channles of 21-by-21
        
    def relprop(self, R):
        return R.view(-1, channel*4*scale, 21, 21)







class CNN_lrp(nn.Module):
    def __init__(self):
        super(CNN_lrp, self).__init__()
       
        self.layers = nn.Sequential(
            Conv2d(best_model.conv1),
            ReLU(),
            Conv2d(best_model.conv2),
            MaxPool2d(nn.MaxPool2d(2)),
            Reshape_net(),
            Linear(best_model.fc1),
            ReLU(),
            Linear(best_model.fc2)

        ) 
        
    def forward(self, x):
        #print ('before',x.size())
        #print ( lrp_model.layers[0].weight.size())
        #print ('channels',lrp_model.layers[0].in_channels,lrp_model.layers[0].out_channels)
        x = self.layers(x)
        #print ('after',x.size())
        return x
        
    def relprop(self, R):
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l-1].relprop(R)
        return R

lrp_model=CNN_lrp()

def forward_hook(self, input, output):
    self.X = input[0]
    self.Y = output
    
for i in range(0, len(lrp_model.layers)):
    lrp_model.layers[i].register_forward_hook(forward_hook)
    
lrp_model.to(device)
lrp_model.eval()

correct_ = 0
buffer_label = []
buffer_lrp_model = []

target_dir=cwd+'/LRP'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


for idx, (input, label) in enumerate(data_loader):
    
    
    output_lrp_model = lrp_model(input)
    pred_ = output_lrp_model.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct_ += pred_.eq(label.data.view_as(pred_)).cpu().sum()
    
    T_lrp_model = pred_.squeeze().cpu().numpy()
    T_lrp_model = (T_lrp_model[:,np.newaxis] == np.arange(4))*1.0
    T_lrp_model = torch.from_numpy(T_lrp_model).type(torch.FloatTensor)
    T_lrp_model = T_lrp_model.to(device)
    LRP_= lrp_model.relprop(output_lrp_model * T_lrp_model)
    
    
    
    buffer_label.append(label.data.cpu().numpy())
    buffer_lrp_model.append(pred_.cpu().numpy())
    
    comm_str=str(duration)+'-'+str(channel)+'_'+str(seed)
    # save results which are classified correctly by VGG16, incorrectly by AlexNet
    for i in range(0,160):
        if pred_.squeeze().cpu().numpy()[i] == label.data.cpu().numpy()[i]:
            img = input[i].data.cpu().numpy()
            img =(img-img.min()) / (img.max()-img.min())
            #img = img.astype('uint8')

            target_dir2=target_dir+'/'+sid
            if not os.path.exists(target_dir2):
                os.makedirs(target_dir2)
            
                    
            heatmap_lrp = LRP_[i].data.cpu().numpy()
            heatmap_lrp = np.absolute(heatmap_lrp)
            heatmap_lrp = (heatmap_lrp-heatmap_lrp.min()) / (heatmap_lrp.max()-heatmap_lrp.min())
            #heatmap_lrp = heatmap_lrp.astype('uint8')

           
            fn=target_dir2+'/I_tr'+comm_str+'_'+str(label.data.cpu().numpy()[i])+'_tr_'+str(i)+'.pt'

            torch.save(torch.from_numpy(img),fn)

            
            fn=target_dir2+'/H_tr'+comm_str+'_'+str(label.data.cpu().numpy()[i])+'_tr_'+str(i)+'.pt'
            torch.save(torch.from_numpy(heatmap_lrp),fn)
         
            target_dir3=target_dir+'/figures/'+sid
            if not os.path.exists(target_dir3):
                os.makedirs(target_dir3)
            
            dims=img.shape
            for j in range(dims[0]):
                pylab.imshow(img[j,:,:],cmap='jet')
                pylab.savefig(target_dir3+'/img_#'+comm_str+'_tr_'+str(i)+'_ch'+str(j)+'_'+str(label.data.cpu().numpy()[i])+'.png')
                pylab.close()

            dims=heatmap_lrp.shape
            for j in range(dims[0]):
                pylab.imshow(heatmap_lrp[j,:,:],cmap='jet')
                pylab.savefig(target_dir3+'/heatmap_#'+comm_str+'_tr_'+str(i)+'-ch'+str(j)+'_'+str(label.data.cpu().numpy()[i])+'.png')
                pylab.close()


            
            
            

print('Done...')






