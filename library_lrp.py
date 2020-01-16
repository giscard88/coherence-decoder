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
import copy


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



# add relprop() method to each layer
########################################


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

class Linear(nn.Linear):
    def __init__(self, linear):
        super(nn.Linear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        
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
        super(nn.MaxPool2d, self).__init__()
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
                                        True)
        self.weight = conv2d.weight
        self.bias = conv2d.bias
        
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



class CNN_lrp(nn.Module):
    def __init__(self):
        super(CNN_lrp, self).__init__()
        padd_1=0
        ker_1=2
        padd_2=0
        ker_2=2
        input_channel=10
        out1=int((44-ker_1+2*padd_1)+1)
        out2=int((out1-ker_2+2*padd_2)+1)
        out2=int(out2/2) # max pooling with the kernel of 2
        final_size=int(out2*out2*input_channel*4)
        self.layers = nn.Sequential(
            Conv2d(alex.features[0]),
            ReLU(),
            MaxPool2d(x,2),
            Conv2d(alex.features[3]),
            ReLU(),
            
            Linear(alex.classifier[1]),
            ReLU(),
            Linear(alex.classifier[4]),
            ReLU(),
            Linear(alex.classifier[6])
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
        
    def relprop(self, R):
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l-1].relprop(R)
        return R





print (model.features)

for i in model.named_parameters():
    print (i)
