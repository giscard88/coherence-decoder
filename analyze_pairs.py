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


parser.add_argument('--duration', type=int, default=4000,
                    help='duration of time series of interest (default: 4000)')

parser.add_argument('--channel', type=int, default=10,
                    help='# of input channels: the half of number of frequency bands (default: 2)')


parser.add_argument('--subject', type=str, default='1',
                    help=' the desired subject (default: 1 )')

args = parser.parse_args()


duration=args.duration
channel=args.channel
sid=args.subject

    
width_freq=125.0/(channel*2.0) # each channel will have correlations in the two sub-frequency bands
edges_freq=np.arange(0,125.0,width_freq)
edges_freq=np.hstack((edges_freq,125.0)) # it adds the final edge (125.0 Hz).



def num2freq(num):
    return (edges_freq[num],edges_freq[num+1])




cwd=os.getcwd()

target_dir=cwd+'/LRP2/'+str(sid)
os.chdir(target_dir)

num_plane=int(4000/duration)*channel


all_=defaultdict(list)

cut_off_=defaultdict(list)
classes_=[]
for xin in [0,1,2,3]: # target classes 
    target_str='H_tr'+str(duration)+'-'+str(channel)+'_10_'+str(xin)+'*.pt' # _10_ is always there and thus redundant. Need to be removed later.    
    temp_cnt=np.zeros(num_plane*44*44)
    file_list=glob(target_str)
    print (xin, len(file_list))
    if len(file_list)>0:
        for fn in file_list:
            data=torch.load(fn)
            data=data.cpu().numpy()
            freq_=np.zeros(channel*2)
            for ch in range(channel):
                for i in range(44):
                    for j in range(44):
                        if i>j:
                            idx=ch*2
                            freq_[idx]=freq_[idx]+data[ch,i,j]
                        elif i<j:
                            idx=ch*2+1
                            freq_[idx]=freq_[idx]+data[ch,i,j]
    pylab.plot(range(channel*2),freq_,'-o',label=str(xin))

    classes_.append(freq_)




#classes_=np.array(classes_)

#plot=np.mean(classes_,0)
#plot_e=np.std(classes_,0)

#pylab.errorbar(range(channel*2),y=plot,yerr=plot_e)
pylab.legend()
pylab.show()



                        
                        

#    idx=np.argsort(freq_)
#    idx=np.flip(idx,0)
#    for i in range(5):
#        print (i, num2freq(idx[i]))



            #print (data.shape)

'''

            data=data.cpu().numpy()
            data=data.flatten()
            temp_cnt=temp_cnt+data
            args_=np.argsort(temp_cnt)
            args_=np.flip(args_,0)
     
        all_[str(xin)].append(temp_cnt)
    else:
        print (str(duration)+'-'+str(channel)+'_10_'+str(xin),'subject',sid)
        sys.exit('Warning ! LRP results do not seem to be available')
        
        
 
final={}  
for xin in [0,1,2,3]: # target classes 
    temp=np.array(all_[str(xin)])
    
    temp=temp.flatten()
    args_=np.argsort(temp)
    args_=np.flip(args_,0)
    final[str(xin)]=args_[:cut_num]
    
#print (final)

freq=defaultdict(list)

for xin in final:
    temp=final[xin]
    for t in temp:
        freq[t].extend(xin)

#print (freq)
scan=defaultdict(list)

for xin in freq:
    temp_=''
    labels=freq[xin]
    for l in labels:

        temp_=temp_+str(l)
    
    
    scan[str(temp_)].append(int(xin))

os.chdir(cwd)
fp=open('scan'+'_'+str(duration)+'_'+str(channel)+'_'+str(sid)+'.json','w')
json.dump(scan,fp)
fp.close()
'''

# end




 




