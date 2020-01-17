'''
This script is based on example.py included as a part of dataset at https://gin.g-node.org/robintibor/high-gamma-dataset/src/master.  

'''


import nitime
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu


import logging
import argparse
import sys
import os.path
from collections import OrderedDict
import numpy as np
import json
import pylab
import torch



import torch 
from torch import optim

parser = argparse.ArgumentParser(description='readout EEG signals from high-gamma datasset')

parser.add_argument('--duration', type=int, default=500, help='duration of time segments (default: 500)')
parser.add_argument('--channel', type=int, default=2, help='number of coherence maps (default: 2)') 

args = parser.parse_args()

duration=args.duration
channel=args.channel




def convert(sid ,period ,attr='train'):
    subject_id = sid
    
    
    target_dir='converted_data/'+attr+'/'+str(subject_id)
    fn=target_dir+'/data_4_'+str(subject_id)+'-'+str(period[0])+'-'+str(period[1])+'.npy'
    data=np.load(fn)
    fn=target_dir+'/label_4_'+str(subject_id)+'-'+str(period[0])+'-'+str(period[1])+'.npy'
    labels=np.load(fn)    
    
    tr, elc, ts_n=data.shape

    slices_=[]

    for t in range(tr):
        #print (train_set.X[t,:,:])
        #print (data_set.X[t,:,:].shape)
        time_series = ts.TimeSeries(data[t,:,:], sampling_rate=250.0,time_unit='ms')
        
        
        NW=3.5
        bw=NW*2.0*time_series.sampling_rate/float(time_series.shape[-1])
       

        #print ('length of time series',time_series.shape[-1])
        #print ('sampling rate',time_series.sampling_rate)
        #print ('bandwidth of tapers',bw)
        C = nta.coherence.MTCoherenceAnalyzer(time_series,bandwidth=bw)

        #print (C.df)

        C.df=int(C.df) #It looks like df is used as index, but it does allow df to be float in nitime (raising error). This trick semes to work. 

        #print ('done and now printing frequencies') 
        freq=C.frequencies
        coh_val=C.coherence
        #print (coh_val.shape)
        #print (freq)


    
        coh_slice=split_power(freq,coh_val,t,sid,attr,period)
        slices_.append(coh_slice)
       
    slices_=np.array(slices_)
    
    print ('slices_',slices_.shape,'labels',labels.shape)
    return slices_,labels



def split_power(freq_,data,t_,sid,attr,period):

    
    width_freq=125.0/(channel*2.0) # each channel will have correlations in the two sub-frequency bands
    edges_freq=np.arange(0,125.0,width_freq)
    edges_freq=np.hstack((edges_freq,125.0)) # it adds the final edge (125.0 Hz).
    coh_array=np.zeros((channel,44,44))
    fig_flag=False
    for c in range(channel):
        for s in range(2):

            idx1=c*2+s
            


            lb=edges_freq[idx1]
            ub=edges_freq[idx1+1]
            idx2=np.where((freq_>=lb) & (freq_<ub))[0]
                #print (idx2)
            for xi in range(44):
                for yi in range(44):
                #print (xi, yi)
                
                    if xi>yi and s==0:    
                        coh_array[c,xi,yi]=np.mean(data[xi,yi][idx2])
                    if xi>yi and s==1:    
                        coh_array[c,yi,xi]=np.mean(data[xi,yi][idx2])

    fig_dir=cwd+'/cohfigs/'+str(channel)+'/'+attr+'/'+str(sid)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if fig_flag:
        for ch_ in range(channel):
            plot_data=coh_array[ch_,:,:]
            pylab.imshow(plot_data,cmap='jet')
            pylab.colorbar()
            pylab.title('tr:'+str(t_)+'_ch:'+str(ch_)+'_lab:'+str(label)+'_'+str(period[0])+'-'+str(period[1])) 
            pylab.savefig(fig_dir+'/coh_'+str(t_)+'_'+str(ch_)+'_'+str(label)+'_'+str(period[0])+'-'+str(period[1])+'.png') 
            pylab.close()
    return  coh_array
       


if __name__ == '__main__':
    cwd=os.getcwd()

    

    residual=4000 % duration

    if residual==0:
        temp=np.arange(0,4000,duration)
        temp=np.hstack((temp,4000))
   

    else:
        sys.exit('4000 ms should be divisible by duration')

    periods_=[]

    for t, value in enumerate(temp[:-1]):
    
        periods_.append([value,temp[t+1]])

    print (periods_)

    labels_=[]
    target_dir=cwd+'/np'+str(duration)+'ch'+str(channel)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for s in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]: #1 done 
        for b in ['train','test']:
            coh_array_flag=False
            for pr in periods_:
            #print (s, b)
                coh_slice,label=convert(s, pr, b)
                if coh_array_flag:
                    coh_tensor=np.concatenate((coh_tensor,coh_slice),1)
                else:
                    coh_tensor=coh_slice
                    coh_array_flag=True
            print (coh_tensor.shape,label.shape)
            tensor_in=torch.from_numpy(coh_tensor)
            tensor_lab=torch.from_numpy(label)
            tensor_in=tensor_in.float()
            tensor_lab=tensor_lab.long()
            torch.save(tensor_in, target_dir+'/input_4_'+b+'_'+str(s)+'.pt')
            torch.save(tensor_lab, target_dir+'/label_4_'+b+'_'+str(s)+'.pt')
            del coh_tensor, label
    
     

              

