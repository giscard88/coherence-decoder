import json
import numpy as np
import sys
import os
import glob
import pylab
import torch



cwd=os.getcwd()

channel=1
width_freq=125.0/(channel*2.0) # each channel will have correlations in the two sub-frequency bands
edges_freq=np.arange(0,125.0,width_freq)
edges_freq=np.hstack((edges_freq,125.0)) # it adds the final edge (125.0 Hz). 

def split_power(data):

    coh_array=np.zeros((channel,44,44))
    flag_lab=False
    for c in range(channel):
        for s in range(2):

            idx1=c*2+s
            

            for arg in data:
                pt1=arg.find(':')
                pt2=arg.find('_')
                temp_data=data[arg]
                freq_=np.array(temp_data[0])
                coh_=np.array(temp_data[1])
                if flag_lab:
                    if temp_data[2]==label_:
                        pass
                    else:
                        sys.exit('label incorrect') # let's catch if there is an error in labels. 
                else:
                    label_=temp_data[2]
                    flag_lab=True
                lb=edges_freq[idx1]
                ub=edges_freq[idx1+1]
                idx2=np.where((freq_>=lb) & (freq_<ub))[0]
                #print (idx2)
                xi=int(arg[pt1+1:pt2])
                yi=int(arg[pt2+1:])
                #print (xi, yi)
                
                if xi>yi and s==0:    
                    coh_array[c,xi,yi]=np.mean(coh_[idx2])
                if xi>yi and s==1:    
                    coh_array[c,yi,xi]=np.mean(coh_[idx2])
        for t in range(44): # let's fill diagonal with 1s. 
            coh_array[c,t,t]=1.0
    return coh_array,label_                
                        
        
        
    


attr=['train'] #,'test']
subjects=['1']
fig_flag=True
for a in attr:
    for sid in subjects:
        target_dir=cwd+'/converted_data/'+a+'/'+sid
        os.chdir(target_dir)
        files=glob.glob('tr*.json')
        tids_=[]
        for name in files:
            pt1=name.find('.json')
            tids_.append(name[2:pt1])
        tids_=np.array(tids_).astype(int)
        tids_=np.sort(tids_)
        #print (sids_)
        all_coh=np.zeros((len(tids_),channel,44,44))
        all_label=np.zeros(len(tids_))
        for t_ in tids_[:12]:
            fp=open('tr'+str(t_)+'.json','r')
            data=json.load(fp)
            fp.close()
            
            coherence_,label_=split_power(data)
            all_coh[t_,:,:,:]=coherence_
            all_label[t_]=label_
            fig_dir=cwd+'/cohfigs/'+str(channel)+'/'+a+'/'+sid
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            if fig_flag:
                for ch_ in range(channel):
                    plot_data=all_coh[t_,ch_,:,:]
                    pylab.imshow(plot_data,cmap='jet')
                    pylab.colorbar()
                    pylab.title('tr:'+str(t_)+'_ch:'+str(ch_)+'_lab:'+str(all_label[t_])) 
                    pylab.savefig(fig_dir+'/coh_'+str(t_)+'_'+str(ch_)+'_'+str(all_label[t_])+'.png') 
                    pylab.close()
      

        tensor_in=torch.from_numpy(all_coh)
        tensor_lab=torch.from_numpy(all_label)
        torch.save(tensor_in, 'input'+a+'_'+sid+'.pt')
        torch.save(tensor_lab, 'label'+a+'_'+sid+'.pt')
            
        
        
        

    



