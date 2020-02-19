import os
import random
import shutil
import time
import warnings
import sys
import os


import copy


import json
import argparse
import numpy as np
from collections import defaultdict
import pylab







cwd=os.getcwd()
chans=[2,4,6,8,10]
color={0:'r',1:'g',2:'b',3:'k'}

for i, d in enumerate([500,1000,2000,4000]):
    best_chans_means=[]
    best_chans_err=[]
    for ch in chans:

        var_=range(1,15)
        plot_data=[]
        plot_data_e=[]
        temp_max=[]
        for xin in var_:
            coll_seed=np.zeros(20)
            sid=xin
            for r in range(1,21):
                fp=open(cwd+'/train_history/best_'+str(d)+'-'+str(ch)+'_'+str(r)+'_'+str(sid)+'.json','r')
                temp=np.array(json.load(fp))
                fp.close()
                coll_seed[r-1]=np.amax(temp)
                temp_max.append(np.amax(temp))
        temp_max=np.array(temp_max)
        best_chans_means.append(np.mean(temp_max))
        best_chans_err.append(np.std(temp_max)/np.sqrt(float(len(temp_max))))


#pylab.bar(chans,best_chans_means)
    pylab.errorbar(chans,best_chans_means,yerr=best_chans_err,fmt='-o',label=str(d))
pylab.xlabel('channel #')
pylab.ylabel('accuracy')
pylab.legend()
pylab.title('errorbar:sde')
pylab.savefig('best_figures/plot_channel.png')
pylab.savefig('best_figures/plot_channel.eps')





pylab.show()






