'''
This script is based on example.py included as a part of dataset at https://gin.g-node.org/robintibor/high-gamma-dataset/src/master.  

'''


import nitime
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu


import logging
import sys
import os.path
from collections import OrderedDict
import numpy as np
import json

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
import torch.nn.functional as F
import torch as th
from torch import optim
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize


log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset



def convert(sid, attr='train'):
    subject_id = sid
    # have to change the data_folder here to make it run.
    data_folder = 'data'
    if attr=='train':
        filename =  os.path.join(data_folder, 'train/{:d}.mat'.format(subject_id))
    elif attr=='test':
        filename =  os.path.join(data_folder, 'test/{:d}.mat'.format(subject_id))
    else:
        sys.exit('choose attribution to be either test or train')
    data_set = load_bbci_data(filename, low_cut_hz=0)
    #print (type(train_set.X),train_set.X.shape)
    #print (train_set.y,len(train_set.y))
    tr, elc, ts_n=data_set.X.shape
    
    target_dir='converted_data/'+attr+'/'+str(subject_id)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    

    
    for t in range(tr):
        #print (train_set.X[t,:,:])
        #print (train_set.X[t,:,:].shape)
        time_series = ts.TimeSeries(data_set.X[t,:,:], sampling_rate=250.0,time_unit='s')
        
        
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



    
        coherence={}
        for xi in range(elc):
            #print 'pre', pre 
            for yi in range(xi):
        
                yval=coh_val[xi,yi,:]
                coherence[str(t)+':'+str(xi)+'_'+str(yi)]=(list(freq),list(yval),int(data_set.y[t]))

        fp=open(target_dir+'/tr'+str(t)+'.json','w')
        json.dump(coherence,fp)
        fp.close()


if __name__ == '__main__':
    for s in [1]: #1 done
        for b in ['train', 'test']:
            print (s, b)
            convert(s, b)

