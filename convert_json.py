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

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt


from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne

from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


parser = argparse.ArgumentParser(description='readout EEG signals from high-gamma datasset')

parser.add_argument('--duration', type=int, default=500, help='duration of time segments (default: 500)')
 

args = parser.parse_args()

duration=args.duration
cwd=os.getcwd()



def load_bbci_data(filename, low_cut_hz, period, debug=False):
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
    ival =period

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset



def convert(sid ,period ,attr='train'):
    subject_id = sid
    # have to change the data_folder here to make it run.
    data_folder = 'data'
    if attr=='train':
        filename =  os.path.join(data_folder, 'train/{:d}.mat'.format(subject_id))
    elif attr=='test':
        filename =  os.path.join(data_folder, 'test/{:d}.mat'.format(subject_id))
    else:
        sys.exit('choose attribution to be either test or train')
    data_set = load_bbci_data(filename, 0,period)
    #print (type(train_set.X),train_set.X.shape)
    #print (train_set.y,len(train_set.y))
    
    slices_=data_set.X
    labels_=data_set.y
    
    return slices_,labels_






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


    
    for s in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        for sel in ['train','test']:
            target_dir='converted_data/'+sel+'/'+str(s)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for pr in periods_:
                data,label=convert(s, pr, sel)
                np.save(target_dir+'/data'+str(s)+'-'+str(pr[0])+'-'+str(pr[1])+'.npy', data)
                np.save(target_dir+'/label'+str(s)+'-'+str(pr[0])+'-'+str(pr[1])+'.npy', label)


    
     

              

