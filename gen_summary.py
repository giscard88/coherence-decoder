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
import sys

from glob import glob
from collections import defaultdict

matrix=np.zeros((14,4))

for xin in range(1,15):
    fp=open('scan_4000_4_'+str(xin)+'.json','r')
    data=json.load(fp)
    fp.close()
    for yin in range(4):
        if str(yin) in data:
            matrix[xin-1,yin]=len(data[str(yin)])

print (matrix)


