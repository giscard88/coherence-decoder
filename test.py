import numpy as np
import sys


duration=250

residual=4000 % duration

if residual==0:
    temp=np.arange(0,4000,duration)
    temp=np.hstack((temp,4000))
    print (temp)

else:
    sys.exit('4000 ms should be divisible by duration')


periods_=[]

for t, value in enumerate(temp[:-1]):
    
    periods_.append([value,temp[t+1]])

print (periods_)


   




