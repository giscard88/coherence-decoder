import numpy as np

a=[1,2,3]
b=[4,5,6]
A=[a,b]
B=[b,a]

D21=np.array(A)
D22=np.array(B)
print (D21)
print (D22)

All=[D21,D22]
All=np.array(All)

flat=All.flatten()

print (flat)

