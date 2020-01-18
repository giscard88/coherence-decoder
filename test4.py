import numpy as np



def lin2Array(arg,grid=44):
    row=int(arg/grid)
    col=int(arg) % int(grid)

    return (row,col)

def Array2lin(args,grid=44):
    arg=args[0]*grid+args[1]

    return arg




A=[[5,3,1],[2,4,9],[8,6,8]]


A=np.array(A)
print ('original\n',A)

A=A.reshape(9)
print ('flatten\n',A)
C=A.reshape((3,3))

print ('reconstruction\n',C)


idx=np.argsort(A)

idx_r=idx[::-1]

for xi, xin in enumerate(idx_r):
    row,col=lin2Array(xin,grid=3)
    print (C[row,col]) 
