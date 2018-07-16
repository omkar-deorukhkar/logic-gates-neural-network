import numpy as np

x = np.array([[0,0,0],
              [0,0,1],
              [0,1,0],
              [0,1,1],
              [1,0,0],
              [1,0,1],
              [1,1,0],
              [1,1,1]])
#x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[1,0,0,1,0,1,1,0]).T
#y = np.array([[0,0,1,1]]).T

def sig(x,der=False):
    op = 1/(1+np.exp(-x))
    if der==False:
        return op
    else:
        return x*(1-x)
        

np.random.seed(1)

syn0 = 2*np.random.random((3,1))-1

for it in range(200000):
    l0 = x
    l1 = sig(np.dot(l0,syn0))

    err = y - l1

    l1_del = err * sig(l1,True)

    syn0 += np.dot(l0.T,l1_del)
print("Output After Training")


l2=[]
for x in l1:
    if x < 0.6:
        l2.append(0)
    else:
        l2.append(1)
        
        

print(l2)
