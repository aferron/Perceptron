import numpy as np

# NOT

def perceptron():
    for data in range(nData):
        for n in range(N):
            activation[data][n] = 0
            for m in range(M+1):
                activation[data][n] += weight[m][n] * inputs[data][m]
        
        if activation[data][n] > 0:
            activation[data][n] = 1
        else:
            activation[data][n] = 0
    


# number of datapoints (0 and 1 are the possible inputs so it's 2) 
nData = 2

# number of nodes (bias node + one node for 0 or 1, so this is 2)
N = 2

# max number of times to run through the data
M = 100 


input = ([1,0], [1,1])

# starting weights
weightN1 = -0.05
weightN2 = -0.02

# weight array. this will be a 2-d array
weights = [(0,0) for i in range(M)]
weights[0] = (weightN1, weightN2)

# activation 
dotProducts = [(0 for i in range(M))] 
output = [(0 for i in range(M))] 

# run it one time on the first datapoint (1,0)
# the answer should be true

# get the dot product
dotProducts[0] = np.dot(weights[0],input[0])
print(dotProduct)


