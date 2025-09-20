import numpy as np
import matplotlib.pyplot as plt

X=np.array([[1,2,3],
            [4,5,6],
            [1,4,2],
            [3,2,1],
            [5,4,3]])
Y=np.array([0,1,0,1,0])

W1=np.array([[0.1,0.2,0.3],
             [0.4,0.5,0.6],
             [0.7,0.8,0.9]])
b1=np.array([0.1,0.2,0.3])

W2=np.array([[0.1,0.2,0.3]])
b2=0.1

def forward(X):
    Z1=np.dot(X,W1.T)+b1
    A1=1/(1+np.exp(-Z1))
    Z2=np.dot(A1,W2.T)+b2
    A2=1/(1+np.exp(-Z2))
    return A1,A2,Z1,Z2

A1,A2,Z1,Z2=forward(X)
print(A2)
print(Z1)
print(Z2)
