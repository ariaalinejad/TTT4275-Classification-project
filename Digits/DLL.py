from email import header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('', header=None)

C = 3
D = 4

W = np.empty([C,D+1])

for i in range(C):
    W[i] = np.random.normal(loc=0.0, scale=0.1, size=D+1)

z = np.empty([30,C])
for k in range(30):
    z[k]=W.dot(np.transpose(x[k]))

g = 1/(1+np.exp(-z))