from email import header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) #Setter de tilfeldige tallene til å være de samme hver gang (bra for testing)

############################Kode jeg hentet fra en forelesning
# activation function
def relu(x):   #Tror den kanskje skrur av noen noder slik at vi bruker mindre tid
    return (x > 0) * x # returns x if  x > 0 
                    # returns 0 otherwise

def relu2deriv(output):
    return output>0 # returns 1 for input > 0
                    # returns 0 otherwise

#eller tror begge func. ene bare gjør negative verdier til 0 

hidden_size = 4

W_0_1 = 2*np.random.random((3,hidden_size)) - 1 #må sette dimensjonene på hver av W-ene til å sammenfalle med den neste
W_1_2 = 2*np.random.random((hidden_size,1)) - 1 # disse tallene må justeres


layer_0 = .... # de x-verdiene liksom 
layer_1 = relu(np.dot(layer_0,W_0_1))
layer_3 = np.dot(layer_1,W_1_2)

#vi vil starte ytterst og justere oss innover fra utgangen

layer_2_delta = (layer_2 - x)
layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)
W_1_2 -= alpha * layer_0.T.dot(layer_2_delta)
W_0_1 -= alpha * layer_1.T.dot(layer_1_delta)
############################
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

'''
- Finne ut av det med activation func.
- Hvor mange ledd skal vi ha?
- Hvor mange noder skal vi ha på hvert ledd?
- Hvilken verdi skal alpha ha? 
- Hvor mange gjennomganger skal vi ha av systemet?
- Hvordan skal vi dele inn dataen for testing og trening?
'''
