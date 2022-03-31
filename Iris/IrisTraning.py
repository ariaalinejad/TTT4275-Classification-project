import pandas as pd
import numpy as np
import wheel
import matplotlib.pyplot as plt

C = 3
D = 4
alpha  = 0.1
Ntrain = 30
Ntest  = 20

df = pd.read_csv('iris.data', header = None)
df_Setosa     = df.iloc[   :50 ]
df_Versicolor = df.iloc[50 :100]
df_Virginica  = df.iloc[100:150]

df_Setosa_train     = df.iloc[      :Ntrain]
df_Setosa_test      = df.iloc[Ntrain:      ]
df_Versicolor_train = df.iloc[      :Ntrain]
df_Versicolor_test  = df.iloc[Ntrain:      ]
df_Virginica_train  = df.iloc[      :Ntrain]
df_Virginica_test   = df.iloc[Ntrain:      ]

df_Versicolor_train[4] = 1


x = np.array(df_Versicolor_train)
#t = np.array([1,0,0])

tVc=np.tile(np.array([[1,0,0]]), (30, 1))
tS =np.tile(np.array([[0,1,0]]), (30, 1))
tVg = np.tile(np.array([[0,0,1]]), (30, 1))
#t = np.array([[tVc,tS,tVg]]) Ble feil tror jeg 
#print(t)


#for i in range(Ntrain):
#    t[i] = t




W = np.empty([C,D+1])
#gir vektlegger de forskjellige x-verdiene ulikt ut fra W
#burde itterere gjennom matrisen og ta en rad om gangen
for i in range(C):
    W[i] = np.random.normal(loc=0.0, scale=0.1, size=D+1)

z = np.empty([Ntrain,C])
for k in range(Ntrain):
    z[k]=W.dot(np.transpose(x[k]))


#funksjonen som tar verdiene fra z og putter det inn i en kalsse ut fra verdien
#g = np.empty([C,D+1])

g = 1/(1+np.exp(-z))

for i in range(30):
    MSE = MSE + ((g-t)*g*(1-g))*np.transpose(x)

for i in range(100):
    W = W - alpha*MSE

print(z)

plt.plot(z)
plt.show()




#Vi må så finne gradienten av MSE, og endre W (vektleggingen)
#ut til å bevege seg i motsatt rettning av den, slik at avstandne til streken blir mer
#alpha bestemmer hvor mye W endres for hver gang
#


'''
Til neste gang:
- fiks t slik at det blir en lang vektor med de tre kategoriene etter hverandre
- sett sammen x slik at det er en lang vektor med alle verdiene nedover 
- sjekk over koden slik at alt henger sammen med at vi bruker alle tre kategorier
- test kode/finn ut hvordan vi skal bruke test data
- x er training

'''