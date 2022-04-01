import pandas as pd
import numpy as np
import wheel
import matplotlib.pyplot as plt

np.random.seed(1)

C = 3
D = 4
alpha  = 0.01
Ntrain = 30
Ntest  = 20
NtrainAll = 90
NtestAll = 60
AllMSE = np.zeros(20)

df = pd.read_csv('iris.data', header = None)
df_S    = df.iloc[   :50 ]
df_Vc   = df.iloc[50 :100]
df_Vg   = df.iloc[100:150]

df_S_train  = df.iloc[      :Ntrain]
df_S_test   = df.iloc[Ntrain:      ]
df_Vc_train = df.iloc[      :Ntrain]
df_Vc_test  = df.iloc[Ntrain:      ]
df_Vg_train = df.iloc[      :Ntrain]
df_Vg_test  = df.iloc[Ntrain:      ]

train_data = np.concatenate((df_S_train,df_Vc_train,df_Vg_train), axis=0)
train_data.T[4] = 1
test_data  = np.concatenate((df_S_test,df_Vc_test,df_Vg_test),    axis=0)
test_data.T[4]  = 1

x = np.array(train_data)

tS = np.tile(np.array([[1,0,0]]), (Ntrain, 1))
tVc= np.tile(np.array([[0,1,0]]), (Ntrain, 1))
tVg= np.tile(np.array([[0,0,1]]), (Ntrain, 1))
t  = np.concatenate((tS,tVc,tVg),axis=0)
#t = np.array([[tVc,tS,tVg]]) Ble feil tror jeg 

#W = np.empty([C,D+1])
#gir vektlegger de forskjellige x-verdiene ulikt ut fra W
#burde itterere gjennom matrisen og ta en rad om gangen
#for i in range(C):
#    W[i] = np.random.normal(loc=0.0, scale=0.1, size=D+1)

W = 2*np.random.random((C,D+1))-1

z = np.empty([NtrainAll,C])

for i in range(20):
    for k in range(NtrainAll):
        z[k]=W.dot(np.transpose(x[k]))


    #funksjonen som tar verdiene fra z og putter det inn i en kalsse ut fra verdien
    #g = np.empty([C,D+1])

    g = 1/(1+np.exp(-z))

    #for i in range(NtrainAll):
    #    MSE = MSE + ((g-t)*g*(1-g))*np.transpose(x)
    MSE = sum(np.dot(((g-t)*g*(1-g)).T, x)) # byttet så den første greia er transposa i stede for x, vet ikke hvorfor d funka hehe

    #for i in range(100): -- tror denne heller skal dekke hele dritten
    W = W - alpha*MSE
        #print(W)
    AllMSE = np.append(AllMSE, AllMSE)



plt.plot(AllMSE) #prøvde å plotte utviklingen av MSE, men gikk ikke helt hehe
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