import pandas as pd
import numpy as np
import wheel
import matplotlib.pyplot as plt
from tabulate import tabulate

np.random.seed(1)

C = 3
D = 4

alpha  = 0.1
Ntrain = 30
Ntest  = 20
NtrainAll = 90
NtestAll = 60
AllMSE = []

df = pd.read_csv('iris.data', header = None)
df_S    = df.iloc[   :50 ]
df_Vc   = df.iloc[50 :100]
df_Vg   = df.iloc[100:150]


df_S_train  =  df_S.iloc[      :Ntrain]
df_S_test   =  df_S.iloc[Ntrain:      ]
df_Vc_train = df_Vc.iloc[      :Ntrain]
df_Vc_test  = df_Vc.iloc[Ntrain:      ]
df_Vg_train = df_Vg.iloc[      :Ntrain]
df_Vg_test  = df_Vg.iloc[Ntrain:      ]

x = np.concatenate((df_S_train,df_Vc_train,df_Vg_train), axis=0)
x.T[4] = 1
test_data  = np.concatenate((df_S_test,df_Vc_test,df_Vg_test), axis=0)
test_data.T[4]  = 1

tS = np.tile(np.array([[1,0,0]]), (Ntrain, 1))
tVc= np.tile(np.array([[0,1,0]]), (Ntrain, 1))
tVg= np.tile(np.array([[0,0,1]]), (Ntrain, 1))
t  = np.concatenate((tS,tVc,tVg),axis=0)

#W = 2*np.random.random((C,D+1))-1
W = np.random.random((C,D+1))

z = np.empty([NtrainAll,C])
g = np.empty([NtrainAll,C])

for i in range(20):
    GradMSE = 0 
    for k in range(90):
        z[k]=np.dot(W,x[k].T)

        for i in range(C):
            g[k][i] = 1/(1+np.exp(-z[k][i]))

        GradMSE += np.outer(((g[k]-t[k])*g[k]*(1-g[k])), x[k])
       

    W = W - alpha*GradMSE
    #print(f'Shapes \n t: {t.shape} \n x: {x.shape} \n g: {g.shape} \n (g-t)*g: {((g-t)*g).shape} \n ((g-t)*g*(1-g)): {np.outer(((g[0]-t[0])*g[0]*(1-g[0])),x[0]).shape} \n GardMSE: {GradMSE.shape}')
    #print(sum(sum(GradMSE)))
    MSE = 0
    for k in range(90):
        MSE += 0.5*np.dot((g[k]-t[k]).T,(g[k]-t[k]))
    AllMSE.append(MSE)

ConfMatrix = np.zeros([C,C])

for k in range(90):
    ConfMatrix[np.argmax(t[k])][np.argmax(g[k])] += 1
    print(t[k])
    print(g[k])

#print(AllMSE)

plt.plot(AllMSE) 
#plt.show()
print(ConfMatrix)

#Printing litt fint, men ikke fint nok
table = [['Event/Decition', 'Septosa', 'Versicolor', 'Virginica'], 
['Septosa', ConfMatrix[0][0],ConfMatrix[0][1] , ConfMatrix[0][2]], 
['Versicolor', ConfMatrix[1][0],ConfMatrix[1][1] , ConfMatrix[1][2]], 
['Virginica', ConfMatrix[2][0], ConfMatrix[2][1], ConfMatrix[2][2]]]
print(tabulate(table))



'''
print ('W \n')
for line in W:
    print ('\n ', '   '.join(map(str, line)))
'''
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