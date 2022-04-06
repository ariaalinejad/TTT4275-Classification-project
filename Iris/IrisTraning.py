import pandas as pd
import numpy as np
import wheel
import matplotlib.pyplot as plt
from tabulate import tabulate




def main():
    np.random.seed(1)

    C = 3
    D = 4
    alpha  = 0.01
    Ntrain = 30
    Ntest  = 20
    NtrainAll = Ntrain*3
    NtestAll = Ntest*3
    ConfMatrixTrain = np.zeros([C,C])
    ConfMarixTest = np.zeros([C,C])
    AllMSE = []
    W = 2*np.random.random((C,D+1))-1 # 3x5 matrix gives number between -1 and 1 

    xTest, xTrain, tTest, tTrain = splitData('iris.data', Ntest)
    #xTrain, xTest, tTrain, tTest = splitData('iris.data', Ntrain)

    for i in range(1000): # run 20 backpropagations

        zTrain, gTrain, MSETrain, GradMSETrain = findMSE(xTrain,tTrain, NtrainAll, C, W)

        W = W - alpha*GradMSETrain
        
        AllMSE.append(MSETrain)

    zTest, gTest, MSETest, GradMSETest = findMSE(xTest, tTest, NtestAll, C, W)

    # makes ConfMatrix for Training data
    for k in range(NtrainAll):
        ConfMatrixTrain[np.argmax(tTrain[k])][np.argmax(gTrain[k])] += 1
    tableTrain = [['Event/Decition', 'Septosa', 'Versicolor', 'Virginica'], 
    ['Septosa', ConfMatrixTrain[0][0],ConfMatrixTrain[0][1] , ConfMatrixTrain[0][2]], 
    ['Versicolor', ConfMatrixTrain[1][0],ConfMatrixTrain[1][1] , ConfMatrixTrain[1][2]], 
    ['Virginica', ConfMatrixTrain[2][0], ConfMatrixTrain[2][1], ConfMatrixTrain[2][2]]]

    # makes ConfMatrix for Testing data
    for k in range(NtestAll):
        ConfMarixTest[np.argmax(tTest[k])][np.argmax(gTest[k])] += 1
    tableTest = [['Event/Decition', 'Septosa', 'Versicolor', 'Virginica'], 
    ['Septosa',    ConfMarixTest[0][0],ConfMarixTest[0][1], ConfMarixTest[0][2]], 
    ['Versicolor', ConfMarixTest[1][0],ConfMarixTest[1][1], ConfMarixTest[1][2]], 
    ['Virginica',  ConfMarixTest[2][0],ConfMarixTest[2][1], ConfMarixTest[2][2]]]
    
    
    '''print ('Weighting matrix: \n')
    for line in W:
        print ('\n ', '   '.join(map(str, line)))
    print('ConfmatrixTrain train: \n'+tabulate(tableTrain))
    print('ConfmatrixTrain test: \n'+tabulate(tableTest))
    print(f'Error rate(train): {round(findErrorRate(ConfMatrixTrain,NtrainAll, C),5)}% \n')
    print(f'Error rate(train): {round(findErrorRate(ConfMarixTest,NtrainAll, C),5)}% \n')'''

    print(histogramList('iris.data', 0, 16))
    #plt.plot(AllMSE) 
    #plt.hist(histogramList('iris.data', 0, 16))
    #plt.show()
    
    #print(f'Error rate(test): {round(MSE_test/NtestAll,5)}%')


    #print(f'Shapes \n t: {t.shape} \n x: {x.shape} \n g: {g.shape} \n (g-t)*g: {((g-t)*g).shape} \n ((g-t)*g*(1-g)): {np.outer(((g[0]-t[0])*g[0]*(1-g[0])),x[0]).shape} \n GardMSE: {GradMSE.shape}')

    #Vi må så finne gradienten av MSE, og endre W (vektleggingen)
    #ut til å bevege seg i motsatt rettning av den, slik at avstandne til streken blir mer
    #alpha bestemmer hvor mye W endres for hver gang





def splitData(fileName, N):
    df = pd.read_csv(fileName, header = None)
    df[4] = 1 

    df_S    = df.iloc[   :50 ]
    df_Vc   = df.iloc[50 :100]
    df_Vg   = df.iloc[100:150]

    '''firstData =  []#np.zeros([N*3,5])
    secondData = []#np.zeros([(50-N)*3,5])
    
    for i in [df_S, df_Vc, df_Vg]:
        firstData.extend(i.iloc[:N])
        secondData.extend(i.iloc[N:])
        #firstData[:N*(index+1)]       = (i.iloc[:N])
        #secondData[:(50-N)] = (i.iloc[N:])
    
    firstData = np.array(firstData)
    secondData = np.array(secondData)'''

    df_S_train  =  df_S.iloc[:N]
    df_S_test   =  df_S.iloc[N:]
    df_Vc_train = df_Vc.iloc[:N]
    df_Vc_test  = df_Vc.iloc[N:]
    df_Vg_train = df_Vg.iloc[:N]
    df_Vg_test  = df_Vg.iloc[N:]

    firstX = np.concatenate((df_S_train,df_Vc_train,df_Vg_train), axis=0) #training data (90x5 matix)
    secondX  = np.concatenate((df_S_test,df_Vc_test,df_Vg_test), axis=0) #testing data

    firstTS = np.tile(np.array([[1,0,0]]), (N, 1))
    firstTVc= np.tile(np.array([[0,1,0]]), (N, 1))
    firstTVg= np.tile(np.array([[0,0,1]]), (N, 1))
    firstT      = np.concatenate((firstTS,firstTVc,firstTVg),axis=0)                #90x3 matrix

    secondTS  = np.tile(np.array([[1,0,0]]), (50-N, 1))
    secondTVc = np.tile(np.array([[0,1,0]]), (50-N, 1))
    secondTVg = np.tile(np.array([[0,0,1]]), (50-N, 1))
    secondT = np.concatenate((secondTS,secondTVc,secondTVg),axis=0) #60x3 matrix

    return firstX, secondX, firstT, secondT


def findMSE(x_local, t_local, N, C, W):
    z_local = np.empty([N,C]) # 90x3 matrix
    g_local = np.empty([N,C]) # 90x3 matrix
    MSE_local = 0
    GradMSE_local = 0 
    for k in range(N):
        z_local[k]=np.dot(W,x_local[k].T)   # 3x90 matrix

        g_local[k]=1/(1+np.exp(-z_local[k]))

        MSE_local += 0.5*np.dot((g_local[k]-t_local[k]).T,(g_local[k]-t_local[k]))
        GradMSE_local += np.dot(((g_local[k]-t_local[k])*g_local[k]*(1-g_local[k])).reshape((3,1)), x_local[k].reshape((1,5))) # 1x3 times 1x5 gives 3x5 matrix
    return z_local, g_local, MSE_local, GradMSE_local


def findErrorRate(ConfMatrixTrain, N, C):
    errorRate = 0
    for i in range(C):
        for j in range(C):
            if (i != j):
                errorRate += ConfMatrixTrain[i][j]/N
    return errorRate
    



def histogramList(fileName, featureIndex, NBin):
    df = pd.read_csv(fileName, header = None)
    df[4] = 1 
    print(df)

    histList = pd.cut(df[1], bins='16')
    print(histList)
    
    
    binWidth = (max(df.T[featureIndex]) - min(df.T[featureIndex]))/NBin
    #print(f'binwidth: {binWidth}')
   
    #histList = pd.Series(np.zeros(NBin))
    #print(f'histList: {histList}')
    for i in range(NBin):
        #histList[i] = df[binWidth*i < df[featureIndex] < binWidth*(i+1)]
        #histList[i] = df[df[featureIndex] < 4]
        print(f'histList i: {histList[i]}')
    
    return np.array(histList)



main()



'''import pandas as pd
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



print ('W \n')
for line in W:
    print ('\n ', '   '.join(map(str, line)))

#Vi må så finne gradienten av MSE, og endre W (vektleggingen)
#ut til å bevege seg i motsatt rettning av den, slik at avstandne til streken blir mer
#alpha bestemmer hvor mye W endres for hver gang
#



Til neste gang:
- fiks t slik at det blir en lang vektor med de tre kategoriene etter hverandre
- sett sammen x slik at det er en lang vektor med alle verdiene nedover 
- sjekk over koden slik at alt henger sammen med at vi bruker alle tre kategorier
- test kode/finn ut hvordan vi skal bruke test data
- x er training

'''

