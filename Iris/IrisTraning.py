import pandas as pd
import numpy as np
import wheel
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import io




def main():
    #np.random.seed(1)

    C = 3
    D = 1
    alpha  = 0.01
    Ntrain = 30
    Ntest  = 20
    NtrainAll = Ntrain*3
    NtestAll = Ntest*3
    Niterations = 1000
    ConfMatrixTrain = np.zeros([C,C])
    ConfMatrixTest = np.zeros([C,C])
    AllMSE = []
    #W = 2*np.random.random((C,D+1))-1 # 3x5 matrix gives number between -1 and 1 
    W = np.zeros((C,D+1))
    #Predicted = np.zeros(90,3)

    xTrain, xTest, tTrain, tTest = splitData('iris.data', Ntrain, [0,1,2])
    #xTrain, xTest, tTrain, tTest = splitData('iris.data', Ntrain)

    for i in range(Niterations): # run 20 backpropagations

        zTrain, gTrain, MSETrain, GradMSETrain = findMSE(xTrain,tTrain, NtrainAll, C, W)

        W = W - alpha*GradMSETrain
        
        AllMSE.append(MSETrain)

    zTest, gTest, MSETest, GradMSETest = findMSE(xTest, tTest, NtestAll, C, W)

    # makes ConfMatrix for Training data
    for k in range(NtrainAll):
        #predicted[k][np.argmax(gTest[k])] = 1
        ConfMatrixTrain[np.argmax(tTrain[k])][np.argmax(gTrain[k])] += 1
    io.savemat('confmatrixTrain.mat', {"ConfmatrixTrain": ConfMatrixTrain })

    # makes ConfMatrix for Testing data
    for k in range(NtestAll):
        ConfMatrixTest[np.argmax(tTest[k])][np.argmax(gTest[k])] += 1
    io.savemat('confmatrixTest.mat', {"ConfmatrixTest": ConfMatrixTest })
    

    print ('Weighting matrix: \n')
    for line in W:
        print ('\n ', '   '.join(map(str, line)))

    print(f'Error rate(train): {round(findErrorRate(ConfMatrixTrain,NtrainAll, C),5)}% \n')
    print(f'Error rate(train): {round(findErrorRate(ConfMatrixTest,NtrainAll, C),5)}% \n')
    histogram('iris.data')
    #plt.show()
    
    #print(histogramList('iris.data', 0, 16))
    #plt.plot(AllMSE) 
    #plt.hist(histogramList('iris.data', 0, 16))
    #plt.show()
    
    #print(f'Error rate(test): {round(MSE_test/NtestAll,5)}%')


    #print(f'Shapes \n t: {t.shape} \n x: {x.shape} \n g: {g.shape} \n (g-t)*g: {((g-t)*g).shape} \n ((g-t)*g*(1-g)): {np.outer(((g[0]-t[0])*g[0]*(1-g[0])),x[0]).shape} \n GardMSE: {GradMSE.shape}')

    
def splitData(fileName, N, col = None):
    df = pd.read_csv(fileName, header = None)
    df[4] = 1 

    if (col != None):
        for i in col:
            df.drop(i, inplace=True, axis=1)
    
    #print(df)
    
    df_S    = df.iloc[   :50 ]
    df_Vc   = df.iloc[50 :100]
    df_Vg   = df.iloc[100:150]


    df_S_train  =  df_S.iloc[:N]
    df_S_test   =  df_S.iloc[N:]
    df_Vc_train = df_Vc.iloc[:N]
    df_Vc_test  = df_Vc.iloc[N:]
    df_Vg_train = df_Vg.iloc[:N]
    df_Vg_test  = df_Vg.iloc[N:]

    firstX      = np.concatenate((df_S_train,df_Vc_train,df_Vg_train), axis=0) #training data (90x5 matix)
    secondX     = np.concatenate((df_S_test,df_Vc_test,df_Vg_test), axis=0) #testing data

    firstTS     = np.tile(np.array([[1,0,0]]), (N, 1))
    firstTVc    = np.tile(np.array([[0,1,0]]), (N, 1))
    firstTVg    = np.tile(np.array([[0,0,1]]), (N, 1))
    firstT      = np.concatenate((firstTS,firstTVc,firstTVg),axis=0)                #90x3 matrix

    secondTS    = np.tile(np.array([[1,0,0]]), (50-N, 1))
    secondTVc   = np.tile(np.array([[0,1,0]]), (50-N, 1))
    secondTVg   = np.tile(np.array([[0,0,1]]), (50-N, 1))
    secondT     = np.concatenate((secondTS,secondTVc,secondTVg),axis=0) #60x3 matrix

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
        GradMSE_local += np.outer(((g_local[k]-t_local[k])*g_local[k]*(1-g_local[k])), x_local[k]) # 1x3 times 1x5 gives 3x5 matrix
    return z_local, g_local, MSE_local, GradMSE_local


def findErrorRate(ConfMatrixTrain, N, C):
    errorRate = 0
    for i in range(C):
        for j in range(C):
            if (i != j):
                errorRate += ConfMatrixTrain[i][j]/N
    return errorRate


def histogram(fileName):
    df = pd.read_csv(fileName, header = None)
    sepalLength = np.array(df[0])
    sepalWidth  = np.array(df[1])
    petalLength = np.array(df[2])
    petalWidth  = np.array(df[3])

    
    classType = [sepalLength, sepalWidth, petalLength, petalWidth]
    className = ['Sepal Length[cm]', 'Sepal Width [cm]', 'Petal length[cm]', 'Petal width[cm]']

    for i in range(4):
        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
        axs[0].hist(classType[i][:50], bins=16, color = "mediumseagreen")
        axs[0].set_title('Iris-setosa')
        axs[0].set_ylabel('samples per bin')
        axs[1].hist(classType[i][50:100], bins=16, color = "mediumaquamarine")
        axs[1].set_title('Iris-versicolor')
        axs[1].set_ylabel('samples per bin')
        axs[2].hist(classType[i][100:150], bins=16, color = "mediumturquoise")
        axs[2].set_title('Iris-virginica')
        axs[2].set_ylabel('samples per bin')
        axs[2].set_xlabel(className[i])

    #Find Overlap
    [overlapSepalLength, overlapSepalWidth, overlapPetalLength, overlapPetalWidth] = [0,0,0,0]
    overlap = [overlapSepalLength, overlapSepalWidth, overlapPetalLength, overlapPetalWidth]
    overlapPrint = ["Overlap Sepal Length: ", "Overlap Sepal Width: ", "Overlap Petal Length: ", "Overlap Petal Width: "]
    ranges = [(4,8), (2,4.5), (0.5,7),(0,2.5)]

    for i in range(4):
        counts1, _ = np.histogram(classType[i][0:50],    bins=16, range = ranges[i])
        counts2, _ = np.histogram(classType[i][50:100],  bins=16, range = ranges[i])
        counts3, _ = np.histogram(classType[i][100:150], bins=16, range = ranges[i])

        overlap[i] = np.sum(np.minimum(counts1,counts2))\
        + np.sum(np.minimum(counts1,counts3))\
        + np.sum(np.minimum(counts2,counts3))
        print(overlapPrint[i], overlap[i])

main()



