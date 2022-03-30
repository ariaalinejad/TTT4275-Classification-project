import pandas as pd
import wheel

df = pd.read_csv('iris.data', header = None)
df_Setosa     = df.iloc[:50]
df_Versicolor = df.iloc[50:100]
df_Virginica  = df.iloc[100:150]

df_Setosa_train     = df.iloc[:30]
df_Setosa_test      = df.iloc[30:]
df_Versicolor_train = df.iloc[:30]
df_Versicolor_test  = df.iloc[30:]
df_Virginica_train  = df.iloc[:30]
df_Virginica_test   = df.iloc[30:]

#print(df_Setosa)
print(df_Versicolor)
#print(df_Virginica)
#print("Shape of new dataframes - {} , {} , {}".format(df_Setosa.shape, df_Versicolor.shape, df_Versicolor.shape))