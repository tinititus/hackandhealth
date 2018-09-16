#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 07:06:37 2018

@author: menon
"""


from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np



import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
r=pd.read_csv('base_treino.txt', delimiter="|")

def resgata_dados(df, coluna):
    dados = list(df[coluna].unique())
    return(dados)



#Hot enconding of the variables

sexo_la = LabelEncoder()
r['sexo_encoded'] = sexo_la.fit_transform(r['base_hackaturing.sexo'])
sexo_ohe = OneHotEncoder()
X_sexo = sexo_ohe.fit_transform(r.sexo_encoded.values.reshape(-1,1)).toarray()

tipo_guia_la = LabelEncoder()
r['tipo_guia_encoded'] = tipo_guia_la.fit_transform(r['base_hackaturing.tipo_guia'])
tipo_guia_ohe = OneHotEncoder()
X_tipo_guia = tipo_guia_ohe.fit_transform(r.tipo_guia_encoded.values.reshape(-1,1)).toarray()

tipo_item_la = LabelEncoder()
r['tipo_item_encoded'] = tipo_item_la.fit_transform(r['base_hackaturing.tipo_item'])
tipo_item_ohe = OneHotEncoder()
X_tipo_item = tipo_item_ohe.fit_transform(r.tipo_item_encoded.values.reshape(-1,1)).toarray()

carater_atendimento_la = LabelEncoder()
r['carater_atendimento_encoded'] = carater_atendimento_la.fit_transform(r['base_hackaturing.carater_atendimento'])
carater_atendimento_ohe = OneHotEncoder()
X_carater_atendimento = carater_atendimento_ohe.fit_transform(r.carater_atendimento_encoded.values.reshape(-1,1)).toarray()

# Pegando a idade do paciente
anos = []
#'base_hackaturing.data_nascimento'
for i in range(len(r.index)):
    data1 = datetime.strptime(r.loc[i][8], "%Y-%m-%d").date() #Converte input em data no formato "aaaa-mm-dd", o parâmetro "%d/%m/%Y" retorna erro caso o usuário digite fora desse formato, mas não o transforma.
    data2 = datetime.strptime(r.loc[i][16], "%Y-%m-%d %H:%M:%S.0").date()
    data1 = data1.toordinal() #Convertendo em dias
    data2 = data2.toordinal() #Convertendo em dias
    dias = abs(data1 - data2) #Diferenca em dias
    anos.append(dias / 365)

# Valor cobrado
valor_cobrado = []
for i in range(len(r.index)):
    valor_cobrado.append(r.loc[i][25])

quantidade = []
for i in range(len(r.index)):
    quantidade.append(r.loc[i][23])

print('etapa1')
x1=np.matrix(X_carater_atendimento)
x2=np.matrix(X_sexo)
x3=np.matrix(X_tipo_guia)
x4=np.matrix(X_tipo_item)
x5=np.matrix(anos)
x6=np.matrix(quantidade)
x7=np.matrix(valor_cobrado)


X=np.concatenate((x1,x2),axis=1)
X=np.concatenate((X,x3),axis=1)
X=np.concatenate((X,x4),axis=1)
X=np.concatenate((X,np.transpose(x5)),axis=1)
X=np.concatenate((X,np.transpose(x6)),axis=1)
X=np.concatenate((X,np.transpose(x7)),axis=1)

Y=[]
for i in range(len(r)):
    if r.iloc[i][24]!=r.iloc[i][25]:
        Y.append(0)
    else:
        Y.append(1)
Y=np.transpose(Y)

print('etapa2')


# Criando o modelo
model = Sequential()
model.add(Dense(25, input_dim=12, kernel_initializer='normal', activation='relu'))
model.add(Dense(12, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
# Compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

import random
train_values=  random.sample(range(0, len(r)), int(len(r)*0.8))


# Normalizando o sistema
scaler = StandardScaler()
scaler.fit(X)
x_proc=scaler.transform(X)
y_proc=Y

# PCA para reduzir a dimensionalidade para acima de 90% de variabilidade
pca = PCA(n_components=12, svd_solver='full')
pca.fit(x_proc)                 
x_pca=pca.transform(x_proc)
print('etapa3')

# Criando teste e treino
trainX=[]
trainY=[]
testX=[]
testY=[]
for i in range(len(x_pca)):
    tag=0
    for j in range(len(train_values)):
        if train_values[j]==i:
            tag=1
            trainX.append(x_pca[i])
            trainY.append(y_proc[i])
    if tag!=1:
        testX.append(x_pca[i])
        testY.append(y_proc[i])        


trainX=np.matrix(trainX)
trainY=np.matrix(trainY)
trainY=np.transpose(trainY)
testX=np.matrix(testX)
testY=np.matrix(testY)
testY=np.transpose(testY)



print('etapa4')
model.fit(trainX.reshape((1200,12)),trainY.reshape((1200,1)),verbose=1, batch_size=200,epochs=3)
score = model.evaluate(testX.reshape((300,12)), testY.reshape((300,1)), batch_size=15)
print('Test_loss: '+str(score[0]))
print('Test_accuracy: '+str(score[1]))