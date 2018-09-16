#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 09:25:54 2018

@author: menon
"""


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


r=pd.read_csv('filtrado.txt', delimiter="|")
# tirando colunas vazias
r=r.dropna(subset=['base_hackaturing.data_nascimento'])




# Fazendo o encoder
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



# Pegando mesmo numero de observacao para as clases (teve ou nao diferenca de preco cobrado e pago)

c=[]
for i in range(len(r)):
    if r.iloc[i][24]!=r.iloc[i][25]:
        c.append(r.iloc[i])

import random
values=  random.sample(range(len(r)), 10000)


for i in values:
    c.append(r.iloc[i])
  
h=pd.DataFrame(c)
h.to_csv('pequeno.txt', sep="|")

