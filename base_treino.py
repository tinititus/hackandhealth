#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:22:45 2018

@author: menon
"""
import pandas as pd


# Codigo para criar uma base de teste e treino que computadores comuns conseguem rodar
r=pd.read_csv('pequeno.txt', delimiter="|")
import random
values=  random.sample(range(len(r)), 1500)

c=[]
for i in values:
    c.append(r.iloc[i])
  
h=pd.DataFrame(c)
h.to_csv('base_treino.txt', sep="|")


