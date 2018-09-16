#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sat Sep 15 16:09:32 2018

@author: menon
"""
import pandas as pd


r=pd.read_csv('hackaturing.dsv', delimiter="|")

r.head()

# Mask para retirar apenas valores importantes pra rede      
h=r.mask(r.eq('None')).dropna(subset=['base_hackaturing.valor_cobrado','base_hackaturing.valor_pago'])
h.to_csv('filtrado.txt', sep='|')