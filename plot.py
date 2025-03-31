# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:08:00 2025

@author: 10449
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("D:\Msc\Thesis\Res\Continental_unet.csv")
df = df[df['epoch']<=40]
plt.plot(df['epoch'], df['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim(0.03, 0.06)
plt.show()