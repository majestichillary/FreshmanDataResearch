# -*- coding: utf-8 -*-
"""
@author: majestichillary
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("imdb_top_1000.csv") 
#df = pd.read_csv(r'‪C://Users//18124//Downloads//imdb_top_1000.csv')

#x = df.groupby('Director').groups

#y = df["No_of_Votes"]

df.groupby('Director')['No_of_Votes'].sum()

#df.pivot(index = "Series_Title", columns = "Director", values = "No_of_Votes").plot(kind = "bar")

df = df.sort_values("No_of_Votes", ascending=False)

fig = plt.figure(figsize=(100,20))

ax1 = fig.add_subplot(1,2,1)

ax1.bar(df['Director'],df['No_of_Votes'])
ax1.set_xticklabels(df['Director'], rotation=60, horizontalalignment='right')

plt.show()
