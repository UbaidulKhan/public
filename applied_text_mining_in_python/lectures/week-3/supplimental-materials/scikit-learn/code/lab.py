"""
 Video: https://www.youtube.com/watch?v=i_LwzRVP7bg
 Data:  https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
"""
 
"""
-------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

"""
Add column headings
"""

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", 
        "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()


"""
Convert class/label(G) to integer - 1, otherwise 0
"""
df["class"] = (df["class"] == "g").astype(int)
print(df.head())



"""
For all classes that are 1, pull all the colunns and map each column 
"""
for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()