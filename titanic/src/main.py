# This script implements a predictive model
# for the passengers that survive or not

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train=pd.read_csv('../data/train.csv')
test=pd.read_csv('../data/test.csv')


train.plot(kind='scatter', x='Survived', y='Fare')
plt.show()