from __future__ import print_function
from random import sample
import pandas as pd
import numpy as np

def sampleFiles(n, smple = None, cv=None):
    if not smple:
        smple = sample(range(1000),1000)
    for i, n_ in enumerate(smple[:n]):
        print (100*i/float(n), end="\r")
        data = pd.read_csv('edited_data/1024_features/train_1024_'+str(1000*(n_+1))+'.csv', index_col=0)
        data.index = range(1000*(n_-1),1000*n_)
        try:
            data_final = pd.concat([data_final, data])
        except UnboundLocalError:
            data_final = data
    if cv:
        return data_final, sampleFiles(cv, smple = smple[n:],)
    else:
        return data_final