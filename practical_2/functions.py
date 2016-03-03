# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from sklearn.cross_validation import train_test_split
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np

import util

def create_data_matrix(direc, size=None, verbose=False, call_list = []):
    X_ = []
    classes = []
    ids = []
    if not size:
        size = len(os.listdir(direc))
    dfs_store = True
    for i, datafile in enumerate(os.listdir(direc)):
        if datafile == '.DS_Store':
            dfs_store = False
            size += 1
            continue 
        if i == size:
            break
        if verbose:
            print "\rNumber of datafiles loaded:", i+dfs_store,            
        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label       
            assert clazz == "X"
            classes.append(-1)
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        X_.append(call_feats(tree, call_list, direc))
    X = np.array(padding_zeros(X_, call_list))
    return X, np.array(classes), ids, call_list

def call_feats(tree, call_list, direc):
    call_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_list and direc == "train":
            call_list.append(call) 
        if call not in call_counter:
            call_counter[call] = 1            
        else:
            call_counter[call] += 1
        call_feat_array = [call_counter[c] if c in call_counter else 0 for c in call_list]
    return call_feat_array
    
#create train and test set
def split_mask(dftouse, split_size = 0.7):
    itrain, itest = train_test_split(xrange(dftouse.shape[0]), train_size=split_size)
    mask=np.ones(dftouse.shape[0], dtype='int')
    mask[itrain]=1
    mask[itest]=0
    mask = (mask==1)
    return mask
    
def padding_zeros(vectors, call_list):
    return [v + [0]*(len(call_list) - len(v)) for v in vectors]
    
def write_to_file(filename, pred_ids, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for pred_id, pred in zip(pred_ids, predictions):
            f.write(str(pred_id) + "," + str(pred) + "\n")