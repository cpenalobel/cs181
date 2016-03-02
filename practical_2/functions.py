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

TRAIN_DIR = "train"

call_set = set([])

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

def create_data_matrix(start_index=0, end_index=None, direc="train", verbose=False):
    X = None
    classes = []
    ids = [] 
    i = -1
    if not end_index:
        end_index = len(os.listdir(direc))
    for i, datafile in enumerate(os.listdir(direc)):
        if datafile == '.DS_Store':
            continue
        if datafile[0] == '.':
            datafile = datafile[2:]
        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break
        if verbose:
            print "\rNumber of datafiles loaded:", i+1,
            
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
        add_to_set(tree)
        this_row = call_feats(tree)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def call_feats(tree):
    good_calls = ['sleep', 'dump_line']

    call_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]

    return call_feat_array
    
#create train and test set
def split_mask(dftouse, split_size = 0.7):
    itrain, itest = train_test_split(xrange(dftouse.shape[0]), train_size=split_size)
    mask=np.ones(dftouse.shape[0], dtype='int')
    mask[itrain]=1
    mask[itest]=0
    mask = (mask==1)
    return mask