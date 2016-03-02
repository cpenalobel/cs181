# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression

import util

# to get features, need my added code to the 
# create_data_matrix function and the 
# Padding_zeros function below it

TRAIN_DIR = "train"
TEST_DIR = "test"

call_set = set([])
call_list = []

def create_data_matrix(start_index, end_index, direc):
    feature_vectors = []
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

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
        tree = ET.parse(os.path.join(direc, datafile))


        call_counter = {}

        for el in tree.iter():
            call = el.tag

            if call not in call_set:
                call_set.add(call)
                call_list.append(call)
       
            if call not in call_counter:
                call_counter[call] = 1
            else:
                call_counter[call] += 1

            call_feat_array = [0 for x in call_list]

            for i in range(len(call_list)):
                call = call_list[i]
                if call in call_counter:
                    call_feat_array[i] = call_counter[call]


        feature_vectors.append(call_feat_array)

    X = np.array(padding_zeros(feature_vectors))
        


    return X, np.array(classes), ids



def padding_zeros(vectors):
    for vector in vectors:
        n = len(vector)
        k = len(call_list)
        if n < k:
            zeros = [0] * (k - n)
            vector.extend(zeros)

    return vectors



## Feature extraction
def main():
    X_train, t_train, train_ids = create_data_matrix(0, 5, TRAIN_DIR)
    X_valid, t_valid, valid_ids = create_data_matrix(10, 15, TRAIN_DIR)
    X_test, t_test, test_ids = create_data_matrix(0, 15, TEST_DIR)
    
    print 'Data matrix (training set):'
    print X_train
    print 'Classes (training set):'
    print t_train

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X_train, t_train)
    preds = logreg.predict(X_test, t_test)
    write_to_file("submissions/LogReg", preds)