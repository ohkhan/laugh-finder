from file_feature_extraction import parse_audio_files
from file_feature_extraction import one_hot_encode
import os

"""
File tree should look something like:

/python-training:
| - train.py
.
.
.
| - /Data:
    | - /training:
        | - XXXX-0-XXXX.wav
        .
        .
        .
"""

parent_dir = 'Data' 
sub_dirs= ['training'] # name of folder containing data

features, labels = parse_audio_files(parent_dir,sub_dirs)

labels = one_hot_encode(labels)

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

