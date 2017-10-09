import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def read_files(train_path, dev_path, test_path, add_delimeter = False):
    """
    Read all files from disk
    """
    train = read_file(train_path, add_delimeter)
    dev = read_file(dev_path, add_delimeter)
    test = read_file(test_path, add_delimeter)
    return train, dev, test


def read_file(file_path, add_delimeter):
    """
    Reads a data file from disk
    :param file_path:
    :return: lists of sentence_1, sentence_2, labels
    """
    labels = []
    data_s1 = []
    data_s2 = []
    infile = open(file_path, 'rt')
    for line in infile:
        #line = line.decode('utf-8').strip()
        line = line.strip()
        if line.startswith('-'):
            continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = items[1].lower()
        sentence2 = items[2].lower()
        if add_delimeter:
            sentence2 = "_ " + sentence2
        labels.append(int(label))
        data_s1.append(sentence1)
        data_s2.append(sentence2)
    infile.close()
    print('finished reading', file_path)
    return data_s1, data_s2, labels
