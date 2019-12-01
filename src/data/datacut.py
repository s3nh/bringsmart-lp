"""
Data preprocessing, divide folder 
into valid and traind dataset, 
based on folder files 
"""
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os 
import sys 
import random
import shutil

DATA_PATH = 'data'

def data_partition(file = DATA_PATH, output = DATA_PATH, ratio = 0.3):
    _files = os.listdir(file)
    n_files = range(len(_files))
    train_ixes = random.sample(n_files, int(ratio * len(_files)))
    test_ixes = [el for el in n_files if el not in train_ixes]
    train_data = [_files[ix] for ix in train_ixes] 
    test_data = [_files[ix] for ix in test_ixes]
    return train_data, test_data 


def data_rename(train_data, test_data, path = DATA_PATH):
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')
    ren_train_data = [os.path.join(train_path, el ) for el in train_data]
    ren_test_data = [os.path.join(test_path, el) for el in test_data]
    return ren_train_data, ren_test_data

def main():
    train_data, test_data  =  data_partition(file=DATA_PATH)
    os.makedirs(os.path.join(DATA_PATH, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_PATH, 'test') , exist_ok=True)
    ren_train_data, ren_test_data = data_rename(train_data, test_data, path=DATA_PATH)  
    for el in range(len(train_data)):
        shutil.move(os.path.join('data', train_data[el]), ren_train_data[el])
    for el in range(len(test_data)):
        shutil.move(os.path.join('data', test_data[el]), ren_test_data[el])


if __name__ == "__main__":
    main()