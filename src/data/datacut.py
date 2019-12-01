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
DATA_PATH = 'data'

def data_partition(file = DATA_PATH, output = DATA_PATH, ratio = 0.3):
    _files = os.listdir(file)
    return len(files)

def main():
    n_files  =  data_partition()
    print(n_files)

if __name__ == "__main__":
    main()