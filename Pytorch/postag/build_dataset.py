# coding=utf8
"""
@author: Yantong Lai
@date: 11/19/2019
"""

import numpy as np
import pandas as pd
import os
import random
SEED = 12345
random.seed(SEED)

from Pytorch.postag.utils import *


# Filename
dataset_path = "../../dataset/TrustPilot/"

total_csv_filename = "../../dataset/TrustPilot_processed/total.csv"
train_csv_filename = "../../dataset/TrustPilot_processed/train.csv"
valid_csv_filename = "../../dataset/TrustPilot_processed/valid.csv"
test_csv_filename = "../../dataset/TrustPilot_processed/test.csv"

# Ratio
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def split_train_valid_test(total_arr):

    # Shuffle data
    random.shuffle(total_arr)

    # Split train_data into train_data, valid_data and test data
    train_arr = total_arr[:int(len(total_arr) * TRAIN_RATIO), :]
    valid_arr = total_arr[int(len(total_arr) * TRAIN_RATIO): int(len(total_arr) * TRAIN_RATIO) + int(len(total_arr) * VALID_RATIO), :]
    test_arr = total_arr[int(len(total_arr) * TRAIN_RATIO) + int(len(total_arr) * VALID_RATIO):, :]

    return train_arr, valid_arr, test_arr


def build_dataset(dataset_path, total_csv, train_csv, valid_csv, test_csv):

    # 1. Load TrustPilot dataset
    total_array = load_TrustPilot(dataset_path)

    # 2. Save total_array to csv
    save_array_csv(total_array, total_csv)

    # 3. Split train_data, valid_data and test_data
    train_array, valid_array, test_array = split_train_valid_test(total_array)
    
    # 4. Save train_array, valid_array, test_array to csv
    save_array_csv(train_array, train_csv)
    save_array_csv(valid_array, valid_csv)
    save_array_csv(test_array, test_csv)


build_dataset(dataset_path, total_csv_filename, train_csv_filename, valid_csv_filename, test_csv_filename)