# coding=utf8
"""
@author: Yantong Lai
@date: 12/16/2019
"""

import numpy as np
import pandas as pd
import os
import argparse

from Pytorch.sentiment.utils import *

# File path
dataset_dir = "../../dataset/WWW2015_data/"
processed_langid_dir = "../../dataset/WWW2015_processed_langid_data/"
processed_fasttext_dir = "../../dataset/WWW2015_processed_fasttext_data"
processed_dataset_dir = "../../dataset/WWW2015_processed/"

file_path = "germany.auto-adjusted_gender.NUTS-regions.jsonl.tmp"
csv_filename = "germany_data.csv"

total_csv = os.path.join(processed_dataset_dir, "total.csv")
train_csv = os.path.join(processed_dataset_dir, "train.csv")
valid_csv = os.path.join(processed_dataset_dir, "valid.csv")
test_csv = os.path.join(processed_dataset_dir, "test.csv")


# Split ratio
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


# Random seed
SEED = 1234567


def load_and_preprocesss():
    """
    It is a function to load and pre-process datasets.
    """
    #####################################
    #         1. Load Dataset           #
    #####################################
    loadDataset = LoadDataset()
    review_list, rating_list, gender_list, location_list, age_list = loadDataset.load_file(dataset_dir + file_path)


    #####################################
    #      2. Data Pre-processing       #
    #####################################
    dataPreprocessing = DataPreprocessing()

    labeled_gender_list = dataPreprocessing.label_gender(gender_list)
    labeled_age_list = dataPreprocessing.label_age(age_list)
    assert len(review_list) == len(rating_list) == len(labeled_age_list) == len(labeled_gender_list) == len(
        location_list)

    # Check if there exists a directory to save processed files
    if not os.path.exists(processed_langid_dir):
        os.mkdir(processed_langid_dir)

    # Form csv files and save
    form_csv(review_list, rating_list, labeled_gender_list, labeled_age_list, location_list,
             processed_langid_dir + csv_filename)

    print("Write to csv successfully!\n")


    #####################################
    #     3. Language Double Check      #
    #####################################
    # Check if there exists a directory to save fasttext processed files
    if not os.path.exists(processed_fasttext_dir):
        os.mkdir(processed_fasttext_dir)

    for file in sorted(os.listdir(processed_langid_dir)):
        if file.endswith(".csv"):
            fasttext_language_detection(filename=os.path.join(processed_langid_dir, file),
                                        new_filename=os.path.join(processed_fasttext_dir, file))


def main():

    #####################################
    #  1. Load and Pre-process Dataset  #
    #####################################
    # load_and_preprocesss()


    #####################################
    #  2. Load and Pre-process Dataset  #
    #####################################
    split_train_valid_test(total_csv=total_csv, seed=SEED, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO,
                           train_csv=train_csv, valid_csv=valid_csv, test_csv=test_csv)


if __name__ == '__main__':

    main()