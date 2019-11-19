# coding=utf8
"""
@author: Yantong Lai
@date: 11/15/2019
"""

import json
import csv
import numpy as np
import pandas as pd
import os
import ast
import random
random.seed(12345)

dataset_path = "../../dataset/TrustPilot/"
total_csv_filename = "../../dataset/TrustPilot_processed/total_data.csv"
train_csv_filename = "../../dataset/TrustPilot_processed/train_data.csv"
test_csv_filename = "../../dataset/TrustPilot_processed/test_data.csv"
valid_csv_filename = "../../dataset/TrustPilot_processed/valid_data.csv"


symbol_list = [",", ".", "-", "/", "[", "]", "?", "<", ">", "{", "}", "|", "\\", ":", ";", "'", "!", "@", "#", "$", "%",
               "_", "(", ")", "\n"]


class GetAttributes:
    """
    It is a class to load TrustPilot dataset and form train.json, test.json.
    For Age:
    - O45: 1
    - U35: 0

    For Gender:
    - M: 1
    - F: 0
    """

    def __init__(self):
        pass

    def getAge(self, filename):
        if "O45" in filename:
            return 1
        else:
            return 0

    def getGender(self, filename):
        if "-M.data" in filename:
            return 1
        else:
            return 0


def data_cleaning(lines):
    clean_lines = []
    for item in lines:
        if item[0] in symbol_list:
            continue
        else:
            clean_lines.append(item)
    return clean_lines


def load_data(filename, age, gender):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Data Cleaning
    clean_lines = data_cleaning(lines)

    word_list = []
    tag_list = []
    age_list = []
    gender_list = []
    for item in clean_lines:
        word = item.split("\t")[0]
        tag = item.split("\t")[1].strip("\n")
        word_list.append(word)
        tag_list.append(tag)
        age_list.append(age)
        gender_list.append(gender)

    arr = np.array((word_list, tag_list, age_list, gender_list)).T
    return arr


def load_TrustPilot(dataset_path):

    print("########## We are now loading TrustPilot! ##########")

    getAttributes = GetAttributes()
    data_list = []
    for file in os.listdir(dataset_path):

        if not file.startswith("en"):
            continue

        age = getAttributes.getAge(file)
        gender = getAttributes.getGender(file)
        print("file = {}".format(file))
        print("age = {}".format(age))
        print("gender = {}".format(gender))

        filename = os.path.join(dataset_path, file)
        file_arr = load_data(filename, age, gender)
        print("{}_arr.shape = {}\n".format(file, file_arr.shape))

        assert file_arr.shape[1] == 4
        data_list.append(file_arr)

    print("########## Loading Successfully! ##########\n")
    return np.concatenate(data_list)


def save_array_csv(arr, csv_filename):

    df = pd.DataFrame(data=arr, columns=['word', 'tag', 'age', 'gender'])
    return df.to_csv(csv_filename, index=False)


def save_to_json(content, json_filename):
    with open(json_filename, "w") as f:
        json.dump(content, f)


def load_json(json_filename):
    with open(json_filename, "r") as f:
        return json.load(f)


def save_to_csv(total_data, csv_filename):

    with open(csv_filename, 'w') as f:
        keys = total_data[0].keys()
        dict_writer = csv.DictWriter(f, keys)
        # Write dict keys
        # dict_writer.writeheader()
        dict_writer.writerows(total_data)