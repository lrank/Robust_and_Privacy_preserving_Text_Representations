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

dataset_path = "../../dataset/TrustPilot"
total_json_filename = "../../dataset/TrustPilot_processed/total_data.json"
train_json_filename = "../../dataset/TrustPilot_processed/train_data.json"
test_json_filename = "../../dataset/TrustPilot_processed/test_data.json"
valid_json_filename = "../../dataset/TrustPilot_processed/valid_data.json"

total_csv_filename = "../../dataset/TrustPilot_processed/total_data.csv"
train_csv_filename = "../../dataset/TrustPilot_processed/train_data.csv"
test_csv_filename = "../../dataset/TrustPilot_processed/test_data.csv"
valid_csv_filename = "../../dataset/TrustPilot_processed/valid_data.csv"


symbol_list = [",", ".", "-", "/", "[", "]", "?", "<", ">", "{", "}", "|", "\\", ":", ";", "'", "!", "@", "#", "$", "%",
               "_", "(", ")"]

# Train: Valid: Test = 8 : 1 : 1
Train_Valid_Test_Ratio = 0.8


# Load all the files and form train.json, test.json
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


def load_TrustPilot(filename, age, gender):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Data Cleaning
    clean_lines = data_cleaning(lines)

    words_list = []
    words_tuple = ()
    tags_list = []
    tags_tuple = ()
    sentence_dict = {}
    file_data = []

    # Load data
    for idx in range(len(clean_lines)):
        if clean_lines[idx] != "\n":
            word = clean_lines[idx].split("\t")[0]
            words_list.append(word)
            words_tuple = tuple(words_list)

            tag = clean_lines[idx].split("\t")[1]
            tags_list.append(tag.strip("\n"))
            tags_tuple = tuple(tags_list)
        if clean_lines[idx] == "\n":
            # sentence_dict["words"] = words_list
            # sentence_dict["tags"] = tags_list

            sentence_dict["words"] = words_tuple
            sentence_dict["tags"] = tags_tuple

            sentence_dict["age"] = age
            sentence_dict["gender"] = gender
            file_data.append(sentence_dict)

            # Clear words_list, tags_list and sentence_dict
            words_list = []
            tags_list = []
            sentence_dict = {}

    return file_data


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


def main():
    getAttributes = GetAttributes()

    total_file_data = []
    for file in os.listdir(dataset_path):
        print("file = ", file)

        age = getAttributes.getAge(file)
        print("age = ", age)

        gender = getAttributes.getGender(file)
        print("gender = ", gender)

        file_data = load_TrustPilot(os.path.join(dataset_path, file), age, gender)
        print("file_data = ", file_data)
        print("len(file_data) = ", len(file_data))

        total_file_data.append(file_data)
        print("total_file_data = ", total_file_data)
        print("len(total_file_data) = ", len(total_file_data))
        print("\n")


    total_data = []
    for item in total_file_data:
        # print("len(item) = ", len(item))
        for child_item in item:
            total_data.append(child_item)

    print("total_data = ", total_data)
    print("len(total_data) = ", len(total_data))
    print("total_data[0] = ", total_data[0])
    print("total_data[0].keys() = ", total_data[0].keys())
    print("\n")

    # Save total_data to json and csv
    save_to_json(total_data, total_json_filename)
    save_to_csv(total_data, total_csv_filename)

    # Shuffle data
    random.shuffle(total_data)

    # Split train_data into train_data, valid_data and test data
    train_data = total_data[:int(len(total_data) * Train_Valid_Test_Ratio)]
    valid_data = total_data[int(len(total_data) * Train_Valid_Test_Ratio):
                            int(len(total_data) * Train_Valid_Test_Ratio) + int(
                                len(total_data) * ((1 - Train_Valid_Test_Ratio) / 2)) + 1]
    test_data = total_data[int(len(total_data) * Train_Valid_Test_Ratio) + int(
        len(total_data) * ((1 - Train_Valid_Test_Ratio) / 2)) + 1:]

    print("len(train_data) = ", len(train_data))
    print("len(valid_data) = ", len(valid_data))
    print("len(test_data) = ", len(test_data))

    # Save to json file
    save_to_json(train_data, train_json_filename)
    save_to_json(test_data, test_json_filename)
    save_to_json(valid_data, valid_json_filename)

    # Save to csv file
    save_to_csv(train_data, train_csv_filename)
    save_to_csv(test_data, test_csv_filename)
    save_to_csv(valid_data, valid_csv_filename)


if __name__ == "__main__":
    main()