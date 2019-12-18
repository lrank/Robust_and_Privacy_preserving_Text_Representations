# coding=utf8
"""
@author: Yantong Lai
@date: 12/13/2019
"""

import ast
import pandas as pd
import numpy as np
import langid
import fasttext


FASTTEXT_MODEL_PATH = "model/lid.176.bin"


class LoadDataset:

    def __init__(self):
        pass

    def str_to_dict(self, strdict):
        """
        It is a function change <Str> to <Dict>
        :param strdict: <Str>, like a dict, "{key: value}"
        :return: real <Dict>, {key, value}
        """
        try:
            if isinstance(strdict, str):
                return ast.literal_eval(strdict)
        except Exception as e:
            print(e)
            return None

    def load_file(self, filename):
        """
        It is a function to read file.
        :param filename: filename
        :return: <List> all file lines
        """

        # 1. Read file
        with open(filename, 'r') as f:
            lines = f.readlines()

        # 2. Change str to dict
        review_list = []
        rating_list = []
        age_list = []
        gender_list = []
        location_list = []

        count = 0
        for line in lines:
            line2dict = self.str_to_dict(line)
            print("No. {}".format(lines.index(line)))

            # 3. Extract review, rating, age, sex and loc
            if 'birth_year' in line2dict.keys() and line2dict['birth_year'] is not None \
                    and 'gender' in line2dict.keys() and line2dict['gender'] is not None \
                    and 'reviews' in line2dict.keys() and line2dict['reviews'] is not None:

                reviews = line2dict['reviews']
                gender = line2dict['gender']
                location = filename.split("/")[-1].split(".")[0]
                birth = line2dict['birth_year']
                for item in reviews:
                    if item['text'] is None:
                        print("item['text'] is None")
                        continue

                    # 4. Check reviews if written in English
                    try:
                        check_en_res = langid.classify(item['text'][0])

                        if check_en_res[0] == "en":
                            review_list.append(item['text'][0])
                            rating_list.append(item['rating'])
                            gender_list.append(gender)
                            location_list.append(location)
                            age_list.append(birth)

                            print("count = {}".format(count + 1))
                            count += 1
                            print("len(review_list) = {}\n".format(len(review_list)))

                            if count == 10000:
                                break

                            if len(review_list) == len(rating_list) == len(age_list) == len(gender_list) == len(location_list) == 10000:
                                break
                        else:
                            continue

                    except IndexError:
                        continue

            else:

                continue

        return review_list, rating_list, gender_list, location_list, age_list


class DataPreprocessing:

    def __init__(self):
        pass

    def check_en(self, review_list):
        pass

    def label_age(self, age_list):
        """
        It is a function to label age.
        If birth year is smaller than 1960, like 1950, label 1; else label 0
        :param age_list: <List> age_list
        :return: <List> labeled_age_list
        """
        labeled_age_list = []
        for age in age_list:
            if isinstance(age, int):
                if int(age) < 1960:
                    labeled_age_list.append(1)
                else:
                    labeled_age_list.append(0)

        return labeled_age_list

    def label_gender(self, gender_list):
        """
        It is a function to label gender.
        If male, label 1; female, label 0
        :param gender_list: <List> gender_list
        :return: <List> labeled_gender_list
        """
        labeled_gender_list = []
        for gender in gender_list:
            if isinstance(gender, str):
                if gender == "M":
                    labeled_gender_list.append(1)
                else:
                    labeled_gender_list.append(0)

        return labeled_gender_list


def form_csv(review_list, rating_list, labeled_gender_list, labeled_age_list, location_list, filename):
    """
    It is a function to form pd.DataFrame.
    :param review_list: review_list
    :param rating_list: rating_list
    :param labeled_gender_list: labeled_gender_list
    :param labeled_age_list: labeled_age_list
    :param location_list: location_list
    :param filename: csv filename
    :return: csv file
    """
    data = np.array((review_list, rating_list, labeled_gender_list, labeled_age_list, location_list)).T
    df = pd.DataFrame(data=data)

    return df.to_csv(filename, index=False, header=None)


def fasttext_language_detection(filename, new_filename):
    """
    It is a function to check language with FastText

    Step 1: Download pre-trained models lid.176.bin via https://fasttext.cc/docs/en/language-identification.html
    :param filename: filename
    :return: checked file
    """
    df = pd.read_csv(filename, header=None)
    reviews = df[0].tolist()
    rating_list = df[1].tolist()
    labeled_gender_list = df[2].tolist()
    labeled_age_list = df[3].tolist()
    location_list = df[4].tolist()

    # Load FastText pre-trained model
    model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    checked_reviews = []
    checked_rating_list = []
    checked_gender_list = []
    checked_age_list = []
    checked_location_list = []
    for idx in range(len(reviews)):
        pred = model.predict(reviews[idx].strip("\n"))
        if pred[0][0].replace("__label__", "") == "en" and float(pred[1][0]) >= 0.9:
            checked_reviews.append(reviews[idx].strip("\n"))
            checked_rating_list.append(rating_list[idx])
            checked_gender_list.append(labeled_gender_list[idx])
            checked_age_list.append(labeled_age_list[idx])
            checked_location_list.append(location_list[idx])

    # Save to a new csv file
    return form_csv(checked_reviews, checked_rating_list, checked_gender_list, checked_age_list, checked_location_list,
                    new_filename)


def split_train_valid_test(total_csv, seed, train_ratio, valid_ratio, train_csv, valid_csv, test_csv):
    """
    It is a function to split total dataset into train dataset, valid dataset and test dataset.
    :param total_filename: total dataset filename
    :return: train dataset file, valid dataset file and test dataset file.
    """
    df_total = pd.read_csv(total_csv)

    # Copy df_total
    df = df_total

    # Shuffle the total dataframe
    df = df.apply(np.random.RandomState(seed=seed).permutation, axis=0)

    # Split df_total into df_train, df_valid and df_test
    df_train = df.iloc[:int(len(df) * train_ratio), :]
    df_valid = df.iloc[int(len(df) * train_ratio): (int(len(df) * train_ratio) + int(len(df) * valid_ratio)), :]
    df_test = df.iloc[(int(len(df) * train_ratio) + int(len(df) * valid_ratio)):, :]

    # Save df into csv
    df_train.to_csv(train_csv, index=False, header=None)
    df_valid.to_csv(valid_csv, index=False, header=None)
    df_test.to_csv(test_csv, index=False, header=None)


