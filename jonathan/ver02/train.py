from sklearn.naive_bayes import GaussianNB # Not allowed for final model. (Only used for training weights.)
import numpy as np
import pandas as pd
from helpers import *

RANDOM_STATE = 10

def parse_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Given the name of a csv file, returns the data it contains as a vectorized array 
    of points, and an array of corresponding labels.
    """
    data = pd.read_csv(filename)

    data["Q1"] = data["Q1"].apply(get_number)
    data["Q2"] = data["Q2"].apply(get_number)
    data["Q3"] = data["Q3"].apply(get_number)
    data["Q4"] = data["Q4"].apply(get_number)

    data["Q6"] = data["Q6"].apply(get_number_list_clean)

    data["Q7"] = data["Q7"].apply(to_numeric).fillna(0)
    data["Q8"] = data["Q8"].apply(to_numeric).fillna(0)
    data["Q9"] = data["Q9"].apply(to_numeric).fillna(0)
    
    temp_names = []
    for i in range(1,7):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
    del data["Q6"]

    new_names = []
    for col in ["Q1", "Q2", "Q3", "Q4"] + temp_names:
        values = [-1, 1, 2, 3, 4, 5, 6] if col in temp_names else [-1, 1, 2, 3, 4, 5]
        indicators = pd.get_dummies(pd.Series(data[col], dtype=pd.CategoricalDtype(categories=values)), prefix=col)
        new_names.extend(indicators.columns)
        data = pd.concat([data, indicators], axis=1)
        del data[col]

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))
    del data["Q5"]

    # vocab = []
    # for line in clean(data["Q10"]):
    #     for word in line.split():
    #         if word.lower() not in vocab:
    #             vocab.append(word.lower())
    # indicators = pd.get_dummies(pd.Series(data['Q10'], dtype=pd.CategoricalDtype(categories=vocab)), prefix='Q10')
    # new_names.extend(indicators.columns)
    # data = pd.concat([data, indicators], axis=1)
    # del data['Q10']
    
    data = data[new_names + ["Q7", "Label"]]

    data = data.sample(frac=1, random_state=RANDOM_STATE)

    x = data.drop("Label", axis=1).values
    t = data["Label"]

    return x, t

if __name__ == "__main__":
    filename = "clean_dataset.csv"
    x, t = parse_data(filename)

    # TODO: remove outliers in data

    n_train = 1200

    x_train = x[:n_train]
    t_train = t[:n_train]

    x_test = x[n_train:]
    t_test = t[n_train:]

    gnb = GaussianNB()
    gnb.fit(x_train, t_train)
    train_acc = gnb.score(x_train, t_train)
    test_acc = gnb.score(x_test, t_test)
    print(f"{type(gnb).__name__} train acc: {train_acc}")
    print(f"{type(gnb).__name__} test acc: {test_acc}")

    # print(gnb.classes_)
    # print(gnb.class_count_)
    # print(gnb.class_prior_)

    # print(gnb.theta_.tolist())  # Mean
    # print(gnb.var_.tolist())  # Variance
    