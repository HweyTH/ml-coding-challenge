"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import numpy as np
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

file_name = "clean_dataset.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return float(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":

    df = pd.read_csv(file_name)
    # print(df)

    # # Clean numerics

    df["Q7"] = df["Q7"].apply(to_numeric)
    df["Q8"] = df["Q8"].apply(to_numeric)
    df["Q9"] = df["Q9"].apply(to_numeric)

    # # Clean for number categories

    # df["Q1"] = df["Q1"].apply(get_number)

    # # Create area rank categories

    df["Q6"] = df["Q6"].apply(get_number_list_clean)

    temp_names = []
    for i in range(1,7):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))

    del df["Q6"]

    # # Create category indicators

    new_names = []
    for col in temp_names:
        temp_df = pd.DataFrame({col: range(1, 7)})
        temp_indicators = pd.get_dummies(temp_df[col], prefix=col)
        df[col] = pd.Categorical(df[col], categories=range(1, 7))
        indicators = pd.get_dummies(df[col], prefix=col)
        new_names.extend(indicators.columns)
        df = pd.concat([df, indicators], axis=1)
        del df[col]

    # # Create multi-category indicators

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    del df["Q5"]

    # # Prepare data for training - use a simple train/test split for now

    df = df[new_names + ["Q1","Q2","Q3","Q7","Q8","Q9", "Label"]]
    print(df)

    df = df.sample(frac=1, random_state=random_state)


    x = df.drop("Label", axis=1)
    x = x.astype(float)
    x.fillna(x.mean(), inplace=True)
    print(x.mean())
    np.savetxt("mean.csv", x.mean(), delimiter=",")
    x = x.values
    y = pd.get_dummies(df["Label"].values)
    y = y.astype(float)
    
    # nan_indices = np.isnan(x)
    # nan_rows = np.any(nan_indices, axis=1)
    # x_cleaned = x[~nan_rows]
    # y_cleaned = y[~nan_rows]
    # print(np.isnan(x_cleaned).any())
    # print(np.isnan(y_cleaned).any())
    
    # print(df)
    # print(x[0])

    n_train = 1200
    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    # # Train and evaluate classifiers

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf.fit(x_train, y_train)
    # train_acc = clf.score(x_train, y_train)
    # test_acc = clf.score(x_test, y_test)
    # print(f"{type(clf).__name__} train acc: {train_acc}")
    # print(f"{type(clf).__name__} test acc: {test_acc}")
    
    # lrm = LogisticRegression(fit_intercept=False)
    
    # print(df)
    # print(x_train)
    # print(np.sqrt(float(5.0)))
    # print(x_train[:,-3:])
    mean = x_train[:,-3:].mean(axis=0)
    print(mean)
    # print(mean)
    # print(np.isinf(x_train).any())
    std = x_train[:,-3:].std(axis=0)
    print(std)
    
    x_train_norm = x_train.copy()
    x_train_norm[:,-3:] = (x_train[:,-3:] - mean) / std
    
    y_idxmax = y_train.idxmax(axis=1)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_idxmax)
    
    x_test_norm = x_test.copy()
    x_test_norm[:,-3:] = (x_test[:,-3:] - mean) / std
    
    
    # print(x_train[0])
    # print(x_train_norm)
    # print(y_train)
    # print(y_train["Dubai"].values)
    lrm = LogisticRegression(max_iter=10000, multi_class='ovr')
    lrm.fit(x_train_norm,y_encoded)
    train_acc = lrm.score(x_train_norm,y_encoded)
    test_acc = lrm.score(x_test_norm, encoder.fit_transform(y_test.idxmax(axis=1)))
    print(f"{type(lrm).__name__} train acc: {train_acc}")
    print(f"{type(lrm).__name__} test acc: {test_acc}")
    # weights = lrm.coef_
    # print(weights)
    test_predictions = lrm.predict(x_test_norm)
    print(test_predictions)
    # weights_file = "logistic_regression_weights.csv"
    # np.savetxt(weights_file, weights, delimiter=",", header=",".join([f"Feature_{i}" for i in range(weights.shape[1])]))
    # print(f"Weights saved to {weights_file}")

    # mean_std = np.vstack([mean, std])
    # np.savetxt("mean_std.csv", mean_std, delimiter=",")