"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from model2 import DecisionTree
from sklearn.model_selection import KFold
file_name = "clean_dataset.csv"
random_state = 42
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

    data = data.sample(frac=1, random_state=42)

    x = data.drop("Label", axis=1).values
    t = data["Label"].values

    return x, t


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
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":

    x, y = parse_data("clean_dataset.csv")
   
    n_train = 1200
    
    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]
    
    max_depth_values = [5, 7, 10, 20]
    min_samples_split_values = [2, 5, 10, 15, 20]

    # Define number of folds for cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits)

    best_accuracy = 0
    best_params = {}

    # Iterate over parameter combinations
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            accuracy_sum = 0
            
            # Iterate over cross-validation folds
            for train_index, val_index in kf.split(x_train):
                # Split data into training and validation sets
                x_train_cv, X_val_cv = x_train[train_index], x_train[val_index]
                y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
                
                # Create and fit the decision tree model with current parameters
                dt_classifier = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
                dt_classifier.fit(x_train_cv, y_train_cv)
                
                # Evaluate the model on the validation set
                y_val_pred = dt_classifier.predict(X_val_cv)
                accuracy = np.mean(y_val_pred == y_val_cv)
                accuracy_sum += accuracy
            
            # Calculate average accuracy across folds
            average_accuracy = accuracy_sum / n_splits
            
            # Check if current parameter combination gives better accuracy
            if average_accuracy > best_accuracy:
                best_accuracy = average_accuracy
                best_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split}
                
    print("Best parameters:", best_params)
    print("Best accuracy:", best_accuracy)
    
    best_dt_classifier = DecisionTree(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
    best_dt_classifier.fit(x_train, y_train)
    best_dt_classifier.save_model('best_tree_model.txt')
    

        

