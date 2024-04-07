import re
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# ignore syntax warning
import warnings
warnings.filterwarnings('ignore')

file_name = "/Users/hwey/Desktop/repo_thaigia/CSC311H5S/ML Coding Challenge/clean_dataset.csv"
RANDOM_STATE = 42

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
    
    data = data[new_names + ["Q7", "Label"]]

    data = data.sample(frac=1, random_state=RANDOM_STATE)

    x = data.drop("Label", axis=1).values
    t = data["Label"]

    return x, t

if __name__ == "__main__":
    # parse data from csv file
    x, y = parse_data(file_name)
   
    # split data into training and test sets
    n_train = 1200
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]

    # train Sklearn MLPClassifier with the given data
    model = MLPClassifier()
    model.fit(x_train, y_train)

    # report validation accuracy
    print(model.score(x_test, y_test))

    # extract sklearn model parameters
    W1 = model.coefs_[0]
    W2 = model.coefs_[1]
    b1 = model.intercepts_[0]
    b2 = model.intercepts_[1]

    # report weights and parameters used in sklearn model 
    print(W1.tolist())
    print(W2.tolist())
    print(b1.tolist())
    print(b2.tolist())




    

    
    
    



