from sklearn.naive_bayes import GaussianNB # Not allowed for final model. (Only used for training weights.)
import re  # Not allowed for final model. (Only used for training weights.)
import numpy as np
import pandas as pd
import sys, csv, random

"""
Question 1
From a scale 1 to 5, how popular is this city? (1 is the least popular and 5 is the most popular)

Question 2
On a scale of 1 to 5, how efficient is this city at turning everyday occurrences into potential viral moments on social media? (1 is the least efficient and 5 is the most efficient)

Question 3
Rate the city's architectural uniqueness from 1 to 5, with 5 being a blend of futuristic wonder and historical charm.

Question 4
Rate the city's enthusiasm for spontaneous street parties on a scale of 1 to 5, with 5 being the life of the celebration.

Question 5
If you were to travel to this city, who would be likely with you?
['Partner', 'Friends', 'Siblings', 'Co-worker']

Question 6
Rank the following words from the least to most relatable to this city. Each area should have a different number assigned to it. (1 is the least relatable and 6 is the most relatable)
(Skyscrapers, Sport, Art and Music, Carnival, Cuisine, Economic)

Question 7
In your opinion, what is the average temperature of this city over the month of January? (Specify your answer in Celsius)

Question 8
How many different languages might you overhear during a stroll through the city?

Question 9
How many different fashion styles might you spot within a 10-minute walk in the city?

Question 10
What quote comes to mind when you think of this city?
"""

RANDOM_STATE = 10

def to_numeric(s: str) -> float:
    """
    Converts string `s` to a float.
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

    data["Q7"] = data["Q7"].apply(to_numeric).fillna(0)
    data["Q8"] = data["Q8"].apply(to_numeric).fillna(0)
    data["Q9"] = data["Q9"].apply(to_numeric).fillna(0)

    data["Q1"] = data["Q1"].apply(get_number)

    data["Q6"] = data["Q6"].apply(get_number_list_clean)

    temp_names = []
    for i in range(1,7):
        col_name = f"rank_{i}"
        temp_names.append(col_name)
        data[col_name] = data["Q6"].apply(lambda l: find_area_at_rank(l, i))
    del data["Q6"]


    new_names = []
    for col in ["Q1"] + temp_names:
        indicators = pd.get_dummies(data[col], prefix=col)
        new_names.extend(indicators.columns)
        data = pd.concat([data, indicators], axis=1)
        del data[col]

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      data[cat_name] = data["Q5"].apply(lambda s: cat_in_s(s, cat))

    del data["Q5"]

    data = data[new_names + ["Q7", "Label"]]  # This statement emoves Q10 for now ... (TODO: maybe use Q10)

    data = data.sample(frac=1, random_state=RANDOM_STATE)

    x = data.drop("Label", axis=1).values
    # t = pd.get_dummies(data["Label"].values)
    t = data["Label"]

    # print(t.shape)
    # print(t)

    return x, t

# def bow():
#     """
#     """
#     ...

# def map(x, t):
#     """
#     """
#     x, t
#     ...

# def mle(x, t):
#     """
#     """
#     x, t
#     ...

# def predict(x) -> str:
#     """
#     """
#     ...

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

    # print(gnb.get_params())

    print(gnb.classes_)
    print(gnb.class_count_)
    print(gnb.class_prior_)

    # print(gnb.theta_.tolist())  # Mean
    # print(gnb.var_.tolist())  # Variance
    
