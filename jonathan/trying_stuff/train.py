from sklearn.naive_bayes import GaussianNB # Not allowed for final model. (Only used for training weights.)
import numpy as np
import pandas as pd
import re
import itertools

RANDOM_STATE = 12

def make_vocab(data) -> list[str]:
    vocab = []
    for line in data:
        for word in line.split():
            if word not in vocab: vocab.append(word)
    return vocab

def make_bow(data, vocab) -> np.ndarray:
    bow = np.zeros([len(data), len(vocab)])
    d = {vocab[i] : i for i in range(len(vocab))}
    for i in range(len(data)):
        for word in data[i].split():
            if word.lower() in d: bow[i][d[word.lower()]] = 1
    return bow

def clean(answers) -> list[str]:
    clean_answers = []
    for line in answers:
        clean_answers.append(re.sub('\W+', ' ', str(line)).lower())
    return clean_answers

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

def normalize(n: float, mean: float, std: float):
    """
    Normalize a given number given the mean and standard deviation.
    """
    return (n - mean) / std

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

def parse_data(filename: str, vocab: list[str], q7_mean, q7_std, q8_mean, q8_std, q9_mean, q9_std) -> tuple[np.ndarray, np.ndarray]:
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
    # data["Q7"] = data["Q7"].apply(lambda s: normalize(s, q7_mean, q7_std))

    data["Q8"] = data["Q8"].apply(to_numeric).fillna(0)
    # data["Q8"] = data["Q8"].apply(lambda s: normalize(s, q8_mean, q8_std))

    data["Q9"] = data["Q9"].apply(to_numeric).fillna(0)
    # data["Q9"] = data["Q9"].apply(lambda s: normalize(s, q9_mean, q9_std))

    data["Q10"] = clean(data["Q10"])
    
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

    bow = pd.DataFrame(make_bow(data["Q10"], vocab), columns=vocab)
    # new_names.extend(bow.columns)  # use all vocab (this may overfit the training data)
    new_names.extend(vocab)  # only use select words
    data = pd.concat([data, bow], axis=1)
    del data["Q10"]
    
    # data = data[new_names + ["Q7", "Q8", "Q9", "Label"]]
    # data = data[new_names + ["Q7", "Q8", "Label"]]
    data = data[new_names + ["Q7", "Label"]]

    data = data.sample(frac=1, random_state=RANDOM_STATE)

    x = data.drop("Label", axis=1).values
    t = data["Label"]

    return x, t

if __name__ == "__main__":

    # Trying some vocab options
    vocab_options = [ 
        make_vocab(clean(pd.read_csv("trying_stuff/trainfile.csv")["Q10"])),
        ["dubai", "york", "rio", "paris"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "janeiro", "paris", "love", "eiffel"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "janeiro", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "janeiro","football", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "new", "york", "dreams", "rio", "de", "janeiro", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "de", "janeiro", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "de", "janeiro", "football", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "new", "york", "dreams", "rio", "de", "janeiro", "football", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "york", "dreams", "rio", "janeiro", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "janeiro", "paris", "love", "eiffel"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "de", "janeiro", "paris", "love", "eiffel"],
        ["dubai", "rich", "money", "habibi", "new", "york", "dreams", "rio", "de", "janeiro", "brazil", "paris", "love", "eiffel", "tower"],
        ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "de", "janeiro", "paris", "lights", "love", "eiffel", "tower"]
    ]

    max_vocab = []  # ['dubai', 'rich', 'money', 'habibi', 'york', 'dreams', 'rio', 'de', 'janeiro', 'paris', 'love', 'eiffel', 'tower']
    max_valid = 0

    # Maybe Normalize data? No.
    q7 = pd.read_csv("trying_stuff/trainfile.csv")["Q7"].apply(to_numeric).fillna(0)
    q8 = pd.read_csv("trying_stuff/trainfile.csv")["Q8"].apply(to_numeric).fillna(0)
    q9 = pd.read_csv("trying_stuff/trainfile.csv")["Q9"].apply(to_numeric).fillna(0)
    q7_mean, q7_std = np.mean(q7), np.std(q7)
    q8_mean, q8_std = np.mean(q8), np.std(q8)
    q9_mean, q9_std = np.mean(q9), np.std(q9)

    # Using validation set to treat which vocab words to use as a hyperparameter.
    for vocab in vocab_options:
        x_train, t_train = parse_data("trying_stuff/trainfile.csv", vocab, q7_mean, q7_std, q8_mean, q8_std, q9_mean, q9_std)
        x_valid, t_valid = parse_data("trying_stuff/validfile.csv", vocab, q7_mean, q7_std, q8_mean, q8_std, q9_mean, q9_std)
        gnb = GaussianNB()
        gnb.fit(x_train, t_train)
        valid_acc = gnb.score(x_valid, t_valid)
        if valid_acc > max_valid: max_vocab, max_valid = vocab, valid_acc

    # max_vocab = make_vocab(clean(pd.read_csv("ver04/trainfile.csv")["Q10"])) #  Accuracy using all vocab words.
    # max_vocab = ["dubai", "rich", "money", "habibi", "york", "dreams", "rio", "janeiro", "paris", "love", "eiffel", "tower"]

    print("vocab:", max_vocab)

    x_train, t_train = parse_data("trying_stuff/trainfile.csv", max_vocab, q7_mean, q7_std, q8_mean, q8_std, q9_mean, q9_std)
    x_valid, t_valid = parse_data("trying_stuff/validfile.csv", max_vocab, q7_mean, q7_std, q8_mean, q8_std, q9_mean, q9_std)
    x_test, t_test = parse_data("trying_stuff/testfile.csv", max_vocab, q7_mean, q7_std, q8_mean, q8_std, q9_mean, q9_std)

    gnb = GaussianNB()
    gnb.fit(x_train, t_train)
    # gnb.class_prior_ = np.array([0.25, 0.25, 0.25, 0.25])  # Manually set all class prior to be equally likely
    train_acc = gnb.score(x_train, t_train)
    valid_acc = gnb.score(x_valid, t_valid)
    test_acc = gnb.score(x_test, t_test)
    print(f"{type(gnb).__name__} train acc: {train_acc}")
    print(f"{type(gnb).__name__} valid acc: {valid_acc}")
    print(f"{type(gnb).__name__} test acc: {test_acc}")

    # print(gnb.classes_)
    # print(gnb.class_count_)
    # print(gnb.class_prior_.tolist())

    # print(gnb.theta_.tolist())  # Mean
    # print(gnb.var_.tolist())  # Variance
    