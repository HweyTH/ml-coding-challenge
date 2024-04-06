import re
import numpy as np
import math
import pandas as pd

file_name = "clean_dataset.csv"
random_state = 42
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = X.shape[0]
    h = sigmoid(np.dot(X, weights))
    epsilon = 1e-5  # To prevent log(0)
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def compute_gradient(X, y, weights):
    m = X.shape[0]
    h = sigmoid(np.dot(X, weights))
    gradient = np.dot(X.T, (h - y)) / m
    return gradient


def gradient_descent(X, y, num_iterations, learning_rate):
    weights = np.zeros(X.shape[1])
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, weights)
        weights = weights - learning_rate * gradient
        # if i % 1000 == 0:
        #     print(f"Iteration {i}, Cost: {compute_cost(X, y, weights)}")
    return weights

def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return probabilities


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

def process_data(df):
        df = pd.read_csv(file_name)
        df["Q7"] = df["Q7"].apply(to_numeric)
        df["Q8"] = df["Q8"].apply(to_numeric)
        df["Q9"] = df["Q9"].apply(to_numeric)
        df["Q6"] = df["Q6"].apply(get_number_list_clean)

        temp_names = []
        for i in range(1,7):
            col_name = f"rank_{i}"
            temp_names.append(col_name)
            df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))
        del df["Q6"]

        new_names = []
        for col in temp_names:
            temp_df = pd.DataFrame({col: range(1, 7)})
            temp_indicators = pd.get_dummies(temp_df[col], prefix=col)
            df[col] = pd.Categorical(df[col], categories=range(1, 7))
            indicators = pd.get_dummies(df[col], prefix=col)
            print(indicators)
            new_names.extend(indicators.columns)
            df = pd.concat([df, indicators], axis=1)
            del df[col]
        
        for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
            cat_name = f"Q5{cat}"
            new_names.append(cat_name)
            df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))
        del df["Q5"]

        return df[new_names + ["Q1","Q2","Q3","Q7","Q8","Q9", "Label"]]

if __name__ == "__main__":
    df = pd.read_csv(file_name)
    df = process_data(df)
    
    #randomize data
    df = df.sample(frac=1, random_state=random_state)
    
    #split data
    x = df.drop("Label", axis=1)
    x = x.astype(float)
    x.fillna(x.mean(), inplace=True)
    x = x.values
    y = pd.get_dummies(df["Label"].values)
    y = y.astype(float)
    
    #split data set
    n_train = 1200
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_test = x[n_train:]
    y_test = y[n_train:]
    
    #normalize data
    mean = x_train[:,-3:].mean(axis=0)
    std = x_train[:,-3:].std(axis=0)
    x_train_norm = x_train.copy()
    x_train_norm[:,-3:] = (x_train[:,-3:] - mean) / std
    x_test_norm = x_test.copy()
    x_test_norm[:,-3:] = (x_test[:,-3:] - mean) / std
    
    #dubai model
    models = {}
    for city in ["Dubai", "New York City", "Paris", "Rio de Janeiro"]:
        y_train_city = y_train[city].values
        models[city] = gradient_descent(x_train_norm, y_train_city, 5000, 0.1)
        
    print(models)
    # Export models to CSV
    weights_data = []
    for city, weights in models.items():
        weights_data.append(weights)
    weights_data = np.array(weights_data)
    print(weights_data)
    np.savetxt("weights.csv", weights_data, delimiter=",", header=",".join([f"Feature_{i}" for i in range(weights_data.shape[1])]))
    
    train_probabilities = np.zeros((x_train_norm.shape[0], len(models)))
    for i, city in enumerate(models):
        weights = models[city]
        train_probabilities[:, i] = predict(x_train_norm, weights)
    train_predictions = np.argmax(train_probabilities, axis=1)
    y_train_indices = np.argmax(y_train.values, axis=1)
    training_accuracy = (train_predictions == y_train_indices).mean()
    # print(f"Training Accuracy: {training_accuracy}")
        
    test_probabilities = np.zeros((x_test_norm.shape[0], len(models)))
    for i, city in enumerate(models):
        # print(x_test_norm)
        # print(weights)
        weights = models[city]
        test_probabilities[:, i] = predict(x_test_norm, weights)
    predictions = np.argmax(test_probabilities, axis=1)
    print(predictions)
    # print(predictions)
    y_test_indices = np.argmax(y_test.values, axis=1)
    accuracy = (predictions == y_test_indices).mean()
    print(f"Test Accuracy: {accuracy}")  