import pandas as pd
import numpy as np
import re

def sigmoid(z):
    with np.errstate(over='ignore'):  # Ignore overflow warnings
        result = 1 / (1 + np.exp(-z))
        if np.any(np.isinf(result)):  # Check if any element of result is infinite
            result[np.isinf(result)] = 0.0  # Set infinite values to 0
        return result

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
    return [int(n) for n in re.findall("(\\d+)", str(s))]


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

def process_data(df, mean_std_file, mean_file):
    # Read mean and standard deviation
    mean_std_df = pd.read_csv(mean_std_file)
    mean_norm = mean_std_df.values[0]
    std_norm = mean_std_df.values[1]

    # Read mean values
    mean_df = pd.read_csv(mean_file, header=None)
    df.fillna(mean_df, inplace=True)

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
    for n in temp_names:
        print(df[n])

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
    
    print(new_names)
    
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        cat_name = f"Q5{cat}"
        new_names.append(cat_name)
        df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))
    del df["Q5"]

    # Use mean and std from provided files

    ret = df[new_names + ["Q1","Q2","Q3","Q7","Q8","Q9"]].astype(float)
    mean_values_columns = ret.columns[-3:]
    print(mean_values_columns)
    ret[mean_values_columns] = (ret[mean_values_columns] - mean_norm) / std_norm
    return ret

def predict_all(filename):
    
    # Load data
    df = pd.read_csv(filename)
    # df = df.sample(frac=1, random_state=42).iloc[-2:]
    print(list(df["Q6"]))

    # Process data
    processed_df = process_data(df, "mean_std.csv", "mean.csv")
    processed_values = processed_df.values
    
    print(processed_df)

    # Load weights
    weights_df = pd.read_csv("weights.csv")
    weights = weights_df.values
    
    #create predictions
    test_probabilities = np.zeros((processed_values.shape[0], len(weights)))
    print(processed_df)
    for i, model_weight in enumerate(weights):
        test_probabilities[:, i] = sigmoid(np.dot(processed_values, model_weight))
        
    #Use argmax (ovr) model
    predictions = np.argmax(test_probabilities, axis=1)
    
    city_map = {0: "Dubai", 1: "New York City", 2: "Paris", 3: "Rio de Janeiro"}
    mapped_predictions = [city_map[prediction] for prediction in predictions]
    return mapped_predictions

# if __name__ == "__main__":
#     predictions = predict_all("clean_dataset.csv")
#     print("Predictions:", predictions)
    
#     # Load clean dataset
#     clean_df = pd.read_csv("clean_dataset.csv").sample(frac=1, random_state=42).iloc[-2:]
#     true_labels = clean_df["Label"].values
    
#     # Calculate accuracy
#     correct_predictions = sum(1 for pred, true_label in zip(predictions, true_labels) if pred == true_label)
#     total_samples = len(true_labels)
#     accuracy = correct_predictions / total_samples
#     print("Accuracy:", accuracy)

