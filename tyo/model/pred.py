import numpy as np
import pandas as pd

from decision_tree import DecisionTree

import pandas as pd

import pandas as pd

def predict(x, tree):
    """
    Helper function to make prediction for a given input x using the decision tree.
    """
    # Traverse the decision tree until a leaf node is reached
    node = tree
    while node.left:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def predict_all(filename):
    """
    Make predictions for the data in filename using the decision tree model loaded from best_decision_tree_model.txt.
    """
    # Load the decision tree model
    decision_tree = DecisionTree.load_model('best_decision_tree_model.txt')

    # Read the file containing the test data using pandas
    data = pd.read_csv(filename)
    predictions = []
    for idx, row in data.iterrows():
        # Convert test example to a format compatible with decision tree prediction
        x = row.values.tolist()
        # Obtain a prediction for this test example
        pred = predict(x, decision_tree)
        predictions.append(pred)
    
    return predictions

# Example usage:
predictions = predict_all('test_data.csv')
print(predictions)

  

    

