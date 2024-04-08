import numpy as np
from collections import Counter
import pickle

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, root=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = root
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_classes == 1 or n_samples < self.min_samples_split:
            return Node(value=Counter(y).most_common(1)[0][0])

        # Find best split
        best_gain = 0
        best_feature = None
        best_threshold = None
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # Check if best feature and threshold are found
        if best_feature is None or best_threshold is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _information_gain(self, X, y, feature_idx, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return 0

        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        num_left = np.sum(left_indices)
        num_right = np.sum(right_indices)
        num_total = num_left + num_right

        child_entropy = (num_left / num_total) * left_entropy + (num_right / num_total) * right_entropy

        return parent_entropy - child_entropy

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Added small value to avoid log(0)
        return entropy
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
            
    
