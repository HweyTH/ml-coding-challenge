import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
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

    def save_grid(self, node, filename):
        with open(filename, 'w') as f:
            self._save_node(node, f)

    def save_model(self, filename):
        with open(filename, 'w') as f:
            f.write(f"max_depth={self.max_depth}\n")
            f.write(f"min_samples_split={self.min_samples_split}\n")
            f.write("tree_structure:\n")
            self._save_node(self.root, f)
            
    def _save_node(self, node, f):
        if node is None:
            f.write("None\n")
            return
        f.write(f"Node: feature={node.feature}, threshold={node.threshold}, value={node.value}\n")
        if node.left:
            f.write("Left:\n")
            self._save_node(node.left, f)
        if node.right:
            f.write("Right:\n")
            self._save_node(node.right, f)
            
    def load_model(cls, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            params = {}
            tree_structure_start = False
            tree_structure_lines = []
            for line in lines:
                if line.strip() == "tree_structure:":
                    tree_structure_start = True
                    continue
                if not tree_structure_start:
                    key, value = line.strip().split('=')
                    params[key] = int(value) if value.isdigit() else value
                else:
                    tree_structure_lines.append(line.strip())
            root = cls._load_node(tree_structure_lines)
            return cls(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], tree_=root)

    @staticmethod
    def _load_node(lines):
        if lines[0] == "None":
            return None
        # Parse node from lines
        parts = lines[0].split(': ')[1].split(', ')
        feature = int(parts[0].split('=')[1])
        threshold = float(parts[1].split('=')[1])
        value = float(parts[2].split('=')[1])
        left_start = None
        left_end = None
        right_start = None
        for idx, line in enumerate(lines):
            if line == "Left:" and left_start is None:
                left_start = idx + 1
            elif line == "Right:" and right_start is None:
                left_end = idx
                right_start = idx + 1
        if right_start is None:
            left_end = len(lines)
        left = DecisionTree._load_node(lines[left_start:left_end])
        if right_start is not None:
            right = DecisionTree._load_node(lines[right_start:])
        else:
            right = None
        return Node(feature=feature, threshold=threshold, left=left, right=right, value=value)
