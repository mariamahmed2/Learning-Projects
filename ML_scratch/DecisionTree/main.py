import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

def entropy(y):
    """Compute entropy of label array y."""
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stop splitting if stopping conditions met
        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            leaf_value = self._majority_vote(y)
            return TreeNode(value=leaf_value)

        feature_indices = np.random.choice(num_features, self.n_features, replace=False)

        # Find the best split
        best_feat, best_thresh = self._best_split(X, y, feature_indices)

        # Create child nodes
        left_indices, right_indices = self._split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return TreeNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _majority_vote(self, y):
        most_common = Counter(y).most_common(1)[0][0]
        return most_common

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = entropy(y)
        left_indices, right_indices = self._split(feature_column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_indices), len(right_indices)
        entropy_left = entropy(y[left_indices])
        entropy_right = entropy(y[right_indices])
        weighted_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        return parent_entropy - weighted_entropy

    def _split(self, feature_column, threshold):
        left_indices = np.argwhere(feature_column <= threshold).flatten()
        right_indices = np.argwhere(feature_column > threshold).flatten()
        return left_indices, right_indices

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == '__main__':
    # Load dataset
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy(y_test, y_pred):.4f}')

