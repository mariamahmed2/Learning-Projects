import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class SimpleDecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        predictions = np.ones(X.shape[0])

        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values > self.threshold] = -1

        return predictions

class AdaBoostClassifier:
    def __init__(self, n_estimators=5):
        self.n_estimators = n_estimators
        self.weak_learners = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            stump = SimpleDecisionStump()
            lowest_error = float('inf')

            # Try every feature and every unique threshold
            for feature in range(n_features):
                values = X[:, feature]
                unique_thresholds = np.unique(values)

                for threshold in unique_thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[values < threshold] = -1

                    misclassified = weights[y != predictions]
                    error = np.sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < lowest_error:
                        lowest_error = error
                        stump.feature_index = feature
                        stump.threshold = threshold
                        stump.polarity = polarity

            EPS = 1e-10
            stump.alpha = 0.5 * np.log((1.0 - lowest_error) / (lowest_error + EPS))

            # Update sample weights
            predictions = stump.predict(X)
            weights *= np.exp(-stump.alpha * y * predictions)
            weights /= np.sum(weights)

            self.weak_learners.append(stump)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])

        for stump in self.weak_learners:
            final_predictions += stump.alpha * stump.predict(X)

        return np.sign(final_predictions)

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == '__main__':
    # Load dataset and prepare labels
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    y[y == 0] = -1  # Convert labels from (0, 1) to (-1, 1)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train AdaBoost classifier
    model = AdaBoostClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    acc = compute_accuracy(y_test, predictions)

    print(f'Model Accuracy: {acc:.4f}')

