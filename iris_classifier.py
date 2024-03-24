from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle as pkl

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a LogisticRegression¶
clf = LogisticRegression()
clf.fit(X, y)


pkl.dump(clf, open('iris_classifier.pkl', 'wb'))
