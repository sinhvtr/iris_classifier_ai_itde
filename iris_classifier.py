from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)


pkl.dump(clf, open('iris_classifier.pkl', 'wb'))
