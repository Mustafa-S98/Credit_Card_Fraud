from pandas import read_csv as read
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split as data_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

dataset = read("creditcard.csv")

# dataset.hist()
# plt.show()

fraud = dataset[dataset['Class'] == 1]
valid = dataset[dataset['Class'] == 0]

outliers = len(fraud) / float(len(valid))

# print(outliers)
# print("Fraud cases : {}".format(len(fraud)))
# print("Valid cases : {}".format(len(valid)))

corr_mat = dataset.corr()
# print(corr_mat)

# fig = plt.figure()
# sns.heatmap(corr_mat)
# plt.show()

Y = dataset['Class']

columns = [c for c in dataset.columns.tolist() if c not in ['Class']]
X = dataset[columns]

# print(X[0 : 100])
# print(Y[0 : 100])

classifiers = {
    "IsolationForest" : IsolationForest(random_state = 1, contamination = outliers, max_samples = len(X)),

    "LocalOutlierFactor" : LocalOutlierFactor(n_neighbors = 20, contamination = outliers)
}

for (name, model) in classifiers.items():
    
