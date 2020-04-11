import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv file and put it into dataframe
data = pd.read_csv('../iris/iris.csv')
print(data.head())

# remove unnecessary columns from the dataframe and split features and the target
X = data.drop(axis=1, columns=['Id', 'Species'])
y = data['Species']

# label encode the target, b/c it's originally a string value
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# lets train some models
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

clf_dt = DecisionTreeClassifier()
clf_nb = GaussianNB()
clf_svc = SVC()
clf_knn = KNeighborsClassifier()

# classfiers = [clf_dt, clf_nb, clf_svc, clf_knn]

# parameter_grid = [
#     {
#         'max_depth': [None, 2, 3, 4],
#         'criterion': ['gini', 'entropy']
#     },
#     {},
#     {
#         'C': [0.03, 0.05, 0.07, 0.1],
#         'kernel': ['linear', 'poly', 'rbf']
#     },
#     {
#         'n_neighbors': [2, 3, 4, 5, 6] 
#     }
# ]

# optimized_classifiers = []

# # find the optimal hyper parameters for each classifer
# for index, clf in enumerate(classfiers):
#     optimized_clf = GridSearchCV(
#         clf,
#         param_grid=parameter_grid[index],
#         scoring='balanced_accuracy',
#         cv=20
#     )
#     # fit the classifier and measure it's performance
#     optimized_clf.fit(X, y)
#     performance = {
#         'classifier': clf.__class__.__name__,
#         'best_params_': optimized_clf.best_params_,
#         'best_score_': optimized_clf.best_score_
#     }
#     print("performance: ",  performance)
#     optimized_classifiers.append(optimized_clf)

# use the decision tree classifier
clf_dt = DecisionTreeClassifier()

# split train and test
from sklearn.model_selection import train_test_split
X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2)

# fit on training
clf_dt.fit(X_t, y_t)
# test on validation
y_v_pred = clf_dt.predict(X_v)

# evaluate via confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_v, y_v_pred)
print(cm)

# plot tree
from sklearn.tree import export_graphviz
export_graphviz(
    clf_dt,
    out_file='tree.dot',
    feature_names=X.columns,
    class_names=data['Species'].unique(),
    filled=True,
    rounded=True,
    leaves_parallel=True,
    impurity=False,
    label='all'
)