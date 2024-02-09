import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv('Twitterdata/featureVectors.csv')
y = X['word.Tag']
X.drop('word.Tag', axis=1, inplace=True)
X = X.astype('float32')
y = y.astype('float32')
X = np.nan_to_num(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
dtc = DecisionTreeClassifier(max_depth=32, class_weight={0:1, 1:1})
gnb = GaussianNB()
clf = RandomForestClassifier(max_depth=10, class_weight={0:1, 1:2})
dtc.fit(X_train, y_train)
gnb.fit(X_train, y_train)
clf.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
target_names = ['I-Loc', 'B-Org', 'I-Per', 'Other', 'B-Per', 'I-Org', 'B-Loc']
print("Results for Decision tree..")
print(classification_report(y_test, y_pred, target_names=target_names))
score = f1_score(y_pred, y_test, average='weighted')
print("Decision Tree F1 score: {:.2f}".format(score))
print("Results for Naive Bayes...")
y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))
score = f1_score(y_pred, y_test, average='weighted')
print("Naive Bayes F1 score: {:.2f}".format(score))
print("Results for Random Forest...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))
score = f1_score(y_pred, y_test, average='weighted')
print("random Forest F1 score: {:.2f}".format(score))