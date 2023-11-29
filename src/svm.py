import sys
import os

from src.utils import optimal_threshold, pre_rec_f1
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import average_precision_score


# train svm
def train_svc(train_data, valid_data, train_label, valid_label, kernel="rbf", cv=3):
    params = {"C": [0.1, 0.01, 0.001, 5, 20, 50], "gamma": ["scale"]}
    svc = SVC(kernel=kernel)

    clf = GridSearchCV(svc, params, scoring="f1", cv=cv)

    clf.fit(train_data, train_label)

    valid_score = clf.decision_function(valid_data)
    # valid_score = minmax_scale(valid_score)
    th, _ = optimal_threshold(valid_label, valid_score)

    return clf.best_estimator_, th


# test svm
def test_svc(test_data, test_label, clf, th):
    pre_score = clf.decision_function(test_data)
    # pre_score = minmax_scale(pre_score)
    ap = round(average_precision_score(test_label, pre_score), 4)

    pre_label = (pre_score > th) * 1
    pre, rec, f1 = pre_rec_f1(test_label, pre_label)
    return [ap, pre, rec, f1]
