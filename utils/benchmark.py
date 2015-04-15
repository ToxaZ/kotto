"""
    python benchmark.py <path-to-train> <path-to-test> <name-of-submission>
"""

from __future__ import division
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFECV


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def dim_reduction(X, threshold=0):
    data = X[:, 1:-1]
    target = X[:, -1]
    svc = SVC(kernel="linear")
    print("creating SVC")
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(target, 2), scoring='accuracy')
    print("creating RFECV")
    rfecv.fit(data, target)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # print "Number of features before reduction (threshold=%f) %d" % (threshold, len(data[0]))
    # data = sel.fit_transform(data,target)
    # print "Number of features after reduction (threshold=%f) %d" % (threshold, len(data[0]))

    return data, target


def load_train_data(path=None, train_size=0.8, reduction_threshold=95):
    global data, target
    path = sys.argv[1] if len(sys.argv) > 1 else path
    if path is None:
        df = pd.read_csv('data/train.csv')
    else:
        df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)

    data, target = dim_reduction(X, reduction_threshold)


    X_train, X_valid, y_train, y_valid = train_test_split(
        data, target, train_size=train_size,
    )
    print(" -- Loaded data.")
    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))

def load_test_data(path=None):
    path = sys.argv[2] if len(sys.argv) > 2 else path
    if path is None:
        df = pd.read_csv('data/test.csv')
    else:
        df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    return X_test.astype(float), ids.astype(str)


def cross_validate_train(path="data/train.csv", reduction_threshold=95):
    global scores
    clf = RandomForestClassifier(n_estimators=10)
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    data, target = dim_reduction(X, reduction_threshold)
    scores = cross_val_score(clf, data, target, cv=5, scoring="log_loss")
    print scores
    print scores.mean()

def train():
    X_train, X_valid, y_train, y_valid = load_train_data()
    # Number of trees, increase this to beat the benchmark ;)
    n_estimators = 10
    clf = RandomForestClassifier(n_estimators=n_estimators)
    print(" -- Start training Random Forest Classifier.")
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_valid)
    print(" -- Finished training.")

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    score = logloss_mc(y_true, y_prob)
    print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))

    return clf, encoder


def make_submission(clf, encoder, path='my_submission.csv'):
    path = sys.argv[3] if len(sys.argv) > 3 else path
    X_test, ids = load_test_data()
    y_prob = clf.predict_proba(X_test)
    with open(path, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + list(map(str, probs.tolist())))
            f.write(probas)
            f.write('\n')
    print(" -- Wrote submission to file {}.".format(path))


def main():
    print(" - Start.")
    cross_validate_train()
    # clf, encoder = train()
    # make_submission(clf, encoder)
    print(" - Finished.")


if __name__ == '__main__':
    main()
