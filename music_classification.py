#coding:utf-8

import os
import sys
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
    
if __name__ == '__main__':
    data = np.load('data.npy')
    labels=np.load('label.npy')
    #交差検定
    accuracys=[]
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.1, random_state=i)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        print "====%d====" % (i)
        print "Accuracy:", lr.score(X_test, Y_test)   
        predict=lr.predict(X_test)
        print classification_report(Y_test, predict, target_names=["classical", "country", "jazz", "metal", "pop", "rock"])