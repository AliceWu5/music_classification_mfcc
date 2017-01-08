#coding:utf-8

import os
import sys
import numpy as np
import librosa
import sklearn
from matplotlib.pyplot import specgram
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
    
if __name__ == '__main__':

    path='./genres/'

    data=[]
    labels=[]

    for i, folder in enumerate(os.listdir(path)):
        for file in os.listdir(path+folder):
            music, fs = librosa.audio.load(path+folder+'/'+file)
            mfcc_feature = librosa.feature.mfcc(music,n_mfcc=13)
            #フレームのうち，最初の10%と最後の10%を利用
            mfcc_feature = np.array([x[int(len(x)*1/10):int(len(x)*9/10)] for x in mfcc_feature])
            data.append(mfcc_feature.mean(axis=1))
            labels.append(folder)

    np.save('data.npy', data)
    np.save('label.npy', labels)