import torch

import re
import pickle
from PIL import Image
from os import listdir

import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def trainClassifier(X, Y, val_split=0.80, num_classes=20):
    train_x, train_y = [], []
    val_x, val_y = [], []

    train_x, val_x, train_y, val_y = train_test_split(X, Y, stratify=Y, test_size=val_split)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', verbose=True, probability=True))
    clf.fit(train_x, train_y)
    
    filename = 'svmmodel_pascalvoc_alexnet_noft_prob.sav'
    pickle.dump(clf, open(filename, 'wb'))

    pred = clf.predict(train_x)
    
    correct = 0
    for i in range(len(train_y)):
        if pred[i] == train_y[i]:
            correct += 1
    print("Train classification accuracy: {}".format(correct / len(train_y)))

    
    pred = clf.predict(val_x)
    
    correct = 0
    for i in range(len(val_y)):
        if pred[i] == val_y[i]:
            correct += 1
    print(correct / len(val_y))
    print("Validation classification accuracy: {}".format(correct / len(val_y)))

if __name__ == "__main__":
    train_path = '../../newFeaturesNoFinTuneAlexnet/newFeatures/'
    
    images, classes = [], []

    for f in listdir(train_path):
        #cls = re.search('_(.+?).pt', f).group(1)
        cls = f.split('_')[5].split('.')[0]
        #print(cls)
        classes.append(int(cls))
        
        im = torch.load(train_path + f, map_location=torch.device('cpu'))
        images.append(im.detach().numpy()[0])

    trainClassifier(images, classes) 
