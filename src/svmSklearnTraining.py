import torch

import re
from PIL import Image
from os import listdir

import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def trainClassifier(X, Y, val_split=0.20, num_classes=20):
    train_x, train_y = [], []
    val_x, val_y = [], []

    train_x, val_x, train_y, val_y = train_test_split(X, Y, stratify=Y, test_size=val_split)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(train_x, train_y)
    
    pred =clf.predict(val_x)
    print(pred == val_y)

if __name__ == "__main__":
    train_path = '../../features/'
    
    images, classes = [], []

    for f in listdir(train_path):
        cls = re.search('_(.+?).pt', f).group(1)
        classes.append(int(cls))
        
        im = torch.load(train_path + f, map_location=torch.device('cpu'))
        images.append(im.detach().numpy()[0])

    trainClassifier(images, classes) 
