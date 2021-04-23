import cv2
from os import path
import torch
from PIL import Image
from selSearch import *
from feature import *
from utils import *
from dataload import *
import numpy as np
import tqdm
import pickle

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

train_im = []
train_features = []
train_labels = []

def region_warpping(data, regions):
    """
    inputs:
        data - instance of voc dataloader class
        regions - region proposals from selective search
    function:
        checks the iou value of all the regional proposals with the actual ground truths
    """
    image = data['im']
    gtbbs = data['boxes']
    gtclasses = data['class']

    for index, region in enumerate(regions):
        for idx, gtbb in enumerate(gtbbs):
            x, y, w, h = region
            iou = getIoU([x, y, x + w, y + h], gtbb)

            if iou > 0.70:
                temp_im = image[y: y + h, x: x + w]
                resized_im = cv2.resize(temp_im, (227, 227), interpolation=cv2.INTER_AREA)
                train_im.append(resized_im)
                train_labels.append(gtclasses[idx])

if __name__ == "__main__":
    if not path.exists("images.pkl"):
        dataLoader = voc('./../../VOC2007')
        for i in tqdm.tqdm(range(dataLoader.__len__())):
            imageData = dataLoader.__getitem__(i)
            pathh = imageData['path']
            im = cv2.imread(pathh)
            rects = selectiveSearch(im)
            region_warpping(imageData, rects)

        save_data = {
            'train_im': train_im,
            'train_features': train_features,
            'train_labels': train_labels
        }

        with open('data.pkl', 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Loading saved data...')
        with open('images.pkl', 'rb') as f:
            data = pickle.load(f)

        train_im = data['train_im']
        train_features = data['train_features']
        train_labels = data['train_labels']

    for idx, i in enumerate(train_im):
        f = Image.fromarray(np.uint8(i))
        f = featureExtractor(f)
        torch.save(f, 'features/' + str(idx) + '_' + str(train_labels[idx]) + '.pt')
        print(idx, '/', len(train_im))
