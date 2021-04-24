from sel import selectiveSearch
from dataload import voc
import numpy as np
import cv2
from utils import getIoU


def getPred(im):
    """
    return predicted class of the image
    """
    pass


def getRes(im, model):
    ss = selectiveSearch(im)
    bbox = []
    cls = []
    for e, result in enumerate(ss):
        x, y, w, h = result
        bbox.append(result)
        imcut = im[y : y + h, x : x + w]
        resized = cv2.resize(imcut, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(resized, axis=0)
        cl = getPred(img)
        cls.append(cl)
    return {"bbox": bbox, "class": cls}


def getScore(gr, out, thresh=0.5):
    outClass = out["class"]
    outbbox = out["class"]
    groundbbox = gr["class"]
    groundClass = gr["class"]
    score = 0
    counter = 0
    for i, cls in enumerate(outClass):
        for j, gcls in enumerate(groundClass):
            if cls != gCls:
                continue
            iou = getIoU(outbbox[i], groundbbox[j])
            if iou > thresh:
                score += 1
                break
        counter += 1
    return score, counter


if __name__ == "__main__":
    trainPath = ""
    obj = voc(trainPath)
    positive = 0
    counter = 0
    maxFiles = 100
    for o in obj[:100]:
        res = getRes(o["im"])
        gCls = o["class"]
        cls = res["class"]
        score, count = getScore(o, res)
        positive += score
        counter += count
    print("Acc: ", positive / counter)
