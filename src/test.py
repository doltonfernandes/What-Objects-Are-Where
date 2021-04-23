from sel import selectiveSearch
from dataload import voc
import numpy as np
import cv2

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


if __name__ == "__main__":
    trainPath = ""
    obj = voc(trainPath)
    positive = 0
    counter = 0
    for o in obj:
        res = getRes(o["im"])
        gCls = o["class"]
        cls = res["class"]
        for c in cls:
            if c in gCls:
                positive += 1
                gCls.remove(c)
            counter += 1
    print("Acc: ", positive / counter)
