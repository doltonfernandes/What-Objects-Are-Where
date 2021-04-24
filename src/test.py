from sel import selectiveSearch
from dataload import voc
import numpy as np
import cv2
import pickle
from PIL import Image
from utils import getIoU
from nms import nms_detections

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f_new import *
import tqdm

loaded_model = pickle.load(open('svmmodel_pascalvoc_alexnet_noft_prob.sav', 'rb'))

def getPred(im):
    """
    return predicted class of the image and score
    """
    f = featureExtractor(im).cpu().data.numpy()
    result = int(loaded_model.predict(f)[0])
    cls_score  = loaded_model.predict_proba(f)[0][result - 1]
    return result, cls_score


def getRes(im):
    ss = selectiveSearch(im)
    H,W,_ = im.shape
    bbox = []
    cls = []
    bbox_class = []
    bbox_score = []
    for i in range(21):
        bbox_class.append([])
        bbox_score.append([])
    for e, result in tqdm.tqdm(enumerate(ss)):
        x, y, w, h = result
        if w*h/H*W < 1000:
            continue
        imcut = im[y : y + h, x : x + w]
        img = cv2.cvtColor(imcut, cv2.COLOR_BGR2RGB) 
        img =  Image.fromarray(img)
        cl, sc = getPred(img)
        if sc < 0.9:
            continue
        bbox_class[cl].append(result)
        bbox_score[cl].append(sc)
    for i,bboxs in enumerate(bbox_class):
        if bboxs != [] and i != 0:
            picks = nms_detections(bboxs, bbox_score[i])
            bbs = np.array(bboxs)
            bbs = bbs[picks]
            for bb in bbs:
                bbox.append(bb)
                cls.append(i)

    print(len(bbox))
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
    trainPath = "../VOC2007"
    obj = voc(trainPath)
    positive = 0
    counter = 0
    maxFiles = 100
    for i in range(1):
        o = obj[22] 
        res = getRes(o["im"])
        gCls = o["class"]
        cls = res["class"]
        for bb in res["bbox"]:
            x,y,h,w = bb	
            im = cv2.rectangle(o["im"], (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite("1_sel.jpg", im)
        print(cls)
        exit()
		
        score, count = getScore(o, res)
        positive += score
        counter += count
    print("Acc: ", positive / counter)
