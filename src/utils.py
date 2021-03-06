import numpy as np

def getIoU(boxA, boxB):
    # boxX = [0., 0., 10., 10.]
    boxA = np.array(boxA).astype(np.int)
    boxB = np.array(boxB).astype(np.int)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    #print(boxA[2], boxA[0], boxA[3], boxA[1])
    #print(boxB[2], boxB[0], boxB[3], boxB[1])
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

print(getIoU([0, 0, 10, 10], [0, 0, 10, 10]))
