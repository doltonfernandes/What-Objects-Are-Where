import cv2
from PIL import Image
from selSearch import *
from feature import *
from utils import *

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

train_im = []
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
    gtclasses = data['classes']

    for index, region in enumerate(regions):
        for idx, gtbb in enumerate(gtbbs):
            x, y, w, h = region
            iou = get_iou([x, y, x + w, y + h], gtbb)

            if iou > 0.70:
                temp_im = image[y: y + h, x: x + w]
                resized_im = cv2.resize(temp_im, (224, 224), interpolation=cv2.INTER_AREA)
                train_im.append(resized_im)
                train.labels.append(gtclasses[idx])

if __name__ == "__main__":
    f = "../img/1.jpg"
    im = cv2.imread(f)
    rects = selectiveSearch(im)
    print("No. of proposals:", len(rects))
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite("../img/1_sel.jpg", im)

    # Get image in bounding box
    im = Image.open(f)
    im = im.crop((50, 10, 450, 400))
    im.save('../img/1_bounding.jpg')

    # Add p=16 pixels on border
    im = add_margin(im, 16, 16, 16, 16, (0, 0, 0))
    im = featureExtractor(im)
