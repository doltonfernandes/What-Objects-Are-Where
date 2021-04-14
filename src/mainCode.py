import cv2
from PIL import Image
from selSearch import *
from feature import *

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

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
