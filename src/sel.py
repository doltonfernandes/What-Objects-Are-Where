import cv2, torch
from torchvision import models, transforms
from PIL import Image

def selectiveSearch(im):
    """
    returns the region proposals for the image
    input: im - image
    return: list of proposals
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects

def featureExtractor(im):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    im = transform(im)
    im = im.unsqueeze(0)
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    return alexnet(im)

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
