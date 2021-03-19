import cv2


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


if __name__ == "__main__":
    f = "../img/1.jpg"
    im = cv2.imread(f)
    rects = selectiveSearch(im)
    print("No. of proposals:", len(rects))
    for rect in rects:
        x, y, w, h = rect
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite("../img/1_sel.jpg", im)
