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
