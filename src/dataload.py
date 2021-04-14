import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2


class voc:
    def __init__(self, root):
        self.root = root
        self.classes = (
            "__background__",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        self.num_classes = len(self.classes)
        self.class_map = dict(zip(self.classes, range(self.num_classes)))
        self.ext = ".jpg"
        # self._image_index = self._load_image_set_index()
        self.pathAnnot = root + "./Annotations/"
        self.annots = os.lisdir(self.pathAnnot)

    def __len__(self):
        return len(self.annots)

    def __get__(self, index):
        filename = self.pathAnnot + self.annots[index]
        tree = ET.parse(filename)
        objs = tree.findall("object")
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        classes = np.zeros((num_objs), dtype=np.int32)
        for ix, obj in enumerate(objs):
            bbox = obj.find("bndbox")
            # Make pixel indexes 0-based
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1
            cls = self._class_to_ind[obj.find("name").text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            classes[ix] = cls
        imgpath = tree.find("filename").text
        impath = self.root + "/JPEGImages/" + imgpath
        im = cv2.imread(impath)
        return {
            "boxes": boxes,
            "im": im,
            "class": classes,
        }
