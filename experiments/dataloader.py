from mlpipeline import log
from mlpipeline.base import DataLoaderABC
from mlpipeline.utils import Datasets
import pandas as pd
import numpy as np
import random
import cv2
from skimage import io, transform
import torchvision
import torch
import json

from mlpipeline.pytorch import DatasetBasicABC

np.random.seed(100)
random.seed(100)
torch.manual_seed(100)


FLAT_USED_LABELS = {"None": 0, "text-component": 127, "component": 255}
# SEMANTIC_USED_LABELS = load_semantic_labels("../data/generated/semantic_colors.json")


class RicoDataSetABC(DatasetBasicABC):
    def __init__(self, used_labels, flat=True, return_img=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat = flat
        self._used_labels = used_labels
        self.return_img = return_img

    def _assign_area(self, img, view, x1, y1, x2, y2, rectangle=False):
        if not rectangle:
            img[y1:y2, x1:x2] = self._get_color(view)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), self._get_color(view), 1)
        return img


class FlatRicolDataSet(RicoDataSetABC):
    def _get_color(self, view):
        if self.flat:
            return [255, 255, 255]

        if "Text" not in view["class"]:
            c = self._used_labels["component"]
        else:
            c = self._used_labels["text-component"]
        return [c, c, c]

    
class ConvModelDataSet(FlatRicolDataSet):
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((256, 144, 3), dtype=np.uint8)  # 2560x1440 divided by 20
        for view in entry["ui_leaf_views"]:
            # if not view["visibility"] == "visible":
            #     color = (255, 0, 0)
            # else:
            #     color = (255, 0, 255)
            x1, y1, x2, y2 = (np.array(view["bounds"]) / 10).round().astype(np.int)
            img = self._assign_area(img, view, x1, y1, x2, y2)
            # print(x1, y1, x2, y2)
            # print(view["focusable"])
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

        # cv2.imshow("", img)
        # cv2.waitKey()
        if self.return_img:
            # original_img = cv2.imread("../data/" + entry["screenshot"])
            # original_img = cv2.resize(original_img, (256, 144))
            # original_img = torchvision.transforms.functional.to_tensor(original_img)
            
            # original_img = io.imread("../data/" + entry["screenshot"])
            # original_img = transform.resize(original_img, (256, 144))
            # original_img = torchvision.transforms.functional.to_tensor(original_img).float()
            
            original_img = torchvision.io.read_image("../data/" + entry["screenshot"])/255
            original_img = torchvision.transforms.functional.resize(original_img, (256, 144))

            return entry["screenshot"], torchvision.transforms.functional.to_tensor(img), original_img
        return entry["screenshot"], torchvision.transforms.functional.to_tensor(img)


class SemanticConvModelDataSet(RicoDataSetABC):
    def _get_color(self, view):
        label = view["componentLabel"]
        if self.flat:
            return self._used_labels[label][None]
        else:
            if label == "Text Button":
                try:
                    return self._used_labels[label][view["textButtonClass"]]
                except KeyError:
                    return self._used_labels[label]["misc_text"]
            elif label == "Icon":
                return self._used_labels[label][view["iconClass"]]
            else:
                return self._used_labels[label][None]

    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((256, 144, 3), dtype=np.uint8)  # 2560x1440 divided by 20
        for view in entry["semantic_data"]:
            # if not view["visibility"] == "visible":
            #     color = (255, 0, 0)
            # else:
            #     color = (255, 0, 255)
            x1, y1, x2, y2 = (np.array(view["bounds"]) / 10).round().astype(np.int)
            img = self._assign_area(img, view, x1, y1, x2, y2)
            # print(x1, y1, x2, y2)
            # print(view["focusable"])
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

        # cv2.imshow("", img)
        # cv2.waitKey()
        if self.return_img:
            # original_img = cv2.imread("../data/" + entry["screenshot"])
            # original_img = cv2.resize(original_img, (256, 144))
            # original_img = torchvision.transforms.functional.to_tensor(original_img)
            
            # original_img = io.imread("../data/" + entry["screenshot"])
            # original_img = transform.resize(original_img, (256, 144))
            # original_img = torchvision.transforms.functional.to_tensor(original_img).float()
            
            original_img = torchvision.io.read_image("../data/" + entry["screenshot"])/255
            original_img = torchvision.transforms.functional.resize(original_img, (256, 144))

            return entry["screenshot"], torchvision.transforms.functional.to_tensor(img), original_img
        return entry["screenshot"], torchvision.transforms.functional.to_tensor(img)
    

class LinearModelDataSet(FlatRicolDataSet):
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((128, 72, 3), dtype=np.uint8)  # 2560x1440 divided by 10
        for view in entry["ui_leaf_views"]:
            # if not view["visibility"] == "visible":
            #     color = (255, 0, 0)
            # else:
            #     color = (255, 0, 255)
            x1, y1, x2, y2 = (np.array(view["bounds"]) / 20).round().astype(np.int)
            img = self._assign_area(img, view,  x1, y1, x2, y2)
            # print(x1, y1, x2, y2)
            # print(view["focusable"])
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

        # cv2.imshow("", img)
        # cv2.waitKey()
        img = img[:, :, 0]
        out = torchvision.transforms.functional.to_tensor(img).flatten()
        return entry["screenshot"], out
    

class ObjectDetectionModelDataSet(SemanticConvModelDataSet):
    def _get_color(self, view):
        label = view["componentLabel"]
        c = self._used_labels[label] * 2
        return (c, c, c)
        
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        # img = np.zeros((256, 144, 3), dtype=np.uint8)  # 2560x1440 divided by 20
        img = cv2.imread("../data/" + entry["screenshot"])
        img = cv2.resize(img, (144, 256))
        targets = {}
        boxes = []
        labels = []
        for view in entry["semantic_data"]:
            _boxes = (torch.tensor(view["bounds"]) / 10).round().type(torch.int64)
            if _boxes[2] - _boxes[0] > 0 and _boxes[3] - _boxes[1] > 0:
                boxes.append(_boxes)
                labels.append(self._used_labels[view["componentLabel"]])
            # x1, y1, x2, y2 = (torch.tensor(view["bounds"]) / 10).round().type(torch.int32)
            # img = self._assign_area(img, view, x1, y1, x2, y2, True)
            # print(x1, y1, x2, y2)
            # print(view["focusable"])
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
            # cv2.imshow("", img)
            # cv2.waitKey()
        targets["boxes"] = torch.stack(boxes).cuda()
        targets["labels"] = torch.tensor(labels).type(torch.int64).cuda()

        # img = cv2.imread("../data/" + entry["screenshot"])
        # img = cv2.resize(img, (144, 256))
        # for view, (x1, y1, x2, y2) in zip(entry["semantic_data"], boxes):
        #     img = self._assign_area(img, view, x1, y1, x2, y2, True)

        #     cv2.imshow("", img)
        #     cv2.waitKey()
        return entry["screenshot"], torchvision.transforms.functional.to_tensor(img), targets

    def collate_fn(self, batch):
        return list(zip(*batch))


def verify_row(row):
    if len(row) == 0:
        return False
    boxes = []
    for view in row:
        _boxes = (np.array(view["bounds"]) / 10).round().astype(np.int32)
        if _boxes[2] - _boxes[0] > 0 and _boxes[3] - _boxes[1] > 0:
            boxes.append(_boxes)
    if len(boxes) == 0:
        return False
    return True
            
    
def load_data(data_path):
    data = pd.read_json(data_path)
    data["valid_row"] = data["semantic_data"].apply(verify_row)
    log("Dropped number of rows (empty semantic_data): {}".format((~data["valid_row"]).sum()))
    data = data[data["valid_row"]]
    return data, []


def load_semantic_labels(path):
    with open(path) as f:
        labels = json.load(f)

    step = int(220 * (1 / len(labels)))
    label_coding = {}
    for idx, (label, sub_categories) in enumerate(labels.items()):
        val = 10 + idx * step
        if len(sub_categories) == 1:
            label_coding[label] = {None: [val, val, val]}
        else:
            sub_step = int(220 * (1 / len(sub_categories)))
            sub_category_coding = {None: [val, val, val]}            
            
            for sub_idx, sub_category in enumerate(sub_categories):
                sub_val = [val, val, val]
                
                sub_val[idx % 3] = 10 + sub_idx * sub_step
                
                sub_category_coding[sub_category] = sub_val

            label_coding[label] = sub_category_coding
    return label_coding

def load_semantic_classes(path):
    with open(path) as f:
        labels = json.load(f)

    label_coding = {}
    for idx, label in enumerate(labels.keys()):
        label_coding[label] = idx

    
    log("labels used:  {}".format(label_coding))
    return label_coding
