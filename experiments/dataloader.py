from mlpipeline.base import DataLoaderABC
from mlpipeline.utils import Datasets
import pandas as pd
import numpy as np
import random
import cv2
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
    def __init__(self, used_labels, flat=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat = flat
        self._used_labels = used_labels

    def _assign_area(self, img, view, x1, y1, x2, y2):
        img[y1:y2, x1:x2] = self._get_color(view)
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
    

def load_data(data_path):
    return pd.read_json(data_path), []


def imshow_tensor(t, in_shape):
    if len(t.shape) != 4:    
        t = t.view(-1, 128, 72)

    # img = random.choices([(_t.cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8).squeeze() for _t in t], k=5)
    
    # print(np.asarray(torchvision.transforms.functional.to_pil_image(t[0])).max())
    # cv2.imshow("", np.asarray(torchvision.transforms.functional.to_pil_image(t[0])))
    # cv2.waitKey()
    choices = random.choices(list(range(in_shape[0])), k=5)
    img = [(torchvision.transforms.functional.to_pil_image(t[_t])) for _t in choices]
    img = np.concatenate(img, axis=1)
    cv2.imshow("", img)
    cv2.waitKey(10)


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
