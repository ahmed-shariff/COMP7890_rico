from mlpipeline.base import DataLoaderABC
from mlpipeline.utils import Datasets
import pandas as pd
import numpy as np
import random
import cv2
import torchvision
import torch

from mlpipeline.pytorch import DatasetBasicABC

np.random.seed(100)
random.seed(100)
torch.manual_seed(100)


FLAT_USED_LABELS = {"None": 0, "text-component": 127, "component": 255}
SEMANTIC_USED_LABELS = {}


class RicoDataSetABC(DatasetBasicABC):
    def __init__(self, used_labels, flat=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat = flat
        self._used_labels = used_labels

    def _assign_area(self, img, view, x1, y1, x2, y2):
        if self.flat:
            img[y1:y2, x1:x2] = 255
        else:
            img[y1:y2, x1:x2] = self._get_color(view)
        return img


class FlatRicolDataSet(RicoDataSetABC):
    def _get_color(self, view):
        if "Text" not in view["class"]:
            return self._used_labels["component"]
        else:
            return self._used_labels["text-component"]

    
class ConvModelDataSet(FlatRicolDataSet):
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((256, 144), dtype=np.uint8)  # 2560x1440 divided by 20
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


class LinearModelDataSet(FlatRicolDataSet):
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((128, 72), dtype=np.uint8)  # 2560x1440 divided by 10
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
        out = torchvision.transforms.functional.to_tensor(img).flatten()
        return entry["screenshot"], out
    

def load_data(data_path):
    return pd.read_json(data_path), []


def imshow_tensor(t):
    if len(t.shape) != 4:
        t = t.view(-1, 128, 72)
    img = random.choices([(_t.cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8).squeeze() for _t in t], k=5)
    img = np.concatenate(img, axis=1)
    cv2.imshow("", img)
    cv2.waitKey(10)
