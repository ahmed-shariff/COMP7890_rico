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

class ConvModelDataSet(DatasetBasicABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((256, 144), dtype=np.uint8)  # 2560x1440 divided by 20
        for view in entry["ui_leaf_views"]:
            # if not view["visibility"] == "visible":
            #     color = (255, 0, 0)
            # else:
            #     color = (255, 0, 255)
            x1, y1, x2, y2 = (np.array(view["bounds"]) / 10).round().astype(np.int)
            img[y1:y2, x1:x2] = 255
            # print(x1, y1, x2, y2)
            # print(view["focusable"])
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

        # cv2.imshow("", img)
        # cv2.waitKey()
        return torchvision.transforms.functional.to_tensor(img)


class LinearModelDataSet(DatasetBasicABC):
    def __getitem__(self, idx):
        entry = self.current_data.iloc[idx]
        img = np.zeros((128, 72), dtype=np.uint8)  # 2560x1440 divided by 10
        for view in entry["ui_leaf_views"]:
            # if not view["visibility"] == "visible":
            #     color = (255, 0, 0)
            # else:
            #     color = (255, 0, 255)
            x1, y1, x2, y2 = (np.array(view["bounds"]) / 20).round().astype(np.int)
            img[y1:y2, x1:x2] = 255
            # print(x1, y1, x2, y2)
            # print(view["focusable"])
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

        # cv2.imshow("", img)
        # cv2.waitKey()
        out = torchvision.transforms.functional.to_tensor(img).flatten()
        return out
    

def load_data(data_path):
    return pd.read_json(data_path), []
