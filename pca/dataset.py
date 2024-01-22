import pandas as pd
import os
from PIL import Image
import numpy as np


class Dataset:
    def __init__(self, path, size=None):
        self.path = path
        self.size = size
        self.data = None

    def load_data(self):
        data_path = os.listdir(self.path)
        paths = []
        classess = []
        for data in data_path:
            full_path = os.path.join(self.path, data)
            class_ = data.split(".")[1][:-1]

            paths.append(full_path)
            classess.append(class_)

        self.data = pd.DataFrame({"Path": paths, "Class": classess})

    def resize(self):
        X = []
        for i in self.data.index:
            image = Image.open(self.data["Path"][i]).resize(self.size)
            image = np.array(image)
            image = image.flatten()
            X.append(image)
        X = np.array(X) / 255
        return np.array(X)
