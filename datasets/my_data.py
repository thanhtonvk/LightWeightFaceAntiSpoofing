import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import cv2
import numpy as np
class MyDataSet(Dataset):
    def __init__(self, test_mode=False, transform=None):
        if test_mode:
            list_path = "datasets/val.csv"
        else:
            list_path = "datasets/train.csv"
        
        df = pd.read_csv(list_path)
        self.paths = df['path'].tolist()
        self.labels =df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(image_path)
        cropped_face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            cropped_face = self.transform(label=label, img=cropped_face)['image']
        cropped_face = np.transpose(cropped_face, (2, 0, 1)).astype(np.float32)
        return (torch.tensor(cropped_face), torch.tensor(label, dtype=torch.long))