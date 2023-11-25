import numpy as np
from PIL import Image
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch



class BIMCV(Dataset):
    def __init__(self, df, root_dir, mode="train", args=None):
        assert mode in ['train', 'val'], "invalid mode"
        self.path = list(df['path'])
        self.label = df['label']
        self.root_dir = root_dir
        self.mode = mode
        self.args = args

        self.transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    def __getitem__(self, index):
        p = self.root_dir + self.path[index]
        img = Image.open(p).convert("RGB")
        label = self.label[index]
        if (self.args is not None) and (self.args.model == "clip_resnet50" or self.args.model == "bit_resnet50"):
            img = self.args.preprocess(img)
        else:
            img = self.transform[self.mode](img)
        return img, label

    def __len__(self):
        return len(self.path)

    
if __name__ == "__main__":
    pass    


