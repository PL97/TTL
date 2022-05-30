import numpy as np
from PIL import Image
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch

import sys
sys.path.extend("./")
from settings import parse_opts


chexpert_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Device']
       
chexpert_to_num = {'No Finding': 0, 
                    'Enlarged Cardiomediastinum':1, 
                    'Cardiomegaly':2,
                    'Lung Opacity':3, 
                    'Lung Lesion':4, 
                    'Edema':5, 
                    'Consolidation':6, 
                    'Pneumonia':7,
                    'Atelectasis':8, 
                    'Pneumothorax':9, 
                    'Pleural Effusion':10, 
                    'Pleural Other':11,
                    'Fracture':12,
                    'Support Device':13}

low_idx = [3, 5, 6, 7, 8, 10, 11]
high_idx = [0, 1, 2, 4, 9, 12, 13]

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



class CheXpert(Dataset):
    def __init__(self, df, root_dir, mode="train", args=None):
        assert mode in ['train', 'val'], "invalid mode"
        self.path = list(df['Path'])
        # self.label = pd.Categorical(df["dx"]).codes
        self.label = np.asarray(df.iloc[:, 1:])
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
        img = expand2square(img, (255, 255, 255))
        # img = padAndResize(img)
        label = self.label[index].astype(np.float32)
        if self.args.target != 'all':
            if self.args.target == 'low':
                label = label[low_idx]
            elif self.args.target == 'high':
                label = label[high_idx]
            else:
                label = label[chexpert_to_num[self.args.target]]
        # label = torch.FloatTensor(label)
        # img = self.transform[self.mode](img)
        if (self.args is not None) and (self.args.model == "clip_resnet50" or self.args.model == "bit_resnet50"):
            img = self.args.preprocess(img)
        else:
            img = self.transform[self.mode](img)
        return img, label

    def __len__(self):
        return len(self.path)

def preprocess(df):
    kept_col = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    df = df.iloc[:, kept_col].fillna(0)
    df = df.replace(-1, 1)
    return df

    
if __name__ == "__main__":

    # pass  
    args = parse_opts()  

    # # test case
    mode = "VAL"
    import matplotlib.pyplot as plt
    # root_path = "/home/le/CheXpert-v1.0-small/"
    root_path = "/home/luo00042/SATASSD2/trunc_tl/CheXpert-v1.0-small/"
    df = pd.read_csv(root_path + "{}.csv".format(mode))
    df = preprocess(df)

    print(df.head())
    count = []
    for i in range(df.shape[1]-1):
        tmp = np.sum(df.iloc[:, i+1])
        count.append(tmp)
    print(count)
    # plt.bar(chexpert_classes, count)
    # plt.savefig(mode)


    # count = np.asarray(count)
    # count = 1/count
    # count = count/np.sum(count)
    # print(count)


    print(args.root_path)
    dl = CheXpert(df, root_dir=args.root_path, args = args)
    dl = DataLoader(dl, batch_size=128, shuffle=False)   

    count = 0
    for x, y in dl:
        count += 1
        print(x.shape)
    print(count)

