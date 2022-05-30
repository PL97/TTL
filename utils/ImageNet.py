from PIL import Image
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch
import torchvision


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



class ImageNet(Dataset):
    def __init__(self, mode="train", args=None):
        assert mode in ['train', 'val'], "invalid mode"
        self.ds = torchvision.datasets.ImageNet('path/to/imagenet_root/', download=True)

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
        img, label = self.ds[i]
        print(img.shape, label)

    def __len__(self):
        return len(self.path)

    
if __name__ == "__main__":
    
    dl = ImageNet()
    for i, j in dl:
        print("pass")
    pass

    # # test case
    # import matplotlib.pyplot as plt
    # root_path = "/home/le/project/TL/skin/archive/"
    # df = pd.read_csv(root_path + "HAM10000_metadata.csv")
    # dl = HAM(df, root_dir=root_path+"jpgs/")

    # for img, label in dl:
    #     print(img.shape)
    #     plt.figure()
    #     plt.imshow(img.numpy().transpose([1, 2, 0]))
    #     plt.show()
    #     break
    
    # import torchvision
    # import torch
    # transform = transforms.Compose(
    # [transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform,
    #                                     download=True,)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
    #                                       shuffle=True, num_workers=2)
    # for x, y in trainloader:
    #     print(y)

