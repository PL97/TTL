import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.io.stata import precision_loss_doc
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler, SGD, Adam
import torch
# from sklearn.metrics import precision_recall_curve, auc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("./")
from main import train, evaluate
sys.path.append("./utils/")
from CheXpert import preprocess, CheXpert
from settings import parse_opts
from models import densenet121, densenet201, resnet50


chexpert_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture']

if __name__ == "__main__":
    args = parse_opts()

    if not args.test:
        train_df = pd.read_csv(os.path.join(args.root_path, "CheXpert-v1.0-small", "TRAIN.csv"))
        train_df = preprocess(train_df)
        train_ds = CheXpert(train_df, root_dir=args.root_path, mode="train", args=args)
        train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers) 
        
        val_df = pd.read_csv(os.path.join(args.root_path, "CheXpert-v1.0-small", "VAL.csv"))
        val_df = preprocess(val_df)
        val_ds = CheXpert(val_df, root_dir=args.root_path, mode="val", args=args)
        val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers) 


        if args.cont:
            save_file_name = os.path.join(args.saved_path, "best.pt")
            print("continue training from: ", save_file_name)
            model = torch.load(save_file_name).module.to(torch.device("cpu"))
        else:
            if args.model == "densenet121":
                model = densenet121(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
            elif args.model == "densenet201":
                model = densenet201(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
            elif args.model == "resnet50":
                model = resnet50(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
            else:
                exit("model not found")
        
        print("training with ", args.model)
        train(model=model, trainloader=train_dl, valloader=val_dl, args=args)

    else:
        test_df = pd.read_csv(os.path.join(args.root_path, "CheXpert-v1.0-small", "TEST.csv"))
        test_df = preprocess(test_df)
        test_ds = CheXpert(test_df, root_dir=args.root_path, mode="val", args=args)
        test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers) 

        model = torch.load(os.path.join(args.saved_path, "best.pt"))
        criterion = args.criterion
        acc, _ = evaluate(model, test_dl, criterion, args)
        print("top one accuracy:", acc)
        print(chexpert_classes)