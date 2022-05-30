from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
from pandas.io.stata import precision_loss_doc
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler, SGD, Adam, AdamW
import torch
from sklearn.metrics import precision_recall_curve, auc
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import time
import random
from torch.profiler import profile, record_function, ProfilerActivity

import sys
sys.path.extend("./utils/")
from utils.models import densenet121, densenet201, resnet50, clip_resnet50, bit_resnet50, freeze_resnet50, freeze_densenet201, cbr_larget, alexnet, efficientnet, layer_wise_freeze_resnet50
from utils.HAM import HAM
from utils.BIMCV import BIMCV
from utils.settings import parse_opts
from utils.slim_resnet import slim_resnet50
from utils.layer_wise_slim_resnet import layer_wise_slim_resnet50
from utils.slim_densenet import slim_densenet201
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

resnet50_model_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
densenet201_model_url = "https://download.pytorch.org/models/densenet201-c1103571.pth"


def plot(dict_loss, dict_acc, args):
    epochs = len(dict_loss['train'])
    plt.figure()
    plt.subplot(211)
    plt.plot(range(epochs), dict_loss['train'], label='train_loss')
    plt.plot(range(epochs), dict_loss['val'], label='val_loss')
    plt.legend()
    plt.subplot(212)
    plt.plot(range(epochs), dict_acc['train'], label='train_acc')
    plt.plot(range(epochs), dict_acc['val'], label='val_acc')
    plt.legend()
    plt.title(args.saved_path.split("/")[-1])
    plt.savefig(os.path.join(args.saved_path, "train_stat.png"))
    
    

def evaluate_single(model, valloader, criterion, args):
    model.eval()
    PRED = []
    LABELS = []
    LOSS = 0
    m = nn.Softmax(dim=1)
    total_run_time = 0
    # time_begin = time.time()
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA,
    #     ]
    # ) as p:
    counter = 0
    model.to(args.device)
    for data, label in valloader:
        input = data.to(args.device)
        target = label.to(args.device).long()
        
        
        # time_begin = time.time()
        # with profile(
        #     activities=[
        #         ProfilerActivity.CPU,
        #         ProfilerActivity.CUDA,
        #     ]
        # ) as p:
        #     model(input)
        # print(p.key_averages().table(
        # sort_by="self_cuda_time_total", row_limit=-1))
        # exit()
        # time_end = time.time()
        # total_run_time += time_end - time_begin
        # counter += 1

        output = m(model(input))
        loss = criterion(output, target)

        LABELS.extend(label.detach().cpu().numpy())
        PRED.extend(output.detach().cpu().numpy())
        LOSS += loss.detach().cpu().numpy() * data.shape[0]
    
    # if args.test:
    #     print(total_run_time/counter)
    #     # print(p.key_averages().table(
    #     # sort_by="self_cuda_time_total", row_limit=-1))
    #     return
        
    
    # if args.test:
    #     print("avgerage runnning time per instance", (time_end - time_begin)/valloader.dataset.__len__())

    LABELS = np.asarray(LABELS)
    PRED = np.asarray(PRED)
    
    TP = [0] * args.classes
    FP = [0] * args.classes
    
    PRED_LABEL = np.argmax(PRED, axis=1)
    acc = np.mean(PRED_LABEL==LABELS)
    LOSS = LOSS / valloader.dataset.__len__()

    if args.test:
        rocs = []
        prcs = []
        for i in range(LABELS.shape[0]):
            if LABELS[i] == PRED_LABEL[i]:
                TP[LABELS[i]] += 1
            else:
                FP[LABELS[i]] += 1

        for i in range(PRED.shape[1]):
            tmp_labels = (LABELS==i).astype(int)
            roc = roc_auc_score(tmp_labels, PRED[:, i])
            p, r, t = precision_recall_curve(tmp_labels, PRED[:, i])
            prc = auc(r, p)
            rocs.append(roc)
            prcs.append(prc)

        
            
        Precision = [TP[i]/(TP[i] + FP[i]) for i in range(args.classes)]
        # print("precision", Precision)
        # for p in Precision:
        #     print(p)
        # print("roc", rocs)
        # for r in rocs:
        #     print(r)
        # print("prc", prcs)
        # for p in prcs:
        #     print(p)
        for p, r in zip(prcs, rocs):
            print("{} {}".format(p, r))
    
    return acc, LOSS


def evaluate_multi(model, valloader, criterion, args):
    model.eval()
    LABELS = []
    PRED = []
    LOSS = 0
    # for data, label in tqdm(valloader):
    # with profile(
    #         activities=[
    #             ProfilerActivity.CPU,
    #             ProfilerActivity.CUDA,
    #         ]
    #     ) as p:
    for data, label in valloader:
        input = data.to(args.device)
        target = label.to(args.device)
        output = model(input)
        loss = criterion(output, target)
        LOSS += loss.detach().cpu().numpy() * data.shape[0]

    PRED.extend(output.detach().cpu().numpy())
    LABELS.extend(label.detach().cpu().numpy())
    
    # if args.test:
    #     print(p.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))



    PRED = np.asarray(PRED)
    LABELS = np.asarray(LABELS)

    acc = []
    for i in range(LABELS.shape[1]):
        tmp_pred = (PRED[:, i] > 0).astype(int)
        tmp_labels = LABELS[:, i]
        acc.append(np.sum(tmp_pred == tmp_labels)/LABELS.shape[0])
    acc = np.mean(acc)

    prc = []
    for i in range(LABELS.shape[1]):
        p, r, t = precision_recall_curve(LABELS[:, i], PRED[:, i])
        prc.append(auc(r, p))
    print(prc)
    print("accuracy: ", acc)
    prc = np.mean(prc)
    LOSS = LOSS / valloader.dataset.__len__()
    return prc, LOSS


def evaluate(model, valloader, criterion, args):
    if args.dataset == "CheXpert" and (args.target == "all" or args.target == 'low' or args.target == 'high'):
        return evaluate_multi(model, valloader, criterion, args)
    elif args.dataset == "HAM" or args.dataset == "BIMCV" or (args.dataset == "CheXpert" and args.target != "all"):
        return evaluate_single(model, valloader, criterion, args)
    else:
        exit("evaluation not supported yet")

        
def getWeights(labels, args):
    new_labels = labels.detach().cpu().numpy()
    count = [np.sum(new_labels == i) for i in range(args.classes)]
    count = torch.Tensor(1/count)
    print(count)
    print(labels)
    return count

def train(model, trainloader, valloader, args):
    # training configs
    model = model.to(args.device)
    if args.data_parallel:
        model = nn.DataParallel(model)
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if "freeze" in args.model:
        modules=list(model.children())[:-1]
        base=nn.Sequential(*modules)
        fc = list(model.children())[-1]

        optimizer = Adam(
        [
            {"params": fc.parameters(), "lr": args.lr},
            {"params": base.parameters()},
        ],
        lr = args.lr/args.ptl_decay)

        # optimizer = SGD(
        # [
        #     {"params": fc.parameters(), "lr": args.lr},
        #     {"params": base.parameters()},
        # ],
        # lr = args.lr/10,
        # momentum=0.9)

    else:
        modules=list(model.children())[:-1]
        base=nn.Sequential(*modules)
        fc = list(model.children())[-1]

        # optimizer = Adam(
        # [
        #     {"params": fc.parameters(), "lr": args.lr},
        #     {"params": base.parameters()},
        # ],
        # lr = args.lr/10)
        optimizer = Adam(model.parameters(), lr = args.lr)
        # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)


    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    # criterion = nn.CrossEntropyLoss(reduction='mean', weight=args.weights)
    criterion = args.criterion
    max_acc = 0
    min_loss = 1e8
    
    hist_loss = {"train":[], "val":[]}
    hist_acc = {"train":[], "val":[]}

    iter_count = 0
    
    # Start Training
    for epoch in range(args.max_epoch):
        model.train()
        # labels, preds = [], []
        # for data, label in tqdm(trainloader, position=0, leave=True):
        for data, label in trainloader:
            iter_count += 1
            input = data.to(args.device)
            target = label.to(args.device)

            # for BCE loss only
            if args.target != 'all' and args.target != 'low' and args.target != 'high':
                target = target.long()

            optimizer.zero_grad()
            output = model(input)

            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()


            if args.eval_iter > 0 and (iter_count % args.eval_iter) == 0:
                train_acc, train_loss = evaluate(model, trainloader, criterion, args=args)
                val_acc, val_loss = evaluate(model, valloader, criterion, args=args)
                log_config = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']}
                print("epoch{epoch}: train: loss:{train_loss} \t acc:{train_acc} | test: loss:{val_loss} \t acc:{val_acc} \t lr:{lr}".format(**log_config))
                scheduler.step(val_loss)
                
                if min_loss > val_loss:
                    min_loss = val_loss
                    save_file_name = os.path.join(args.saved_path, "best.pt")
                    torch.save(model, save_file_name)

                if optimizer.param_groups[0]['lr'] < 1e-6:
                    break


        if args.eval_iter <= 0:
            train_acc, train_loss = evaluate(model, trainloader, criterion, args=args)
            val_acc, val_loss = evaluate(model, valloader, criterion, args=args)
            log_config = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]['lr']}
            print("epoch{epoch}: train: loss:{train_loss} \t acc:{train_acc} | test: loss:{val_loss} \t acc:{val_acc} \t lr:{lr}".format(**log_config))
            scheduler.step(val_loss)
            
            if min_loss > val_loss:
                min_loss = val_loss
                save_file_name = os.path.join(args.saved_path, "best.pt")
                torch.save(model, save_file_name)

            if optimizer.param_groups[0]['lr'] < 1e-6:
                break
            
        # save for plot
        hist_loss['train'].append(train_loss)
        hist_loss['val'].append(val_loss)
        hist_acc['train'].append(train_acc)
        hist_acc['val'].append(val_acc)
        
    
    # save the final model
    save_file_name = os.path.join(args.saved_path, "final.pt")
    torch.save(model, save_file_name)
    plot(hist_loss, hist_acc, args)
        

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



if __name__ == "__main__":
    set_random_seed(1)
    torch.cuda.empty_cache()
    args = parse_opts()
    mode = "noisy"
    if not args.test:

        if args.dataset == "HAM":
            if args.sub == 100:
                train_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "train.csv"))
                val_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "val.csv"))
            else:
                train_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "train_{}.csv".format(args.sub/100)))
                val_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "val_{}.csv".format(args.sub/100)))          
            
            train_ds = HAM(train_df, root_dir=args.root_path+"jpgs/", mode='train', args=args)
            train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers, drop_last=True)
            val_ds = HAM(val_df, root_dir=args.root_path+"jpgs/", mode='val', args=args)
            val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)     

        elif args.dataset == "BIMCV":
            
            if args.sub == 100:
                train_df = pd.read_csv(os.path.join(args.root_path, mode,  str(args.exp), "train.csv"))
                val_df = pd.read_csv(os.path.join(args.root_path, mode, str(args.exp), "val.csv"))
            else:
                print("train with sub set")
                train_df = pd.read_csv(os.path.join(args.root_path, mode,  str(args.exp), "train_{}.csv".format(args.sub/100)))
                val_df = pd.read_csv(os.path.join(args.root_path, mode, str(args.exp), "val_{}.csv".format(args.sub/100)))

            train_ds = BIMCV(train_df, root_dir=args.root_path+"crop/", mode='train', args=args)
            train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
            
            
            val_ds = BIMCV(val_df, root_dir=args.root_path+"crop/", mode='val', args=args)
            val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)   
        
        elif args.dataset == "IMGNET":
            imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')

        
        if args.model == "densenet121":
            model = densenet121(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
        elif args.model == "densenet201":
            model = densenet201(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes)
        elif args.model == "resnet50":
            model = resnet50(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes, args=args)
        elif args.model == "resnet50_FAT":
            model = resnet50(pretrained=args.pretrained, trunc=args.trunc, classes=args.classes, args=args, freeze=True)
        elif args.model == "clip_resnet50":
            model, preprocess = clip_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        elif args.model == "bit_resnet50":
            model, preprocess = bit_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        elif args.model == "slim_resnet50":
            model = slim_resnet50(shrink_coefficient=args.slim_factor, load_up_to=args.slim_from, num_classes=args.classes)
            state_dict = load_state_dict_from_url(resnet50_model_url, progress=True)
            model.load_up_to_block(state_dict)
        elif args.model == "layer_wise_slim_resnet50":
            model = layer_wise_slim_resnet50(shrink_coefficient=args.slim_factor, load_up_to=args.slim_from, num_classes=args.classes, pretrained=True)
        elif args.model == "slim_densenet201":
            model = slim_densenet201(shrink_coefficient=args.slim_factor, load_up_to=args.slim_from, num_classes=args.classes)
            state_dict = load_state_dict_from_url(densenet201_model_url, progress=True)
            model.load_up_to_block(state_dict)
        elif args.model == "freeze_resnet50":
            model = freeze_resnet50(finetune_from=args.finetune_from, classes=args.classes)
        elif args.model == "layer_wise_freeze_resnet50":
            model = net = layer_wise_freeze_resnet50(finetune_from=args.finetune_from, classes=args.classes)
        elif args.model == "freeze_densenet201":
            model = freeze_densenet201(finetune_from=args.finetune_from, classes=args.classes)
        elif args.model == "cbr_larget":
            model = cbr_larget(pretrained=args.pretrained, classes=args.classes)
        elif args.model == 'alexnet':
            model = alexnet(pretrained=args.pretrained, classes=args.classes)
        elif args.model == 'layerttl_resnet50':
            model = resnet50(pretrained=args.pretrained, trunc=args.trunc, layer_wise=True, classes=args.classes, args=args)
        elif args.model == "efficientnet":
            model = efficientnet(num_classes=args.classes, pretrained=args.pretrained, trunc=args.trunc)
        else:
            exit("model not found")
        
        print("training with ", args.model)
        train(model=model, trainloader=train_dl, valloader=val_dl, args=args)
    else:
        # torch.set_num_threads(1)
        # torch.set_num_threads(1)
        # args.bs = 1
        args.pin_memory = True
        # args.num_workers = 1
        print(args.bs)
        if args.dataset == "HAM":
            test_df = pd.read_csv(os.path.join(args.root_path, str(args.exp), "test.csv"))
            test_ds = HAM(test_df, root_dir=args.root_path+"jpgs/", mode='val', args=args)
            test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)
        
        elif args.dataset == "BIMCV":
            test_df = pd.read_csv(os.path.join(args.root_path, mode, str(args.exp), "test.csv"))
            test_ds = BIMCV(test_df, root_dir=args.root_path+"crop/", mode='val', args=args)
            test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)   

        if args.model == "clip_resnet50":
            _, preprocess = clip_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        elif args.model == "bit_resnet50":
            _, preprocess = bit_resnet50(trunc=args.trunc, classes=args.classes)
            args.preprocess = preprocess
        else:
            pass

        print(os.path.join(args.saved_path, "best.pt"))
        model = torch.load(os.path.join(args.saved_path, "best.pt"))
        model = model.module.to(args.device)
        # model = model.module
        criterion = nn.CrossEntropyLoss(reduction='mean')


        acc, _ = evaluate(model, test_dl, criterion, args)

        

        # with profile(
        #     activities=[
        #         ProfilerActivity.CPU,
        #         ProfilerActivity.CUDA,
        #     ]
        # ) as p:
        #     acc, _ = evaluate(model, test_dl, criterion, args)
        # print(p.key_averages().table(
        #     sort_by="self_cuda_time_total", row_limit=-1))

        
        
        print("top one accuracy:", acc)
        

    
    
    

