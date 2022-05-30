import argparse
import time
import torch.nn as nn
import torch
import os
import numpy as np

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
                    'Support Device': 13}

low_idx = [3, 5, 6, 7, 8, 10, 11]
high_idx = [0, 1, 2, 4, 9, 12, 13]


def parse_opts():
    # working_folder = "checkpoints_AdamW"
    # working_folder = "checkpoints_repeat"
    # working_folder = "checkpoints_pooling"
    # working_folder = "checkpoints_lr"
    # working_folder = "checkpoints_lr_large_random"

    # working_folder = "checkpoints"
    working_folder = "checkpoints_noisy_nopooling"
    # working_folder = "checkpoints_nopooling"
    # working_folder = "ablation_study/checkpoints_SGD"
    # working_folder = "cca_layer_wise/checkpoints_SGD"


    parser = argparse.ArgumentParser()


    # training hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--bs', type=int, default= 64, help ='batch size')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--eval_iter', type=int, default=-1, help='evalate training iteration-wise')
    
    
    # model config
    parser.add_argument('--model', type=str, default='densenet121', help='model archicture')
    parser.add_argument('--pretrained', type=str, default="imagenet", help="pretrained model weights")
    parser.add_argument('--trunc', type=int, default=-1, help="truncation level")
    # parser.add_argument('--classes', type=int, default=7)
    parser.add_argument('--slim_from', type=int, default=-1, help="slim model startring from x blocks")
    parser.add_argument('--finetune_from', type=int, default=-1, help="finetune model from x blocks")
    parser.add_argument('--ptl_decay', type=int, default=10, help="finetune model from x blocks")
    parser.add_argument('--slim_factor', type=int, default=2, help="finetune model from x blocks")
    
    


    # others
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1, help="number of workers")
    parser.add_argument('--data_parallel', action='store_true')
    parser.add_argument('--exp', type=int, default=1)
    parser.add_argument('--sub', type=int, default=100)
    parser.add_argument('--target', type=str, default="all")
    parser.add_argument('--cont', action='store_true', help ='continue training')
    parser.add_argument('--pooling', action='store_true', help ='continue training')
    
    
    # work space
    parser.add_argument('--dataset', type=str, default= 'HAM', help='specify work space')
    
    
    args = parser.parse_args()

    # modify this accordingly
    # args.root_path = "/home/luo00042/SATASSD2/trunc_tl/"
    # args.root_path = "/home/le/TL/sync/truncatedTL/"
    args.root_path = "/home/jusun/shared/"
    
    
    # hardcode, modify if necessary
    # args.root_path = "/home/le/project/TL/skin/archive/"
    # general for all dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # for HAM1000
    if args.dataset == "HAM":
        args.root_path = "{}/HAM/".format(args.root_path)
        args.weights = torch.Tensor([0.0065231 , 0.0393127 , 0.03981599, 0.08533731, 0.13457038, 0.3110071 , 0.38343341]).to(args.device)
        args.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=args.weights)
        args.classes = 7

    # for chexpert
    elif args.dataset == "CheXpert":
        if args.target == "all":
            args.weights = torch.Tensor([0.06844464, 0.07128722, 0.04574183, 0.01522274, 0.14422622, 0.02694098, 0.03982271, 0.06548477, 0.02583909, 0.07088983, 0.01689942, 0.2282677, 0.16624332, 0.01468954]).to(args.device)
            args.weights = args.weights / torch.min(args.weights)
            args.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=args.weights) 
            args.classes = 14
        elif args.target == "low" or args.target == "high":
            tmp_weight = np.asarray([17279.0, 16590.0, 25855.0, 77690.0, 8200.0, 43898.0, 29698.0, 18060.0, 45770.0, 16683.0, 69982.0, 5181.0, 7114.0, 80510.0])
            if args.target == "low":
                tmp_weight = tmp_weight[low_idx]
                args.classes = len(low_idx)
            else:
                tmp_weight = tmp_weight[high_idx]
                args.classes = len(high_idx)
            tmp_weight = 1/tmp_weight
            tmp_weight = tmp_weight/np.sum(tmp_weight)
            args.weights = torch.Tensor(tmp_weight).to(args.device)
            args.weights = args.weights / torch.min(args.weights)
            args.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=args.weights) 
            
        else:
            tmp_weight = np.asarray([17279.0, 16590.0, 25855.0, 77690.0, 8200.0, 43898.0, 29698.0, 18060.0, 45770.0, 16683.0, 69982.0, 5181.0, 7114.0, 80510.0])
            pos = tmp_weight[chexpert_to_num[args.target]]
            neg = sum(tmp_weight) - pos
            args.weights = torch.Tensor([pos/(pos+neg), neg/(neg+pos)]).to(args.device)
            args.weights = args.weights / torch.min(args.weights)
            args.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=args.weights) 
            args.classes = 2
        
    elif args.dataset == "BIMCV":
        args.root_path = "{}/BIMCV/".format(args.root_path)
        args.weights = torch.Tensor([0.59724951, 0.40275049]).to(args.device)
        args.weights = args.weights / torch.min(args.weights)
        args.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=args.weights) 
        args.classes = 2


    print("weights for positive classes:", args.weights)
    # saving config
    if args.target == "all":
        if "slim" in args.model:
            save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "slim":args.slim_from, "exp":args.exp, "dataset":args.dataset}
            args.saved_path = "../{working_folder}/{dataset}/{model}_{pretrained}_{slim}_{exp}/".format(**save_info)   
        elif "freeze" in args.model:
            save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "finetune":args.finetune_from, "exp":args.exp, "dataset":args.dataset}
            args.saved_path = "../{working_folder}/{dataset}/{model}_{pretrained}_{finetune}_{exp}/".format(**save_info)  
        else:
            save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "trunc":args.trunc, "exp":args.exp, "dataset":args.dataset}
            args.saved_path = "../{working_folder}/{dataset}/{model}_{pretrained}_{trunc}_{exp}/".format(**save_info)
    else:
        if args.dataset == "CheXpert" or args.pretrained != "random":
            if "slim" in args.model:
                save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "slim":args.slim_from, "exp":args.exp, "dataset":args.dataset, "target":args.target}
                args.saved_path = "../{working_folder}/{dataset}/{model}_{pretrained}_{slim}_{target}_{exp}/".format(**save_info)   
            elif "freeze" in args.model:
                save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "finetune":args.finetune_from, "exp":args.exp, "dataset":args.dataset, "target":args.target}
                args.saved_path = "../{working_folder}/{dataset}/{model}_{pretrained}_{finetune}_{target}_{exp}/".format(**save_info)  
            else:
                save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "trunc":args.trunc, "exp":args.exp, "dataset":args.dataset, "target":args.target}
                args.saved_path = "../{working_folder}/{dataset}/{model}_{pretrained}_{trunc}_{target}_{exp}/".format(**save_info)
        else:
            exit("target indexing not support for the specifided dataset")

    if args.sub != 100:
        args.saved_path = os.path.join(args.saved_path, str(args.sub))

    print(args.saved_path)

    save_file_name = os.path.join(args.saved_path, "final.pt")
    if (not args.test) and os.path.exists(save_file_name):
    # if False:
        exit("working folder already exists")
    else:
        os.makedirs(args.saved_path, exist_ok=True)

    if args.test:
        print("Testing on {}, load model from at {}".format(args.dataset, args.saved_path))
    else:
        print("Training on {}, create new exp container at {}".format(args.dataset, args.saved_path))

    return args


if __name__ == "__main__":
    args = parse_opts()




































# import argparse
# import time
# import torch.nn as nn
# import torch
# import os
# import numpy as np

# chexpert_classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
#        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
#        'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
#        'Fracture', 'Support Device']
       
# chexpert_to_num = {'No Finding': 0, 
#                     'Enlarged Cardiomediastinum':1, 
#                     'Cardiomegaly':2,
#                     'Lung Opacity':3, 
#                     'Lung Lesion':4, 
#                     'Edema':5, 
#                     'Consolidation':6, 
#                     'Pneumonia':7,
#                     'Atelectasis':8, 
#                     'Pneumothorax':9, 
#                     'Pleural Effusion':10, 
#                     'Pleural Other':11,
#                     'Fracture':12,
#                     'Support Device': 13}

# low_idx = [3, 5, 6, 7, 8, 10, 11]
# high_idx = [0, 1, 2, 4, 9, 12, 13]


# def parse_opts():
#     # working_folder = "checkpoints_AdamW"
#     # working_folder = "checkpoints_repeat"
#     # working_folder = "checkpoints_pooling"
#     # working_folder = "checkpoints_lr"
#     # working_folder = "checkpoints_lr_large_random"

#     working_folder = "checkpoints"
#     # working_folder = "checkpoints_sensitivity"
#     # working_folder = "checkpoints_ttest"
#     # working_folder = "checkpoints_network"
#     # working_folder = "ablation_study/checkpoints_SGD"
#     # working_folder = "cca_layer_wise/checkpoints_SGD"


#     parser = argparse.ArgumentParser()


#     # training hyper-parameters
#     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
#     parser.add_argument('--bs', type=int, default= 64, help ='batch size')
#     parser.add_argument('--max_epoch', type=int, default=50)
#     parser.add_argument('--eval_iter', type=int, default=-1, help='evalate training iteration-wise')
    
    
#     # model config
#     parser.add_argument('--model', type=str, default='densenet121', help='model archicture')
#     parser.add_argument('--pretrained', type=str, default="imagenet", help="pretrained model weights")
#     parser.add_argument('--trunc', type=int, default=-1, help="truncation level")
#     # parser.add_argument('--classes', type=int, default=7)
#     parser.add_argument('--slim_from', type=int, default=-1, help="slim model startring from x blocks")
#     parser.add_argument('--finetune_from', type=int, default=-1, help="finetune model from x blocks")
#     parser.add_argument('--ptl_decay', type=int, default=10, help="finetune model from x blocks")
#     parser.add_argument('--slim_factor', type=float, default=2, help="finetune model from x blocks")
    
    


#     # others
#     parser.add_argument('--test', action='store_true', help ='test the pretrained model')
#     parser.add_argument('--pin_memory', action='store_true')
#     parser.add_argument('--num_workers', type=int, default=1, help="number of workers")
#     parser.add_argument('--data_parallel', action='store_true')
#     parser.add_argument('--exp', type=int, default=1)
#     parser.add_argument('--sub', type=int, default=100)
#     parser.add_argument('--target', type=str, default="all")
#     parser.add_argument('--cont', action='store_true', help ='continue training')
#     parser.add_argument('--pooling', action='store_true', help ='continue training')
    
    
#     # work space
#     parser.add_argument('--dataset', type=str, default= 'HAM', help='specify work space')
    
    
#     args = parser.parse_args()

#     # modify this accordingly
#     # args.root_path = "/home/luo00042/SATASSD2/trunc_tl/"
#     args.root_path = "/home/le/TL/sync/truncatedTL/"
    
    
#     # hardcode, modify if necessary
#     # args.root_path = "/home/le/project/TL/skin/archive/"
#     # general for all dataset
#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
#     # for HAM1000
#     if args.dataset == "HAM":
#         args.root_path = "{}/HAM/".format(args.root_path)
#         args.weights = torch.Tensor([0.0065231 , 0.0393127 , 0.03981599, 0.08533731, 0.13457038, 0.3110071 , 0.38343341]).to(args.device)
#         args.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=args.weights)
#         args.classes = 7

#     # for chexpert
#     elif args.dataset == "CheXpert":
#         if args.target == "all":
#             args.weights = torch.Tensor([0.06844464, 0.07128722, 0.04574183, 0.01522274, 0.14422622, 0.02694098, 0.03982271, 0.06548477, 0.02583909, 0.07088983, 0.01689942, 0.2282677, 0.16624332, 0.01468954]).to(args.device)
#             args.weights = args.weights / torch.min(args.weights)
#             args.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=args.weights) 
#             args.classes = 14
#         elif args.target == "low" or args.target == "high":
#             tmp_weight = np.asarray([17279.0, 16590.0, 25855.0, 77690.0, 8200.0, 43898.0, 29698.0, 18060.0, 45770.0, 16683.0, 69982.0, 5181.0, 7114.0, 80510.0])
#             if args.target == "low":
#                 tmp_weight = tmp_weight[low_idx]
#                 args.classes = len(low_idx)
#             else:
#                 tmp_weight = tmp_weight[high_idx]
#                 args.classes = len(high_idx)
#             tmp_weight = 1/tmp_weight
#             tmp_weight = tmp_weight/np.sum(tmp_weight)
#             args.weights = torch.Tensor(tmp_weight).to(args.device)
#             args.weights = args.weights / torch.min(args.weights)
#             args.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=args.weights) 
            
#         else:
#             tmp_weight = np.asarray([17279.0, 16590.0, 25855.0, 77690.0, 8200.0, 43898.0, 29698.0, 18060.0, 45770.0, 16683.0, 69982.0, 5181.0, 7114.0, 80510.0])
#             pos = tmp_weight[chexpert_to_num[args.target]]
#             neg = sum(tmp_weight) - pos
#             args.weights = torch.Tensor([pos/(pos+neg), neg/(neg+pos)]).to(args.device)
#             args.weights = args.weights / torch.min(args.weights)
#             args.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=args.weights) 
#             args.classes = 2
        
#     elif args.dataset == "BIMCV":
#         args.root_path = "{}/BIMCV/".format(args.root_path)
#         args.weights = torch.Tensor([0.59724951, 0.40275049]).to(args.device)
#         args.weights = args.weights / torch.min(args.weights)
#         args.criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=args.weights) 
#         args.classes = 2


#     print("weights for positive classes:", args.weights)
#     # saving config
#     if args.target == "all":
#         if "slim" in args.model:
#             save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "slim":args.slim_from, "exp":args.exp, "dataset":args.dataset, "slim_factor": args.slim_factor}
#             args.saved_path = "../{working_folder}/{dataset}/slim_{model}_{pretrained}_{slim}_{exp}_{slim_factor}/".format(**save_info)   
#         elif "freeze" in args.model:
#             save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "finetune":args.finetune_from, "exp":args.exp, "dataset":args.dataset, "ptl_decay": args.ptl_decay}
#             args.saved_path = "../{working_folder}/{dataset}/freeze_{model}_{pretrained}_{finetune}_{exp}_{ptl_decay}/".format(**save_info)  
#         else:
#             save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "trunc":args.trunc, "exp":args.exp, "dataset":args.dataset}
#             args.saved_path = "../{working_folder}/{dataset}/trunc_{model}_{pretrained}_{trunc}_{exp}/".format(**save_info)
#     else:
#         if args.dataset == "CheXpert" or args.pretrained != "random":
#             if "slim" in args.model:
#                 save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "slim":args.slim_from, "exp":args.exp, "dataset":args.dataset, "target":args.target, "slim_factor": args.slim_factor}
#                 args.saved_path = "../{working_folder}/{dataset}/slim_{model}_{pretrained}_{slim}_{target}_{exp}_{slim_factor}/".format(**save_info)   
#             elif "freeze" in args.model:
#                 save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "finetune":args.finetune_from, "exp":args.exp, "dataset":args.dataset, "target":args.target, "ptl_decay": args.ptl_decay}
#                 args.saved_path = "../{working_folder}/{dataset}/freeze_{model}_{pretrained}_{finetune}_{target}_{exp}_{ptl_decay}/".format(**save_info)  
#             else:
#                 save_info = {"working_folder":working_folder, "pretrained": args.pretrained, "model": args.model, "trunc":args.trunc, "exp":args.exp, "dataset":args.dataset, "target":args.target}
#                 args.saved_path = "../{working_folder}/{dataset}/trunc_{model}_{pretrained}_{trunc}_{target}_{exp}/".format(**save_info)
#         else:
#             exit("target indexing not support for the specifided dataset")

#     if args.sub != 100:
#         args.saved_path = os.path.join(args.saved_path, str(args.sub))

#     print(args.saved_path)

#     save_file_name = os.path.join(args.saved_path, "final.pt")
#     if (not args.test) and os.path.exists(save_file_name):
#         exit("working folder already exists")
#     else:
#         os.makedirs(args.saved_path, exist_ok=True)

#     if args.test:
#         print("Testing on {}, load model from at {}".format(args.dataset, args.saved_path))
#     else:
#         print("Training on {}, create new exp container at {}".format(args.dataset, args.saved_path))

#     return args


# if __name__ == "__main__":
#     args = parse_opts()
