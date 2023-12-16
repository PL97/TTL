# [Rethinking Transfer Learning for Medical Image Classification](https://arxiv.org/abs/2106.05152)
This is the offical implemention of paper **[Rethinking Transfer Learning for Medical Image Classification](https://arxiv.org/abs/2106.05152)** [BMVC'23 **oral**]



Overview            |  example of TTL on resent50
:-------------------------:|:-------------------------:
<img src="figures/overview.png" width="500" /> |  <img src="figures/svcca.png" width="450" />




# Usage
## Setup
### **pip**
Requires python>=3.10+


See the `requirements.txt` for environment configuration
```bash
pip install -r requirements.txt
```

## Dataset
<figure><img src="figures/examples.png"></figure>

### **BIMCV**
- Please download our pre-processed datasets [TBA](), put under `data/` directory and perform following commands:
    ```bash
    cd ./data
    unzip digit_dataset.zip
    ```

<!-- ### **MIDOG 2022**
- Please download the dataset [here](https://midog2022.grand-challenge.org/), put under `data/MIDOG/` -->

### **HAM10000**
- Please download the dataset [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T), put under `data/HAM10000/` 

### **PENet Dataset**
- Please download the dataset [here](https://github.com/marshuang80/penet), put under `data/PENet/` directory and perform following commands:


## Train
### 2D experiment (BIMCV & HAM1000)
**block-wise TTL**

Please using following commands to train a model with federated learning strategy.
- **--model** specify model archicture: resnet50 | densenet201
- **--pretrained** specify source domain: imagenet | chexpert
- **--dataset** specify target dataset: BIMCV | HAM10000
- **--trunc** specify truncation point: {-1, 1, 2, 3}

```bash
python main.py --model resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 --sub 100
```


**layer-wise TTL**

**--trunc** specify truncation point: {-1, 1, 2, ..., 16}

```bash
python main.py --model layerttl_resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 --sub 100
```

### Test
**block-wise TTL**

```bash
python main.py --model resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 --sub 100
```

**layer-wise TTL**

```bash
python main.py --model layerttl_resnet50 --bs 64 --data_parallel --num_workers 12 --max_epoch 200 --pretrained imagenet --dataset BIMCV --trunc -1 --exp 1 --sub 100
```

<figure><img src="figures/BIMVC.png"></figure>


If you use this code or dataset in you research, please consider citing our paper with the following Bibtex code:

```
@article{peng2022rethinking,
  title={Rethinking Transfer Learning for Medical Image Classification},
  author={Peng, Le and Liang, Hengyue and Luo, Gaoxiang and Li, Taihui and Sun, Ju},
  journal={medRxiv},
  pages={2022--11},
  year={2022},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
