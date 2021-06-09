# Rethink Transfer Learning in Medical Image Classification

Le Peng<sup>1</sup>, Hengyue Liang<sup>2</sup>, Taihui Li<sup>1</sup>, Ju Sun<sup>1</sup>

<sup>1</sup> Computer Science and Engineering, University of Minnesota

<sup>2</sup> Electrical and Computer Engineering, University of Minnesota
 
**[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)** (TL) with deep convolutional neural networks (DCNNs) has proved successful in many medical image classification (MIC). In the conventional TL practice, the following two steps are taken :

   1. **Pretraining**: A DCNN is first trained on a general-purpose, large-scale benchmark computer vision dataset (e.g., [ImageNet](https://www.image-net.org/)).

   2. **Fine-tuning**: The pretrained DCNN is then trained on the target dataset of interest (e.g., chest x-ray images to identify diseases).

It is generally believed that the pretraining step helps the DCNN to learn general features of the natural images, which can be reused to the target task. As a result, TL can not only speed up the training, but also improve the performace.

However, the above mentioned TL practive is puzzling, in the sense that **MIC typically relies on ONLY low- and/or mid-level features** that are learned in the bottom layers of DCNNs. For example, in a chest x-ray image shown below, doctors relies on low-/mid-level visual features (such as blobs, oppacities) to diagnose pathologies, while high-level features (e.g., to tell that this is a human chest) is irrelevant to our diagnostic task.

<div align="center">
<figure><img src="figures/chest-Xray.png" width="512"></figure>

<figcaption>Figure 1: An example chest X-ray with observable pathologies. Area marked in colored box are pathologies annotated by experienced doctors.</figcaption>
</div>

Follow this intuition, it is naturally questionable if the current TL practice is the best possible in MIC domain, since we may not need to reuse the high-level pretrained features. In our **Conference Name** paper, **Insert Paper name and Link Here**, we perform careful experimental comparisons on shallow and deep networks, with different TL strategies, to answer this question. Indeed, we find that:

   1. Deep models are **not** always favorable, but TL most often benefits the model performance, no matter the network is shallow or deep. This conclusion challenges part of the conclusion made in a prior work: [Transfusion:Understanding Transfer Learning for Medical Imaging](https://ai.googleblog.com/2019/12/understanding-transfer-learning-for.html).

   2. Fine-tuning **truncated** version of DCNNs almost always yields the best performance in the target MIC task. This, if confirmed further in other medical imaging tasks such as segmentation, can be a new pattern to practice TL in medical image domain.

   3. Point 2 is especially significant in **data-poor** regime.

In what follows, we briefly introduce the main messages we would like to convey in our paper.


## Both ROC and PRC should be used to evaluate classifier performance in MIC

In MIC tasks, it is common for the dataset to be highly imbalanced. (For example, one can expect that most medical scans attempting to diagnose a rare disease will return negative result.) In machine learning community, it is well know that precision-recall-curve **(PRC)** is more informative and indicative of the true performance than the receiver-operating-characteristic **(ROC)** curve under dominant negative classes. Here is a simple examplt to illustate the difference between **ROC** and **PRC**:

Let's consider a dataset consisting of 10 positives and 990 negatives for a rare disease. Assume classifier A (CA) scores the positives uniformly random distributed over its top 12 predictions; and classifier B (CB) scores the positives uniformly random distributed over its top 30 predictions. Intuitively, CA is a much better classifer as they detect 1 true-positive (TP) at the cost of 0.2 false-positive (FP), comparing with 1 TP:2 FP for CB --- this information is captured by **Precision**, but not recall.

<div align="center">
<figure><img src="figures/roc.png" width="215"><img src="figures/prc.png" width="215"><img src="figures/conf_table.png" width="215"></figure>

<figcaption>Figure 2: ROC (left) and PRC (middle) of CA and CB for the example classification task. The ROC curves for both classifiers are high and close, almost showing no significant performance gap; while the PRC curve seperate two classifiers apart, reflecting the precision gap of the two classifiers. The confusion table (Right) recaps the definition of ROC, PRC and relevant terminologies.</figcaption>
</div>


Figure 2 depicts the performance of CA and CB using ROC and PRC metric respectively. As PRC takes into account the precision, it is able to separate CA and CB by a large margin and give a clear indication which is the better classifier under the example imbalanced classification problem. Indeed, one of the main reasons that our paper draws conclusions that challenge the previous work [Transfusion](https://ai.googleblog.com/2019/12/understanding-transfer-learning-for.html) is that they only evaluate their models with ROC curve, whereas we evalute our models with both ROC and PRC curves.

## Transfer Learning (TL) v.s. Random Initialization (RI)

We compare TL and RI on deep networks (DenseNet121, ResNet50, as both are popular choices in Medical Imaging Classification tasks) and shallow networks (CBR families as introduced in [Transfusion](https://ai.googleblog.com/2019/12/understanding-transfer-learning-for.html)). We choose the public dataset [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) as the data-rich regime, since the whole dataset consists of 224,316 chest radiographs, much more than typicle medical datasets that are maintained in local hospitals/institutions; we simulate the data-poor regime with our [local COVID dataset obtained from M Health Fairview, Minnesota](https://www.medrxiv.org/content/10.1101/2021.06.04.21258316v1) (the smallest subset consists of only 88 positives and 1451 negatives), whose size is more commonly seen in real-life practice.

We find from our experiment that:

  1. Under data-rich regime:

  2. Under data-poor regime:


## Truncated Transfer Learning


## Discussion


## Acknowledgements


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/HengyueL/MedTL/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.


First posted by Hegyue Liang, Wednesday, June 9, 2021.
