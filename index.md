# Rethink Transfer Learning in Medical ImageClassification
Wednesday, June 9, 2021

Posted by Hegyue Liang, PhD student, Electrical and Computer Enginerring, University of Minnesota
 
**[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)** (TL) with deep convolutional neural networks (DCNNs) has proved successful in many medical image classification (MIC). In the conventional TL practice, the following two steps are taken :

    1. **Pretraining**: A DCNN is first trained on a general-purpose, large-scale benchmark computer vision dataset (e.g., [Imagenet](https://www.image-net.org/)).

    2. **Fine-tuning**: The pretrained DCNN is then trained on the target dataset of interest (e.g., chest x-ray images to identify diseases).

It is generally believed that the pretraining step helps the DCNN to learn general features of the natural images, which can be reused to the target task. As a result, TL can not only speed up the training, but also improve the performace.

However, the above mentioned TL practive is puzzling, in the sense that **MIC typically relies on ONLY low- and/or mid-level features** that are learned in the bottom layers of DCNNs. For example, in a chest x-ray image shown below, doctors relies on low-/mid-level visual features (such as blobs, oppacities) to diagnose pathologies, while high-level features (e.g., to tell that this is a human chest) is irrelevant to our diagnostic task.

![Chest x-ray](https://github.com/HengyueL/MedTL/chest_xray.pdf)

Follow this intuition, it is naturally questionable if the current TL practice is the best possible in MIC domain, since we may not need to reuse the high-level pretrained features. In our **Conference Name** paper, **Insert Paper name and Link Here**, we perform careful experimental comparisons on shallow and deep networks, with different TL strategies, to answer this question. Indeed, we find that:

    1. Deep models are **not** always favorable, but TL most often benefits the model performance, no matter the network is shallow or deep. This conclusion challenges part of the conclusion made in a prior work: [Transfusion:Understanding Transfer Learning for Medical Imaging](https://ai.googleblog.com/2019/12/understanding-transfer-learning-for.html).

    2. Fine-tuning **truncated** version of DCNNs almost always yields the best performance in the target MIC task. This, if confirmed further in other medical imaging tasks such as segmentation, can be a new pattern to practice TL in medical image domain.

    3. Point 2 is especially significant in **data-poor** regime.




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
