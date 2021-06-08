# Rethink Transfer Learning in Medical ImageClassification
Wednesday, June 9, 2021

Posted by Hegyue Liang, PhD student, Electrical and Computer Enginerring, University of Minnesota
 
**some test bold text**
[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) (TL) with deep convolutional neural networks (DCNNs) has proved successful in medical image classification (MIC). The following two steps are taken in the conventional TL practice:
1) Pretraining: A DCNN is first trained on a general-purpose, large-scale benchmark computer vision dataset (e.g., [Imagenet](https://www.image-net.org/)).
2) Fine-tuning: The pretrained DCNN is then trained on the target dataset of interest (e.g., chest x-ray images to identify diseases).

It is generally believed that the pretraining step helps the DCNN to learn general features of the natural images, which can be reused to the target task, only speeding up the training, but also improving the performace.

However, the above mentioned practice of TL is puzzling, in the sense that MIC typically relies only on low- and/or mid-level features that are learned in the bottom layers of DCNNs. For example, in a chest x-ray image shown below, doctors relies on low-/mid-level visual features (such as blobs, oppacities) to diagnose pathologies.
[chest_xray.pdf](https://github.com/HengyueL/MedTL/files/6619498/chest_xray.pdf)





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
