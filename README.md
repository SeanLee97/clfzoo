<p align="center">/ clfzoo /</p>

Eng / [CN](https://github.com/SeanLee97/clfzoo/blob/master/docs/ZH_README.md)

clfzoo is a toolkit for text classification. We have implemented some baseline models, such as TextCNN, TextRNN, RCNN, Transformer, HAN, DPCNN. And We have designed a unified and friendly API to train / predict / test the models. Looking forward to your code contributions and suggestions.

## Requiements
```
python3+
numpy
sklearn
tensorflow>=1.6.0
```

## Installation
```
git clone https://github.com/SeanLee97/clfzoo.git
cd clfzoo
```

## Overview
```
project
│    README.md
│
└─── docs
│
└─── clfzoo    # models
│   │  base.py       # base model template
│   │  config.py     # default configure
│   │  dataloader.py
│   │  instance.py   # data instance
│   │  vocab.py      # vocabulary
│   │  libs          # layers and functions
│   │  dpcnn         # implement dpcnn model
│   │   │  __init__.py  # model apis
│   │   │  model.py     # model
│   │  ...           # implement other models
└───examples
    │   ...
```

### Data Prepare
Each line is a document. The line format is "label \t sentence". The default word tokenizer is split by blank space, so words in sentence should split by blank space.

for english sample

```
greeting    how are you.
```

for chinese sample
```
打招呼  你 最近 过得 怎样 啊 ？
```

### Usage

#### train
```python
# import model api
import clfzoo.textcnn as clf  

# import model config
from clfzoo.config import ConfigTextCNN

"""define model config

You can assign value to hy-params defined on base model config (here is ConfigTextCNN)
"""

class Config(ConfigTextCNN):
    def __init__(self):
        # it is required to implement super() function
        super(Config, self).__init__()

    # it is required to provide dataset
    train_file = '/path/to/train'
    dev_file = '/path/to/test'
    
    # ... other hy-params

# `training` is flag to indicate train mode.
clf.model(Config(), training=True)

# start to train
clf.train()
```

The train log will output to `log.txt`, the model weights and checkpoint summaries will output to `models` folder.

#### predict

Predit the labels and probability scores.

```python
import clfzoo.textcnn as clf
from clfzoo.config import ConfigTextCNN

class Config(ConfigTextCNN):
    def __init__(self):
        super(Config, self).__init__()
    
    # the same hy-params as train

# inject config to model
clf.model(Config())

"""
Input: a list
    each item in list is a sentence string split by blank space (for chinese sentence you should prepare your input data first)
"""
datas = ['how are u ?', 'what is the weather ?', ...]

"""
Return: a list
    [('label 1', 'score 1'), ('label 2', 'score 2'), ...]
"""
preds = clf.predict(datas)
```

#### test

Predit the labels and probability scores and get result metrics. In order to calculate metrics you should provide ground-truth label.

```python
import clfzoo.textcnn as clf
from clfzoo.config import ConfigTextCNN

class Config(ConfigTextCNN):
    def __init__(self):
        super(Config, self).__init__()
    
    # the same hy-params as train

# inject config to model
clf.model(Config())

"""
Input: a list
    each item in list is a sentence string split by blank space (for chinese sentence you should prepare your input data first)
"""
datas = ['how are u ?', 'what is the weather ?', ...]
labels = ['greeting', 'weather', ...]

"""
Return: a tuple
    - predicts: a list
        [('label 1', 'score 1'), ('label 2', 'score 2'), ...]
    - metrics: a dict
        {'recall': '', 'precision': '', 'f1': , 'accuracy': ''}
"""
preds, metrics = clf.test(datas, labels)
```


## Benchmark Results
here we use [smp2017-ECDT](https://arxiv.org/abs/1709.10217) dataset as an example, which is a multi-label (31 labels)、short-text and chinese dataset.

We train all models in 20 epochs, and calculate metrics by sklearn metrics functions. As we all know [fasttext](https://github.com/facebookresearch/fastText) is a strong baseline in text-classification, so here we give the result on fasttext

|  Models  | Precision   | Recall   | F1   |
| ------------ | ------------ | ------------ | ------------ |
|  fasttext  |  0.81  | 0.81  | 0.81  |
|   TextCNN |  0.83  | 0.84   | 0.83   |
|   TextRNN |  0.84  | 0.83   |  0.82  |
|   RCNN |  0.86  | 0.85   | 0.85   |
|   DPCNN |  0.87  | 0.85  | 0.85 |
|   Transformer |  0.67  | 0.65   | 0.64  |
|   HAN |  TODO  | TODO   | TODO   |

**Attention!**  It seems that Transformer and HAN can`t perform well now, We will fix bugs and update their result later.

## Contributors
- sean lee
    - a single coder 
    - [seanlee97@github.io](https://seanlee97.github.io/)
- x.m. li
    - a undergraduate student from Shanxi University
    - [holahack@github](https://github.com/holahack)
- ...

## Refrence
Some code modules from

- [transformer](https://github.com/Kyubyong/transformer)
- [artf](https://github.com/SeanLee97/artf)

Papers

- TextCNN: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- DPCNN: [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- HAN: [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

## Contact Us
Any questions please mailto xmlee97#gmail.com
