# mcb-model-for-vqa


This is an implementation of [Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Groudning](https://arxiv.org/abs/1606.01847).  
A theorical background of this paper is in [mcb_for_vqa.pdf](https://github.com/shmsw25/mcb-model-for-vqa/blob/master/mcb_for_vqa.pdf).  
Thanks to Yunseok Jang, Hyugjin Ko, and [Sangeon Park](https://github.com/pse1202).  

### Codes

qa_model.py : an implementation of a model
train.py : a code for training
CBP : compact bilinear model from [here](https://github.com/therne/compact-bilinear-pooling-tf).


### Datasets

Datasets are available on [here](http://visualqa.org/download.html). Only Real Images and OpenEnded questions are used.


### You might also want to look at

- Other MCB models  
-- A. Fukui et al. Multimodal compact bilinear pooling for visual question answering and visual grounding. 2016.
- Bilinear pooling  
-- T.-Y. Lin et al. Bilinear CNN models for fine-grained visual recognition. 2015.  
-- J. Carreira et al. Semantic segmentation with second-order pooling. 2012.  
- Compact bilinear pooling & Count sketch  
-- Y. Gao et al. Compact bilinear pooling. 2016.  
-- N. Pham and R. Paph. Fast and scalable polynomial kernels via explicit feature maps. 2013.  
-- M. Charikar et al. Finding frequent items in data streams. 2002.  
-- K. Q. Weinberger et al. Feature hashing for large scale multitask learning. 2009  
-- R. Pagh. Compressed matrix multiplication. 2012.  
- Models referenced for Visual Grounding  
-- L. A. Hendricks et al. Generating Visual Explanations. 2016.  
-- A. Rohrbach et al. Grounding of Textual Phrases in Images by Reconstruction. 2016.  
- More for MCB for Visual QA  
-- [VQA model slides from UCB](http://visualqa.org/static/slides/vqa_final.pdf)  
-- [VQA Demo](demo.berkeleyvision.org)  
-- [VQA Code from MCB](https://github.com/akirafukui/vqa-mcb)  
-- [VQA challenge](http://visualqa.org/challenge.html)  

