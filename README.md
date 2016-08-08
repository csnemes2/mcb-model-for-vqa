# mcb-model-for-vqa


This is an implementation of [Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Groudning](https://arxiv.org/abs/1606.01847) with Tensorflow.  
A theorical background of this paper is in [mcb_for_vqa.pdf](https://github.com/shmsw25/mcb-model-for-vqa/blob/master/mcb_for_vqa.pdf).  
Thanks to Yunseok Jang, Hyugjin Ko, and [Sangeon Park](https://github.com/pse1202).  


### Codes

Before Training(Caffe is used)
- build_train_datasets.py : build annotations for train(with idx of image feature vector, question, and idx of answer) and feature vectors. Also build word2ix and ix2word for questions and answers.
- build_val_datasets.py : build_annotations for valid(with image_id, idx of image feature vector, question, question_id) and feature vectors.


Models and Train & Test codes(Tensorflow is used)
- config.py : configuration file (support model without Attention & with Attention)
- model
    - vqamodel.py : an abstract model for VQA
    - mcb_vqamodel.py : a MCB model which do not have attention mapping
    - mcbAtt_vqamodel.py : a MCB model which have attention mapping
    - concat_vqamodel.py : a non-bilinear model(Use concatenate to make feature)
    - CBP : Module for compact bilinear pooling from [here](github.com/therne/compact-bilinear-pooling-tf).
- train.py : a code for training
- test.py : a code for test
- vqaEvaluation, vqaTools : Metric Evaluation Tool for VQA dataset from [here](https://github.com/VT-vision-lab/VQA/).


### Datasets

Datasets are available on [here](http://visualqa.org/download.html). Only Real Images and OpenEnded questions are used.
3 questions for an image, 10 answers per a question.
**I only used 50000 images and 150000 questions. Split training & validation set with 9 : 1 rate.**
- In questions, 10525 different words exist and most frequent 5000 words are selected.(threshold 3) In paper, 13K~20K words were selected.
- In answers, 50697 different words exist and most frequent 5000 words are selected.(threshold 9) In paper, 3000 words were selected.
- Length of questions
    - 0~9 : 126520, 10~19 : 8465, 20~29 : 16
    - Consider only 20 words in front when embedding questions
- Use answers with confidence 'yes' only : Total 1049879 numbers of (Image, Question, Answer) are used.

### You might also want to look at

- Other MCB models  
  - A. Fukui et al. Multimodal compact bilinear pooling for visual question answering and visual grounding. 2016.
- Bilinear pooling  
  - T.-Y. Lin et al. Bilinear CNN models for fine-grained visual recognition. 2015.  
  - J. Carreira et al. Semantic segmentation with second-order pooling. 2012.  
- Compact bilinear pooling & Count sketch  
  - Y. Gao et al. Compact bilinear pooling. 2016.  
  - N. Pham and R. Paph. Fast and scalable polynomial kernels via explicit feature maps. 2013.  
  - M. Charikar et al. Finding frequent items in data streams. 2002.  
  - K. Q. Weinberger et al. Feature hashing for large scale multitask learning. 2009  
  - R. Pagh. Compressed matrix multiplication. 2012.  
- Models referenced for Visual Grounding  
  - L. A. Hendricks et al. Generating Visual Explanations. 2016.  
  - A. Rohrbach et al. Grounding of Textual Phrases in Images by Reconstruction. 2016.  
- More for MCB for Visual QA  
  - [VQA model slides from UCB](http://visualqa.org/static/slides/vqa_final.pdf)  
  - [VQA Demo](demo.berkeleyvision.org)  
  - [VQA Code from MCB](https://github.com/akirafukui/vqa-mcb)  
  - [VQA challenge](http://visualqa.org/challenge.html)  

