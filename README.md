# Face_Recognition
## Introduction
My backbone network is Resnet-50 V1 ,whose output dimension is 512, trained by "ArcLoss" loss layer. My training data is provided by Trillion Pairs which is cleaned from MS-Celeb-1M and Asian-Celeb.The best performance for face recognition so far is 1.LFW: 99.3% accurace 2. 93% true accept rate under 0.0001 % false accept rate on 1:1 verification
## Contents

### Model
  Feature extraction : Resnet-50 V1. My [Pre-trained model] (https://github.com/tensorflow/models/tree/master/research/slim) is provided by Tensorflow
  
  Loss layer : [Arc Loss layer](https://github.com/deepinsight/insightface)
### Training Data
Training data is downloaded from [Trillion Pairs](http://trillionpairs.deepglint.com/overview) (both MS-Celeb-1M-v1c and Asian-Celeb)

All face images are aligned by [Dlib: 5-point landmark](http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html)

