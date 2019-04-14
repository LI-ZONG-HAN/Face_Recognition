# Face_Recognition
## Introduction
My backbone network is Resnet-50 V1 ,whose output dimension is 512, trained by "ArcLoss" loss layer. My training data is provided by Trillion Pairs which is cleaned from MS-Celeb-1M and Asian-Celeb.The best performance for face recognition so far is 1.LFW: 99.3% accurace 2. Megaface: 93% true accept rate under 0.0001% false accept rate on 1:1 verification
## Contents

### Model
  Backbone network : Resnet-50 V1 with 512 dim output. My [Pre-trained model] (https://github.com/tensorflow/models/tree/master/research/slim) is provided by Tensorflow
  
  Loss layer : [Arc Loss layer](https://github.com/deepinsight/insightface)
### Training Data
Training data is downloaded from [Trillion Pairs](http://trillionpairs.deepglint.com/overview) (both MS-Celeb-1M-v1c and Asian-Celeb)

All face images are aligned by [Dlib: 5-point landmark](http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html) and and cropped to 224x224 and packed in binary format

### Train
Environment : windows 10 + Python 3.5(installed by Anaconda) + Tensorflow 1.5.0 GPU version

GPU : 1080Ti

Batch Size : 50

### How to run training

Step 1: Put the packed binary format data in the floder named "training"

Step 2: run train_Arc_loss_multi_task.py
```
python train_Arc_loss_multi_task.py
```

About learning rate decay.Instead of steps or polynomial decay, the learning rate will dacay automatically when loss stop improving.
it will make the network trained better because you never know how many steps the training need before you train it. 

[How to determine loss stop improving](http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html)
