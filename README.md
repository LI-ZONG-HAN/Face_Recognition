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

How to packed in binary format ? pls see [Data](https://github.com/LI-ZONG-HAN/Face_Recognition/tree/master/Data)

### Train
Environment : windows 10 + Python 3.5(installed by Anaconda) + Tensorflow 1.5.0 GPU version

GPU : 1080Ti

Batch Size : 50

### How to run training

Step 1: Put the packed binary format data in the floder named "training"

Step 2: run train_Arc_loss_multi_task.py. First argument is to chose FR or Gender or Age which means which task you want to train.
-train_sets and -valid_sets mean the training and valid binary file name (not including Extension)
```
python train_Arc_loss_multi_task.py FR -train_sets bin_file_name -valid_sets bin_file_name
```
If you want to see all arguments, pls typing
```
python train_Arc_loss_multi_task.py -h
```
In the training process, trained model and training log will saved at "Model" floder

It will create a floder named loss_debug to save the loss of each step.

About learning rate decay.Instead of steps or polynomial decay, the learning rate will dacay automatically when loss stop improving.
it will make the network trained better because you never know how many steps the training need before you train it. 
[How to determine loss stop improving](http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html)

## Evaluation

### FR
```
python eval_FR.py 0 -model pb_file_path
```
The fisrt argument determine which dataset to evaluate.There are six options 0:LFW 1:asian_valid 2:asian_training 3:west_valid 4:west_training 5:Geo_test set

optional arguments is image width and feature dimetion

The result will show best average accuracy at same persion and different persion and the gap. The definition of gap is that distance between same and different people group divided by stdev of different people group.

Option 0~4 will run 5 times because we only evaluate a small set sampling from the whole data set. Option 5 evaluate whole data set so run 1 time only.

### Gender and age
```
python eval_gender_Age_with_label.py Age -dir eval_floder -model pb_file_path
```
Fisrt argument determine evaluate whether age or gender

the floder need to be structured as follows
```
eval_floder/
         label1/            
               image1.jpg
               image2.jpg            
               image3.jpg        
         label2/            
               image4.jpg            
               image5.jpg            
               image6.jpg        
         label3/            
               image7.jpg            
               image8.jpg            
               image9.jpg
```
The result will show the average accuracy over all labels and confusion matrix. 

## On going
### Training with larger batch size
Bigger batch size seems to make the network trained better


