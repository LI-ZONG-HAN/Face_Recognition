# Make images data to binary files for training
## Why
The older way i used is that load separate images in HD in each batch.It takes a long time at open each image files. Therefor i packed all image files in one big binary file and the position for each image in the bin file is recorded in the idx file. 
## content
After data is packed.Two file will be created.

bin: image data

idx: record original_file_path / label / start position in bin / end position in bin


### Training Data
Training data is downloaded from [Trillion Pairs](http://trillionpairs.deepglint.com/overview) (both MS-Celeb-1M-v1c and Asian-Celeb)

All face images are aligned by [Dlib: 5-point landmark](http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html) and and cropped to 224x224 and packed in binary format

How to packed in binary format ? pls see Data floder

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

## On going
### Training with larger batch size
Bigger batch size seems to make the network trained better

### Multi-task training 
I plan to train three tasks which are FR, gender and age classification in the same network. They will share the resnet-50 network. I have finished the code. 
