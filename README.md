# Face_Recognition
## Contents

### Model
  Feature extraction : Resnet-50 V1. My Pre-trained model is provided by Tensorflow [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim)
  
  Loss layer : [Arc Loss layer](https://github.com/deepinsight/insightface)
### Training Data
Training data is downloaded from [Trillion Pairs](http://trillionpairs.deepglint.com/overview) (both MS-Celeb-1M-v1c and Asian-Celeb)

All face images are aligned by [Dlib: 5-point landmark](http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html)

