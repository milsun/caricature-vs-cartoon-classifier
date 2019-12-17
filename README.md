# Caricature Vs Cartoon Classifier

This repo contains code for classifying caricature and cartoon images using Convolutional neural network(CNN) and transfer learning.

## Abstract

![VGG16](https://raw.githubusercontent.com/milsun/caricature-vs-cartoon-classifier/master/images/vgg16.png)


* A CNN based model trained on augmented 3.5k original images.
* Model takes advantage of transfer learning and uses pre-trained VGG16 model weights.
* Trained model is able to achieve 87% accuracy.
* Developed as a sub-project to filter cartoons from caricature dataset.

## Getting Started


### Prerequisites

Run below command if you don't have python3 installed

```
sudo apt-get install python3.6
```

### Installing

Dependencies:

```
pip install tensorflow
pip install keras
pip install PyDrive
pip install numpy
pip install matplotlib
```

## Results
As you can see model works pretty well, and only gets confused with cartoon images containing a single face.


![Results](https://raw.githubusercontent.com/milsun/caricature-vs-cartoon-classifier/master/images/results.png)
# caricature-vs-cartoon-classifier
