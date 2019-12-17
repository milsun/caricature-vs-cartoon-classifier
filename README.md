# Caricature Vs Cartoon Classifier

This repo contains code for classifying caricature and cartoon images using Convolutional neural network(CNN) and transfer learning.

## Abstract

![VGG16](https://raw.githubusercontent.com/milsun/caricature-vs-cartoon-classifier/master/images/vgg16.png)


* A CNN based model trained on augmented 3.5k original images.
* Model takes advantage of transfer learning and uses pre-trained VGG16 model weights.
* Trained model is able to achieve 87% accuracy.
* Developed as a sub-project to filter cartoons from caricature dataset.
* Trained model can be found [here](https://github.com/milsun/caricature-vs-cartoon-classifier/tree/master/model).

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

### Usage

```
def load_image(filename):
    img = image.load_img(filename, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255. 

    return img_tensor

from keras.models import load_model

model = load_model('model.h5')
threshold = 0.2

if model.predict(load_image(img_name))[0][0] < threshold:
	print('Caricature')
else:
    print('Cartoon')

```


## Results
As you can see model works pretty well, and only gets confused with cartoon images containing a single face.


![Results](https://raw.githubusercontent.com/milsun/caricature-vs-cartoon-classifier/master/images/results.png)
