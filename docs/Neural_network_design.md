# Design of the neural network

## [VanillaCNN](https://github.com/ishay2b/VanillaCNN)

The design of the neural network is inspired by VanillaCNN.

The steps VanillaCNN uses to produce the landmark results are as follows:

1. VanillaCNN uses an external face detector to detect the faces present in an image. This face detector generates bounding boxes for every detected face.
2. The bounding box is used to crop the original image. The cropped image is resized and fed to the VanillaCNN neural network.
3. VanillaCNN predicts the location of the facial landmarks, relative to the bounding box.
4. The facial landmarks predicted by the neural network are projected to the coordinate system of the original image. The landmarks can now be overlayed on top of the original image.

VanillaCNN detects 5 facial landmarks (left pupil, right pupil, nose, left mouth corner, right mouth corner) and is trained on the MTFL, AFLW and AFW datasets. VanillaCNN uses a convolutional neural network with an input size of 40 by 40 pixels. Each convolutional layer is followed by a hyperbolic tangent layer, and absolute value layer and a pooling layer. The network consists of a total of 4 convolutional layers, followed by 2 inner product layers. The last innerproduct layer has 10 outputs. These are the x and y coordinates of every landmark. For training mean squared error loss is used. This mean squared error loss is divided by the interocular distance to normalize for different sized faces.

## 98 landmark design

The 98 landmark network is designed to be run on the DEEPHi DPU. Because the DPU as of yet does not support the tanh function, the tanh and absolute value units in the VanillaCNN network were replaced by the ReLU activation function. ReLU is also less computationally expensive and can be faster to train due to the gradient being non-zero for a larger part of the activation function. The network was retrained to confirm its function. This reduced the testing loss of the neural network from 0.022 to 0.019. To try to improve the spatial resolution of the neural network, the network was redesigned with dilated convolutions instead of pooling layers. This further improved the testing loss of the network from 0.019 to 0.011.

This 5 landmark design was scaled up to a 98 landmark design. To try to improve the performance of the model after quantization, the model was trained with added noise. This noise is generated as a uniform distribution with a mean of 1.0. The noise is then multiplied by the outputs of the ReLU layers.