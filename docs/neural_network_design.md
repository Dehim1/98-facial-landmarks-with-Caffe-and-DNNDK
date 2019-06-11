# Design of the neural network

## [VanillaCNN](https://github.com/ishay2b/VanillaCNN)

The design of the neural network is inspired by VanillaCNN.

The steps VanillaCNN uses to produce the landmark results are as follows:

1. VanillaCNN uses an external face detector to detect the faces present in an image. This face detector generates bounding boxes for every detected face.
2. The bounding boxes are used to crop the original image. Each cropped image is resized and fed to the VanillaCNN neural network.
3. VanillaCNN predicts the location of the facial landmarks, relative to the bounding box.
4. The facial landmarks predicted by the neural network are projected to the coordinate system of the original image. The landmarks can now be overlayed on top of the original image.

VanillaCNN detects 5 facial landmarks (left pupil, right pupil, nose, left mouth corner, right mouth corner) and is trained on the MTFL, AFLW and AFW datasets. VanillaCNN uses a convolutional neural network with an input size of 40 by 40 pixels. Each convolutional layer is followed by a hyperbolic tangent layer, an absolute value layer and a pooling layer. The network consists of a total of 4 convolutional layers, followed by 2 fully connected layers. The last fully connected layer has 10 outputs. These are the x and y coordinates of every landmark. For training mean squared error loss is used. This mean squared error loss is divided by the interocular distance to normalize for different sized faces.

## 98 landmark design

The 98 landmark network is designed to be run on the DEEPHi DPU. Because the DPU as of yet does not support the tanh function, the tanh and absolute value units in the VanillaCNN network were replaced by the ReLU activation function. ReLU is also less computationally expensive and can be faster to train due to the gradient being non-zero for a larger part of the activation function. The network was retrained to confirm its function. This reduced the testing loss of the neural network from 0.022 to 0.019. To try to improve the spatial resolution of the neural network, the network was redesigned with dilated convolutions instead of pooling layers. This further improved the testing loss of the network from 0.019 to 0.011.

This 5 landmark design was scaled up to a 98 landmark design. The input was changed to a size of 80 by 80. Since the 98 landmark network must predict 98 x and 98 y coordinates, the 10 output fully connected layer was replaced by a 196 output fully connected layer. To try to improve the performance of the model after quantization, the model was trained with added noise. This noise is generated as a uniform distribution with a mean of 1.0 and a maximum absolute deviation of 0.050. The noise is then multiplied by the outputs of the ReLU layers. The deviation of this noise can be increased to increase regularization, or decreased to allow the network to converge to a lower training loss.

To get more data, the 68 landmark datasets from ibug are also used during training. Because these landmarks are not a subset of the WFLW dataset, an extra top layer is added to the model. This top layer is only used during training. The other layers of the network are shared with the 98 landmark top layer. This increases the training set size from 7500 to 11936. Similarly more datasets can be addded to further increase the dataset size.

To further reduce testing loss, the neural network is also trained to predict the attribute flags provided by the WFLW dataset. This is done by adding another fully connected layer with a sigmoid activation function for each of the attribute flags. The neural network can use the knowledge it learns from these attribute flags to get better at predicting landmark locations.

Because this neural network will be used as a preprocessing step for face recognition, another fully connected layer is added. This fully connected layer will be trained to predict a transformation matrix which transforms the image such that the mean squared error between the original landmarks and a list of reference landmarks is as small as possible. This should align each image such that each face has approximately the same size, position and orientation when it is fed to the face recognition network. For this the python layer [Transform2DPoints](../python/Transform2DPoints.py) was created.

All of the top layers use a Euclidean loss layer. The outputs of these loss layers are fed to a custom layer, which averages each loss over multiple iterations using an exponentially weighted average. The derivative of each of the loss layers is set to the average of all of the exponentially weighted averages, divided by each of the individual exponentially weighted averages. This should cause each loss layer to converge at approximately the same rate. The idea behind this layer was obtained from https://arxiv.org/abs/1705.07115. To implement this the python layer [MultitaskLoss](../python/MultitaskLoss.py) was created.

&nbsp;

At first only dilated convolutions were used to reduce the height and width of the neural network. No strided or padded convolutions were used. This resulted in a network with a reasonable accuracy, but the network was fairly slow to train. This seemed to be because a high dilation rate causes training to slow down.

To try to find a well performing model, multiple different neural network architectures were tried. The best performing models used same padding in almost every layer and used strided convolutions to reduce the height and width of the neural network. The strided convolutions used a filter kernel of size 2 by 2 with a stride of 2, whereas the other layers used filter kernels of size 3 by 3 with a stride of 1. Once the height and width of the network got as low as 5 by 5, no padding was used and the height and width of the neural network were further reduced to 1 by 1 using 3 by 3 convolutions. All fully connected layers are connected to this 1 by 1 layer. Each hidden layer uses a ReLU activation function.

This resulted in 2 different neural network models. one with 11 layers and one with 15 layers. The structure of these neural networks is illustrated in figure 1 and figure 2.
