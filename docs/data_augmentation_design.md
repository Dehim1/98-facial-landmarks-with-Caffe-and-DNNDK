# Data augmentation design

This document explains how to perform data augmentation on the WFLW and ibug facial landmark datasets and how to write the resulting dataset to .h5 files, so that it can be used with Caffe.

Data augmentation increases the size of the dataset by generating multiple alterations of the dataset. For images this can mean mirroring, rotating, shearing, cropping, translating, scaling, color transformations, adding noise, removing parts of an image, etcetera. Because the dataset is larger and more diverse after augmentation, the degree to which the network will overfit is likely to be reduced. This increases the neural networks performance on data that was not seen during the training process.

For information on how to use the data augmentation scripts, refer to [data_augmentation_readme.md](data_augmentation_readme.md)


## Design

### Global design

Each image and its landmarks will first be randomly mirrored, rotated, translated and scaled. The amount by which this is done is randomly determined following a predefined uniform distribution. The reason this is done randomly is such that every image is augmented differently. This will allow for a highly varied dataset, without it being excessively large. Mirroring, rotation, translation and scaling were chosen, because the developers of the WFLW dataset also used these augmentations when training their neural network<sup><a href="#ref-3">[3]</a></sup>.

Data augmentation procedes as follows:

<a id="figure-1">
    <figure class="image">
        <a href="./images/data_augmentation.png">
            <img src="./images/data_augmentation.png" alt="drawing" width="150">
        </a>
    </figure>
</a>

Figure 1: Image visualizing the data augmentation process.

<a id="1-through-12"></a>

1. A single full image is loaded at a time. This will reduce RAM usage.
2. The image and landmarks are randomly rotated.
3. The bounding box of the face is determined from the facial landmarks. The bounding boxes provided by the WFLW dataset are not used, because at large angles of rotation, the provided bounding boxes are no longer accurate.
4. The bounding boxes are randomly translated in both the x and the y dimensions. The image and landmarks are not translated, but the bounding box is. This has the same effect as translation of the image and landmarks, but is less compute intensive. The amount by which the bounding boxes are translated is defined as a percentage of the dimensions of the bounding box. This makes translation independent of image size.
5. The bounding boxes are randomly scaled in both the x and the y dimensions.  The image and landmarks are not scaled, but the bounding box is. This has the same effect as scaling the image and landmarks, but is less compute intensive. Scaling is also done as a percentage of the dimensions of the bounding box.
6. The bounding boxes are clipped. That is, if the bounding box has coordinates that lie outside the image, The bounding boxes are clipped, such that all coordinates lie inside the image.
7. The landmarks are projected to the bounding box. The landmarks in the WFLW dataset are defined as coordinates in the original image. This will not mean anything to a neural network, because the network will not see the original image. Instead the coordinates should be defined by their position in the bounding box. To do this, the center position of the bounding box is subtracted from every landmark, after which the x and y coordinates of the landmarks are divided by the width and the height of the bounding box.
8. The image is cropped by the bounding box.
9. The cropped image is resized to the input dimensions of the neural network.
10. The image and landmarks are randomly mirrored or not mirrored around the central x line of the image.
11. A loss weight is created. This weight normalizes the loss by the interocular distance of the face. This way the network trains as fast on large faces as it does on small faces. Furthermore, for both the WFLW dataset and the ibug dataset a lossgate is generated. This is used to turn training on or off depending on if the input data is from the WFLW dataset or the ibug dataset.
12. The image, landmarks and loss weights and gates as well as the attributes provided by WFLW are written to arrays. The images and loss weights for both the WFLW and ibug datasets are written to the same array. The other data is written to separate arrays for both datasets. 
<a id="13-and-14"></a>

13. Once the arrays are full, the arrays are shuffled. This ensures the data is randomly ordered, which will improve training.
14. The arrays are written to .h5 files.

This process will repeat until the desired dataset size is achieved.

To make data augmentation more manageable, the classes `BBox` and `LandmarkDataUnit` are created. A `LandmarkDataUnit` object holds the image, landmarks and bounding box during the augmentation process. The bounding box in turn is a `BBox` object. The `LandmarkDataUnit` class defines functions for mirroring, rotation, bounding box creation, translation, scaling, clipping, projecting the image landmarks to the bounding box and cropping. The translate and scale functions of the `LandmarkDataUnit` class make use of translate and scale functions defined in the `BBox` class.

Steps [1 through 12](#1-through-12) are encapsulated in the `run` function of a `Threading.Thread` class `AugmentDataThread`. This will allow the data augmentation process to be executed across multiple threads, significantly accelerating this process.

These threads are created in the `GenerateDataset` function. This function also handles the creation of the arrays that hold the images and landmarks as well as steps [13 and 14](#13-and-14).
