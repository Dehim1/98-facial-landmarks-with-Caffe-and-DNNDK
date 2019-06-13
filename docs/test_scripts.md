# Test scripts

This document explains the different test scripts used to verify the functionality of various parts of this project. The test script [face_landmark.py](../python/face_landmark.py) is already explained in [README.md](../README.md) along with [Train.py](../python/Train.py)

## [TestLDU.py](../python/TestLDU.py)

[TestLDU.py](../python/TestLDU.py) is used to test the LandmarkDataUnit class and the Data Augmentation functions provided by it. It loads the original dataset, loads one of the images, augments it in various ways and writes each augmentation back to the [ldu_test](../ldu_test) directory. Here each augmented image can be inspected to make sure data augmentation was performed correctly.

Important variables in [TestLDU.py](../python/TestLDU.py) are:

* `dataset_dir`: Directory in which you placed the original dataset.
* `output_dir`: Directory in which you want the output images to be stored.
* `data`: Contains the original dataset in memory. Use GetData_68 if you want to use the IBug 68 landmark dataset.
* `i`: Index of image in dataset to use during this test.

## [CheckData.py](../python/CheckData.py)

[CheckData.py](../python/CheckData.py) is used to check whether the datasets created by [AugData.py](../python/AugData.py) are valid. It does this by opening the dataset and taking one data sample. Of this sample it will check whether it has 68 landmarks or 98 landmarks. After this it will draw the landmarks on the image and display the image on screen.

Important variables in [CheckData.py](../python/CheckData.py) are:

* `h5_dir`: The directory in which the h5 files composing the dataset are located.
* `prefix_test`: The prefix of the .h5 files composing the test dataset.
* `prefix_train`: The prefix of the .h5 files composing the train dataset.
* `h5_file_idx`: The index of the .h5 file within either the test dataset or the train dataset.
* `i`: The index of the data sample within a .h5 file within a dataset.

## [DrawNetwork.py](../python/DrawNetwork.py)
[DrawNetwork.py](../python/DrawNetwork.py) creates a graph drawing of the train, test and deploy networks of a model defined in the zoo directory. For this it uses caffe.draw, which is provided by Caffe for this purpose. The results are saved to the [net_drawings](../net_drawings) directory.

Important variables are:

`net_prefix`: prefix of .prototxt network definition file for network to draw.