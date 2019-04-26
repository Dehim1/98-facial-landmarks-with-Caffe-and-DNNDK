# Data augmentation readme

The [WFLW dataset](https://wywu.github.io/projects/LAB/WFLW.html) contains 10000 faces annotated with 98 different landmarks. This README describes how to use the provided python scripts to prepare the WFLW dataset for use with Caffe.

The scripts also handle data augmentation which introduces variation into the dataset effectively increasing its size. This reduces overfitting allowing the network to better extract landmarks of faces that were not seen during the training process.

For more information on the inner workings of these scripts including data augmentation refer to [data_augmentation_design.md](data_augmentation_design.md)

## Dependencies

The getData.py script makes use of the following external python libraries:

* os. Used to perform portable path manipulation.
* threading. Used to better utilize the available compute power of the CPU.
* copy. Used for deepcopy function. This function is used because pythons assignment operator does not copy the right-hand side to the left-hand side. Instead it binds the right-hand side to the left-hand side, such that when the left-hand side variable is altered, the right-hand side variable is also altered.
* math. Used for floor and ceil functions.
* cv2 (OpenCV). Used to load and manipulate images.
* numpy. Used to create and manipulate multi dimensional contiguous arrays of data.
* random. Used to generate random data augmentations for each data entry.
* sklearn.utils. Used for shuffle function to randomly shuffle data, such that the order at which the network sees the training data is random.
* h5py. Used to write dataset to .h5 file. This is one of the acceptable input formats for Caffe.

## Download WFLW

Create a directory named `WFLW`.
It is recommended to put this directory in a common directory (`datasets`) containing all the datasets on your computer.

[Download WFLW Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz) and extract it to `WFLW/WFLW_annotations`.

[Download WFLW Training and Testing Images](https://drive.google.com/open?id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC) and extract it to `WFLW/WFLW_images`.

Copy the contents of `WFLW/WFLW_annotations/list_98py_rect_attr_train_test` to `WFLW/WFLW_images`

Create a directory `WFLW/WFLW_h5`. 

This results in the following directory structure.

```bash
datasets
└──WFLW
    ├──WFLW_annotations
    │   ├──list_98pt_rect_attr_train_test
    │   └──list_98pt_test
    ├──WFLW_h5
    └──WFLW_images
        ├──0--Parade
        ┋
        └──61--Street_Battle
```

## Edit getData.py

Open getData.py and scroll down to the bottom. Starting at line 111 you will see:

```python
#Get dataset
dataset_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_images'
h5_dir = '/home/dehim/Downloads/datasets/WFLW/WFLW_h5'
h5_prefix_test = 'test_aug'
h5_prefix_train = 'train_aug'
maxFileSize = 7500
minFreeMemory = 3.0 #GiB

imgSize = (80, 80)

N_testAugmentations = 3
N_trainAugmentations = 9

mirror = True
angleRange = (-30.0, 30.0)
xTranslateRange = (-0.15, 0.15)
yTranslateRange = (-0.15, 0.15)
xScaleRange = (0.8, 1.2)
yScaleRange = (0.8, 1.2)
augmentationRange = (mirror, angleRange, (xTranslateRange, yTranslateRange), (xScaleRange, yScaleRange))

test_data = GetDataFromTXT_98(os.path.join(dataset_dir, 'list_98pt_rect_attr_test.txt'))
train_data = GetDataFromTXT_98(os.path.join(dataset_dir, 'list_98pt_rect_attr_train.txt'))

GenerateDataset(test_data, N_testAugmentations, augmentationRange, imgSize, maxFileSize, minFreeMemory, h5_dir, h5_prefix_test)
GenerateDataset(train_data, N_trainAugmentations, augmentationRange, imgSize, maxFileSize, minFreeMemory, h5_dir, h5_prefix_train)
```

Here `dataset_dir` should point to `WFLW_images` on your PC and `h5_dir` should point to `WFLW_h5` on your PC.
`h5_prefix_test` and `h5_prefix_train` store the prefixes for the dataset files. These prefixes are used to create .h5 files that store the augmented dataset and a .txt file that links all .h5 files together.

For example: the value of `h5_prefix_train` is equal to `'train_aug'`. Because of this, the first .h5 file will be named train_aug_0.h5, the second .h5 file will be named train_aug_1.h5, etc. The .txt file will be named train_aug.txt. All these files will be stored in the `h5_dir` directory.

`maxFileSize` determines the maximal number of annotated faces in each .h5 file. If you choose maxFileSize too large, Caffe may determine it has too little memory to run the training process.

If the amount of free memory on your PC is less than `minFreeMemory`, the python script will shuffle the augmented data and write it to multiple .h5 files. 

`imgSize` determines the size at which the images will be stored in the .h5 files. This also determines the size of the input layer of the neural network you will train with this dataset.

`N_testAugmentations` and `N_trainAugmentations` determine how many different augmentations of every face will be generated. Given that the original WFLW test set contains 2500 images and `N_testAugmentations` is equal to 3, this results in 2500\*3=7500 images in the test dataset. Likewise the training set will contain 7500\*9=67500 images.

`mirror` determines whether randomly mirroring the data should be part of the data augmentation process.

`angleRange`, `xTranslateRange`, `yTranslateRange`, `xScaleRange` and `yScaleRange`  all determine uniform ranges from which a random number will be selected. These are all used during augmentation.

`angleRange` determines by how many degrees the image and its annotations are rotated.

`xTranslateRange` determines by how much the bounding box of the face should be shifted to the right as a fraction of the width of the bounding box.

`yTranslateRange` determines by how much the bounding box of the face should be shifted down as a fraction of the height of the bounding box.

`xScaleRange` determines by how much the width of the bounding box should be scaled.

`yScaleRange` determines by how much the height of the bounding box should be scaled.

To run the augmentation process execute the following command.

```bash
python getData.py
```