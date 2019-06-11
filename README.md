# 98 facial landmarks with DNNDK
This repository contains a 98 facial landmark detection network intended for deployment on a Zynq FPGA SoC configured with a DEEPHi DPU IP. Inspiration for this project came from [ishay2b/VanillaCNN](https://github.com/ishay2b/VanillaCNN), [lsy17096535/face-landmark](https://github.com/lsy17096535/face-landmark) and [cunjian/face_alignment](https://github.com/cunjian/face_alignment). This project makes use of the [WFLW 98 landmark](https://wywu.github.io/projects/LAB/WFLW.html) and [ibug 68 landmark](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) datasets.

The [docs](./docs) directory of this repository contains useful documents to get familiar with the design of this project and reproduce the results.

The [python](./python) directory contains all the python scripts that are used in this project.

The [zoo](./zoo) directory contains the network and solver .prototxt files and the trained weights .caffemodel files.

The [dnndk](./dnndk) directory contains the dnndk project and scripts to help quantize and compile the trained neural network to run it on the DPU platform.

[neural_network_design.md](./docs/neural_network_design.md) Walks you through the design of the neural network.<br/>
[datasets.md](./docs/datasets.md) Contains useful information about how the datasets that were used during training are formatted.<br/>
[data_augmentation_design.md](./docs/data_augmentation_design.md) Explains what data augmentation is and how it is implemented in this project.<br/>
[data_augmentation_readme.md](./docs/data_augmentation_readme.md) Show how to use the [AugData.py](./python/AugData.py) script to perform data augmentation on the WFLW and ibug datasets.<br/>
[dpu_neural_network_quantization_and_compilation.md](./docs/dpu_neural_network_quantization_and_compilation.md) Explains how to perform neural network quantization and compilation on the trained neural network, so that it can be run on the DPU platform.<br/>
[dpu_neural_network_application.md](./docs/dpu_neural_network_application.md) Explains how the application for the DPU platform works and how to run it.<br/>
[python_scripts.md](./docs/python_scripts.md) Explains what the python scripts do that are not explained in either of the other documents.

## training the neural network

To train the neural network, a python script [Train.py](./python/Train.py) was created. Important variables in [Train.py](./python/Train.py) are:

* `solver_path`: Path to solver file to use during training.
* `snap_path`: Directory to load weights or solverstate from. This is the parent directory of snapshot_prefix in the solver file.
* `solverstate_path`: Path to solverstate file. Used to restore solverstate.
* `caffemodel_path`: Path to caffemodel file. Used to restore weights.

## testing the neural network

To test the neural network, [face_landmark.py](./python/face_landmark.py) was created. This script takes images from the images folder and uses dlib to detect faces in each image. For each detected face a bounding box is generated. The images are then cropped by the bounding boxes and resized to the input dimensions of the landmark detection network. The landmark detection network is then used to predict the landmark locations for each face. For each landmark, a circle is drawn on the image. The resulting image is stored in the results folder.

Important variables in [face_landmark.py](./python/face_landmark.py) are:

* `network_path`: Path to network definition .prototxt file.
* `weights_path`: Path to weight values .caffemodel file.