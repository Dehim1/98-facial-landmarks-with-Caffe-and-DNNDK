# DPU neural network application

This document describes how the DPU application provided by this repository works and how to run it.

The DPU application project currently relies on the densebox dpu model provided by [dnndk for sdscoc](https://www.xilinx.com/member/forms/download/dnndk-eula-xef.html?filename=xilinx_dnndk_v2.08_for_sdsoc_190214.tar.gz) and has a large part of the code in common with the gstsdxfacedetect project.

## Application design

The DPU application like gstsdxfacedetect is a [GStreamer](https://gstreamer.freedesktop.org/) plugin. GStreamer is a widely used multimedia framework. With GStreamer you can create multimedia pipelines that handle a stream of media from media sources like cameras, hdmi inputs, microphones or video files to media sinks like displays, hdmi outputs, speakers or windows. In this pipeline you can insert plugins that will process the media stream. 

In this case the plugin detects the faces present in each still image, detects 98 landmarks for each face and draws the landmarks on the original image. 

The faces present in the image are detected using the DenseBox class provided by gstsdxfacedetect. This class uses the densebox dpu model. 

Densebox takes in an image and detects the faces present in the image. For every face detected in the image, densebox returns a list of numbers. These numbers are coordinates for the uppper left-hand corner and the lower right-hand corner of a bounding box containing the image of the detected face.

To handle landmark detection on these faces, the classes BBox and LandmarkDetector were created. A BBox object is just a different representation of a bounding box. Instead of storing just the boundaries of the bounding box (the corner coordinates returned by densebox), a BBox object also stores the coordinates of the center of the bounding box as well as the width and the height of the bounding box.

The LandmarkDetector class handles the detection of the face landmarks and can also be used to draw the landmarks and the bounding box on the original image. To detect the landmarks, the image and a BBox object are passed to the Run function of a LandmarkDetector object named landmarkdetector. This Run function does the following things:

1. Clip the bounding box. This clips the bounding box such that neither of the coordinates fall outside the image.
2. Check bounding box size. This ensures neither of the dimensions of the bounding box is less then or equal to 0.
3. Crop image. In this step the image is cropped by the bounding box.
4. Resize image. This resizes the cropped input image to the input dimensions of the neural network.
5. Detect landmarks. In this step the resized input image is fed to the neural network and the neural network is run. The neural network return 196 numbers, which form 98 x and y coordinates for each detected landmark.
6. Project bounding box landmarks to image. The coordinates detected by the network are in a coordinate frame where (-0.5, -0.5) is the upper left-hand corner of the bounding box and (0.5, 0.5) is the lower right-hand corner of the bounding box. This step brings the landmarks into the coordinate frame of the original image.

After executing the run function, both the clipped bounding box and the landmarks are drawn onto the original image. This is done using the LandmarkDetector member functions DrawBBox() and DrawLandmarks(). This step can be omitted if you do not want to draw the landmarks onto the image, but instead want to use the landmarks for another purpose. For this the function GetLandmarks() is defined. This returns a pair of a dynamically allocated array of landmark coordinates and an integer indicating the number of landmarks in the array.

To build the project follow the instructions in [UG1331](https://www.xilinx.com/support/documentation/user_guides/ug1331-dnndk-sdsoc-ug.pdf) on building a GStreamer plugin project.

## Running the application on ZCU102 or ZCU104


To run this application on ZCU102 or ZCU104 it must be loaded from an SD Card. To prepare this SD Card do the following:

1. Format the SD Card to a Fat32 file system.
2. From the dnndk_prebuilt directory in [DNNDK for SDSoC](https://www.xilinx.com/member/forms/download/dnndk-eula-xef.html?filename=xilinx_dnndk_v2.08_for_sdsoc_190214.tar.gz) copy the contents of bootfiles_<board_name> to the root directory of the SD Card.
3. Create a lib directory in the root directory of the SD Card and copy the libraries listed in Table 1 to this lib directory.
4. Copy the [scripts](../dnndk/project/scripts) folder from this repository to the root directory of the SD Card. The scripts in this directory are listed in Table 2.

| Description      | Libraries                                         |
|:-----------------|:--------------------------------------------------|
| API libraries    | libdputils.so, libn2cube.so                       |
| Model libraries  | libdpumodeldensebox.so, libdpumodel98_landmark.so |
| Plugin libraries | gstsdxlandmarkdetect.so                           |

*Table 1: Shared libraries required to run landmark detection application. libdputils.so, libn2cube.so and libdpumodeldensebox.so can be found in the dnndk_prebuilt directory in [DNNDK for SDSoC](https://www.xilinx.com/member/forms/download/dnndk-eula-xef.html?filename=xilinx_dnndk_v2.08_for_sdsoc_190214.tar.gz)*

| Script | Function |
|:--- |:--- |
| run_landmarkdetect_hdmi.sh | Runs landmark detection using the hdmi input as a video source. |
| run_landmarkdetect_usb.sh | Runs landmark detection using a USB camera as a video source. |

*Table 2: Scripts to launch landmark detection*

Once the SD Card is configured, the board can be booted from the SD Card. Make sure the hdmi output of your board is connected to a monitor. The bash scripts required to run the application are located at `/media/card/scripts`.
