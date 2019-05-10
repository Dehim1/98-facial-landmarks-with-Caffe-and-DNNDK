# 98 facial landmarks with DNNDK
This repository contains a 98 facial landmark detection network intended for deployment on a Zynq FPGA SoC configured with a DEEPHi DPU IP. Inspiration for this project came from [VanillaCNN](https://github.com/ishay2b/VanillaCNN), [lsy17096535/face-landmark](https://github.com/lsy17096535/face-landmark) and [cunjian/face_alignment](https://github.com/cunjian/face_alignment)

The docs directory of this repository contains useful documents to get familiar with the design of this project and reproduce the results.

The python directory contains all the python scripts that are used in this project.

The zoo directory contains the network and solver .prototxt files and the trained weights .caffemodel files.