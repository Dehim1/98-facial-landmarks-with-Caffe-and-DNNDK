#!/usr/bin/env bash

#working directory
work_dir=$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

decent    quantize                                    \
          -model ${model_dir}/float.prototxt     \
          -weights ${model_dir}/float.caffemodel \
          -output_dir ${output_dir} \
          -calib_iter 10000 \
          -gpu 0 -auto_test 
