#!/bin/bash
net=98_landmark_12
model_dir=decent_output
output_dir=dnnc_output

echo "Compiling network: ${net}"

dnnc --prototxt=${model_dir}/deploy.prototxt     \
       --caffemodel=${model_dir}/deploy.caffemodel \
       --output_dir=${output_dir}                  \
       --net_name=${net}                           \
       --dpu=4096FA                                 \
       --mode=debug \
       --cpu_arch=arm64 \
       --abi=0
