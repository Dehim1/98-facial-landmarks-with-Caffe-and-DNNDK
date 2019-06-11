#!/bin/bash
#Abort if any command fails
set -e
set -o pipefail
if [[ $0 != $BASH_SOURCE ]]; then
    echo "Do not use '.' or 'source'. Use './' or 'bash' instead."
else
    script_dir=$(dirname "$(readlink -f "$0")")
    #path and name of float model
    model_dir=${script_dir}
    model_name=15_layer_float
    #output directory
    output_dir=${script_dir}/decent_output

    decent  quantize \
            -model ${model_dir}/${model_name}.prototxt \
            -weights ${model_dir}/${model_name}.caffemodel \
            -output_dir ${output_dir} \
            -calib_iter 1000 \
            -gpu 0
fi