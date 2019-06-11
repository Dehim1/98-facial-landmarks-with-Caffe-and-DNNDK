#!/bin/bash
#Abort if any command fails
set -e
set -o pipefail
if [[ $0 != $BASH_SOURCE ]]; then
    echo "Do not use '.' or 'source'. Use './' or 'bash' instead."
else
    script_dir=$(dirname "$(readlink -f "$0")")

    model_dir=${script_dir}/decent_output
    output_dir=${script_dir}/dnnc_output
    net=98_landmark

    echo "Compiling network: ${net}"
    dnnc    --prototxt=${model_dir}/deploy.prototxt \
            --caffemodel=${model_dir}/deploy.caffemodel \
            --output_dir=${output_dir} \
            --net_name=${net} \
            --dpu=4096FA \
            --mode=normal \
            --cpu_arch=arm64 \
            --abi=0
fi