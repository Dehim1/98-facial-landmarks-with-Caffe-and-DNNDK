# Quantization and compilation of the neural network

Quantization of the neural network requires a .prototxt file which is similar to the deploy.prototxt file which is used during deployment of the neural network on the PC, but differs in a few ways. The input data layer is an HDF5Data layer and points to the hdf5 training dataset. It is a requirements of DECENT that the top blob of the input layer which is the input of the neural network is called `data`. A second HDF5Data layer pointing to the hdf5 test dataset can be included, but this also requires a loss layer. However to scale the losses by the interocular distance, scale layers with two bottom blobs are used. These do not appear to work with Decent. Because of this both the test HDF5Data layer and the loss layer were omitted.

To run quantization [decent.sh](../dnndk/decent.sh) was created.

```bash
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
```

* `model` specifies the path to the network .prototxt file.
* `weights` specifies the path to the weights .caffemodel file.
* `output_dir` specifies the output directory
* `calib_iter` specifies the number of calibration iterations for which to run quantization.
* `gpu` specifies the device id of the gpu to use during quantization.

Executing this script produces [deploy.prototxt](../dnndk/decent_output/deploy.prototxt) and [deploy.caffemodel](../dnndk/decent_output/deploy.caffemodel) in the path specified by `output_dir`. You will have to modify [deploy.prototxt](../dnndk/decent_output/deploy.prototxt) before compiling the neural network.

modify the Input layer like so.

Before:

```protobuf
layer {
  name: ""
  type: "Input"
  top: "data"
  transform_param {
  }
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 80
      dim: 80
    }
  }
}
```

After:

```protobuf
layer {
  name: ""
  type: "Input"
  top: "data"
  transform_param {
    mean_value: 0.0
    mean_value: 0.0
    mean_value: 0.0
  }
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 80
      dim: 80
    }
  }
}

```

mean_value in transform_param specifies a value that will be subtracted from the blue, green and red channels of each input image. Because no value was subtracted from the input channels in the creation of the hdf5 datasets that were used to train the neural network, these values are all set to 0.0.

The HDF5Data layer must be removed from [deploy.prototxt](../dnndk/decent_output/deploy.prototxt).

To run compilation [dnnc.sh](../dnndk/dnnc.sh) was created.

```bash
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
```

* `prototxt` specifies the path to deploy.prototxt.
* `caffemodel` specifies the path to deploy.caffemodel.
* `output_dir` specifies the output directory.
* `net_name` specifies the name of the neural network.
* `dpu` specifies for what dpu architecture the network should be compiled.
* `mode` specifies whether to compile the neural network in debug mode or normal mode. In debug mode each layer is compiled as a separate entity which makes profiling and debugging of each separate layer possible. In normal mode the entire network is compiled to a single unit allowing for faster execution.
* `cpu_arch` Specifies the cpu architecture for which to compile the neural network.

Running this script produces [dpu_98_landmark.elf](../dnndk/dnnc_output/dpu_98_landmark.elf) in the directory specified by output_dir. This file must be converted to a .so file. to do this execute the following command:

```bash
aarch64-linux-gnu-gcc -fPIC -shared dpu_98_landmark.elf -o libdpumodel98_landmark.so
```
