import caffe
import os
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

python_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(python_path, '..')
zoo_path = os.path.join(root_path, 'zoo')
solver_path = os.path.join(zoo_path, '12_layer_adam_solver.prototxt')
weights_path = os.path.join(zoo_path, '12_layer_weights.caffemodel')

os.chdir(zoo_path)
caffe.set_mode_gpu()
solver = caffe.get_solver(solver_path)
# # uncomment to load trained network weights from weight_path.
# solver.net.copy_from('weights_path')
solver.solve()