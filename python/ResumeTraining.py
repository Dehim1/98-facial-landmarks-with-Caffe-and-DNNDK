import caffe
import os
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

python_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(python_path, '..')
zoo_path = os.path.join(root_path, 'zoo')
solver_path = os.path.join(zoo_path, '12_layer_adam_solver.prototxt')
snap_path = os.path.join(root_path, 'caffe_data/12_layer')

snap_prefix = 'snap'
Iter = 5000

solverstate_path = os.path.join(snap_path, (snap_prefix + '_iter_' + str(Iter) + '.solverstate'))

os.chdir(zoo_path)
caffe.set_mode_gpu()
solver = caffe.get_solver(solver_path)
solver.restore(solverstate_path)
solver.solve()