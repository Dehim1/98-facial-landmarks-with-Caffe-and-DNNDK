import caffe
import os
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

PYTHON_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(PYTHON_PATH, '..')
ZOO_PATH = os.path.join(ROOT_PATH, 'ZOO')
SOLVER_PATH = os.path.join(ZOO_PATH, 'vanilla_adam_solver.prototxt')
# NET_PATH = os.path.join(ZOO_PATH, 'large_train_relu_no_pool_noise_uniform_euclidean.prototxt')
# SNAP_PATH = os.path.join(ROOT_PATH, 'caffeData/snapshots/relu_no_pool_aug_noise')
# NET_PATH = os.path.join(ZOO_PATH, 'large_train_relu_no_pool_noise_uniform_euclidean.prototxt')
# SNAP_PATH = os.path.join(ROOT_PATH, 'caffeData/snapshots/19_layer_retrain')
# NET_PATH = os.path.join(ZOO_PATH, '17_layer_dilated_train.prototxt')
# SNAP_PATH = os.path.join(ROOT_PATH, 'caffeData/snapshots/17_layer_retrain')
# NET_PATH = os.path.join(ZOO_PATH, '12_layer_dilated_train.prototxt')
# SNAP_PATH = os.path.join(ROOT_PATH, 'caffeData/snapshots/12_layer_dilated')
NET_PATH = os.path.join(ZOO_PATH, '12_layer_large_train.prototxt')
SNAP_PATH = os.path.join(ROOT_PATH, 'caffeData/snapshots/12_layer_large')


net_proto = caffe_pb2.NetParameter()
text_format.Merge(open(NET_PATH).read(), net_proto)
caffe.draw.draw_net_to_file (net_proto, 'train_net.png', 'TB',
                            phase=caffe.TRAIN)
caffe.draw.draw_net_to_file (net_proto, 'test_net.png', 'TB',
                            phase=caffe.TEST)

snap_prefix = 'snap'
Iter = 420000

solverStateFile = (snap_prefix + '_iter_' + str(Iter) + '.solverstate')
SOLVER_STATE = os.path.join(SNAP_PATH, solverStateFile)

os.chdir(ZOO_PATH)
caffe.set_mode_gpu()
solver = caffe.get_solver(SOLVER_PATH)
# solver.restore(SOLVER_STATE)
solver.solve()
os.chdir(ROOT_PATH)