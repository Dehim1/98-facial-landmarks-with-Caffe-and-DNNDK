import caffe
import os
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

python_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(python_path, '..')
zoo_path = os.path.join(root_path, 'zoo')
draw_path = os.path.join(root_path, 'net_drawings')
net_prefix = '12_layer'

train_net = os.path.join(zoo_path, net_prefix + '_train.prototxt')
deploy_net = os.path.join(zoo_path, net_prefix + '_deploy.prototxt')

net_proto = caffe_pb2.NetParameter()
text_format.Merge(open(train_net).read(), net_proto)
caffe.draw.draw_net_to_file (net_proto, os.path.join(draw_path, net_prefix + '_train.png'), 'TB',
                            phase=caffe.TRAIN)
caffe.draw.draw_net_to_file (net_proto, os.path.join(draw_path, net_prefix + '_test.png'), 'TB',
                            phase=caffe.TEST)

net_proto = caffe_pb2.NetParameter()             
text_format.Merge(open(deploy_net).read(), net_proto)
caffe.draw.draw_net_to_file (net_proto, os.path.join(draw_path, net_prefix + '_deploy.png'), 'TB')