import caffe
import numpy as np

class MultitaskLoss_2(caffe.Layer):
    #ORIGINAL EXAMPLE
    def setup(self, bottom, top):
        self.bottoms = np.zeros((len(bottom)), np.float32)
        self.bottoms_diff = np.zeros((len(bottom)), np.float32)
        top[0].reshape(len(bottom))
        # top[0].data.fill(1.0)

    def reshape(self, bottom, top):
        # check input dimensions match
        for idx in range(len(bottom)):
            if bottom[idx].data.shape != ():
                raise Exception('Bottom blob ' + str(idx) + ' must have shape (). shape is ' + str(bottom[idx].data.shape))
        # top[0].reshape(len(bottom))

    def forward(self, bottom, top):
        for idx in range(len(bottom)):
            self.bottoms[idx] = bottom[idx].data

        beta_mu = 0.9995
        top[0].data[...] =  beta_mu * top[0].data[...] + (1.0-beta_mu)*self.bottoms
        self.bottoms_diff = np.mean(top[0].data[...])/(top[0].data[...]+0.000001)

    def backward(self, top, propagate_down, bottom):
        for idx in range(len(bottom)):
            if not propagate_down[idx]:
                continue
            bottom[idx].diff[...] = self.bottoms_diff[idx]
