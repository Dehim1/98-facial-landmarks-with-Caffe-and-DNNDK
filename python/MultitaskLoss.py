import caffe
import numpy as np

class MultitaskLoss(caffe.Layer):
    #ORIGINAL EXAMPLE
    def setup(self, bottom, top):
        self.blobs.add_blob(1)
        self.blobs[0].reshape(len(bottom))
        self.blobs[0].data.fill(0.0)

        self.bottoms = np.zeros((len(bottom)), np.float32)
        self.bottoms_diff = np.zeros((len(bottom)), np.float32)
        self.losses_exp = np.zeros((len(bottom)), np.float32)
        self.losses = np.zeros((len(bottom)), np.float32)
        self.blobs_diff = np.zeros((len(bottom)), np.float32)

    def reshape(self, bottom, top):
        # check input dimensions match
        for idx in range(len(bottom)):
            if bottom[idx].data.shape != ():
                raise Exception('Bottom blob ' + str(idx) + ' must have shape (). shape is ' + str(bottom[idx].data.shape))
        top[0].reshape(1)

    def forward(self, bottom, top):
        for idx in range(len(bottom)):
            self.bottoms[idx] = bottom[idx].data
        self.bottoms_diff = np.exp(self.blobs[0].data)
        self.losses_exp = self.bottoms_diff*self.bottoms
        self.losses = self.losses_exp - self.blobs[0].data
        self.blobs_diff = self.losses_exp - 1.0
        top[0].data[...] = np.sum(self.losses)

        print(np.exp(-self.blobs[0].data))

    def backward(self, top, propagate_down, bottom):
        for idx in range(len(bottom)):
            if not propagate_down[idx]:
                continue
            bottom[idx].diff[...] = self.bottoms_diff[idx]
            self.blobs[0].diff[idx] = self.blobs_diff[idx]
