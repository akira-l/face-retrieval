import numpy as np
from PIL import Image
import seg.surgery
import caffe
import cv2

import pdb

class seg_attention(object):
    def __init__(self, size):
        # init
        self.size = size
        caffe.set_device(0)
        caffe.set_mode_gpu()

        # load net
        self.net = caffe.Net('./seg/face_seg_fcn8s_deploy.prototxt', 
                            './seg/face_seg_fcn8s.caffemodel', 
                            caffe.TEST)

    def get_att(self, im):
        im = im.resize((500, 500), Image.BILINEAR)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        
        # shape for input (data blob is N x C x H x W), set data
        self.net.blobs['data'].reshape(1, *in_.shape)
        self.net.blobs['data'].data[...] = in_

        # run net and take argmax for prediction
        self.net.forward()
        out = self.net.blobs['score'].data[0].argmax(axis=0)
        _, thresh=cv2.threshold(np.uint8(out),2,255,cv2.THRESH_BINARY)
        att_map = Image.fromarray(np.uint8(out))
        att_map = att_map.resize((self.size, self.size), Image.BILINEAR)
        #att_map = att_map.convert('L')
        #att_map.save('att_check2.jpg')
        return att_map
        


if __name__ == '__main__':
    att = seg_attention(224)
    num = 1
    im = Image.open(str(num)+'.jpg')
    out, att_map = att.get_att(im)
    pdb.set_trace()