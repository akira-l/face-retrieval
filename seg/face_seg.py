import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import surgery

import caffe

import pdb

# load image, switch to BGR, subtract mean, 
# and make dims C x H x W for Caffe
#im = Image.open('../../data/images/Alison_Lohman_0001.jpg')
num = 1
im = Image.open(str(num)+'.jpg')
im = im.resize((500, 500))
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# init
#caffe.set_device(0)
#caffe.set_mode_gpu()

# load net
net = caffe.Net('./face_seg_fcn8s_deploy.prototxt', 
                './face_seg_fcn8s.caffemodel', 
                caffe.TEST)

# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

_, thresh=cv2.threshold(np.uint8(out),2,255,cv2.THRESH_BINARY)

att_map = Image.fromarray(np.uint8(out))
att_map = att_map.resize((224, 224), Image.BILINEAR)
#att_map = att_map.convert('L')
pdb.set_trace()
cv2.imwrite('thresh.jpg', thresh)

plt.imshow(out)
plt.draw()
plt.savefig('test'+str(num)+'.png')

'''
plt.pause(0.001)
plt.waitforbuttonpress()
'''