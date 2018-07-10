import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from resnet_features import resnet50_features
from resnet_utilities.layers import conv1x1, conv3x3

class attention_block(nn.Module):
    def __init__(self):
        super(attention_block, self).__init__()

        self.scale3_conv1 = nn.Conv2d(256*3, 256*3, kernel_size=3, stride=1, padding=1)
        self.scale3_conv2 = nn.ConvTranspose2d(256*3, 256, 3, stride=2,
                                      padding=1, output_padding=1,
                                      groups=1, bias=True, dilation=1)
        self.scale3_conv3 = nn.ConvTranspose2d(256, 256, 3, stride=2, 
                                      padding=1, output_padding=1,
                                      groups=1, bias=True, dilation=1)
        
        
        self.scale2_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.scale2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.scale2_conv3 = torch.nn.ConvTranspose2d(256, 256, 3, stride=2, 
                                      padding=1, output_padding=1,
                                      groups=1, bias=True, dilation=1)

        self.scale1_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.scale1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.scale1_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.sum_conv1 = nn.Conv2d(3*256, 256, kernel_size=3, stride=1, padding=1)
        self.sum_conv2 = nn.ConvTranspose2d(256, 256, 3, stride=4, 
                                      padding=0, output_padding=1,
                                      groups=1, bias=True, dilation=1)
        self.sum_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.sum_conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        '''
        (Pdb) features[0].shape
        torch.Size([32, 256, 28, 28])
        (Pdb) features[1].shape
        torch.Size([32, 256, 14, 14])
        (Pdb) features[2].shape
        torch.Size([32, 256, 7, 7])
        (Pdb) features[3].shape
        torch.Size([32, 256, 4, 4])
        (Pdb) features[4].shape
        torch.Size([32, 256, 2, 2])
        '''

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :height, :width]
        
    def forward(self, layer1, layer2, layer3, layer4, layer5):
        layer4_up = self._upsample(layer4, layer3, scale_factor=2)
        layer5_up = self._upsample(layer5, layer3, scale_factor=4)

        s3 = torch.cat((layer3, layer4_up, layer5_up), 1)
        s3 = F.relu(self.scale3_conv1(s3))
        s3 = F.relu(self.scale3_conv2(s3))
        s3 = F.relu(self.scale3_conv3(s3))

        s2 = F.relu(self.scale2_conv1(layer2))
        s2 = F.relu(self.scale2_conv2(s2))
        s2 = F.relu(self.scale2_conv3(s2))

        s1 = F.relu(self.scale1_conv1(layer1))
        s1 = F.relu(self.scale1_conv2(s1))
        s1 = F.relu(self.scale1_conv3(s1))

        s = torch.cat((s1,s2,s3), 1)
        out = F.relu(self.sum_conv1(s))
        out = F.relu(self.sum_conv2(out))
        out = F.relu(self.sum_conv3(out))
        out = F.relu(self.sum_conv4(out))
        out = torch.exp(out)

        return out


class FeaturePyramid(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)

        # both based around resnet_feature_5
        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)




    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :height, :width]

    def forward(self, x):

        # don't need resnet_feature_2 as it is too large
        # resnet feature shape: 
        #     3 torch.Size([2, 512, 28, 28]) 
        #     4 torch.Size([2, 1024, 14, 14]) 
        #     5 torch.Size([2, 2048, 7, 7])

        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(pyramid_feature_6))

        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)

        pyramid_feature_4 = self.upsample_transform_1(
            torch.add(upsampled_feature_5, pyramid_feature_4)
        )

        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)

        pyramid_feature_3 = self.upsample_transform_2(
            torch.add(upsampled_feature_4, pyramid_feature_3)
        )

        return pyramid_feature_3, pyramid_feature_4, pyramid_feature_5, pyramid_feature_6, pyramid_feature_7
               


class SubNet(nn.Module):

    def __init__(self, mode, anchors=9, classes=80, depth=4,
                 base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation

        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1)
                                          for _ in range(depth)])

        if mode == 'boxes':
            self.subnet_output = conv3x3(256, 4 * self.anchors, padding=1)
        elif mode == 'classes':
            # add an extra dim for confidence
            self.subnet_output = conv3x3(256, (1 + self.classes) * self.anchors, padding=1)

        self._output_layer_init(self.subnet_output.bias.data)

    def _output_layer_init(self, tensor, pi=0.01):
        fill_constant = - math.log((1 - pi) / pi)
        '''
        if isinstance(tensor, Variable):
            self._output_layer_init(tensor.data)
        '''
        return tensor.fill_(fill_constant)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))

        x = self.subnet_output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0),
                                                    x.size(2) * x.size(3) * self.anchors, -1)

        return x

'''
class RetinaNet(nn.Module):

    def __init__(self, classes):
        super(RetinaNet, self).__init__()
        self.classes = classes

        _resnet = resnet50_features(pretrained=True)
        self.feature_pyramid = FeaturePyramid(_resnet)

        self.subnet_boxes = SubNet(mode='boxes')
        self.subnet_classes = SubNet(mode='classes')

    def forward(self, x):

        boxes = []
        classes = []

        get_all = self.feature_pyramid(x)
        features = get_all[:5]
        attention  = get_all[5:]

        # how faster to do one loop
        boxes = [self.subnet_boxes(feature) for feature in features]
        classes = [self.subnet_classes(feature) for feature in features]

        return torch.cat(boxes, 1), torch.cat(classes, 1), attention
'''

if __name__ == '__main__':
    import time
    net = RetinaNet(classes=1)
    x = Variable(torch.rand(1, 3, 100, 100))

    now = time.time()
    predictions = net(x)
    later = time.time()

    print(later - now)

    for prediction in predictions:
        print(prediction.size())
