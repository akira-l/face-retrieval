from __future__ import print_function

import os
import sys
import argparse
import pdb
import datetime
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from loss_originalFAN import FocalLoss
from retinanet_originalFAN import RetinaNet
from datagen import ListDataset
from testgen import TestListDataset

from torch.autograd import Variable

from encoder import DataEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--state', '-s', help='training or test stage')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
global tensorboard_counter
tensorboard_counter = 0


# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root="./../../wider face/WIDER_train/images",
                       list_file="./../../wider face/wider_face_split/wider_face_train_bbx_gt.txt", 
                       train=True, 
                       transform=transform, 
                       input_size=224)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=36, 
                                          shuffle=True, 
                                          num_workers=1, 
                                          collate_fn=trainset.collate_fn)
'''
testset = ListDataset(root="./../../wider face/WIDER_val/images",
                      list_file="./../../wider face/wider_face_split/wider_face_val_bbx_gt.txt", 
                      train=False, 
                      transform=transform, 
                      input_size=224)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         num_workers=1, 
                                         collate_fn=testset.collate_fn)
'''

testset = TestListDataset(root="./../face_a/train",
                      train=False, 
                      transform=transform, 
                      input_size=224)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         num_workers=1, 
                                         collate_fn=testset.collate_fn)



# Model
net = RetinaNet()
'''
net.load_state_dict(torch.load('./model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
'''
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net = torch.nn.DataParallel(net, device_ids=[0])
net.cuda()

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

# Training
def train(epoch, file_obj, writer, tensorboard_counter=tensorboard_counter):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    '''
    loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
    loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
    cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
    cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
    '''
    #for test_count in range(10):
    for batch_idx, (inputs, loc_targets, cls_targets, att_gt, _) in enumerate(trainloader):
        
        #batch_idx = 1
        #inputs = Variable(torch.rand(4, 3, 224, 224))
        #loc_targets = Variable(torch.rand(4, 9441, 4))
        #cls_targets = Variable(3*torch.rand(4, 9441))
        
        #cls_targets = cls_targets.float()
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        #loc_preds, cls_preds = net(inputs)
        #### pdb.set_trace()
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        
        # add attention loss 
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        file_obj.write('loss|'+str(loss.data[0])+'|'+'avg_loss|'+str(train_loss/(batch_idx+1))+'\n')
        print('epoch: %d | train_loss: %.3f | avg_loss: %.3f' % (epoch, loss.data[0], train_loss/(batch_idx+1)))
        writer.add_scalar('data/loss', loss.data[0], tensorboard_counter)
        writer.add_scalar('data/avg_loss', train_loss/(batch_idx+1), tensorboard_counter)
        tensorboard_counter += 1
    save_model(net, 'originalFAN_model.pth')



def test():
    print('\nTest')
    net.load_state_dict(torch.load("./originalFAN_model.pth"))
    '''
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./model.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    print('recorded loss:', best_loss)
    '''

    net.eval()
    test_loss = 0
    coder = DataEncoder()
    pdb.set_trace()
    for batch_idx, (inputs, img_path) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        
        loc_preds, cls_preds = net(inputs)

        boxes = []
        labels = []
        boxes, labels, score = coder.decode(loc_preds[0].data.cpu(), cls_preds[0].data.cpu(), (224, 224))
        
        img = cv2.imread(img_path[0])
        img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        
        for (x,y,w,h), score_loop in zip(boxes, score):
            cv2.rectangle(img,(x,y),(w,h),(0,255,0),1)
            cv2.putText(img,str(float(score_loop)),(x,y),cv2.FONT_HERSHEY_PLAIN,0.6,(255,255,255),2)
        cv2.imwrite('result_test.jpg', img)
        pdb.set_trace()




if __name__ == "__main__":
    if args.state == 'train':
        writer = SummaryWriter()
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        save_file = './save_file/save_file-'+date_now+'.txt'
        file_obj = open(save_file, 'a')
        file_obj.write('description: retinanet with attention, 224, optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4), epoch: 200, record loss and average loss ')
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch, file_obj, writer)  
            #test(epoch)
        file_obj.close()    
    elif args.state == 'test':
        test()
    else:
        raise ValueError('need training or test state input')
    