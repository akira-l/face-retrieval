from __future__ import print_function

import os
import sys
import argparse
import pdb
import datetime
import cv2
import numpy as np
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss_originalFAN import FocalLoss
from retinanet_originalFAN import RetinaNet
from Idnet2 import Idnet
from data import face_dataset

from torch.autograd import Variable
from encoder import DataEncoder
from arcface_loss2 import Arcface
from cosface_loss import MarginCosineProduct

from detector import FANdetector


def parse_args():
    parser = argparse.ArgumentParser(description='train face retrieval')
    #data 
    parser.add_argument('--data-version', default='face_a', 
                        help='input data type:original: face_a,\
                                              detected: face_a_image,\
                                              addition with lfw: lfw_masked ')
    parser.add_argument('--occ', default='null', 
                        help='occlusion attention add or not')
    parser.add_argument('--align', default=True,
                        help='with alignment or not')

    #train
    parser.add_argument('--task', default='', help='task description')
    parser.add_argument('--idnet', default='sphere', 
                        help='which net choosen for idnet')
    parser.add_argument('--static-lr', default=True, 
                        help='set learning rate variation')
    parser.add_argument('-lr-setting', default=0.001)
    parser.add_argument('--batch-size', default=32)
    parser.add_argument('--epoch', default=120)

    parser.add_argument('--detect-dir', default='originalFAN', 
                        help='detect pretrain model location')
    parser.add_argument('--face-loss', default='arcface', 
                        help='arcface/cosface/softmax')
    parser.add_argument('--out-feature', default=512,
                        help='id net output feature size')
    parser.add_argument('--pre-idnet', default=False, 
                        help='idnet pretrained or not')
    parser.add_argument('--class-num', default=2874, 
                        help='cross entropy classify number')
    
    parser.add_argument('--stage', default='',
                        help='code application stage, train/test')
    parser.add_argument('--eval-num', default=400, 
                        help='evalute number per epoch')

    args = parser.parse_args()
    return args

class trainer(object):
    def __init__(self, args):
        self.detector = FANdetector()
        self.id_net = Idnet(classnum=8623)
        self.id_net = torch.nn.DataParallel(self.id_net, device_ids=range(torch.cuda.device_count()))
        self.id_net.cuda()
        self.out_feature = args.out_feature
        self.class_num = args.class_num
        self.idnet_name = args.idnet
        self.eval_num = args.eval_num
        self.batch_size = args.batch_size
        self.stage = 'train'#args.stage
        self.detect_flag = True
        if args.face_loss == 'arcface':
            self.MCP = Arcface(self.out_feature, self.class_num).cuda()
            self.MCP.train()
        if args.face_loss == 'cosface':
            self.MCP = MarginCosineProduct(self.out_feature, self.class_num).cuda()
            self.MCP.train()
        if args.face_loss == 'softmax':
            self.MCP = None
            if self.idnet_name is not 'sphere':
                assert self.out_feature==self.class_num, 'Error for input feature size, do not match to class number'
            else:
                classify_feature = self.id_net.fc5.in_feature
                self.id_net.fc5 = nn.Linear(classify_feature, self.class_num)
        
        self.dataloader = self.get_loader(
                                data_version=args.data_version,
                                occ_flag_in=args.occ, 
                                align_flag_in=args.align,
                                batch_size=self.batch_size,
                                train_=True)
        self.eval_loader = self.get_loader(
                                data_version=args.data_version,
                                occ_flag_in=args.occ, 
                                align_flag_in=args.align,
                                batch_size=100,
                                eval_=True)
        
        self.id_net.train()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        if args.static_lr:
            if args.face_loss == 'softmax':
                self.optimizer = optim.SGD(self.id_net.parameters(), 
                                            lr=1e-3, 
                                            momentum=0.9, 
                                            weight_decay=1e-4)
            else:
                self.optimizer = optim.SGD([{'params': self.id_net.parameters()}, 
                                            {'params':self.MCP.parameters()}],
                                            lr=1e-3, 
                                            momentum=0.9, 
                                            weight_decay=1e-4) 
        else:
            if args.face_loss == 'softmax':
                optimizer_ = optim.SGD(self.id_net.parameters(), 
                                            lr=0.004, 
                                            momentum=0.9, 
                                            weight_decay=1e-4)
                self.optimizer = torch.optim.lr_scheduler.MultiStepLR(optimizer_, milestones = [20, 80], gamma=0.5)
            else:
                optimizer_ = optim.SGD([{'params': self.id_net.parameters()}, 
                                            {'params':self.MCP.parameters()}],
                                            lr=0.004, 
                                            momentum=0.9, 
                                            weight_decay=1e-4)
                self.optimizer = torch.optim.lr_scheduler.MultiStepLR(optimizer_, milestones = [20, 80], gamma=0.5)

        data_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        if not os.path.exists('./save_file'):
            os.mkdir('./save_file')
        self.save_file = './save_file/save_file-' + data_now + '.txt'
        self.file_obj = open(self.save_file, 'a')
        self.file_obj.write(args.task+'\n')
        self.file_obj.write('data: '+args.data_version+'\n')
        self.file_obj.write('static optimizer: '+str(args.static_lr)+'\n')
        self.file_obj.write('optimizer setting: '+'\n')
        self.file_obj.write('face loss: '+args.face_loss+'\n')
        self.file_obj.write('output feature length: '+str(args.out_feature)+'\n')
        self.file_obj.write('classify number: '+str(args.class_num)+'\n')
        self.file_obj.write('id net pretrained: '+str(args.pre_idnet)+'\n')
        self.file_obj.close() 

    def train(self, epoch):
        total_image_counter = 0
        train_total_right = 0
        train_acc = 0
        eval_acc = 0
        for batch_idx, (inputs, img_path, ids, att_map) in enumerate(self.dataloader):
            total_image_counter += inputs.size(0)
            with torch.no_grad():
                if self.detect_flag:
                    detect_output = self.detector.detect(inputs, att_map) 
                    inputs = Variable(detect_output.cuda())
                else:
                    inputs = Variable(inputs.cuda())
            self.optimizer.zero_grad()
            if self.MCP is not None:
                id_net_out = self.id_net(inputs)
                id_shape = id_net_out.size()
                target = torch.tensor(ids).view(id_shape[0]).cuda()
                output = self.MCP(id_net_out, target)
            else:
                output = id_net_out
                target = torch.tensor(ids).view(id_shape[0]).cuda()
            loss = self.criterion(output, target)

            (_, estimate_id) = torch.max(output, dim=1)
            print('|epoch: ', epoch, '|batch idx: ', batch_idx, '|loss: ', loss, ' |')
            print('train estimted:', estimate_id[:8])
            print('train ids', ids[:8])
            print('\n')
            batchCorrect = torch.eq(estimate_id, target ).sum().item()
            train_total_right += batchCorrect
            loss.backward()
            self.optimizer.step()

            if batch_idx % 20 == 0:
                train_acc = 1.0*train_total_right / total_image_counter
                train_total_right = 0
                total_image_counter = 0

                self.id_net.eval()
                if self.MCP is not None:
                    self.MCP.eval()
                eval_counter = 0
                eval_right = 0
                for eval_batch_idx, (eval_inputs, eval_img_path, eval_ids, eval_att) in enumerate(self.eval_loader):
                    eval_counter += eval_inputs.size(0)
                    with torch.no_grad():
                        if self.detect_flag:
                            eval_detect = self.detector.detect(eval_inputs, eval_att)
                            eval_inputs = Variable(eval_detect.cuda())
                        else:
                            eval_inputs = Variable(eval_inputs)
                    if self.MCP is not None:
                        eval_net_out = self.id_net(eval_inputs)
                        eval_shape = eval_net_out.size()
                        target = torch.tensor(eval_ids).view(eval_shape[0]).cuda()
                        output = self.MCP(eval_net_out, target)
                    else:
                        output = eval_net_out
                        target = torch.tensor(eval_ids).view(eval_shape[0]).cuda()
                    
                    (_, eval_id) = torch.max(output, dim=1)
                    print('\neval estimted:', estimate_id[:8])
                    print('eval ids', ids[:8])
                    print('\n')
                    eval_batch_correct = torch.eq(eval_id, target).sum().item()
                    eval_right += eval_batch_correct
                eval_acc = 1.0*eval_right / eval_counter

                print('-------train acc: ', train_acc, '\n\n-------test acc: ', eval_acc)
                
                record = 'epoch:' + str(epoch) \
                       + '|train acc:' + str(train_acc) \
                       + '|test acc:' + str(eval_acc)\
                       + '|loss:' + str(loss) + '\n'
                print(record)
                self.record_writer(record)

        if train_acc > 0.9 and epoch % 5 == 0:
            save_folder = args.task
            if not os.path.exists('./'+save_folder):
                os.mkdir('./'+str(save_folder))
            save_name = './'+str(save_folder)+'/epoch-'+str(epoch)+'train_acc-'\
                        +str(train_acc)+'eval_acc-'+str(eval_acc)+'.pth'
            self.save_model(self.id_net, save_name)

    def save_model(self, model, filename):
        state = model.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(state, filename)

    def get_loader(self, data_version, occ_flag_in, align_flag_in, batch_size, train_=False, eval_=False):
        if data_version == 'face_a':
            root_ = './../face_a/train'
            list_file_ = './../face_a/train.csv'
            addition_root_ = None
            addition_list_ = None
        elif data_version == 'face_a_image':
            root_ = './../face_a/image/train'
            list_file_ = './../face_a/train.csv'
            addition_root_ = None
            addition_list_ = None
        elif data_version == 'lfw_masked':
            root_ = './../face_a/train'
            list_file_ = './../face_a/train.csv'
            addition_root_ = './../face_a/lfw_masked'
            addition_list_ = './../face_a/lfwTrain.csv'
        else:
            raise ValueError('do not have this data version!!!')

        if self.stage == 'train':
            shuffle_flag_ = True
        else:
            shuffle_flag_ = False
        att_flag_ = occ_flag_in
        align_flag_ = align_flag_in
        eval_num_ = self.eval_num
        input_size_ = 224

        if train_:
            state = 'train'
        if eval_:
            state = 'test'
        dataset = face_dataset(root = root_,
                      list_file = list_file_,
                      addition_root = addition_root_,
                      addition_list = addition_list_, 
                      input_size = input_size_,
                      eval_num = eval_num_, 
                      align_flag = align_flag_, 
                      att_flag = att_flag_, 
                      state = state)
        
        dataloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size, 
                                          shuffle=shuffle_flag_, 
                                          num_workers=0, 
                                          collate_fn=dataset.collate_fn)
        
        return dataloader


            


    def record_writer(self, content):
        self.file_obj = open(self.save_file, 'a')
        self.file_obj.write(content+'\n')
        self.file_obj.close()

    







class test(object):
    def __init__(self, args):
        pass
    
    def re_ranking(self):
        pass





if __name__ == '__main__':
    global args
    args = parse_args()
    assert len(args.task)>0, 'Error: need desciption for task'
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    if args.stage == '':
        raise ValueError("enter specific stage")
    
    if args.stage == 'train':
        trainer_ = trainer(args)
        for epoch_counter in range(args.epoch):
            trainer_.train(epoch_counter)

    if args.stage == 'test':
        test_ = test(args)


