from __future__ import print_function

import os
import sys
import random
import numpy as np
import csv
import dlib, cv2, re

import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from seg.get_face_seg import seg_attention
from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_pure = transforms.Compose([
        transforms.ToTensor()
])

class face_dataset(data.Dataset):
    def __init__(self, 
                 root, 
                 list_file,  
                 input_size, 
                 eval_num,
                 addition_root=None, 
                 addition_list=None,
                 transform=transform, 
                 transform_att=transform_pure, 
                 att_flag=False, 
                 flip_flag=True, 
                 random_crop_flag=False,
                 center_crop_flag=True,
                 align_flag=True, 
                 state = 'train'):
        self.root = root
        self.list_file = list_file
        if addition_root is not None:
            self.addition_root = addition_root
            self.addition_list = addition_list
        self.input_size = input_size
        self.eval_num = eval_num
        self.att_flag = att_flag
        self.align_flag = align_flag
        self.transform = transform
        self.transform_att = transform_att
        self.random_crop_flag = random_crop_flag
        self.center_crop_flag = center_crop_flag
        self.flip_flag = flip_flag

        fnames = []
        ids = []
        self.encoder = DataEncoder()

        file_list = list(csv.reader(open(list_file, 'r')))
        for content_counter in file_list:
            fnames.append(os.path.join(self.root, content_counter[0]))
            ids.append(int(content_counter[1]))
        
        im_name_list = []
        for id_counter in range(2874):
            seq_num = ids.index(id_counter)
            im_name_list.append(fnames[seq_num])
            del(ids[seq_num])
            del(fnames[seq_num])

        ids_list = list(range(2874))
        self.im_name_valid = fnames[:self.eval_num]
        self.im_name_train = fnames[self.eval_num:]+im_name_list
        self.ids_valid = ids[:self.eval_num]
        self.ids_train = ids[self.eval_num:]+ids_list
        
        if self.align_flag:
            self.align_detect = dlib.get_frontal_face_detector()
            predicter_path = "./model/shape_predictor_5_face_landmarks.dat"
            self.sp = dlib.shape_predictor(predicter_path)

        if addition_root is not None:
            self.addition_root = addition_root
            self.addition_list = addition_list

            add_csv = list(csv.reader(open(self.addition_list, 'r')))
            self.add_id = []
            self.add_img = []
            for img_list in add_csv:
                self.add_id.append(int(img_list[1]))
                folder_dir = re.sub(r'_\d\d\d\d.jpg', '', img_list[0])
                img_dir = os.path.join(os.path.join(self.addition_root, folder_dir), img_list[0])
                self.add_img.append(img_dir)

        if state == 'train':
            self.excuse_list = self.im_name_train
            self.excuse_ids = self.ids_train
        else:
            self.excuse_list = self.im_name_valid
            self.excuse_ids = self.ids_valid
        
        if addition_root is not None:
            self.excuse_list += self.add_img
            self.excuse_ids += self.add_id

        self.thresh = torch.nn.Hardtanh(min_val=0, max_val=1)
        self.get_att = seg_attention(input_size)


    def check_list(self):
        return self.excuse_ids, self.excuse_list

    def alignment(self, img):
        img_cv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        dets = self.align_detect(img_cv, 1)
        if len(dets)!=0:
            faces = dlib.full_object_detections()
            for det in dets:
                faces.append(self.sp(img_cv, det))
            face_img = dlib.get_face_chips(img_cv, faces, size=160)
            return Image.fromarray(cv2.cvtColor(face_img[0],cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        size = self.input_size
        img_path = self.excuse_list[idx]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.att_flag:
            att_map = self.get_att.get_att(img)
        if self.align_flag:
            img = self.alignment(img)
        img = resize(img, size)
        id_ = self.excuse_ids[idx]
        #return self.transform(img), img_path, id_
        if self.att_flag:
            att_map = resize(att_map, size)
            if self.flip_flag:  img, att_map = random_flip([img, att_map])
            att_map = self.transform_att(att_map)
            att_map = torch.floor(100*att_map)
            att_map = self.thresh(att_map)
            img = self.transform(img)
            return img, img_path, id_, att_map
        else:
            if self.flip_flag:  img = random_flip(img)
            img = self.transform(img)
            return img, img_path, id_


    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        img_path = [x[1] for x in batch]
        id_ = [x[2] for x in batch]
        if self.att_flag:
            att = [x[3] for x in batch]
        
        img_path_ = []
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        att_inputs = torch.zeros(num_imgs, h, w)
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            img_path_.append(img_path[i])
            if self.att_flag:
                att_inputs[i] = torch.FloatTensor(att[i])
        if self.att_flag:
            return inputs, img_path_, id_, att_inputs
        else:
            att_inputs = None
            return inputs, img_path_, id_, att_inputs

    def __len__(self):
        return len(self.excuse_ids)


if __name__ == '__main__':
    trainset = face_dataset(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      addition_root = './../face_a/lfw_masked',
                      addition_list = './../face_a/lfwTrain.csv',
                      input_size=224,
                      eval_num = 400)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=4, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset.collate_fn)
    
    for batch_idx, (inputs, img_path, ids) in enumerate(trainloader):
        pdb.set_trace()