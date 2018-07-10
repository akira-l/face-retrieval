import torch
import torch.nn.functional as F
from torch.autograd import Variable

from retinanet_originalFAN import RetinaNet
from encoder import DataEncoder
class FANdetector(object):
    def __init__(self):
        self.net = RetinaNet()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.net.cuda()
        self.net.eval()
        self.net.load_state_dict(torch.load("./trained_model/originalFAN_model.pth"))
        self.coder = DataEncoder()

    def detect(self, inputs, att=None):
        with torch.no_grad():
            inputs = Variable(inputs.cuda())
            loc_preds, cls_preds = self.net(inputs)
        boxes = []
        for box_counter in range(inputs.size(0)):
            box, label, score = self.coder.decode(loc_preds[box_counter].data.cpu(), 
                                             cls_preds[box_counter].data.cpu(), 
                                             (224, 224))
            if box.size(0) == 1:
                boxes.append([float(x) for x in box[0]])
                continue
            tmp_box = box[0]
            for box_loop in box: ###shape should be 224!!!!! 
                select_box = [float(x) for x in box_loop]
                cond1 = abs((select_box[0]+select_box[2])/2-112)<abs((tmp_box[0]+tmp_box[2])/2-112)
                cond2 = abs((select_box[1]+select_box[3])/2-112)<abs((tmp_box[1]+tmp_box[3])/2-112)
                if cond1 and cond2:
                    tmp_box = select_box
            boxes.append(tmp_box)

        img_input = torch.zeros(inputs.size(0), 3, 112, 96)
        for img_counter in range(inputs.size(0)):
            face_box = boxes[img_counter]
            face_box = [int(x) for x in face_box]
            face_box[0] = max(face_box[0], 0)
            face_box[1] = max(face_box[1], 0)
            face_box[2] = min(face_box[2], inputs.size(2))
            face_box[3] = min(face_box[3], inputs.size(2))

            height = face_box[3]-face_box[1]
            width = face_box[2]-face_box[0]

            sampled = F.upsample(inputs[img_counter, :, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,3,width,height), 
                                            size=(112, 96), 
                                            mode='bilinear')
            if att is not None:
                att_sampled = F.upsample(att[img_counter, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,1,width,height),
                                            size=(112, 96), 
                                            mode='bilinear')
                sampled = sampled*att_sampled.cuda()
            img_input[img_counter, :,:,:] = sampled
        return img_input



        


