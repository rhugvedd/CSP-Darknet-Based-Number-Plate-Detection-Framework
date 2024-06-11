import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
from datetime import datetime

from DataExtractor import *
from Darknet import *
from YoloLayer import *
from YoloLoss import *

class NumberPlateDetector(nn.Module):
    def __init__(self, image_size, scaling_fact, anchors, device, norm_momentum):
        super(NumberPlateDetector, self).__init__()

        self.anchor_nos = len(anchors)

        self.Darknet53 = CSPDarknet(in_dim=image_size, norm_momentum = norm_momentum)
        self.YoloConv = YoloConv(in_channels=512, hidden_channels=1024, out_channels=512)
        self.YoloHead = YoloHead(anchors.to(device), num_classes=0, scaling_fact=torch.tensor(scaling_fact).to(device), in_channels=512)
        self.YoloLoss = YoloLoss(lambda_cord=5, lambda_obj=None, lambda_no_obj=0.5, anchor_nos=self.anchor_nos)

    def forward(self, X, score_threshold):

        X = self.Darknet53(X)
        X = self.YoloConv(X)
        X = self.YoloHead(X)

        X = X.reshape(-1, 5)

        boxes_list = []

        # TODO: Can improve this implementation, for more speed.
        for box in X:
            if box[4] > score_threshold:
                boxes_list.append(box[:4])
    
        boxes = torch.stack(boxes_list, dim = 0)
        
        bbox_coords = lambda box:  torch.stack([box[..., 0] - (box[..., 2] / 2), 
                                                box[..., 1] - (box[..., 3] / 2),
                                                box[..., 0] + (box[..., 2] / 2), 
                                                box[..., 1] + (box[..., 3] / 2)], dim = -1)

        boxes = bbox_coords(boxes)

        return boxes

    def train_data(self, X, Y):
        
        m, w, h, _ = Y.shape

        gt_anchor_idx = torch.argmax(Y.view(m, w, h, self.anchor_nos, -1).any(-1).type(torch.int), dim = 3)

        X = self.Darknet53(X)
        X = self.YoloConv(X)
        X = self.YoloHead.train_data(X, gt_anchor_idx)

        loss = self.YoloLoss(X, Y)

        return loss