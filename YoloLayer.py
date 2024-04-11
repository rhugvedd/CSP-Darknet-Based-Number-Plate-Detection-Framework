import torch
import torch.nn as nn

class YoloConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(YoloConv, self).__init__()

        self.mish = nn.Mish()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding='same')
        self.norm1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same')
        self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1, padding='same')
        self.norm3 = nn.BatchNorm2d(in_channels)
        
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding='same')
        self.norm4 = nn.BatchNorm2d(hidden_channels)
        
        self.conv5 = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1, padding='same')
        self.norm5 = nn.BatchNorm2d(in_channels)
        
        self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.norm6 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):

        inputs = self.mish(self.norm1(self.conv1(inputs)))
        
        inputs = self.mish(self.norm2(self.conv2(inputs)))
        
        inputs = self.mish(self.norm3(self.conv3(inputs)))

        inputs = self.mish(self.norm4(self.conv4(inputs)))

        inputs = self.mish(self.norm5(self.conv5(inputs)))
        
        inputs = self.mish(self.norm6(self.conv6(inputs)))

        return inputs

class YoloHead(nn.Module):
    def __init__(self, anchors, num_classes, scaling_fact, in_channels):
        super(YoloHead, self).__init__()

        self.anchors_nos = len(anchors)
        self.anchors = anchors
        self.numclasses = num_classes
        self.scaling_fact = scaling_fact
        
        self.finalConv = nn.Conv2d(in_channels=in_channels, out_channels = self.anchors_nos * (5 + num_classes), kernel_size=1, padding='same')

    def forward(self, inputs, gt_anchor_idx):

        inputs = self.finalConv(inputs)

        # TODO: Here is the permute 
        inputs = inputs.permute(0,2,3,1)

        grid_size = (inputs.shape[1:3])

        inputs = inputs.view(-1, grid_size[0], grid_size[1], self.anchors_nos, 5 + self.numclasses)
        inputs_shape = inputs.shape

        inputs[..., 0:2] = nn.Sigmoid()(inputs[..., 0:2])
        inputs[..., 2:4] = torch.exp(inputs[..., 2:4])
        inputs[..., 4] = nn.Sigmoid()(inputs[..., 4])

        gt_anchor_idx_shape = gt_anchor_idx.shape

        gt_anchors = torch.index_select(self.anchors, 0, gt_anchor_idx.view(-1)).view(gt_anchor_idx_shape + (2,))
        inputs[..., 2:4] = inputs[..., 2:4] * gt_anchors[:,:,:,None,:]

        # TODO: Check this, should we add to all anchor boxes?
        inputs[..., 1] += torch.arange(inputs.shape[1], device=inputs.device)[:, None, None]
        inputs[..., 0] += torch.arange(inputs.shape[2], device=inputs.device)[:, None]
        
        inputs[..., 0:2] = inputs[..., 0:2] * self.scaling_fact

        return inputs