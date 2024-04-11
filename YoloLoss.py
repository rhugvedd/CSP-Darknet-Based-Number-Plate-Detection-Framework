import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, lambda_cord, lambda_obj, lambda_no_obj, anchor_nos):
        super(YoloLoss, self).__init__()

        self.lambda_cord = lambda_cord
        self.lambda_obj = lambda_obj
        self.lambda_no_obj = lambda_no_obj
        self.anchor_nos = anchor_nos
        
    def forward(self, predictions, targets):
        """
        Inputs:
        predictions of shape: batch_size, width, height, anchor_data
        targets of shape: batch_size, width, height, anchor_data
        anchor_nos: No. of Anchors

        Returns:
        Loss normalized with batch size
        """

        m, w, h, _ = targets.shape

        predictions = predictions.view(m, w, h, self.anchor_nos, -1)
        targets = targets.view(m, w, h, self.anchor_nos, -1)

        object_present = torch.zeros(m, w, h, self.anchor_nos, dtype = torch.int8, device = targets.device)
        no_object = torch.ones(m, w, h, self.anchor_nos, dtype = torch.int8, device = targets.device)
        
        object_present[targets.view(m, w, h, self.anchor_nos, -1).any(axis = -1)] = 1
        no_object[targets.view(m, w, h, self.anchor_nos, -1).any(axis = -1)] = 0

        # TODO: Should the loss computation be with respect to all anchor boxes?
        loss = self.lambda_cord * torch.sum(object_present * self.CIoU_Loss(predictions, targets))

        BCE_loss = nn.BCELoss(reduction='none')(predictions[..., 4], targets[..., 4])

        loss += torch.sum(object_present * BCE_loss)

        loss += self.lambda_no_obj * torch.sum(no_object * BCE_loss)

        loss /= targets.size(0)

        return loss

    def CIoU_Loss(self, pred_boxes, gt_boxes, epsilon = 1e-10):

        bbox_coords = lambda box:  torch.stack([box[..., 0] - (box[..., 2] / 2), 
                                                box[..., 1] - (box[..., 3] / 2),
                                                box[..., 0] + (box[..., 2] / 2), 
                                                box[..., 1] + (box[..., 3] / 2)], dim = -1)

        pred_box_coords = bbox_coords(pred_boxes)
        gt_box_coords = bbox_coords(gt_boxes)

        # IoU Loss ----------------------------------------------------------------------------------------------
        ix1 = torch.max(pred_box_coords[..., 0], gt_box_coords[..., 0])
        iy1 = torch.max(pred_box_coords[..., 1], gt_box_coords[..., 1])
        ix2 = torch.min(pred_box_coords[..., 2], gt_box_coords[..., 2])
        iy2 = torch.min(pred_box_coords[..., 3], gt_box_coords[..., 3])

        inter_area = torch.max(ix2 - ix1, torch.tensor(0)) * torch.max(iy2 - iy1, torch.tensor(0))
        union_area = (pred_boxes[..., 2] * pred_boxes[..., 3]) + (gt_boxes[..., 2] * gt_boxes[..., 3]) - inter_area

        iou_loss = inter_area / (union_area + epsilon)

        # Distance Loss -----------------------------------------------------------------------------------------
        sq_dist = ((gt_boxes[..., 0] - pred_boxes[..., 0]) ** 2) + ((gt_boxes[..., 1] - pred_boxes[..., 1]) ** 2)

        min_x1 = torch.min(pred_box_coords[..., 0], gt_box_coords[..., 0])
        min_y1 = torch.min(pred_box_coords[..., 1], gt_box_coords[..., 1])
        max_x2 = torch.max(pred_box_coords[..., 2], gt_box_coords[..., 2])
        max_y2 = torch.max(pred_box_coords[..., 3], gt_box_coords[..., 3])

        sq_diag = ((max_x2 - min_x1) ** 2) + ((max_y2 - min_y1) ** 2)

        dist_loss = sq_dist / (sq_diag + epsilon)
        
        # Aspect Ratio Loss -----------------------------------------------------------------------------------------

        v = 0.40528 *   (
                            (
                                torch.atan(gt_boxes[..., 2] / (gt_boxes[..., 3] + epsilon))
                                -
                                torch.atan(pred_boxes[..., 2] / (pred_boxes[..., 3] + epsilon))
                            ) ** 2
                        )

        alpha = v / (1 - iou_loss + v)

        return 1 - iou_loss + dist_loss + (alpha * v)