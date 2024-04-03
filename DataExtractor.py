import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
import time

class DataExtractor(nn.Module):
    def __init__(
                    self,
                    data_path,
                    annotations_csv,
                    scaled_image_size: tuple,
                    anchor_nos: int,
                    X_name: str,
                    Y_name: str,
                    Anchor_name: str,
                    scaling_fact: int,
                    device,
                    anchor_boxes: list = None
                ):
        super(DataExtractor, self).__init__()
    
        self.data_path = data_path
        self.annotations_csv = annotations_csv
        self.scaled_image_size = scaled_image_size
        self.X_name = X_name
        self.Y_name = Y_name
        self.scaling_fact = scaling_fact
        self.Anchor_name = Anchor_name

        if(anchor_boxes != None):
            self.anchor_nos = len(anchor_boxes)
        else:
            self.anchor_nos = anchor_nos

        self.to(device)

    def extractX_toMem(self):
        print("Extraction Started:")
        
        transform = transforms.Compose([
            transforms.Resize(self.scaled_image_size),
            transforms.ToTensor()
        ])
        
        csv_file = self.data_path + self.annotations_csv

        data = pd.read_csv(csv_file, header = None)
        
        image_names = data.iloc[:, 0]
        
        X_train = torch.zeros(image_names.shape[0], 3, self.scaled_image_size[0], self.scaled_image_size[1], dtype=torch.float16)
        
        for idx, image_name in enumerate(image_names):

            image = Image.open(self.data_path + image_name).convert("RGB")
            
            image = transform(image)
            
            X_train[idhx] = image

            print(idx)

        torch.save(X_train, self.data_path + self.X_name + '.pt')

    def getX_fromMem(self):
        return torch.load(self.data_path + self.X_name + '.pt')
    
    def get_max_iou_anchor(self, box, anchors):
        grid_cell = (box[0] // self.scaling_fact, box[1] // self.scaling_fact)
        
        max_iou = 0
        sel_anchor = 0

        for idx, anchor in enumerate(anchors):
            ix1 = torch.max(box[0] - (box[2] / 2), box[0] - (anchor[0] / 2))
            iy1 = torch.max(box[1] - (box[3] / 2), box[1] - (anchor[1] / 2))
            ix2 = torch.min(box[0] + (box[2] / 2), box[0] + (anchor[0] / 2))
            iy2 = torch.min(box[1] + (box[3] / 2), box[1] + (anchor[1] / 2))

            inter_area = torch.max(ix2 - ix1, torch.tensor(0)) * torch.max(iy2 - iy1, torch.tensor(0))
            union_area = (box[2] * box[3]) + (anchor[0] * anchor[1]) - inter_area
            
            iou = inter_area / union_area
            if (iou) > max_iou:
                sel_anchor = idx

                max_iou = iou

        return sel_anchor

    def get_bounding_boxes(self):
        
        csv_file = self.data_path + self.annotations_csv

        data = pd.read_csv(csv_file, header = None)

        image_names = data.iloc[:, 0]

        return image_names, torch.tensor(data.iloc[:, 1:].values)

    def extractAnchors_toMem(self):
        StTime = time.time()

        image_names, bounding_boxes = self.get_bounding_boxes()

        for idx, image_name in enumerate(image_names):

            image = Image.open(self.data_path + image_name).convert("RGB")

            X_scale = self.scaled_image_size[0] / image.width
            Y_scale = self.scaled_image_size[1] / image.height

            # transforms.Resize(self.scaled_image_size)(image).save("scaled.jpg", "JPEG")
            
            box = bounding_boxes[idx]

            centre_x = ((box[0] + box[2]) / 2) * X_scale
            centre_y = ((box[1] + box[3]) / 2) * Y_scale

            width = (box[2] - box[0]) * X_scale
            height = (box[3] - box[1]) * Y_scale

            bounding_boxes[idx][0] = centre_x
            bounding_boxes[idx][1] = centre_y
            bounding_boxes[idx][2] = width
            bounding_boxes[idx][3] = height

            if idx % 100 == 0: print(idx)
            
        anchors = self.detect_anchors(bounding_boxes[:, 2:], self.anchor_nos)
        
        print(anchors)
        print(f"Time Taken: {time.time() - StTime} s")
        
        if str(input("Save? ('N' for NO)")) != 'N': 
            torch.save(anchors, self.data_path + self.Anchor_name + '.pt')
            print("Anchors saved succesfully!")
        else:
            print("Anchors not saved :(")
        
    def extractY_toMem(self):
        _, bounding_boxes = self.get_bounding_boxes()
        anchors = self.getAnchors_fromMem()

        final_gird_size = (self.scaled_image_size[0] // self.scaling_fact, self.scaled_image_size[1] // self.scaling_fact)
        
        Y_train = torch.zeros(
                                bounding_boxes.shape[0], 
                                self.anchor_nos * 5, 
                                final_gird_size[0], 
                                final_gird_size[1], 
                                dtype=torch.float16
                            )

        for idx, box in enumerate(bounding_boxes):
            
            grid_cell = (box[0] // self.scaling_fact, box[1] // self.scaling_fact)
            max_iou_anchor = self.get_max_iou_anchor(box, anchors)

            Y_train[idx][max_iou_anchor * 5]    [grid_cell[0]][grid_cell[1]] = (box[0] - (grid_cell[0] * self.scaling_fact)) / final_gird_size[0]
            Y_train[idx][max_iou_anchor * 5 + 1][grid_cell[0]][grid_cell[1]] = (box[1] - (grid_cell[1] * self.scaling_fact)) / final_gird_size[1]
            Y_train[idx][max_iou_anchor * 5 + 2][grid_cell[0]][grid_cell[1]] = box[0] / anchors[max_iou_anchor][0]
            Y_train[idx][max_iou_anchor * 5 + 3][grid_cell[0]][grid_cell[1]] = box[1] / anchors[max_iou_anchor][1]
            Y_train[idx][max_iou_anchor * 5 + 4][grid_cell[0]][grid_cell[1]] = 1

            break
    
        # torch.save(Y_train, self.data_path + self.Y_name + '.pt')

    def getY_fromMem(self):
        return torch.load(self.data_path + self.Y_name + '.pt')

    def getAnchors_fromMem(self):
        return torch.load(self.data_path + self.Anchor_name + '.pt')

    def detect_anchors(self, box_dim, anchor_nos):

        kmeans = KMeans(n_clusters=anchor_nos)

        kmeans.fit(box_dim)

        cluster_centers = kmeans.cluster_centers_

        return cluster_centers

data_path = "NumPlateData/"

image_scale = 7
image_size = (image_scale*32, image_scale*32)
scaling_fact = 32

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

ExtractTrainData = DataExtractor (
                                data_path=data_path + 'train/', 
                                annotations_csv="_annotations.csv", 
                                scaled_image_size=image_size, 
                                anchor_nos=5,
                                X_name='X_train',
                                Y_name='Y_train',
                                Anchor_name='Anchor_train',
                                scaling_fact=scaling_fact,
                                device=device
                            )

ExtractTrainData.extractAnchors_toMem()
# ExtractTrainData.extractY_toMem()