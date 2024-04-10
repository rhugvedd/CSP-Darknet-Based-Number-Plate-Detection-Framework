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
    def __init__(self, image_size, scaling_fact, anchors, device):
        super(NumberPlateDetector, self).__init__()

        self.anchor_nos = len(anchors)

        self.Darknet53 = CSPDarknet(in_dim=image_size, training=True)
        self.YoloConv = YoloConv(in_channels=512, hidden_channels=1024, out_channels=512)
        self.YoloHead = YoloHead(anchors.to(device), num_classes=0, scaling_fact=torch.tensor(scaling_fact).to(device), in_channels=512)
        self.YoloLoss = YoloLoss(lambda_cord=5, lambda_obj=None, lambda_no_obj=0.5)

    def forward(self, X, Y):
        X = self.Darknet53(X)
        X = self.YoloConv(X)

        gt_anchor_idx = torch.argmax(Y.any(-1).type(torch.int), dim = 3)
        
        X = self.YoloHead(X, gt_anchor_idx)

        loss = self.YoloLoss(X, Y, self.anchor_nos)

        return loss

"""
List of Hyperparameters
"""
model_name = 'NumPlate-5000'
checkpoint_path = './CheckPoints/'
data_path = "NumPlateData/"
image_scale = 9
image_size = (image_scale*32, image_scale*32)
scaling_fact = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 48
num_examples = 5000
num_epochs = 100
loss_eval_interval = 0
loss_eval_interval += 1
checkpoint_save_epoch = 1
"""
List ends.
"""

num_examples = num_examples - (num_examples % batch_size)

TrainDataExtractor = DataExtractor  (
                                        data_path=data_path + 'train/', 
                                        save_path=data_path,
                                        annotations_csv="_CleanedData.csv", 
                                        scaled_image_size=image_size, 
                                        scaling_fact=scaling_fact,
                                        device=device
                                    )

anchors = TrainDataExtractor.getAnchors_fromMem('Anchor_train')
anchor_nos = anchors.size(0)

X_train, Y_train = TrainDataExtractor.getXY_fromMem(X_Name='X_Train-2024-04-09 19-12-15', Y_Name='Y_Train-2024-04-09 19-12-15')[:num_examples]

NumPlatesTrainer = NumberPlateDetector(image_size, scaling_fact, anchors, device)
NumPlatesTrainer.to(device)

optimizer = optim.Adam(NumPlatesTrainer.parameters(), lr=0.001)
num_batches = len(X_train) // batch_size

start_time = time.time()

print("Computing Started")

for epoch in range(num_epochs + 1):
    running_loss = 0.0
    epoch_loss = 0.0
    
    epoch_st_time = time.time()
    
    for i in range(num_batches):
        
        X_batch = X_train[i * batch_size : (i + 1) * batch_size].to(device).float()
        Y_batch = Y_train[i * batch_size : (i + 1) * batch_size].to(device).float()

        optimizer.zero_grad()
        loss = NumPlatesTrainer(X_batch, Y_batch)
        running_loss += loss
        epoch_loss += loss
        loss.backward()
        optimizer.step()

        if i % loss_eval_interval == 0:
            print('[Epoch: %d, Batch: %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / loss_eval_interval))
            running_loss = 0.0

    if (epoch % checkpoint_save_epoch == 0) and (epoch != 0):
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

        torch.save  ({
                        'epoch': epoch,
                        'model_state_dict': NumPlatesTrainer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, checkpoint_path + model_name + '-Epoch-' + str(epoch) + '-' + date_time + '.pth')
        
        print("Checkpoint Saved")
    
    print(f"Time Taken Epoch: {epoch} - {time.time() - epoch_st_time:.2f} seconds")
    print(f"Loss after Epoch {epoch}: {epoch_loss / num_batches}\n")
