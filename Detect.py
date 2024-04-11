from NumberPlateDetector import *
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

"""
Hyperparameter List:
"""
checkpoint_path = './CheckPoints/'
# checkpoint_name = 'NumPlate-Final-5000-Epoch-5-2024-04-11 23-28-49' + '.pth'
checkpoint_name = 'NumPlate-Final-5000-Pre-Train-Epoch-10-2024-04-11' + '.pth'
data_path = "NumPlateData/"
image_path = "./TestImages/"
image_name = 'Custom.jpg'
image_scale = 9
image_size = (image_scale*32, image_scale*32)
scaling_fact = 32
device = torch.device('cpu')
Anchor_name = 'Anchor_train_9'
score_threshold = 0.5
norm_momentum = 0.075
"""
List ends
"""

TrainDataExtractor = DataExtractor  (
                                        data_path=data_path + 'train/', 
                                        save_path=data_path,
                                        annotations_csv="_Test.csv", 
                                        scaled_image_size=image_size, 
                                        scaling_fact=scaling_fact,
                                        device=device
                                    )

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

anchors = TrainDataExtractor.getAnchors_fromMem(Anchor_name)
anchor_nos = anchors.size(0)

DetectNumPlate = NumberPlateDetector(image_size, scaling_fact, anchors, device, norm_momentum)
DetectNumPlate.to(device)

checkpoint = torch.load(checkpoint_path + checkpoint_name)
DetectNumPlate.load_state_dict(checkpoint['model_state_dict'])   

DetectNumPlate.eval()

start_time = time.time()

print("Computing Started")

STtime = time.time()

image = Image.open(image_path + image_name).convert("RGB")
orig_image = image.copy()

X_scale = image_size[0] / image.width
Y_scale = image_size[1] / image.height

image = transform(image)

image = image[None,:,:,:]

boxes = DetectNumPlate(image, score_threshold)

boxes[:, 0::2] = boxes[:, 0::2] / X_scale
boxes[:, 1::2] = boxes[:, 1::2] / Y_scale

image_tensor = F.to_tensor(orig_image)

# cropped_image = image_tensor[]

print()

image_tensor = (image_tensor * 255).to(torch.uint8)

drawn_image_tensor = draw_bounding_boxes(image_tensor, boxes, colors=(0, 255, 0), width=2)

print(f"Time Taken: {time.time()- STtime} s")

plt.imshow(drawn_image_tensor.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()

drawn_image = F.to_pil_image(drawn_image_tensor)
drawn_image.save('./PredictedNumPlates.jpg')