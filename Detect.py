from NumberPlateDetector import *
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import re
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

import easyocr
import string
from PIL import Image
import warnings

OnlyOCR = False

state_codes = [
    "AN",  # Andaman and Nicobar Islands
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CH",  # Chandigarh
    "CT",  # Chhattisgarh
    "DL",  # Delhi
    "DN",  # Dadra and Nagar Haveli and Daman and Diu
    "GA",  # Goa
    "GJ",  # Gujarat
    "HP",  # Himachal Pradesh
    "HR",  # Haryana
    "JH",  # Jharkhand
    "JK",  # Jammu and Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "MH",  # Maharashtra
    "ML",  # Meghalaya
    "MN",  # Manipur
    "MP",  # Madhya Pradesh
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha
    "PB",  # Punjab
    "PY",  # Puducherry
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TG",  # Telangana
    "TN",  # Tamil Nadu
    "TR",  # Tripura
    "TS",  # Telangana
    "UK",  # Uttarakhand
    "UP",  # Uttar Pradesh
    "WB"   # West Bengal
]

replacements = {"NH": "MH",
                "HH": "MH"}

char_similar_to_nos =  {'O': '0',
                        'I': '1',
                        'J': '3',
                        'A': '4',
                        'G': '6',
                        'S': '5'}

nos_similar_to_char =   {'0': 'O',
                        '1': 'I',
                        '3': 'J',
                        '4': 'A',
                        '6': 'G',
                        '5': 'S'}

"""
Hyperparameter List:
"""
checkpoint_path = './CheckPoints/'
# checkpoint_name = 'NumPlate-Final-20000-LR-0.00025-Epoch-33-2024-04-13 06-44-20' + '.pth'
checkpoint_name = 'NumPlate-Final-5000-Pre-Train-LR-0.0001-Epoch-20-2024-04-12 02-05-50' + '.pth'
data_path = "./NumPlateData/"
image_path = "./TestImages/"
image_name = 'Test7.jpg'
image_scale = 9
image_size = (image_scale*32, image_scale*32)
scaling_fact = 32
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
Anchor_name = 'Anchor_Train_9'
score_threshold = 0.3
norm_momentum = 0.075
box_extension = 0.05
data_base_path = './Database.txt'
out_code_path = './Out.txt'
"""
List ends
"""

def get_num_plate_box():

    image_name = str(input("Enter Image Name: "))
    
    try:
        image = Image.open(image_path + image_name).convert("RGB")
    except:
        print("Wrong Image name") 
        return
    
    print("Locating Number Plate")
    
    image_width = image.width
    image_height = image.height

    min_dim = image_width
    if image_height < image_width:
        min_dim = image_height

    width_extension = box_extension * image_width
    height_extension = box_extension * image_height

    orig_image = image.copy()

    X_scale = image_size[0] / image.width
    Y_scale = image_size[1] / image.height

    transform = transforms.Compose([
        transforms.CenterCrop((min_dim, min_dim)),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    image = transform(image)
    image.to(device)

    # F.to_pil_image(image).save('./Temp.jpg')

    # x = input('Done')

    image = image[None,:,:,:]

    boxes = DetectNumPlate(image, score_threshold)

    boxes[:, 0::2] = (boxes[:, 0::2] / X_scale)
    boxes[:, 1::2] = boxes[:, 1::2] / Y_scale

    return boxes, orig_image, image_height

def drawBoxes_cropImg(orig_image, image_height):

    print("Drawing Bounding Boxes")

    box_int = []
    for idx in range(boxes.size(1)):
        box_int.append(int(boxes[0][idx].detach().numpy()))

    # transforms.Resize(orig_image)
    image_tensor = F.to_tensor(orig_image)

    cropped_image = image_tensor[:, box_int[1]: (box_int[3] + 1), box_int[0]: (box_int[2] + 1)]
    # cropped_image = cropped_image.resize(cropped_image.width)

    image_tensor = (image_tensor * 255).to(torch.uint8)
    cropped_image = (cropped_image * 255).to(torch.uint8)

    drawn_image_tensor = draw_bounding_boxes(image_tensor, boxes, colors=(0, 255, 0), width=int(image_height / 200))

    plt.imshow(drawn_image_tensor.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

    drawn_image = F.to_pil_image(drawn_image_tensor)
    drawn_image.save('./PredictedNumPlates.jpg')

    number_plate = F.to_pil_image(cropped_image)
    number_plate.save('./ExtractedNumPlate.jpg')

def check_num_plate(NumPlate):
    Numplates = '\n'.join([line for line in open(data_base_path, 'r', encoding = 'utf-8').read().splitlines()])

    if(Numplates.find(NumPlate) != -1):
        with open(out_code_path, 'w') as file:
            file.write('1')

def get_number(NumPlateImg):

    print("Extracting Number")

    reader = easyocr.Reader(['en'], gpu=True)
    text = reader.readtext(NumPlateImg)

    for chunk in text:
        raw_chunk = chunk[1]
        
        for old, new in replacements.items():
            raw_chunk = raw_chunk.replace(old, new)

        contains_state_code = any(code in raw_chunk for code in state_codes)
        contains_nos = bool(re.search(r'\d', raw_chunk))

        if contains_nos:
            NumPlate = ""
            if contains_state_code:

                least_st_indx = np.inf
                
                for code in state_codes:
                    st_index = raw_chunk.find(code)

                    if st_index != -1:
                        if st_index < least_st_indx: 
                            least_st_indx = st_index

                NumPlate += raw_chunk[least_st_indx: least_st_indx + 2]
                
                crnt_indx = least_st_indx + 2
                detection_part = 1
                detected_chars_in_phase = 0

                while crnt_indx < len(raw_chunk):
                    
                    crnt_char = raw_chunk[crnt_indx]

                    if (crnt_char == ' ' and crnt_char != ' '): 
                        NumPlate += ' '
                        continue
                    
                    if (detection_part == 1) or (detection_part == 3):
                        if (crnt_char.isdigit()) or (crnt_char in char_similar_to_nos.keys()):
                            if crnt_char.isdigit(): NumPlate += crnt_char
                            else: NumPlate += char_similar_to_nos[crnt_char]

                            detected_chars_in_phase += 1

                    elif detection_part == 2:
                        if (crnt_char in string.ascii_uppercase) or (crnt_char in nos_similar_to_char.keys()):
                            if crnt_char in string.ascii_uppercase: NumPlate += crnt_char
                            else: NumPlate += nos_similar_to_char[crnt_char]
                       
                            detected_chars_in_phase += 1

                    if ((detected_chars_in_phase == 2) and (detection_part <= 2)) or (detected_chars_in_phase == 4):
                        detected_chars_in_phase = 0
                        detection_part += 1
                        if NumPlate[-1] != ' ': NumPlate += ' '
                        if detection_part == 4: break

                    crnt_indx += 1

                # if detection_part == 4:
                NumPlate = NumPlate.rstrip()
                print("\nDetected Number: " + NumPlate + "\n")

                check_num_plate(NumPlate)

print("Computing Started")
STtime = time.time()

print("Initializing")
print('Loading the Trained Checkpoint')

# TrainDataExtractor = DataExtractor  (
#                                         data_path=data_path + 'train/', 
#                                         save_path=data_path,
#                                         annotations_csv="_Test.csv", 
#                                         scaled_image_size=image_size, 
#                                         scaling_fact=scaling_fact,
#                                         device=device
#                                     )

# anchors = TrainDataExtractor.getAnchors_fromMem(Anchor_name)
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor,.*Detect\.py:113")

anchors = torch.load(data_path + Anchor_name + '.pt')
anchor_nos = anchors.size(0)

DetectNumPlate = NumberPlateDetector(image_size, scaling_fact, anchors, device, norm_momentum)
DetectNumPlate.to(device)

checkpoint = torch.load(checkpoint_path + checkpoint_name)
DetectNumPlate.load_state_dict(checkpoint['model_state_dict'])   

DetectNumPlate.eval()

print('Initialization Complete')
print()


while True:
    boxes, orig_image, image_height = get_num_plate_box()
    drawBoxes_cropImg(orig_image, image_height)
    get_number('./ExtractedNumPlate.jpg')

print(f"Time Taken: {time.time()- STtime} s")