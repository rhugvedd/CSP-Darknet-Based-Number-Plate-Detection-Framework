# CSP-Darknet Based Number Plate Detection Framework  

This repository contains the code for a **Number Plate Detection Framework** based on the **CSP-Darknet53 architecture**. The project includes implementations of YOLO layers, loss functions, data preprocessing, and detection pipelines.

## Features  
- Implements the CSP-Darknet53 architecture for efficient feature extraction.  
- Incorporates YOLO layers and CIoU loss for accurate bounding box predictions.
- Custom data handling with tools to extract, preprocess, and format images and targets into PyTorch tensors.  
- Training the model from scratch and then testing.
- End-to-end training and evaluation pipeline.  
- Detection functionality for input images.

## File Descriptions  

- **Darknet.py**: This file implements the CSPDarknet architecture, a highly efficient convolutional neural network designed for feature extraction in object detection tasks, such as number plate recognition. It incorporates custom convolutional layers with optional down-sampling, batch normalization with configurable momentum, and densely connected CSPDenseBlocks that partition input channels to optimize computational efficiency. The network uses the Mish activation function to ensure smooth gradient flow and improved learning. The architecture progresses through multiple down-sampling layers and CSPDenseBlocks, effectively extracting hierarchical features from input data, making it a powerful backbone for YOLO-based detection frameworks.
- **DataExtractor.py**: The DataExtractor class is a versatile data preprocessing module designed for preparing training and testing datasets for number plate detection. It handles tasks such as resizing images, normalizing bounding boxes, and mapping them to a grid structure suitable for object detection models. The class includes methods for extracting features and targets from raw data, saving them in a tensor format, and dynamically generating anchor boxes using k-means clustering for optimal bounding box predictions. With additional functionalities to load precomputed anchors and preprocessed data, it ensures integration into the training pipeline, making it a critical component for efficient model training.
- **Detect.py**: The Detect.py script facilitates the detection and recognition of vehicle license plates from images, leveraging our own trained number plate detection model and EasyOCR. The process begins by loading a trained model checkpoint and anchor data to localize license plates in input images, followed by bounding box extraction and drawing on the original image. OCR is employed to read the extracted license plate text, which undergoes formatting and validation against predefined state codes and possible misreadings. The script supports live interaction, allowing users to process images iteratively while ensuring efficient data handling and user-friendly outputs.  
- **Train.py**: The Train.py script is designed for training a vehicle number plate detection model. It initializes hyperparameters, including the model name, checkpoint paths, learning rate, and batch size. The script loads training data using the DataExtractor class, sets up the NumberPlateDetector model, and trains it using the Adam optimizer. During training, it saves checkpoints periodically, allowing for resumption of training, and prints loss statistics after each epoch. The model's performance is evaluated by calculating the loss after each batch and epoch, and checkpoints are saved at regular intervals for future reference or further training.  
- **YoloLayer.py**: The Yololayer.py file defines the core layers used in the number plate detection model, focusing on the YoloConv and YoloHead classes. The YoloConv class is responsible for feature extraction, utilizing a series of convolutional layers with batch normalization and Mish activation functions to process input images. The YoloHead class follows, performing the final prediction steps, including bounding box coordinates, objectness score, and class probabilities, by applying a 1x1 convolution followed by appropriate transformations like sigmoid and exponential functions. It also includes a custom method (train_data) for handling the training phase, which adjusts the predicted outputs using ground truth data and anchor boxes. Together, these layers enable the network to learn and make accurate predictions on object detection tasks.  
- **YoloLoss.py**: The YoloLoss.py file defines the custom Yolo-loss function used for training. The primary class, YoloLoss, calculates the total loss by combining several components such as the coordinates loss, objectness loss, and the binary cross-entropy (BCE) loss for both object and non-object detections. It uses the Complete Intersection over Union (CIoU) loss to refine the bounding box predictions by considering factors like overlap, distance, and aspect ratio between predicted and ground truth boxes. The loss function is weighted by hyperparameters like lambda_cord, lambda_obj, and lambda_no_obj to balance the different components. The forward method computes the total loss over a batch of predictions and targets, ensuring the model learns to predict accurate bounding boxes and objectness scores while handling both object-present and object-absent scenarios.  
- **DarknetTest.py**: Tests the implementation of CSP-Darknet to ensure correctness.

## Dataset  

The model has been trained on the **Roboflow License Plate Dataset**.