from NumberPlateDetector import *

"""
List of Hyperparameters
"""
<<<<<<< HEAD
model_name = 'NumPlate-Final-20000-LR-0.00025'
checkpoint_path = './CheckPoints/'
load_check_point = True
checkpoint_name = 'NumPlate-Final-5000-Pre-Train-LR-0.0001-Epoch-20-2024-04-12 02-05-50' + '.pth'
data_path = "NumPlateData/"
X_Name = 'X_Train_9_20000-2024-04-12 02-18-41'
Y_Name = 'Y_Train_9_20000-2024-04-12 02-18-41'
=======
model_name = 'NumPlate-Final-5000-Pre-Train-LR-0.0001'
checkpoint_path = './CheckPoints/'
load_check_point = True
checkpoint_name = 'NumPlate-Final-5000-Pre-Train-Epoch-10-2024-04-11' + '.pth'
data_path = "NumPlateData/"
X_Name = 'X_Train_9-2024-04-11 22-53-24'
Y_Name = 'Y_Train_9-2024-04-11 22-53-24'
>>>>>>> 27ad35cec7d716757e4fb40398f31e7e6bf38057
anchor_name = 'Anchor_train_9'
image_scale = 9
image_size = (image_scale*32, image_scale*32)
scaling_fact = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 12
<<<<<<< HEAD
norm_momentum = 0.045
num_examples = 17500
num_epochs = 1000
loss_eval_interval = 0
loss_eval_interval += 1
checkpoint_save_epoch = 1
learning_rate = 0.00025
=======
norm_momentum = 0.075
num_examples = 5000
num_epochs = 1000
loss_eval_interval = 0
loss_eval_interval += 1
checkpoint_save_epoch = 5
learning_rate = 0.0001
>>>>>>> 27ad35cec7d716757e4fb40398f31e7e6bf38057
"""
List ends.
"""

num_examples = num_examples - (num_examples % batch_size)

TrainDataExtractor = DataExtractor  (
                                        data_path=data_path + 'train/', 
                                        save_path=data_path,
<<<<<<< HEAD
                                        annotations_csv="_annotations.csv", 
=======
                                        annotations_csv="_CleanedData.csv", 
>>>>>>> 27ad35cec7d716757e4fb40398f31e7e6bf38057
                                        scaled_image_size=image_size, 
                                        scaling_fact=scaling_fact,
                                        device=device
                                    )

anchors = TrainDataExtractor.getAnchors_fromMem(anchor_name)
anchor_nos = anchors.size(0)

X_train, Y_train = TrainDataExtractor.getXY_fromMem(X_Name=X_Name, Y_Name=Y_Name)[:num_examples]

# to_pil_image(X_train[0]).save('X_2.jpg')

NumPlatesTrainer = NumberPlateDetector(image_size, scaling_fact, anchors, device, norm_momentum)
NumPlatesTrainer.to(device)
optimizer = optim.Adam(NumPlatesTrainer.parameters(), lr=learning_rate)

st_epoch = 0
if load_check_point:
    checkpoint = torch.load(checkpoint_path + checkpoint_name)
    NumPlatesTrainer.load_state_dict(checkpoint['model_state_dict'])   
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print('\n\nLoaded Checkpoint: ' + checkpoint_path + checkpoint_name)
    
    lr = optimizer.param_groups[0]['lr']
    print("Learning rate of loaded model:", lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print("Setting learning rate to:", optimizer.param_groups[0]['lr'])
    
    st_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    print(f'Starting Epoch for Training: {st_epoch}')

num_batches = len(X_train) // batch_size

start_time = time.time()

print("Training Started")

TotalLoss_List = []

for epoch in range(st_epoch, num_epochs + 1):
    running_loss = 0.0
    epoch_loss = 0.0
    
    TotalEpoch_List = []

    epoch_st_time = time.time()
    
    for i in range(num_batches):
        
        X_batch = X_train[i * batch_size : (i + 1) * batch_size].to(device).float()
        Y_batch = Y_train[i * batch_size : (i + 1) * batch_size].to(device).float()

        optimizer.zero_grad()
        loss = NumPlatesTrainer.train_data(X_batch, Y_batch)
        running_loss += loss
        epoch_loss += loss
        loss.backward()
        optimizer.step()

        TotalEpoch_List.append(loss)

        if i % loss_eval_interval == 0:
            print('[Epoch: %d, Batch: %5d] loss: %.3f' % (epoch, i + 1, running_loss / loss_eval_interval))
            running_loss = 0.0

    TotalLoss_List.append(TotalEpoch_List)

<<<<<<< HEAD
=======
    print(TotalLoss_List)

>>>>>>> 27ad35cec7d716757e4fb40398f31e7e6bf38057
    if (epoch % checkpoint_save_epoch == 0) and (epoch != 0):
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '-')

        torch.save  ({
                        'epoch': epoch,
                        'model_state_dict': NumPlatesTrainer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': TotalLoss_List,
                    }, checkpoint_path + model_name + '-Epoch-' + str(epoch) + '-' + date_time + '.pth')
        
        TotalLoss_List = []
        print("Checkpoint Saved")
    
    print(f"Time Taken Epoch: {epoch} - {time.time() - epoch_st_time:.2f} seconds")
    print(f"Loss after Epoch {epoch}: {epoch_loss / num_batches}\n")
