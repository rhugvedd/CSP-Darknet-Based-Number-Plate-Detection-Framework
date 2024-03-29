# Test Darknet on CIFAR-10

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

from Darknet import CSPDarknet

class ChangeView(nn.Module):
    def __init__(self, Batch_Size):
        super(ChangeView, self).__init__()
        self.Batch_Size = Batch_Size
    
    def forward(self, inputs):
        return inputs.view(self.Batch_Size, -1)


print("Loading Dataset")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

X_train = torch.stack([sample[0] for sample in trainset])
y_train = torch.tensor([sample[1] for sample in trainset])

X_test = torch.stack([sample[0] for sample in testset])
y_test = torch.tensor([sample[1] for sample in testset])

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image_shape = (416, 416)
image_shape = (32, 32)
Batch_Size = 256

# CIFAR-10 (Time for 1 Epoch)
# 16 -> 140s
# 32 -> 85s
# 64 -> 56s
# 128 -> 44s
# 256 -> 39s
# 512 -> 43s
# 1024 -> 41.s

# X_train = X_train[:Batch_Size]
# y_train = y_train[:Batch_Size]

# X_test = X_test[:Batch_Size]
# y_test = y_test[:Batch_Size]

X_train = X_train[:(len(X_train) - (len(X_train) % Batch_Size))]
y_train = y_train[:(len(y_train) - (len(y_train) % Batch_Size))]

X_test = X_test[:(len(X_test) - (len(X_test) % Batch_Size))]
y_test = y_test[:(len(y_test) - (len(y_test) % Batch_Size))]

print("Initializing Darknet")
Darknet = nn.Sequential(CSPDarknet(image_shape, training = True), ChangeView(Batch_Size), nn.Linear(512, 256), nn.Linear(256, 10))

Darknet.to(device)

LossCriteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(Darknet.parameters(), lr=0.001)

num_batches = len(X_train) // Batch_Size

start_time = time.time()

print("Computing Started")

for epoch in range(5):
    running_loss = 0.0
    
    epoch_st_time = time.time()
    
    for i in range(num_batches):
        
        inputs = X_train[i*Batch_Size:(i+1)*Batch_Size].to(device)
        labels = y_train[i*Batch_Size:(i+1)*Batch_Size].to(device)

        optimizer.zero_grad()

        outputs = Darknet(inputs)
        loss = LossCriteria(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    print(f"Time Taken Epoch: {epoch + 1} - {time.time() - epoch_st_time:.2f} seconds")

# Test the trained model
correct = 0
total = 0
with torch.no_grad():
    for i in range(0, len(X_test), Batch_Size):
        inputs = X_test[i:i+Batch_Size].to(device)
        labels = y_test[i:i+Batch_Size].to(device)
        outputs = Darknet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

end_time = time.time()
print("Total time taken: {:.2f} seconds".format(end_time - start_time))