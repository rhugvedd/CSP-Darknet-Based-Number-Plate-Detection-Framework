import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

class CSPDenseBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks, in_dim):
        super(CSPDenseBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.DenseBlockInChannels = in_channels // 2

        self.blocks.append(
                        nn.Sequential   (
                                        nn.Conv2d(self.DenseBlockInChannels, hidden_channels, kernel_size=1, padding='same'),
                                        nn.LayerNorm((hidden_channels, in_dim[0], in_dim[1])),
                                        nn.Mish(),
                                        nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding='same'),
                                        nn.LayerNorm((out_channels, in_dim[0], in_dim[1])),
                                        nn.Mish()
                                        ))

        for block in range(1, num_blocks):
            self.blocks.append(
                nn.Sequential   (
                                nn.Conv2d(self.DenseBlockInChannels + (out_channels * block), hidden_channels, kernel_size=1, padding='same'),
                                nn.LayerNorm((hidden_channels, in_dim[0], in_dim[1])),
                                nn.Mish(),
                                nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding='same'),
                                nn.LayerNorm((out_channels, in_dim[0], in_dim[1])),
                                nn.Mish()
                                ))

        self.transLayer = nn.Conv2d(out_channels + self.DenseBlockInChannels, out_channels, kernel_size=3, padding='same')
        self.norm1 = nn.LayerNorm((out_channels, in_dim[0], in_dim[1]))
        self.mish = nn.Mish()

    def forward(self, inputs):
        dims = inputs.shape
        chunk1 = inputs[:, :(dims[1] // 2), :, :]
        chunk2 = inputs[:, (dims[1] // 2):, :, :]

        out = 0
        for block in self.blocks:
            out = block(chunk2)
            chunk2 = torch.cat([chunk2, out], dim=1)

        out = torch.cat([chunk1, out], dim=1)

        out = self.mish(self.norm1(self.transLayer(out)))

        return out

class CSPDarknet(nn.Module):
    def __init__(self, in_dim, training, data_format='channels_first'):
        super(CSPDarknet, self).__init__()

        self.Mish = nn.Mish()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.norm1 = nn.LayerNorm((32, in_dim[0], in_dim[1]))

        self.downSamp1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.downSampNorm1 = nn.LayerNorm((64, in_dim[0] // 2, in_dim[1] // 2))
        self.CSP_DB1 = CSPDenseBlock(in_channels=64, hidden_channels=32, out_channels=64, num_blocks=1, in_dim=(in_dim[0]//2, in_dim[1]//2))
        
        self.downSamp2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.downSampNorm2 = nn.LayerNorm((128, in_dim[0] // 4, in_dim[1] // 4))
        self.CSP_DB2 = CSPDenseBlock(in_channels=128, hidden_channels=64, out_channels=64, num_blocks=2, in_dim=(in_dim[0]//4, in_dim[1]//4))
        
        self.downSamp3 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.downSampNorm3 = nn.LayerNorm((256, in_dim[0] // 8, in_dim[1] // 8))
        self.CSP_DB3 = CSPDenseBlock(in_channels=256, hidden_channels=128, out_channels=128, num_blocks=8, in_dim=(in_dim[0]//8, in_dim[1]//8))
        
        self.downSamp4 = nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1)
        self.downSampNorm4 = nn.LayerNorm((512, in_dim[0] // 16, in_dim[1] // 16))
        self.CSP_DB4 = CSPDenseBlock(in_channels=512, hidden_channels=256, out_channels=256, num_blocks=8, in_dim=(in_dim[0]//16, in_dim[1]//16))

        self.finalConvOutChannels = 512

        self.downSamp5 = nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1)
        self.downSampNorm5 = nn.LayerNorm((1024, in_dim[0] // 32, in_dim[1] // 32))
        self.CSP_DB5 = CSPDenseBlock(in_channels=1024, hidden_channels=512, out_channels=self.finalConvOutChannels, num_blocks=4, in_dim=(in_dim[0]//32, in_dim[1]//32))


        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, inputs):
        inputs = self.Mish(self.norm1(self.conv1(inputs)))

        inputs = self.Mish(self.downSampNorm1(self.downSamp1(inputs)))
        inputs = self.CSP_DB1(inputs)

        inputs = self.Mish(self.downSampNorm2(self.downSamp2(inputs)))
        inputs = self.CSP_DB2(inputs)

        inputs = self.Mish(self.downSampNorm3(self.downSamp3(inputs)))
        inputs = self.CSP_DB3(inputs)

        inputs = self.Mish(self.downSampNorm4(self.downSamp4(inputs)))
        inputs = self.CSP_DB4(inputs)

        inputs = self.Mish(self.downSampNorm5(self.downSamp5(inputs)))
        inputs = self.CSP_DB5(inputs)
        
        inputs = inputs.view(Batch_Size, -1)
        
        inputs = self.fc2(self.fc1(inputs))

        return inputs

print("Loading Dataset")
# Download and prepare CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Load entire dataset into RAM
X_train = torch.stack([sample[0] for sample in trainset])
y_train = torch.tensor([sample[1] for sample in trainset])

X_test = torch.stack([sample[0] for sample in testset])
y_test = torch.tensor([sample[1] for sample in testset])

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image_shape = (416, 416)
image_shape = (32, 32)
Batch_Size = 256

# CIFAR-10
# 16 -> 140s
# 32 -> 85s
# 64 -> 56s
# 128 -> 44s
# 256 -> 39s
# 512 -> 43s
# 1024 -> 41.s

X_train = X_train[:(len(X_train) - (len(X_train) % Batch_Size))]
y_train = y_train[:(len(y_train) - (len(y_train) % Batch_Size))]

X_test = X_test[:(len(X_test) - (len(X_test) % Batch_Size))]
y_test = y_test[:(len(y_test) - (len(y_test) % Batch_Size))]

# inputs = torch.randn(8, 3, image_shape[0], image_shape[1], device=device)

print("Initializing Darknet")
Darknet = CSPDarknet(image_shape, training = True)
Darknet.to(device)

LossCriteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(Darknet.parameters(), lr=0.001)

# Train the model
num_batches = len(X_train) // Batch_Size

start_time = time.time()

print("Computing Started")

for epoch in range(1):  # adjust number of epochs as needed
    running_loss = 0.0
    
    epoch_st_time = time.time()
    
    for i in range(num_batches):
        # Get minibatch
        inputs = X_train[i*Batch_Size:(i+1)*Batch_Size].to(device)
        labels = y_train[i*Batch_Size:(i+1)*Batch_Size].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = Darknet(inputs)
        loss = LossCriteria(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    print(f"Time Taken Epoch: {epoch} - {time.time() - epoch_st_time:.2f} seconds")

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

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# Calculate and print the total time taken
end_time = time.time()
print("Total time taken: {:.2f} seconds".format(end_time - start_time))

# st_time = time.time()
# outputs = Darknet(inputs)

# print(f"Input Shape = {inputs.shape}")
# print(f"Output Shape = {outputs.shape}")
# print(f"Time Taken: {time.time() - st_time} s")