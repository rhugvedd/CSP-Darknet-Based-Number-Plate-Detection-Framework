import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

def main():

    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv7 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

            self.norm1 = nn.LayerNorm(())
            self.norm2 = nn.LayerNorm((-3, -2, -1)) 
            self.norm3 = nn.LayerNorm((64, 8, 8)) 
            self.norm4 = nn.LayerNorm((64, 8, 8)) 
            self.norm5 = nn.LayerNorm((64, 8, 8)) 
            self.norm6 = nn.LayerNorm((64, 8, 8)) 
            self.norm7 = nn.LayerNorm((128, 8, 8)) 

            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.norm1(self.conv1(x))))
            x = self.pool(torch.relu(self.norm2(self.conv2(x))))
            x = torch.relu(self.norm3(self.conv3(x)))
            x = torch.relu(self.norm4(self.conv4(x)))
            x = torch.relu(self.norm5(self.conv5(x)))
            x = torch.relu(self.norm6(self.conv6(x)))
            x = self.pool(torch.relu(self.norm7(self.conv7(x))))
            x = x.view(-1, 128 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))

            x = self.fc3(x)
            return x

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

    # Define minibatch size
    batch_size = 64

    # Initialize the model and move it to GPU if available
    model = CNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if str(device) == 'cuda':
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print("GPU:", gpu_name)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_batches = len(X_train) // batch_size

    start_time = time.time()

    for epoch in range(5):  # adjust number of epochs as needed
        running_loss = 0.0
        for i in range(num_batches):
            # Get minibatch
            inputs = X_train[i*batch_size:(i+1)*batch_size].to(device)
            labels = y_train[i*batch_size:(i+1)*batch_size].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    # Test the trained model
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            inputs = X_test[i:i+batch_size].to(device)
            labels = y_test[i:i+batch_size].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    # Calculate and print the total time taken
    end_time = time.time()
    print("Total time taken: {:.2f} seconds".format(end_time - start_time))

if __name__ == '__main__':
    main()
