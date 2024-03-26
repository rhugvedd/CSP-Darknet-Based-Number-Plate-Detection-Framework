import torch
import torch.nn as nn
import time

def Norm(channels):
    # TODO: Check settings of BatchNorm for training and inference
    return nn.BatchNorm2d(channels)

def Conv(in_channels, out_channels, kernel_size, downSamp):
    if downSamp:
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)
    else:
        return  nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
    
class CSPDenseBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks):
        super(CSPDenseBlock, self).__init__()

        self.blocks = nn.ModuleList()
        self.DenseBlockInChannels = in_channels // 2

        self.blocks.append(
                        nn.Sequential   (
                                        Conv(self.DenseBlockInChannels, hidden_channels, kernel_size=1, downSamp=False),
                                        Norm(hidden_channels),
                                        nn.Mish(),
                                        Conv(hidden_channels, out_channels, kernel_size=3, downSamp=False),
                                        Norm(out_channels),
                                        nn.Mish()
                                        ))

        for block in range(1, num_blocks):
            self.blocks.append(
                nn.Sequential   (
                                Conv(self.DenseBlockInChannels + (out_channels * block), hidden_channels, kernel_size=1, downSamp=False),
                                Norm(hidden_channels),
                                nn.Mish(),
                                Conv(hidden_channels, out_channels, kernel_size=3, downSamp=False),
                                Norm(out_channels),
                                nn.Mish()
                                ))

        self.transLayer = Conv(out_channels + self.DenseBlockInChannels, out_channels, kernel_size=3, downSamp=False)
        self.norm1 = Norm(out_channels)
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

        self.conv1 = Conv(3, 32, kernel_size=3, downSamp=False)
        self.norm1 = Norm(32)

        self.downSamp1 = Conv(32, 64, kernel_size=3, downSamp=True)
        self.downSampNorm1 = Norm(64)
        self.CSP_DB1 = CSPDenseBlock(in_channels=64, hidden_channels=32, out_channels=64, num_blocks=1)
        
        self.downSamp2 = Conv(64, 128, kernel_size=3, downSamp=True)
        self.downSampNorm2 = Norm(128)
        self.CSP_DB2 = CSPDenseBlock(in_channels=128, hidden_channels=64, out_channels=64, num_blocks=2)
        
        self.downSamp3 = Conv(64, 256, kernel_size=3, downSamp=True)
        self.downSampNorm3 = Norm(256)
        self.CSP_DB3 = CSPDenseBlock(in_channels=256, hidden_channels=128, out_channels=128, num_blocks=8)
        
        self.downSamp4 = Conv(128, 512, kernel_size=3, downSamp=True)
        self.downSampNorm4 = Norm(512)
        self.CSP_DB4 = CSPDenseBlock(in_channels=512, hidden_channels=256, out_channels=256, num_blocks=8)

        self.downSamp5 = Conv(256, 1024, kernel_size=3, downSamp=True)
        self.downSampNorm5 = Norm(1024)
        self.CSP_DB5 = CSPDenseBlock(in_channels=1024, hidden_channels=512, out_channels=512, num_blocks=4)

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

        return inputs