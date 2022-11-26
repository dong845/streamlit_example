import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class CifarClassifier(nn.Module):
    def __init__(self, classes):
        super(CifarClassifier, self).__init__()
        self.conv1 = ConvBlock(3, 32, 3, 1, 0)
        self.conv2 = ConvBlock(32, 32, 3, 1, 0)
        self.conv3 = ConvBlock(32, 32, 3, 2, 1)
        self.conv4 = ConvBlock(32, 32, 3, 1, 0)
        self.conv5 = ConvBlock(32, 4, 3, 1, 0)
        self.linear1 = nn.Linear(400, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.relu1(self.linear1(x))
        x = self.linear2(x)
        return x
        