import torch
from torch import nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
        
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # MaxPooling (same for each layer)
        self.pool = nn.MaxPool2d(2, stride=2)
        
        # Encoder UNet
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        

        # Bottle neck
        self.bottleneck = DoubleConv(512, 1024)


        # Decoder Unet
        self.updec1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, output_padding=(1, 0))
        self.dec1 = DoubleConv(1024, 512)
        self.updec2 = nn.ConvTranspose2d(512, 256, 2, stride=2, output_padding=(1, 0))
        self.dec2 = DoubleConv(512, 256)
        self.updec3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.updec4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        self.final_dec = nn.Conv2d(64, 1, 1)


    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.updec1(b)
        cat4 = torch.cat([d4, e4], 1)
        d3 = self.updec2(self.dec1(cat4))
        cat3 = torch.cat([d3, e3], 1)
        d2 = self.updec3(self.dec2(cat3))
        cat2 = torch.cat([d2, e2], 1)
        d1 = self.updec4(self.dec3(cat2))
        # print("d1: ", d1.shape)
        # print("e1: ", e1.shape)
        cat1 = torch.cat([d1, e1], 1)
        
        z = self.final_dec(self.dec4(cat1))
        return z