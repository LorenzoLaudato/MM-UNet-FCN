""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.se1=SqEx(64)
        self.down2 = Down(128, 256)
        self.se2=SqEx(128)
        self.down3 = Down(256, 512)
        self.se3=SqEx(256)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.se4=SqEx(512)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        se1=self.se1(x1)
        x2 = self.down1(x1)
        se2=self.se2(x2)
        x3 = self.down2(x2)
        se3=self.se3(x3)
        x4 = self.down3(x3)
        se4=self.se4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, se4)
        x = self.up2(x, se3)
        x = self.up3(x, se2)
        x = self.up4(x, se1)
        logits = self.outc(x)
        return logits,x
