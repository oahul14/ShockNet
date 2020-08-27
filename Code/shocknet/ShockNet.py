import torch
import torch.nn as nn
from .ResPart import ReSEUpPart, ReSEDownPart, ReSEBridgePart, ReSEStartPart, conv3x3


class ShockSENet_2(nn.Module):
    def __init__(self, has_se=True, reduction=16, up_conv=True):
        super(ShockSENet_2, self).__init__()
        
        self.start = ReSEStartPart(3, 64)
        
        self.down1 = ReSEDownPart(64, 64, has_se=has_se, reduction=reduction)
        self.down2 = ReSEDownPart(64, 128, has_se=has_se, reduction=reduction)

        self.bridge = ReSEBridgePart(128, 256, has_se=has_se, reduction=reduction)
        
        self.up2 = ReSEUpPart(128 + 128, 128, has_se=False, up_conv=up_conv)
        self.up1 = ReSEUpPart(64 + 64, 64, has_se=False, up_conv=up_conv)
 
        self.end = conv3x3(64, 3)

    def forward(self, x):
        x = self.start(x)
        
        # down sampling
        x, init_x1 = self.down1(x)
        x, init_x2 = self.down2(x)

        x = self.bridge(x)

        # up sampling
        x = self.up2(x, init_x2)
        x = self.up1(x, init_x1)

        x = self.end(x)

        return x

class ShockSENet_3(nn.Module):
    def __init__(self, has_se=True, reduction=16, up_conv=True):
        super(ShockSENet_3, self).__init__()
        
        self.start = ReSEStartPart(3, 64)
        
        self.down1 = ReSEDownPart(64, 64, has_se=has_se, reduction=reduction)
        self.down2 = ReSEDownPart(64, 128, has_se=has_se, reduction=reduction)
        self.down3 = ReSEDownPart(128, 256, has_se=has_se, reduction=reduction)

        self.bridge = ReSEBridgePart(256, 512, has_se=has_se, reduction=reduction)
        
        self.up3 = ReSEUpPart(256 + 256, 256, has_se=False, up_conv=up_conv)
        self.up2 = ReSEUpPart(128 + 128, 128, has_se=False, up_conv=up_conv)
        self.up1 = ReSEUpPart(64 + 64, 64, has_se=False, up_conv=up_conv)
 
        self.end = conv3x3(64, 3)

    def forward(self, x):
        x = self.start(x)
        
        # down sampling
        x, init_x1 = self.down1(x)
        x, init_x2 = self.down2(x)
        x, init_x3 = self.down3(x)

        x = self.bridge(x)

        # up sampling
        x = self.up3(x, init_x3)
        x = self.up2(x, init_x2)
        x = self.up1(x, init_x1)

        x = self.end(x)

        return x

class ShockSENet_4(nn.Module):
    def __init__(self, has_se=True, reduction=16, up_conv=True):
        super(ShockSENet_4, self).__init__()
        
        self.start = ReSEStartPart(3, 64)
        
        self.down1 = ReSEDownPart(64, 64, has_se=has_se, reduction=reduction)
        self.down2 = ReSEDownPart(64, 128, has_se=has_se, reduction=reduction)
        self.down3 = ReSEDownPart(128, 256, has_se=has_se, reduction=reduction)
        self.down4 = ReSEDownPart(256, 512, has_se=has_se, reduction=reduction)

        self.bridge = ReSEBridgePart(512, 1024, has_se=has_se, reduction=reduction)

        self.up4 = ReSEUpPart(512 + 512, 512, has_se=has_se, reduction=reduction, up_conv=up_conv)
        self.up3 = ReSEUpPart(256 + 256, 256, has_se=has_se, reduction=reduction, up_conv=up_conv)
        self.up2 = ReSEUpPart(128 + 128, 128, has_se=has_se, reduction=reduction, up_conv=up_conv)
        self.up1 = ReSEUpPart(64 + 64, 64, has_se=has_se, reduction=reduction, up_conv=up_conv)
 
        self.end = conv3x3(64, 3)

    def forward(self, x):
        x = self.start(x)
        
        # down sampling
        x, init_x1 = self.down1(x)
        x, init_x2 = self.down2(x)
        x, init_x3 = self.down3(x)
        x, init_x4 = self.down4(x)

        x = self.bridge(x)

        # up sampling
        x = self.up4(x, init_x4)
        x = self.up3(x, init_x3)
        x = self.up2(x, init_x2)
        x = self.up1(x, init_x1)

        x = self.end(x)

        return x
