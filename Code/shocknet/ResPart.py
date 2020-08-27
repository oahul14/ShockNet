import torch
import torch.nn as nn

def conv1x1(in_ch, out_ch):
    """For down sampling which retains the size of the input"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, 
                padding=0, bias=False, dilation=1)

def conv3x3(in_ch, out_ch):
    """For down sampling which retains the size of the input"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, 
                padding=1, bias=False, dilation=1)

def conv4x4(in_ch, out_ch):
    """For down sampling which narrows the size of the input"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                    padding=1, bias=False, dilation=1)

def conv5x5(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1,
                    padding=2, bias=False, dilation=1)

def convT4x4(in_ch, out_ch):
    """For up sampling which expands the size of the input"""
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                    padding=1, bias=False, dilation=1)

def up_sample(scale_factor=2, mode='bilinear', align_corners=True):
    return nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)

class SEPart(nn.Module):
    def __init__(self, ch, reduction=16, act=nn.ReLU(inplace=True)):
        super(SEPart, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        inter_ch = ch // reduction
        self.fc = nn.Sequential(
            nn.Linear(ch, inter_ch, bias=False),
            act,
            nn.Linear(inter_ch, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ts_size = x.size()
        se = self.avgpool(x).view(ts_size[0], ts_size[1])
        se = self.fc(se).view(ts_size[0], ts_size[1], 1, 1)
        x = x * se
        return x
    
class ReSEStartPart(nn.Module):
    def __init__(self, in_ch, out_ch, act=nn.ReLU(inplace=True)):
        super(ReSEStartPart, self).__init__()
        self.start = nn.Sequential(
            conv5x5(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            act
        )
        
    def forward(self, x):
        return self.start(x)

class ConvConvSE(nn.Module):
    def __init__(self, in_ch, out_ch, has_se=True, reduction=16, act=nn.ReLU(inplace=True)):
        super(ConvConvSE, self).__init__()
        self.conv_conv_se = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            act,
            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
        )
        self.has_se = has_se
        self.se = SEPart(out_ch, reduction)
    
    def forward(self, x):
        x = self.conv_conv_se(x)
        if self.has_se:
            x = self.se(x)
        return x

class ReSEDownPart(nn.Module):
    def __init__(self, in_ch, out_ch, has_se=True, stride=1, base_width=64, 
                dilation=1, reduction=16, act=nn.ReLU(inplace=True)):
        super(ReSEDownPart, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv_conv_se = ConvConvSE(out_ch, out_ch, 
                                    has_se=has_se, reduction=reduction)
        self.act = act
        self.down_sample = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        init = self.conv1(x)
        x = self.conv1(x)
        x = self.conv_conv_se(x)
        x += init
        for_up = self.act(x)
        for_down = self.down_sample(for_up)
        return for_down, for_up

class ReSEBridgePart(nn.Module):
    def __init__(self, in_ch, out_ch, has_se=True, stride=1, base_width=64, 
                dilation=1, reduction=16, act=nn.ReLU(inplace=True)):
        super(ReSEBridgePart, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv_conv_se = ConvConvSE(out_ch, out_ch, 
                                    has_se=has_se, reduction=reduction)
        self.act = act
        
    def forward(self, x):
        init = self.conv1(x)
        x = self.conv1(x)
        x = self.conv_conv_se(x)

        x += init
        x = self.act(x)
        
        return x

class ReSEUpPart(nn.Module):
    def __init__(self, in_ch, out_ch, has_se=True, stride=1, base_width=64, 
                dilation=1, reduction=16, up_conv=True, act=nn.ReLU(inplace=True)):
        super(ReSEUpPart, self).__init__()
        self.up_sample = None
        if up_conv:
            self.up_sample = convT4x4(in_ch, out_ch)
        else:
            self.up_sample = up_sample()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv_conv_se = ConvConvSE(out_ch, out_ch, 
                                    has_se=has_se, reduction=reduction)
        self.act = act
        
    def forward(self, x, x_left):
        x = self.up_sample(x)
        x = torch.cat([x, x_left], dim=1)
        init = self.conv1(x)
        x = self.conv1(x)
        x = self.conv_conv_se(x)

        x += init
        x = self.act(x)

        return x
