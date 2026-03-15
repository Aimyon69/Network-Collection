import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Dict
from collections import OrderedDict

def conv_bn(in_channels: int,out_channels: int,stride: int = 1,leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True)
    )

def conv_bn_no_relu(in_channels: int,out_channels: int,stride :int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride,bias=False),
        nn.BatchNorm2d(out_channels)
    )

def conv1x1(in_channels: int,out_channels: int,stride: int = 1,leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=stride,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True)
    )

def conv_dw(in_channels: int,out_channels: int,stride: int = 1,leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1,stride=stride,bias=False,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky,inplace=True) if leaky != 0 else nn.ReLU(inplace=True)
    )

class SSH(nn.Module):
    def __init__(self,in_channels: int,out_channels: int) -> None:
        super(SSH,self).__init__()

        assert out_channels % 4 == 0

        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
            self.relu = nn.LeakyReLU(negative_slope=leaky,inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.conv3x3 = conv_bn_no_relu(in_channels=in_channels,out_channels=out_channels//2,stride=1)
        
        self.conv5x5_1 = conv_bn(in_channels=in_channels,out_channels=out_channels//4,stride=1,leaky=leaky)
        self.conv5x5_2 = conv_bn_no_relu(in_channels=out_channels//4,out_channels=out_channels//4,stride=1)

        self.conv7x7_1 = conv_bn(in_channels=out_channels//4,out_channels=out_channels//4,stride=1,leaky=leaky)
        self.conv7x7_2 = conv_bn_no_relu(in_channels=out_channels//4,out_channels=out_channels//4,stride=1)

        

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_1 = self.conv7x7_1(conv5x5_1)
        conv7x7 = self.conv7x7_2(conv7x7_1)

        output = torch.cat([conv3x3,conv5x5,conv7x7],dim=1)
        output = self.relu(output)

        return output
    
class FPN(nn.Module):
    def __init__(self,in_channels_list: List[int],out_channels: int) -> None:
        super(FPN,self).__init__()

        leaky = 0
        if out_channels <= 64:
            leaky = 0.1

        self.conv1 = conv1x1(in_channels=in_channels_list[0],out_channels=out_channels,leaky=leaky)
        self.conv2 = conv1x1(in_channels=in_channels_list[1],out_channels=out_channels,leaky=leaky)
        self.conv3 = conv1x1(in_channels=in_channels_list[2],out_channels=out_channels,leaky=leaky)

        self.merge1 = conv_bn(in_channels=out_channels,out_channels=out_channels,leaky=leaky)
        self.merge2 = conv_bn(in_channels=out_channels,out_channels=out_channels,leaky=leaky)

    def forward(self,x: OrderedDict[str,torch.Tensor]) -> torch.Tensor:
        x = list(x.values())

        output1 = self.conv1(x[0])
        output2 = self.conv2(x[1])
        output3 = self.conv3(x[2])

        up3 = F.interpolate(output3,size=[output2.shape[2],output2.shape[3]],mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2,size=[output1.shape[2],output1.shape[3]],mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1,output2,output3]

class MobileNetV1(nn.Module):
    def __init__(self) -> None:
        super(MobileNetV1,self).__init__()

        self.stage1 = nn.Sequential(
            conv_bn(in_channels=3,out_channels=8,stride=2,leaky=0.1),
            conv_dw(in_channels=8,out_channels=16,stride=1),
            conv_dw(in_channels=16,out_channels=32,stride=2),
            conv_dw(in_channels=32,out_channels=32,stride=1),
            conv_dw(in_channels=32,out_channels=64,stride=2),
            conv_dw(in_channels=64,out_channels=64,stride=1)
        )

        self.stage2 = nn.Sequential(
            conv_dw(in_channels=64,out_channels=128,stride=2),
            conv_dw(in_channels=128,out_channels=128,stride=1),
            conv_dw(in_channels=128,out_channels=128,stride=1),
            conv_dw(in_channels=128,out_channels=128,stride=1),
            conv_dw(in_channels=128,out_channels=128,stride=1),
            conv_dw(in_channels=128,out_channels=128,stride=1),
        )

        self.stage3 = nn.Sequential(
            conv_dw(in_channels=128,out_channels=256,stride=2),
            conv_dw(in_channels=256,out_channels=256,stride=1)
        )

        

    def forward(self,x: torch.Tensor) -> OrderedDict[str,torch.Tensor]:
        output = OrderedDict()

        x = self.stage1(x)
        output['stage1'] = x
        x = self.stage2(x)
        output['stage2'] = x
        x = self.stage3(x)
        output['stage3'] = x
        
        return output

