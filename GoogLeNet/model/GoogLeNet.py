import torch
import torch.nn as nn
from typing import Optional,Tuple,List,Any

class BasicConv2d(nn.Module):
    def __init__(self,in_channels: int,out_channels: int,**kwargs: Any) -> None:
        super(BasicConv2d,self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels,eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Inception(nn.Module):
    def __init__(
            self,
            in_channels: int,
            ch1x1: int,
            ch3x3_red: int,
            ch3x3: int,
            ch5x5_red: int,
            ch5x5: int,
            pool_proj: int
    ) -> None:
        super(Inception,self).__init__()

        self.branch1 = BasicConv2d(in_channels,ch1x1,kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels,ch3x3_red,kernel_size=1),
            BasicConv2d(ch3x3_red,ch3x3,kernel_size=3,padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels,ch5x5_red,kernel_size=1),
            BasicConv2d(ch5x5_red,ch5x5,kernel_size=5,padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            BasicConv2d(in_channels,pool_proj,kernel_size=1)
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        branches = [branch1,branch2,branch3,branch4]
        return torch.cat(branches,dim=1)

class InceptionAux(nn.Module):
    def __init__(self,in_channels: int,num_classes: int) -> None:
        super(InceptionAux,self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = BasicConv2d(in_channels,128,kernel_size=1)
        self.fc1 = nn.Linear(4*4*128,1024)
        self.fc2 = nn.Linear(1024,num_classes)
        self.dropout = nn.Dropout(p=0.7)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            aux_logits: bool = True,
            init_weights: bool = True
    ) -> None:
        super(GoogLeNet,self).__init__()

        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.conv2 = BasicConv2d(64,64,kernel_size=1)
        self.conv3 = BasicConv2d(64,192,kernel_size=3,padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512,num_classes)
            self.aux2 = InceptionAux(528,num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024,num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.normal_(m.weight,1)
                nn.init.normal_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)

        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return aux1,aux2,x
        
        return x
