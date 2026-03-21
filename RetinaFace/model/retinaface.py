import torch
import torch.nn as nn
from typing import Dict,List
import torch.nn.functional as F
from .net import SSH,FPN,MobileNetV1

class ClassHead(nn.Module):
    def __init__(self,in_channels: int,anchor_nums: int) -> None:
        super(ClassHead,self).__init__()

        self.classifier = nn.Conv2d(in_channels=in_channels,out_channels=anchor_nums*2,kernel_size=1,padding=0,stride=1)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        output = self.classifier(x)
        output = output.permute(0,2,3,1).contiguous()
        output = output.view(output.shape[0],-1,2)

        return output

class BboxHead(nn.Module):
    def __init__(self,in_channels: int,anchor_nums: int) -> None:
        super(BboxHead,self).__init__()

        self.Bbox = nn.Conv2d(in_channels=in_channels,out_channels=anchor_nums*4,kernel_size=1,padding=0,stride=1)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        output = self.Bbox(x)
        output = output.permute(0,2,3,1).contiguous()
        output = output.view(x.shape[0],-1,4)

        return output

class LandmarkHead(nn.Module):
    def __init__(self,in_channels: int,anchor_nums: int) -> None:
        super(LandmarkHead,self).__init__()

        self.landmark = nn.Conv2d(in_channels=in_channels,out_channels=anchor_nums*10,kernel_size=1,padding=0,stride=1)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        output = self.landmark(x)
        output = output.permute(0,2,3,1).contiguous()
        output = output.view(output.shape[0],-1,10)

        return output
    
class RetinaFace(nn.Module):
    def __init__(self,cfg: Dict,phase: str = 'train') -> None:
        super(RetinaFace,self).__init__()

        self.backbone = MobileNetV1()
        self.phase = phase

        in_channels = cfg['in_channels']
        out_channels = cfg['out_channels']
        fpn_nums = len(cfg['steps'])
        anchor_nums = len(cfg['min_sizes'][0])
        in_channels_list = [
            in_channels * 2,
            in_channels * 4,
            in_channels * 8
        ]
        self.fpn = FPN(in_channels_list=in_channels_list,out_channels=out_channels)
        self.ssh1 = SSH(in_channels=out_channels,out_channels=out_channels)
        self.ssh2 = SSH(in_channels=out_channels,out_channels=out_channels)
        self.ssh3 = SSH(in_channels=out_channels,out_channels=out_channels)
        self.ClassHead = make_ClassHead(fpn_nums=fpn_nums,in_channels=out_channels,anchor_nums=anchor_nums)
        self.BboxHead = make_BboxHead(fpn_nums=fpn_nums,in_channels=out_channels,anchor_nums=anchor_nums)
        self.LandmarkHead = make_LandmarkHead(fpn_nums=fpn_nums,in_channels=out_channels,anchor_nums=anchor_nums)
        if phase == 'train':
            self.init_weights()

    def forward(self,x: torch.Tensor) -> tuple[torch.Tensor]:
        output = self.backbone(x)

        fpn_out = self.fpn(output)

        feature1 = self.ssh1(fpn_out[0])
        feature2 = self.ssh2(fpn_out[1])
        feature3 = self.ssh3(fpn_out[2])
        features = [feature1,feature2,feature3]

        classification = torch.cat([self.ClassHead[i](feature) for i,feature in enumerate(features)],dim=1)
        bbox_regression = torch.cat([self.BboxHead[i](feature) for i,feature in enumerate(features)],dim=1)
        landmark_regression = torch.cat([self.LandmarkHead[i](feature) for i,feature in enumerate(features)],dim=1)
        if self.phase == 'train':
            return (classification,bbox_regression,landmark_regression)
        else:
            return (F.softmax(classification,dim=2),bbox_regression,landmark_regression)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias,0)
            elif isinstance(module,nn.BatchNorm2d):
                nn.init.constant_(module.weight,1)
                nn.init.constant_(module.bias,0)


def make_ClassHead(fpn_nums: int,in_channels: int,anchor_nums: int) -> nn.ModuleList:
    classhead = nn.ModuleList()

    for i in range(fpn_nums):
        classhead.append(ClassHead(in_channels=in_channels,anchor_nums=anchor_nums))

    return classhead

def make_BboxHead(fpn_nums: int,in_channels: int,anchor_nums: int) -> nn.ModuleList:
    bboxhead = nn.ModuleList()

    for i in range(fpn_nums):
        bboxhead.append(BboxHead(in_channels=in_channels,anchor_nums=anchor_nums))

    return bboxhead

def make_LandmarkHead(fpn_nums: int,in_channels: int,anchor_nums: int) -> nn.ModuleList:
    landmarkhead = nn.ModuleList()

    for i in range(fpn_nums):
        landmarkhead.append(LandmarkHead(in_channels=in_channels,anchor_nums=anchor_nums))

    return landmarkhead
        
        