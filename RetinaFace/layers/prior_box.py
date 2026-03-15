import torch
import torch.nn as nn
from typing import Dict,List
from math import ceil
from itertools import product

class PriorBox:
    def __init__(self,cfg: Dict) -> None:
        self.steps = cfg['steps']
        self.image_size: tuple[int,int] = cfg['image_size']
        self.feature_maps = [[ceil(self.image_size[0] / step),ceil(self.image_size[1] / step)] for step in self.steps]
        self.min_sizes = cfg['min_sizes']
        self.clip = cfg['clip']

    def forward(self) -> torch.Tensor:
        anchors: List[List[float]] = list()

        for idx,feature in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[idx]
            for i,j in product(range(feature[0]),range(feature[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[idx] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[idx] / self.image_size[0] for y in [i + 0.5]]
                    for cx,cy in product(dense_cx,dense_cy):
                        anchors.append([cx,cy,s_kx,s_ky])
        
        output = torch.Tensor(anchors)

        if self.clip:
            output.clamp_(max=1,min=0)

        return output




