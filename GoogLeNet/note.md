# Go Deeper Cautiously（审慎地走向更深处）

---

## 前言（Foreword）

不得不说，其实原本准备放弃写这篇笔记，因为我实在做不到在自己对GoogLeNet没有深刻理解的情况下，强行写一篇滥竽充数的笔记仅仅只为了完成计划。正如所见，我最终还是说服了自己，并不是因为我理解了GoogLeNet的设计哲学，其实更像是一篇纠正了自己执念的自省。我仅仅只表达当下我的思考，希望后面看到这篇笔记的未来的我或其他人能够不吝指正。

---

## 困境（Predicament）

前面说到的放弃写下这篇笔记的原因是：对GoogLeNet设计哲学的不解。其中最重要的一环则是**稀疏性架构（Sparse Structure）**的理解。当我结合了多方说法并结合自己的理解说服了自己，是的，只是暂时的说服，后面的我重新开始审视之前的理解的时候发现，我理解中的稀疏总是散发着密集的气味，与网络设计的初衷背道而驰。直到我写下这篇笔记，我才发现我太过于执着为GoogLeNet的成功冠以一个充要的解释，也这是这份执着让我忽略了这个网络真正值得我吸收学习的东西，这也是我为什么说这篇笔记更像是自我执念的自省。论文将GoogLeNet的成功归结于稀疏性思想的胜利，但是与其将这份功劳归于玄之又玄的稀疏性阐述，不如将其归于论文团队的**学术直觉和实验验证的成功**。不可否定的是稀疏性确实为网络设计带来了新的思路和启发，接下来我就阐述下我理解下的GoogLeNet的稀疏性：

---

## 稀疏架构（Sparse Structure）

> Their main result states that if the probability distribution of the data-set is representable by a large, very sparse deep neural network, then the optimal network topology can be constructed layer by layer by analyzing the correlation statistics of the activations of the last layer and clustering neurons with highly correlated outputs.These clusters form the units of the next layer and are connected to the units in the previous layer.

>Although the strict math- ematical proof requires very strong conditions, the fact that this statement resonates with the well known Hebbian principle – neurons that fire together, wire together – suggests that the underlying idea is applicable even under less strict conditions, in practice.

> One must be cautious though: although the proposed architecture has become a success for computer vision, it is still questionable whether its quality can be attributed to the guiding principles that have lead to its construction.

上述就是论文中对Inception模块的指导原则的叙述。总结来说就是：在此之前的网络结构是“密集”的，因为它们没有考虑到units之间的相关性，只是“暴力”的将所有units相互连接，得到结果似乎并不坏，但是引入了两个关键的问题：**1.过拟合（Overfitting）2.计算资源增大（Increased Computation Resource）**。因为最简单提升网络性能的方法就是增加网络深度，但是增加深度的同时上面的两个主要问题是不可避免的。所以根据Arora的成果，可以得到一个可能的解决方法。就是将原先原来的密集架构转向稀疏架构，我们考虑到units之间的相关性，通过计算这种相关性将相关性高的units连接在一起，而摒弃之前“暴力全连接”的方法，这可能不但不降低网络性能，反而在提升网络性能的同时减少计算量以及过拟合问题。Arora的结论和Hebbian的观念不谋而合，即使Arora结论的前提是苛刻的数学条件，也许也正是这份相似，能够使得这个结论在不那么严格的数学条件下适用。

---

## Inception模块（Inception Module）

根据上面的稀疏架构思想来推导Inception模块的构造：

> We assume that each unit from the earlier layer corresponds to some region of the input image and these units are grouped into filter banks.

`我们假设来自较早层的每个单元对应于输入图像的某个区域，并且这些单元被分组为滤波器组（filter banks）。`

这很符合卷积感受野的定义，我们在上述的条件下继续分析：

在靠近输入的网络层，一开始网络的感受野很小，相关的units一定对应原图像中很小的局部区域，也就表明在同一很小的局部区域中我们会得到许多的聚类，将上面得到的许多的聚类进一步通过其相关性进行聚类得到下一层的units。也正是因为一开始的网络感受野很小且units对应局部小区域，网络此时提取的是图像中的低级特征，这样的低级特征耦合性或者说相关性是很高的，我们要将其进行进一步聚类，自然而然就可以用1x1卷积核，它不会增大感受野，只是对当前空间维度的线性映射，达到聚类的效果，而且1x1正好对应浅层units对应局部小区域的特点。随着层数的增加，低级特征变得高级，此时特征的空间稀疏性变强，高级特征整合了不同空间尺度的低级特征，现在要继续实现稀疏连接，达到高级特征聚类的效果显然1x1卷积核不能满足空间尺度要求，那么大尺寸卷积核是必须的。值得说明的是，即使是浅层时大尺寸卷积核，以及深层时的小卷积核都是必要的，它们并非独立，而是互补的关系，他们能在同一层网络，拥有同样大小的感受野的情况下，更好的聚类不同空间尺度的相关性高的units。这也是Inception并行结构的原因之一。也很容易得出，随着网络层数的加深，大卷积核的数量应该增加。考虑到图块对齐的问题，将大卷积核的大小固定为3x3，5x5。

但是问题随之而来，这样的结构似乎并没有达到限制计算量的初衷，这样的模块嵌入网络之中只会加剧计算量。那么如何解决这个问题呢？

>judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise.

`在计算需求会急剧增加的地方，明智地（审慎地）应用降维和投影。`

注意这里的**judiciously（审慎地）**，也是这篇的笔记的主题。既然是审慎地使用，那么降维投影一定是一个兼并优劣的方法。

***劣：***显然降维一定会带来**信息的损失**，论文中说：`even low dimensional embeddings might contain a lot of information about a relatively large image patch.`，也就是说这样的降维虽然会导致信息的损失，但是留下来的信息依然是丰富的，这样的舍是可以接受的。降维会导致信息的压缩：`However, embeddings represent information in a dense, compressed form and compressed information is harder to model.`，密集压缩的信息会导致模型难以建模。

***优：***在损失可接受的情况下，显著减少计算资源。且引入额外的非线性，隐含的正则化。

所以一个策略被提出：

>compress the signals only whenever they have to be aggregated en masse. That is, 1×1 convolutions are used to compute reductions before the expensive 3×3 and 5×5 convolutions.

只在显著增加消耗计算资源的大卷积核前使用降维。其他地方维持信息稀疏性。模块的输出把所有分支的输出在通道维度上**拼接（Concatenate）**起来，通道数又重新变宽（从密集的压缩态，重新展开为高维的稀疏态），让下一层网络更容易进行“建模”。

最后鉴于其他架构中池化操作的成功，Inception添加一条池化分支。最后得到如下的模块结构：

![image-20260221205901772](D:\CodeRepository\Network-Collection\GoogLeNet\Inception.png)

---

## 全局平均池化（Global Average Pool）

全局平均池化（GAP）通过对每个特征通道求均值来彻底取代全连接层，在暴减千万级参数以极大降低过拟合风险的同时，强制网络提取出不受物体位置影响的全局鲁棒特征，彻底重塑了现代CNN“轻量化且高泛化”的尾部设计范式。

---

## 辅助分类器（Auxiliary Inception）

> One interesting insight is that the strong performance of relatively shallower networks on this task suggests that the features produced by the layers in the middle of the network should be very discriminative.

在那个时代，此模块的设计意义是：为了**强行注入梯度**而存在的网络辅助训练器。

而当**Kaiming 初始化和 Batch Normalization（BN）** 在数学上彻底解决了“梯度消失”后，它的新时代意义是：

- ***正则化手段：***它强迫网络在中间层就对齐最终的语义目标，防止网络在深层的前向传播中“走偏”或“死记硬背”无用特征。
- ***深度监督：***分割需要精细的局部特征，通过在中层施加监督，可以确保网络没有把那些“极具判别力”的边缘和轮廓信息在深层的池化中丢掉。
- ***动态退出：***当在网络的中间层，推理结果已经高于置信度阈值时，可以提前退出网络。

---

## Polyak平均（Polyak Average）

Polyak 平均法（EMA）是一种“时间维度上的模型集成（Temporal Ensembling）”，它不使用网络在训练结束时最后一刻的权重，而是维护一个历史权重的滑动平均值作为最终的推理模型，从而以零额外训练成本，换取更平滑的收敛和更强大的泛化能力。

---

## 随机裁剪与缩放（Random Resized Crop）

与 VGG 仅仅在固定宽高比下随机缩放短边（256 到 512）然后再截取 224x224 正方形的“温和”策略相比，GoogLeNet 的方法极其“激进”：它允许模型看到小到原图 8% 的极端特写，并且强行扭曲了物体的真实长宽比（3/4 到 4/3）。这种在尺度（Scale）和形变（Distortion）上的极限施压，迫使网络放弃死记硬背物体的全局标准形状，转而学习最核心、最鲁棒的局部判别特征。后面成为经典的训练策略。

---

## 代码实现（Code Implementation）

```python
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

```

## 末语

回到笔记的标题：Go Deeper Cautiously（审慎地走向更深处）。GoogLeNet团队并没有盲目增加网络深度来提升网络性能，他们兼顾计算量的限制设计出来了Inception模块，这一模块的成功得益于其团队的敏锐学术直觉以及艰辛的实验验证，以及没有完全迷信于网络加深带来的性能提升，而忽略时代的硬件条件和计算资源的无谓消耗，他们在审慎地走向更深处。即使是在Inception naive模块遇到计算瓶颈时，他们也是审慎地使用了降维映射。作为学习者的我，也不应该不加审慎地走向更深处，因为这种深一定会带来学习生涯的“过拟合”以及“资源浪费”。如果一味追求深度，那必然会陷入偏执和认知与地位不符的困境，走的慢不可怕，可怕的是前进得毫无意识，认识事物一定不能浮于表面，应该解开事物包装的外表，深究其本质的优劣，所以请：Go Deeper Cautiously。

<div align='right'> --Ljh  2026.2.21 </div>

