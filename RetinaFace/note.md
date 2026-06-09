# RetinaFace

本文对RetinaFace框架做一个总结分析，涉及**模型架构分析，搭建，训练策略，前后处理逻辑**。需要强调的是：本次实现框架是基于**计算资源紧张的边缘设备端**部署场景，摒弃了原架构中**DCN（Deformable Convolution Network--可变形卷积）以及 Dense Regression Branch**，得到的轻量级简化框架以达到实时性与精准性之间的平衡。

---

## 模型架构分析搭建

### 基本子模块（Basic Sub-Module）

```python
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
```

对于基本的卷积模块，分为普通卷积和深度可分离卷积两种，后面运用到对应的模块我会特别指出对应类型。值得注意的是，这里的激活函数使用了**LeakyReLU**，根本原因是在计算资源受限的使用场景下，由于输入图像尺寸导致的信息量贫乏或者特殊处理下导致图像信息密度下降，如果继续采用经典的**ReLU**激活，结合之前提到的信息瓶颈的问题，如果采用**ReLU的激进负值截断**，只会加剧问题，导致信息丢失更加严重，模型性能下降是可以预测的。所以**LeakyReLU**采用较为温和的策略：部分负值保留，其余截断。则会缓解信息瓶颈问题，在此情景下是更加好的选择。

接着对深度可分离卷积进行一个阐述：这里的实现是经典的**MobileNetV1Block**，只包含两个阶段：第一个阶段是**depthwise**阶段，第二个阶段是**pointwise**阶段。第一阶段是将输入图像的输入通道分组（经典实现是每组一个通道），每个分组对应一个3x3卷积核进行计算，最后得到与原始输入通道数量一致的临时输出（临时输出的尺寸H和W不保证与原始输入一致，取决于stride参数），再对临时输出进行第二阶段处理，一共有设定的输出通道参数数量的1x1卷积核对临时输出结果进行卷积（此时只涉及特征通道的映射融合，不涉及尺寸变化），最后得到最终输出。与普通卷积对比，有以下优点：

**1.参数量更少，对边缘设备和计算资源受限设备友好**。

**2.更多的非线性激活（两个激活函数 VS 一个激活函数），提升模型的表征拟合能力**。

对于上述介绍的**MobileNetV1Block**来说存在一个致命的问题：低维特征表达能力不足。在窄特征通道下，后续的卷积难以提取有效特征，导致模型能力不强，以及后面的depthwise卷积并不涉及通道之间的信息交互，更加导致了模型能力低下。在**MobileNetV2Block**中，在depthwise之前引入了一个1x1升维，它有效解决了上面的两个问题：

**1.丰富特征维度，使通道间特征信息交互：**这个就有效解决了V1模块不涉及通道之间信息交互的问题，以及窄特征通道，无法有效提取有效特征的问题。

**2.将低维特征升到高维空间：**原先的低维特征，信息密度较高，耦合强，后续难以建模和提取特征。但是1x1升维使得特征到了高维空间，有效信息密度减小，耦合度降低，后续的卷积更容易提取有效特征，建模更加容易，模型性能变强，特征表达能力也变强。
