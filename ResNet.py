import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4       # block.expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):    # x = (16, 16, 56, 56)
        identity = x        # identity = (16, 16, 56, 56)

        out = self.conv1(x)  # out = (16, 4, 56, 56)   self.conv1 = nn.Conv2d(inplanes=16, width=4, kernel_size=1, stride=1, bias=False)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # out = (16, 4, 56, 56)   Conv2d(4, 4, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) #　out = (16, 64, 56, 56)　Conv2d(4, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # identity = (16, 64, 56, 56)    self.downsample() =  nn.Conv2d(16, 64, kernel_size=1, stride=1, bias=False)

        out += identity             # 总结，四个量相同，然后能直接相加,结果，四个量不变     out = (16, 64, 56, 56)
        out = self.relu(out)
        return out



class ResNet(nn.Module):          # nn.Module 是nn\modules\module.py文件里面定义的类
    def __init__(
            self,
            block,
            layers,
            num_classes=33,
            zero_init_residual=False,
            groups=1,
            width_per_group=64 // 4,
            replace_stride_with_dilation=None,
            norm_layer=None):   # replace_stride_with_dilation=None 这里只是默认值, 但之前ContextPath里面设置了[1,2,4]
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d     # 使用BatchNorm 标准化方法后就不用使用drop_out方法了
        self._norm_layer = norm_layer
        self.inplanes = 64 // 4           # 输入是十六个通道。width_per_group也是十六个通道
        self.dilation = 2


        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )


        self.groups = groups      # groups = 1
        self.base_width = width_per_group   # 64 // 4
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)   # self.inplanes = 64 // 4
        self.bn1 = norm_layer(self.inplanes)  # self.inplanes = 64 // 4    对输入的十六个通道做批归一化
        self.relu = nn.ReLU(inplace=True)     # 覆盖原先的数据
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 对邻域内特征点取最大，减小卷积层参数误差造成估计均值的偏移的误差，更多的保留纹理信息。
        # 注意这里的kernel_size跟卷积核不是一个东西。 kernel_size可以看做是一个滑动窗口，这个窗口的大小由自己指定，如果输入是单个值，例如
        # 3 ，那么窗口的大小就是3 × 3，还可以输入元组，例如(3, 2) ，那么窗口大小就是3 × 2，最大池化的方法就是取这个窗口覆盖元素中的最大值。
        self.layer1 = self._make_layer(block, 64 // 4, layers[0])   # block = Bottleneck
        self.layer2 = self._make_layer(block, 128 // 4, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        # 由于要做Conv2d(), 让输出通道为 planes * block.expansion 就是 * 4，所以这里的输出通道最后还是128
        self.layer3 = self._make_layer(block, 256 // 4, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512 // 4, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # 图片的大小变为 (1, 1)   在参数H、W 比 输入图像的长宽 小 的情况下效果更好
        # 构造模型的时候，AdaptiveAvgPool2d()的位置一般在卷积层和全连接层的交汇处，以便确定输出到Linear层的大小。
        self.fc = nn.Linear(512 * block.expansion, num_classes)   # block.expansion = 4
        # 这里 in_features为前一层输出通道数  * block.expansion


        for m in self.modules():             # self.modules() 是继承nn.Module自带的
            if isinstance(m, nn.Conv2d):     # isinstance() 函数来判断一个对象是否是一个已知的类型         这里是针对于卷积层来用这种方式初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")       # nonlinearity 非线性
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:    # residual 就是残差模块     主分支的输出的 H W C 和 侧分支输出的 H W C 必须相同
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        # 在相同维度进行相加 和 在相同的维度进行拼接是两个完全不同的概念

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,          # 赋值进来为 2
            dilate=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation          # 这个类里面def __init__里面的变量在这里也可以引用

# 有dilate 和 stride的赋值  或者 self.inplanes != planes * block.expansion 才看下面语句
###########################################################################################################
        if dilate:      # 如果dilate不为空
            self.dilation *= stride                # 每执行一次 _make_layer， self.dilation的值就要乘以2
            stride = stride

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
###########################################################################################################

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion          # self.inplanes 发生改变
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x):   # x = (16, 3, 224, 224)
        out = []
        x = self.conv1(x)         # x = (16, 16, 112, 112)  self.conv1 = nn.Conv2d(3, self.inplanes=16, kernel_size=7, stride=2, padding=3, bias=False)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # x = (16, 16, 56, 56)     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        x = self.layer1(x)        # x = (16, 64, 56, 56)
        x = self.layer2(x)        # x = (16, 128, 28, 28)
        x = self.layer3(x)        # x = (16, 256, 14, 14)
        out.append(x)             # 对应feat16浅一些
        x = self.layer4(x)        # x = (16, 512, 7, 7)
        out.append(x)             # 对应feat32更深
        return out

    def forward(self, x):
        return self._forward_impl(x)

    def _resnet(block, layers, pretrained_path=None, **kwargs, ):

        model = ResNet(block, layers, **kwargs)

        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path), strict=False)     # load_state_dict()是nn\modules\module.py文件里面定义的函数
        return model

    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3], pretrained_path, **kwargs)

    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3], pretrained_path, **kwargs)
