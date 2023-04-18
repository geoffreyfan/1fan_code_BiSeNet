import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ResNet import ResNet



class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()

        self.downpath = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

    def forward(self, x):     # x = (16, 3, 224, 224)
        x = self.downpath(x)
        return x           # x = (16, 3, 28, 28)    变为 原来的 1/8



# class ARM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ARM, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
#         self.bn_atten = nn.BatchNorm2d(out_channels)
#         self.sigmoid_atten = nn.Sigmoid()
#
#         self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
#         self.bn_out = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         identity = x
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         atten = torch.mean(x, dim=(2, 3), keepdim=True)
#         atten = self.conv_atten(atten)
#         atten = self.bn_atten(atten)
#         atten = self.sigmoid_atten(atten)
#
#         x = x * atten
#
#         x = self.conv_out(x)
#         x = self.bn_out(x)
#
#         x += identity
#         x = self.relu(x)
#
#         return x




class ARM(BaseModule):      # self.arm16 = AttentionRefinementModule(context_channels[1], context_channels[0])
    def __init__(self,                 # (256, 128)     (512, 128)
                 in_channels,          # contex_channels 分别赋值到in_channels 和 out_channels
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(ARM, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,          # 1*1 Conv
                bias=False,             # 为什么这里没有使用 stride = 1, padding = 1呢
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            nn.Sigmoid())

    def forward(self, x):   # x = (16, 512, 7, 7)
        x = self.conv_layer(x)  # x = (16, 128, 7, 7)  # ConvModule((conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)(bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)(activate): ReLU(inplace=True))
        x_atten = self.atten_conv_layer(x)  # x = (16, 128, 1, 1)
        x_out = x * x_atten  # x_out = (16, 128, 7, 7) *　(16, 128, 1, 1)　＝　(16, 128, 7, 7)
        return x_out


class ContextPath(nn.Module):
    def __init__(self, out_channels=128):
        super(ContextPath, self).__init__()
        self.resnet = ResNet.resnet50(replace_stride_with_dilation=[1, 2, 4])  # 替换步长用膨胀率
        # 引入膨胀卷积的原因：卷积网络中一般使用池化卷积来增大感受野，但池化只有一些细节会丢失，
        # 后续上采样无法恢复丢失的细节。所以使用了膨胀卷积来替代（膨胀卷积的作用：增大感受野，保持特征的宽和高）
        self.ARM16 = ARM(256, 128)
        self.ARM32 = ARM(512, 128)
        self.conv_head32 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_head16 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                 # 自适应平均池化，指定输出（H，W）
            nn.Conv2d(512, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.up32 = nn.Upsample(scale_factor=2., mode="bilinear")
        self.up16 = nn.Upsample(scale_factor=2., mode="bilinear")

    def forward(self, x):          # x = (16, 3, 224, 224)
        feat16, feat32 = self.resnet(x)     # feat16 = (16, 256, 14, 14)   feat32 = (16, 512, 7, 7)
        avg = self.conv_avg(feat32)         # avg = (16, 256, 1, 1)

        feat32_arm = self.ARM32(feat32) + avg  # feat32_arm = (16, 128, 7, 7) + (16, 256, 1, 1) = (16, 128, 7, 7)
        feat32_up = self.up32(feat32_arm)   # feat32_up = (16, 128, 14, 14)
        feat32_up = self.conv_head32(feat32_up)  # feat32_up = (16, 128, 14, 14)

        feat16_arm = self.ARM16(feat16) + feat32_up  # feat16_arm = (16, 128, 14, 14) + (16, 128, 14, 14) = (16, 128, 14, 14)
        feat16_up = self.up16(feat16_arm)  #  feat16_up =  (16, 128, 28, 28)    # Upsample(scale_factor=2.0, mode=bilinear)
        feat16_up = self.conv_head16(feat16_up)  # feat16_up =  (16, 128, 28, 28)
        return feat16_up, feat32_up






class FFM(nn.Module):
    def __init__(self, channels=128):
        super(FFM, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.skip_forward = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, SP_input, CP_input):

        x = torch.cat([SP_input, CP_input], dim = 1)  # 拼接之后通道数加倍
        x = self.fuse(x)
        identify = self.skip_forward(x)
        out = torch.mul(x, identify) + x
        return out



class BiSeNet(nn.Module):    # num_classes = 3放进来
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.num_classes = num_classes

        self.SpatialPath = SpatialPath()
        self.ContexPath = ContextPath()
        self.FFM = FFM()
        self.cls_seg = nn.Sequential(       # nn.Sequential 和 A.Compose 有异曲同工之妙
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=8., mode="bilinear"),
            nn.Conv2d(128, self.num_classes, 3, padding=1),
        )
        self.up16 = nn.Upsample(scale_factor=2., mode="bilinear")

    def CPdown16(self, x):
        down16 = F.interpolate(x, scale_factor=0.5)
        return down16

    def forward(self, x):  # x = (16, 3, 224, 224)
        SP_out16 = self.SpatialPath(x)  # SP_out16 = (16, 128, 28, 28)

        CP_out16, CP_Out32 = self.ContexPath(x)  # CP_out16 = (16, 128, 28, 28)  CP_Out32=(16, 128, 14, 14)

        FFM_out = self.FFM(SP_out16, CP_out16)
        return self.cls_seg(FFM_out)
