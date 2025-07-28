from mmcv.cnn import build_norm_layer, build_activation_layer
from mmengine.model import BaseModule
from mmcv.ops import DeformConv2d, ModulatedDeformConv2dPack
from torch import nn
from mmseg.registry import MODELS

class ConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)

        self.activate = build_activation_layer(act_cfg)
        self.init_weights()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.activate(self.norm(self.conv(x)))
        return x

@MODELS.register_module()
class KyNet(BaseModule):
    def __init__(self,
                 in_channels=3,
                 embed_dims=(96, 192, 384, 768),
                 depths=(2, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.stem = nn.Sequential(
            ConvModule(in_channels, embed_dims[0] // 2, kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')),
            ConvModule(embed_dims[0] // 2, embed_dims[0], kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')),
            ResDeformConvBlock(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1)
        )
        # stage 0
        self.stage1 = nn.Sequential(
            ConvModule(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')),
            ResDeformConvBlock(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1)
        )
        self.stage2 = nn.Sequential(
            ConvModule(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')),
            ResDeformConvBlock(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1)
        )
        self.stage3 = nn.Sequential(
            ConvModule(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1,
                       norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU')),
            ResDeformConvBlock(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1)
        )



    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        return (x1, x2, x3, x4)


class DeformConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)

        self.activate = build_activation_layer(act_cfg)

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.activate(self.norm(self.deform_conv(x, offset)))


class ModulatedDeformConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 ):
        super().__init__()
        self.defrom_conv = ModulatedDeformConv2dPack(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    
    def forward(self, x):
        return self.defrom_conv(x)


class ResDeformConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 ):
        super().__init__()
        self.conv1 = DeformConvBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = DeformConvBlock(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + input
        return x


if __name__ == "__main__":
    model = KyNet()
    import torch
    x = torch.randn(1, 3, 512, 512)
    outs = model(x)
    for out in outs:
        print(out.shape)