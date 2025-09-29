# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ...utils import MODELS
from ...utils import resize
from ..base_module import BaseModule
from ...utils.activation import ConvModule


@MODELS.register_module()
class SegformerHead(BaseModule):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
        use_conv_bias_in_convmodules (bool | str): If True, ConvModules will use bias.
            If False, they won't. If 'auto', they follow ConvModule's default.
            This is added for compatibility with models trained with no conv bias
            when followed by BatchNorm, while keeping default local behavior.
    """

    def __init__(self,
                 in_channels=[32, 64, 160, 256],
                 in_index=[0, 1, 2, 3],
                 channels=256,
                 dropout_ratio=0.1,
                 out_channels=19,
                 norm_cfg=None,
                 align_corners=False,
                 interpolate_mode='bilinear',
                 use_conv_bias_in_convmodules: bool | str = 'auto'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.interpolate_mode = interpolate_mode
        self.use_conv_bias_in_convmodules = use_conv_bias_in_convmodules # Speichern des neuen Parameters

        self.act_cfg = dict(type='ReLU')
        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        num_inputs = len(self.in_channels)

        conv_module_bias_setting = use_conv_bias_in_convmodules
        if use_conv_bias_in_convmodules == 'auto':
            pass
        elif isinstance(use_conv_bias_in_convmodules, bool):
            # Wenn True/False explizit übergeben wird, verwenden wir das
            conv_module_bias_setting = use_conv_bias_in_convmodules
        else:
            raise ValueError("use_conv_bias_in_convmodules must be 'auto', True, or False")


        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=conv_module_bias_setting # Verwende den bestimmten Bias-Wert
                ))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=conv_module_bias_setting # Verwende den bestimmten Bias-Wert
        )

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out