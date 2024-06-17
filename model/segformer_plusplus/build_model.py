import os

from mmengine import registry
from mmengine.config import Config
from mmengine.model import BaseModule

from .utils import MODELS, imagenet_weights
from .utils import tome_presets


class SegFormer(BaseModule):
    """
    This class represents a SegFormer model that allows for the application of token merging.

    Attributes:
         backbone (BaseModule): MixVisionTransformer backbone
         decode_head (BaseModule): SegFormer head

    """
    def __init__(self, cfg):
        """
        Initialize the SegFormer model.

        Args:
            cfg (Config): an mmengine Config object, which defines the backbone, head and token merging strategy used.

        """
        super().__init__()
        self.backbone = registry.build_model_from_cfg(cfg.backbone, registry=MODELS)
        self.decode_head = registry.build_model_from_cfg(cfg.decode_head, registry=MODELS)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: output tensor

        """
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


def create_model(
        backbone: str = 'b0',
        tome_strategy: str = None,
        out_channels: int = 19,
        pretrained: bool = False,
):
    """
    Create a SegFormer model using the predefined SegFormer backbones from the MiT series (b0-b5).

    Args:
        backbone (str): backbone name (e.g. 'b0')
        tome_strategy (str | list(dict)): select strategy from presets ('bsm_hq', 'bsm_fast', 'n2d_2x2') or define a
            custom strategy using a list, that contains of dictionaries, in which the strategies for the stage are
            defined
        out_channels (int): number of output channels (e.g. 19 for the cityscapes semantic segmentation task)
        pretrained: use pretrained (imagenet) weights

    Returns:
        BaseModule: SegFormer model

    """
    backbone = backbone.lower()
    assert backbone in [f'b{i}' for i in range(6)]

    wd = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(os.path.join(wd, 'configs', f'segformer_mit_{backbone}.py'))

    cfg.decode_head.out_channels = out_channels

    if tome_strategy is not None:
        if tome_strategy not in list(tome_presets.keys()):
            print("Using custom merging strategy.")
        cfg.backbone.tome_cfg = tome_presets[tome_strategy]

    # load imagenet weights
    if pretrained:
        cfg.backbone.init_cfg = dict(type='Pretrained', checkpoint=imagenet_weights[backbone])

    return SegFormer(cfg)


def create_custom_model(
        model_cfg: Config,
        tome_strategy: list[dict] = None,
):
    """
    Create a SegFormer model with customizable backbone and head.

    Args:
        model_cfg (Config): backbone name (e.g. 'b0')
        tome_strategy (list(dict)): custom token merging strategy

    Returns:
        BaseModule: SegFormer model

    """
    if tome_strategy is not None:
        model_cfg.backbone.tome_cfg = tome_strategy

    return SegFormer(model_cfg)
