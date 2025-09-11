# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..utils.registry import Registry


MODEL_WRAPPERS = Registry('model_wrapper')

def is_model_wrapper(model: nn.Module, registry: Registry = MODEL_WRAPPERS):
    """Check if a module is a model wrapper.

    Args:
        model (nn.Module): The model to be checked.
        registry (Registry): The parent registry to search for model wrappers.

    Returns:
        bool: True if the input model is a model wrapper.
    """
    module_wrappers = tuple(registry.module_dict.values())
    if isinstance(model, module_wrappers):
        return True

    if not registry.children:
        return False

    return any(
        is_model_wrapper(model, child) for child in registry.children.values())
