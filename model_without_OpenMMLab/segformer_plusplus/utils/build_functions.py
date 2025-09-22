from typing import Any, Optional, Union
import inspect
import torch.nn as nn
import torch

from ..configs.config.config import Config, ConfigDict
from .registry import Registry
from ..utils.manager import ManagerMixin


TORCH_VERSION = torch.__version__

def build_from_cfg(
        cfg: Union[dict, ConfigDict, Config],
        registry: Registry,
        default_args: Optional[Union[dict, ConfigDict, Config]] = None) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.

    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.

    Args:
        cfg (dict or ConfigDict or Config): Config dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. Defaults to None.

    Returns:
        object: The constructed object.
    """

    if not isinstance(cfg, (dict, ConfigDict, Config)):
        raise TypeError(
            f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}')

    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a mmengine.Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args,
                       (dict, ConfigDict, Config)) or default_args is None):
        raise TypeError(
            'default_args should be a dict, ConfigDict, Config or None, '
            f'but got {type(default_args)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.scope}::{registry.name} registry. '  # noqa: E501
                    f'Please check whether the value of `{obj_type}` is '
                )
        # this will include classes, functions, partial functions and more
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        # If `obj_cls` inherits from `ManagerMixin`, it should be
        # instantiated by `ManagerMixin.get_instance` to ensure that it
        # can be accessed globally.
        if inspect.isclass(obj_cls) and \
                issubclass(obj_cls, ManagerMixin):  # type: ignore
            obj = obj_cls.get_instance(**args)  # type: ignore
        else:
            obj = obj_cls(**args)  # type: ignore
        return obj


def build_model_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
) -> 'nn.Module':
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """
    from ..model.base_module import Sequential
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(_cfg, registry, default_args) for _cfg in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)
    

class SyncBatchNorm(torch.nn.SyncBatchNorm):  # type: ignore

    def _check_input_dim(self, input):
        if TORCH_VERSION == 'parrots':
            if input.dim() < 2:
                raise ValueError(
                    f'expected at least 2D input (got {input.dim()}D input)')
        else:
            super()._check_input_dim(input)