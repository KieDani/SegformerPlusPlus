from mmengine import Registry

MODELS = Registry(
    'models',
    locations=['segformer_plusplus.model.backbone', 'segformer_plusplus.model.head']
)
