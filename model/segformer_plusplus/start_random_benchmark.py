from .build_model import create_model
from .random_benchmark import random_benchmark

model = create_model('b5', 'bsm_hq', pretrained=True)
v = random_benchmark(model)