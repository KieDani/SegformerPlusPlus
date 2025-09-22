from setuptools import find_packages, setup

setup(
    name="segformer_plusplus",
    version="0.2",
    author="Marco Kantonis",
    description="Segformer++: Efficient Token-Merging Strategies for High-Resolution Semantic Segmentation",
    install_requires=[
        'torch>=2.0.1',
        'tomesd',
        'omegaconf',
        'pyyaml',
        'numpy',
        'rich',
        'yapf',
        'addict',
        'tqdm',
        'packaging',
        'Pillow',
        'torchvision'
    ],
    packages=find_packages(),
    license='MIT',
    long_description="https://arxiv.org/abs/2405.14467"
)
