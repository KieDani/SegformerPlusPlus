import sys
import os
import torch
import torchvision.transforms as T
from typing import List, Tuple
# Import the necessary functions for custom download
from torch.hub import download_url_to_file
import urllib.parse

# --- WICHTIG: TorchHub Dependencies ---
# These are informational only. Users must install these packages.
dependencies = [
    'tomesd',
    'omegaconf',
    'numpy',
    'rich',
    'yapf',
    'addict',
    'tqdm',
    'packaging',
    'torchvision'
]

# Adds the path to the 'model_without_OpenMMLab' subdirectory to the sys.path list.
model_dir = os.path.join(os.path.dirname(__file__), 'model_without_OpenMMLab')
sys.path.insert(0, model_dir)

# Imports all entry points from the subdirectory.
from segformer_plusplus.build_model import create_model
from segformer_plusplus.random_benchmark import random_benchmark

# Removes the added path again to keep the sys.path list clean.
sys.path.pop(0)


def _get_local_cache_path(url: str, filename: str) -> str:
    """
    Creates the full local path to the checkpoint file in the PyTorch Hub cache.
    """
    # Retrieves the root folder of the PyTorch Hub cache (~/.cache/torch/)
    torch_home = torch.hub._get_torch_home()

    # The default checkpoint directory
    checkpoint_dir = os.path.join(torch_home, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Adds a hash component for the URL to ensure uniqueness,
    # as the URL itself does not contain a unique file name.
    # We use the URL path as part of the hash.
    url_path_hash = urllib.parse.quote_plus(url)

    # The final local file name, including the base name + URL hash.
    local_filename = f"{filename}_{url_path_hash[:10]}.pt"

    return os.path.join(checkpoint_dir, local_filename)


# --- ENTRYPOINT 1: Main Model (ADJUSTED) ---
def segformer_plusplus(
        backbone: str = 'b5',
        tome_strategy: str = 'bsm_hq',
        out_channels: int = 19,
        pretrained: bool = True,
        checkpoint_url: str = None,
        **kwargs
) -> torch.nn.Module:
    """
    Segformer++: Efficient Token-Merging Strategies for High-Resolution Semantic Segmentation.

    Loads a SegFormer++ model with the specified backbone and head architecture.
    Install requirements via:
        pip install tomesd omegaconf numpy rich yapf addict tqdm packaging torchvision

    Args:
        backbone (str): The backbone type. Selectable from: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'].
        tome_strategy (str): The token merging strategy. Selectable from: ['bsm_hq', 'bsm_fast', 'n2d_2x2'].
        out_channels (int): Number of output classes (e.g., 19 for Cityscapes).
        pretrained (bool): Whether to load the ImageNet pre-trained weights.
        checkpoint_url (str, optional): A URL to a specific checkpoint.
                                        **Important:** The download uses torch.hub.download_url_to_file(),
                                        which may be required for non-direct links.

    Returns:
        torch.nn.Module: The instantiated SegFormer++ model.
    """
    model = create_model(
        backbone=backbone,
        tome_strategy=tome_strategy,
        out_channels=out_channels,
        pretrained=pretrained
    )

    if checkpoint_url:
        # Generate a unique file path in the PyTorch cache
        # We use the backbone name as part of the file name
        local_filepath = _get_local_cache_path(
            url=checkpoint_url,
            filename=f"segformer_plusplus_{backbone}"
        )

        print(f"Attempting to load checkpoint from {checkpoint_url}...")

        if not os.path.exists(local_filepath):
            # Use download_url_to_file for the non-direct download
            try:
                print(f"File not in cache. Downloading to {local_filepath}...")

                # This replaces load_state_dict_from_url and saves the file in the cache
                download_url_to_file(
                    checkpoint_url,
                    local_filepath,
                    progress=True
                )
                print("Download successful.")

            except Exception as e:
                print(f"Error downloading checkpoint from {checkpoint_url}. Check the URL or use a direct asset link. Error: {e}")
                # If the download fails, we return an un-loaded model
                return model

        # Load the state dictionary from the downloaded file
        try:
            state_dict = torch.load(local_filepath, map_location='cpu')

            # Perform state_dict cleanup here if necessary,
            # e.g., if the state is nested under a 'model' key
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded successfully.")

        except Exception as e:
            print(f"Error loading state dict from file {local_filepath}: {e}")
            # Again, return the un-loaded/ImageNet pre-trained model
            print("The model was instantiated, but the checkpoint could not be loaded.")

    return model


# --- ENTRYPOINT 2: Data Processing ---
def data_transforms(
        resolution: Tuple[int, int] = (1024, 1024),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
) -> T.Compose:
    """
    Provides the appropriate data transformations for a given dataset.

    This function is an entry point to get the necessary preprocessing steps
    for images based on typical ImageNet values.

    Args:
        resolution (Tuple[int, int]): The desired size for the images (width, height).
                                     Defaults to (1024, 1024).
        mean (List[float]): The mean values for normalization. Defaults to the
                             ImageNet means.
        std (List[float]): The standard deviations for normalization. Defaults to the
                           ImageNet standard deviations.

    Returns:
        torchvision.transforms.Compose: A composition of transforms
                                        that can be applied to input images.

    Example:
        >>> # Load transforms with default parameters
        >>> transform = torch.hub.load('user/repo_name', 'data_transforms')
        >>>
        >>> # Load transforms with resize to custom image resolution and default normalization
        >>> transform_small = torch.hub.load('user/repo_name', 'data_transforms', resolution=(512, 512))
    """
    transform = T.Compose([
        T.Resize(resolution),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform


# --- ENTRYPOINT 3: Random Benchmark ---
def random_benchmark_entrypoint(**kwargs):
    """
    Runs a random benchmark for SegFormer++.
    """
    return random_benchmark(**kwargs)