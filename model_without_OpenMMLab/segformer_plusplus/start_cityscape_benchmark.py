import os
import torch
import numpy as np
import argparse

from .build_model import create_model
from .cityscape_benchmark import cityscape_benchmark

parser = argparse.ArgumentParser(description="Segformer Benchmarking Script")
parser.add_argument('--backbone', type=str, default='b5', choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'], help='Model backbone version')
parser.add_argument('--head', type=str, default='bsm_hq', choices=['bsm_hq', 'bsm_fast', 'n2d_2x2'], help='Model head type')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pth checkpoint file (optional)')
args = parser.parse_args()

model = create_model(args.backbone, args.head, pretrained=True)

if args.checkpoint:
    checkpoint_path = os.path.expanduser(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
else:
    print("No checkpoint provided – using model as initialized.")

cwd = os.getcwd()

image_path = os.path.join(cwd, 'segformer_plusplus', 'cityscape', 'berlin_000543_000019_leftImg8bit.png')
result = cityscape_benchmark(model, image_path)

reference_txt_path = os.path.join(cwd, 'segformer_plusplus', 'cityscapes_prediction_output__reference_b05_bsm_hq_checkpoint.txt')
generated_txt_path = os.path.join(cwd, 'segformer_plusplus', 'cityscapes_prediction_output.txt')

if os.path.exists(reference_txt_path) and os.path.exists(generated_txt_path):
    ref_arr = np.loadtxt(reference_txt_path, dtype=int)
    gen_arr = np.loadtxt(generated_txt_path, dtype=int)

    if ref_arr.shape != gen_arr.shape:
        print(f"Files have different shapes: {ref_arr.shape} vs. {gen_arr.shape}")
    else:
        total_elements = ref_arr.size
        equal_elements = np.sum(ref_arr == gen_arr)
        similarity = equal_elements / total_elements

        threshold = 0.999
        if similarity >= threshold:
            print(f"Outputs are {similarity*100:.4f}% identical (>= {threshold*100}%).")
        else:
            print(f"Outputs differ by {100 - similarity*100:.4f}%.")
else:
    if not os.path.exists(reference_txt_path):
        print(f"Reference file not found: {reference_txt_path}")
    if not os.path.exists(generated_txt_path):
        print(f"Generated output file not found: {generated_txt_path}")
