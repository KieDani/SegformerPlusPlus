# Install the SegFormer++ without MMSegmentation/MMPose

**Step 0.** Prerequisites

- Pytorch: 2.3 (CUDA 12.1) (older versions should also work fine)

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 2.** Install Segformer++

```shell
cd model
pip install .
```

**Step 3.** [Run the SegFormer++](../run/run_mmeng.md)