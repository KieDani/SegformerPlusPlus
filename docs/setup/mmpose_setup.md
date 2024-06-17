# MMSegmentation Setup

**Step 0.** Prerequisites

- Pytorch: 2.0.1 (CUDA 11.8)

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
```

**Step 2.** Install MMPose

```shell
# starting from the project root
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

**Step 3.** [Run the SegFormer++ using MMPose](../run/run_mmpose.md)