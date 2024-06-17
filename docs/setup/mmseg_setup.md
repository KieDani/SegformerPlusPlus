# MMSegmentation Setup

**Step 0.** Prerequisites

- Pytorch: 2.0.1 (CUDA 11.8)

**Step 1.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
```

**Step 2.** Install MMSegmentation

```shell
# starting from the project root
cd mmsegmentation
pip install -v -e .
pip install -r requirements.txt
```

**Step 3.** [Run the SegFormer++ using MMSegmentation](../run/run_mmseg.md)

