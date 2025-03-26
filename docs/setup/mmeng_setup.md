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



**Troubleshooting**
There might be installation troubles with openmim, mmengine, and mmcv for new python versions. Thus, Step 1 might not work correctly. 
In this case, try the following alternative for step 1:
```shell
pip install torch torchvision numpy
pip install mmcv-full==1.2.7
pip install mmcv
```
