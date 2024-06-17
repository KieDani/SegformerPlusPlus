## Specify Algorithms and Parameters for Token Merging

In order for token merging to be applied, a list of python dictionary called *tome_cfg* must be added to the configuration file within the context of the SegFormer backbone.

**Example**
Bipartite Soft Matching high quality preset:

```python
model = dict(
    backbone=dict(
        tome_cfg=[
            dict(kv_mode='bsm', kv_r=0.6, kv_sx=2, kv_sy=2), # Stage 1
            dict(kv_mode='bsm', kv_r=0.6, kv_sx=2, kv_sy=2), # Stage 2
            dict(q_mode='bsm', q_r=0.8, q_sx=4, q_sy=4),     # Stage 3
            dict(q_mode='bsm', q_r=0.8, q_sx=4, q_sy=4)      # Stage 4
        ]
    )
)
```

Both a *q_mode* and a *kv_mode* can be specified for each stage.

The following modes are available:


#### Bipartite Soft Matching (SegFormer++<sub>HQ</sub> and SegFormer++<sub>fast</sub>)
```python
dict(q_mode='bsm', q_scale_factor=0.8, q_sx=2, q_sy=2)
```
This works in the same way as *bsm*. However, additionally *q_sx* and *q_sy* have to be specified that are the strides
used to select the tokens in the destination set.

#### 1D Neighbour Merging
```python
dict(q_mode='n1d', q_s=2)
```
The stride/kernel size employed for the average pooling has to be specified using the *q_scale_factor*.

#### 2D Neighbour Merging 
```python
dict(q_mode='n2d', q_s=(2, 2))
```
Here, two strides for both directions have to be specified.

If token merging is applied for keys and values exclusively, the *q_mode* has to be set to *None* and the parameters
have to be specified for keys and values. It is also possible to merge the query tokens as well as the key and value tokens.
Therefore, parameters for both sequences have to be specified:
```python
dict(q_mode='n2d', kv_mode='bsm', q_s=(2, 2), kv_r=0.6, kv_sx=2, kv_sy=2)
```