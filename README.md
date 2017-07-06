## Caffe Re-implementation of ShuffleNet: An Extremely Efficient CNN for Mobile Devices
**[WARNING]** **Unofficial version** (**STILL UNDER TEST, Any BUG Reports are Welcomed**), may differ from ShuffleNet([arXiv pre-print](https://arxiv.org/abs/1707.01083)) own idea.

## How to understand Channel Shuffle?
*Our narrow view*:
1. **[Pattern 0]** Output Featuremap from `N x C x [H x W]` to `N x [H x W] x C` but still keeps former shape format.`* `<br>
`*`: Based on paper's description *"Tansposing and then flattening it back ..."*, to be honest, we didn't get it, just a **Pretty Naive** re-implementation. In this pattern, next layer won't know the input data structure has been changed to `N x [H x W] x C`, the spatial relationship of next layer input featuremap will be different.
2. **[Pattern 1]** Output Featuremap keeps the previous shape `N x C x [H x W]`, just exchange some channels between different groups.`**` <br>
`**`:This pattern also has a limitation, `CHECK((channels/group)%group == 0)`. Otherwise, We havn't figured out how to make unbalanced channel shuffle.

## How to use?
#### caffe.proto:
```
message LayerParameter {
...
optional ChannelShuffleParameter channel_shuffle_param = 164;
...
}
...
message ChannelShuffleParameter {
  optional int32 shuffle_pattern = 1 [default = 1]; // 0 or 1, so far two patterns.
  optional int32 shuffle_channel_num = 2 [default = 1]; // exchange how many channels each time [pattern 1]
  // group number, the same as previous ConV layer setting [pattern 1]
  optional uint32 group = 3 [default = 1]; 
}
```
#### .prototxt :
```
layer {
  name: "ChnlShf"
  type: "ChannelShuffle"
  bottom: "Gconv"
  top: "ChnlShf"
  channel_shuffle_param {
    shuffle_pattern: 0 # 0 or 1 as explained above
    group: 4 # the same as previous GConV Layer
    shuffle_channel_num: 1 # exchange how many channels each time
  }
}
```
## Citation
```
@article{ShuffleNet,
  Author = {Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun},
  Journal = {arXiv preprint arXiv:1707.01083},
  Title = {ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices},
  Year = {2017}
}
```

