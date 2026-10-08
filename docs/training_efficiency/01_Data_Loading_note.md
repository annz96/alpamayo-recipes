这里注明一些代办项。

1. 预先分配每帧数据内存占用，实现一次解码内存搬运
- 如何public github里面收益？
开源训练代码训练时走的确实是physical_ai_av这个外部包,而且目前没有优化。
PAIDataset.__getitem__(src/alpamayo/data/pai.py:97)是SFT训练真正会跑的入口(recipes/alpamayo1_sft、recipes/alpamayo1_5_sft两个recipe的训练配置都在用),它加载图像帧的路径最终会调用到 physical_ai_av 这个包——这是个独立发布在PyPI/GitHub上的NVIDIA开源包(github.com/NVlabs/physical_ai_av),recipes/alpamayo1_5_sft/pyproject.toml里锁定的版本是physical_ai_av>=0.2.0。
因为alpamayo-recipes本身不拥有这份解码代码(它是pip依赖,不在这个repo源码树里),不能直接在alpamayo-recipes里改完就生效。
直接给 github.com/NVlabs/physical_ai_av 提PR,改SeekVideoReader.decode_images_from_frame_indices
区别只是内部版本用的是torch.Tensor(torch.as_tensor+.copy_()),这里是numpy.ndarray(需要用np.empty+直接切片赋值),逻辑完全对应:
```
# 改成跟内部video_reader.py一样的预分配模式(numpy版)
output_slots = {frame_idx: i for i, frame_idx in enumerate(unique_frame_idxs)}
stacked: np.ndarray | None = None
# 解码循环里
frame_arr = frame.to_ndarray(format="rgb24")
if stacked is None:
    stacked = np.empty((len(unique_frame_idxs), *frame_arr.shape), dtype=frame_arr.dtype)
stacked[output_slots[cur_frame_idx]] = frame_arr
```
