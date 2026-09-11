## Summary
1. 预先分配每帧数据内存占用，实现一次解码内存搬运
2. 相机懒加载
3. 受限解码并发 `camera_decode_max_concurrency`(Part 3,ch3)

---

## 1. 预先分配每帧数据内存占用，实现一次解码内存搬运

**思路**:第一帧解码出来时就直接申请好最终的`[N, H, W, 3]`连续buffer,后续每帧解码完直接`.copy_()`写进它在buffer里该在的位置,不再需要"先分散存、最后再stack"这两步。

**收益**:独立测:loader吞吐 **+33.9%**,stall负担 **−82.7%**,单样本加载 −20~30%。

**实现**:
- public github [TBD]
- internal gitlab [`src/alpamayo/data/video_reader.py`](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/blob/0d53ef73e8/src/alpamayo/data/video_reader.py) `SeekVideoReader.decode_images_from_frame_indices`

改动前:
```python
collected: dict[int, torch.Tensor] = {}
# 解码循环里,每解出一帧就存进字典
collected[cur_frame_idx] = torch.as_tensor(frame.to_ndarray(format="rgb24"))
...
# 全部解码完后,按顺序取出、再 stack 成一个大tensor
collected = [collected[i] for i in unique_frame_idxs]
tensors = torch.stack(collected)
```

改动后:
```python
output_slots = {frame_idx: i for i, frame_idx in enumerate(unique_frame_idxs)}
tensors: torch.Tensor | None = None

# 解码循环里
frame_tensor = torch.as_tensor(frame.to_ndarray(format="rgb24"))
if tensors is None:
    # 第一帧解码出来时,才知道frame的shape,这时候一次性申请好最终大小的buffer
    tensors = frame_tensor.new_empty((len(unique_frame_idxs), *frame_tensor.shape))
tensors[output_slots[cur_frame_idx]].copy_(frame_tensor)  # 直接原地拷进最终位置
```

没有开关、没有兜底分支——如果传进来的frame尺寸跟预分配的buffer不匹配,`copy_()`会直接报错崩溃,而不是悄悄地做错事。配套测试 `test_seek_reader_preallocates_without_stack_and_preserves_order` 直接给`torch.stack`打了monkeypatch,一旦被调用就抛异常,把"不再调用stack"这个事实钉死。

---

## 2. 相机懒加载(ch2,已合并 [`43b6c87bbb`](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/commit/43b6c87bbb))

**思路**:每个训练sample来自一个6相机拍摄的driving clip,但一个sample平均只用到约4.1个相机。改动前,clip admission阶段会**提前打开全部6个相机的MP4文件**并建好keyframe索引,不管这个相机后面用不用得到。改动后引入 `DeferredVideoReader`,只记录相机路径(`video_path`字符串),把真正打开文件、构造具体reader(`SeekVideoReader`等)这一步**推迟到第一次真正解码这个相机时才做**——没被选中的相机永远不会被打开。

**收益**:`mean_load_s_per_sample` 1.640→**1.475**(−10.1%);全SFT A/B:全步wall **−5.7%**,severe stalls **−30%**;204次相机payload读取消除90次(**−44%**)。

**实现**:[`src/alpamayo/data/video_reader.py`](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/blob/43b6c87bbb/src/alpamayo/data/video_reader.py) 新增 `DeferredVideoReader`;[`src/alpamayo/data/dataset/formats/clipgt.py`](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/blob/43b6c87bbb/src/alpamayo/data/dataset/formats/clipgt.py) 改用它构造reader

改动前:
```python
# clip admission阶段,eagerly打开全部6个相机
owned_handles = []
for camera_name in self._camera_names_in_order():
    reader, handle = self._build_camera_reader(camera_name)  # 立刻 open(video_path, "rb")
    clip[camera_name] = reader
    owned_handles.append(handle)
```

改动后:
```python
class DeferredVideoReader(VideoReader):
    """Path-backed reader that opens its concrete reader on first decode."""

    def __init__(self, video_path, timestamps, reader_cls, thread_count):
        super().__init__(io.BytesIO(), timestamps, thread_count)
        self._video_path = video_path      # 只记路径,不open文件
        self._reader_cls = reader_cls
        self._reader: VideoReader | None = None

    @property
    def initialized(self) -> bool:
        return self._reader is not None

    def _get_reader(self) -> VideoReader:
        """构造具体reader(open文件+建keyframe索引),只在第一次解码时才触发。"""
        ...  # 首次调用 decode_images_from_frame_indices 时才真正 open(self._video_path, "rb")
```

admission阶段现在只是把6个 `DeferredVideoReader`(不含文件句柄)塞进clip字典,谁都没有真正打开文件;之后 `decode_images_from_frame_indices` 第一次被调用时才 lazy 构造出真正的 `SeekVideoReader` 并打开MP4。

---

## 3. 受限解码并发 `camera_decode_max_concurrency`(ch3,已合并 [`!2859`](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/merge_requests/2859))

**思路**:worker解码器严格单线程串行(`video_decode_thread_count=1`,因为80个worker进程/node不能各配一个FFmpeg线程池),六相机、4帧的sample要解30-50帧HEVC 1080p,重样本的长尾延迟会拖垮整批(FIFO交付,一个worker慢了全rank等)。改动引入一个**每个worker进程一个的复用helper线程**(`os.register_at_fork`保证fork安全):相机k在worker自己的线程上解码的同时,相机k+1在helper线程上并发解码,结果严格按输入顺序落地,并发上限用schema锁死为2(`Literal[1, 2]`)。

**收益**:6-worker场景:severe steps 18.27%→**10.00%**;S1同节点对照:stall gap −0.050s/step,severe负担−33%/−45%,干净路径代价仅+1.6%;⚠️在S2/10-worker场景下**测出无效**(该场景瓶颈是allocator,见[02_Dataloader_Preprocessing_cn.md](02_Dataloader_Preprocessing_cn.md))。

**实现**:[`src/alpamayo/data/dataset/formats/clipgt.py`](https://gitlab-master.nvidia.com/alpamayo/alpamayo/-/blob/21e85c2d9d/src/alpamayo/data/dataset/formats/clipgt.py)

改动前:
```python
# 相机严格串行解码
image_data = {}
for camera_name in cameras_to_load:
    video_cache: VideoReader = clip.cached_data[camera_name]
    images, timestamps = video_cache.decode_images_from_timestamps(t0s)
    image_data[camera_name] = ImageSample(images=images, timestamps=timestamps)
```

改动后:
```python
class _CameraDecodeHelper:
    """每个DataLoader worker进程懒加载出一个复用的单线程helper executor。"""
    def get(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, ...)
            return self._executor

_CAMERA_DECODE_HELPER = _CameraDecodeHelper()
os.register_at_fork(after_in_child=_CAMERA_DECODE_HELPER.reset_after_fork)  # fork-safe

def _decode_camera_pair(first_reader, second_reader, t0s):
    """第二个相机丢给helper线程解码,第一个相机留在调用者线程,结果按输入顺序返回。"""
    second_future = _CAMERA_DECODE_HELPER.get().submit(_decode_camera, second_reader, t0s)
    first_sample = _decode_camera(first_reader, t0s)
    return first_sample, second_future.result()

# 按 camera_decode_max_concurrency(1或2)分组处理
for start in range(0, len(camera_order), group_size):
    camera_names = camera_order[start : start + group_size]
    readers = [clip.cached_data[name] for name in camera_names]
    samples = _decode_camera_pair(readers[0], readers[1], t0s) if len(readers) == 2 \
        else (_decode_camera(readers[0], t0s),)
```

`camera_decode_max_concurrency`是走pydantic配置校验的显式开关:必须是1或2,且`=2`时强制要求`video_cache_type="seek"`且`video_decode_thread_count=1`,配置不满足直接报错,不会静默生效成错误行为。
