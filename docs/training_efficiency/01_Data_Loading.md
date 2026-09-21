# 01 Data Loading

> Covers storage → decode → CPU preprocessing → worker CPU scheduling and memory (previously split
> across three files — storage/decode, CPU preprocessing, CPU scheduling — now merged into one
> category to match MR !2911's `01_data_loading.md` categorization). Content currently only covers
> storage/decode; CPU preprocessing and worker scheduling are still to be added.

## Summary
1. Preallocate per-frame output memory to fold decode into a single copy
2. Multi-camera lazy loading (lazy initialization)
3. Camera decode thread concurrency (`camera_decode_max_concurrency`)

---

## 1. Preallocate per-frame output memory to fold decode into a single copy

**Approach**: Allocate the final `[N, H, W, 3]` contiguous buffer as soon as the first frame is decoded, then `.copy_()` every subsequent decoded frame directly into its slot — no more "collect scattered tensors, then stack at the end."

**Result**: Isolated measurement: loader throughput **+33.9%**, stall burden **−82.7%**, per-sample load time −20~30%.

**Implementation**: video frame decode module (`decode_images_from_frame_indices`)

Before:
```python
collected: dict[int, torch.Tensor] = {}
# inside the decode loop, each decoded frame is stored in a dict
collected[cur_frame_idx] = torch.as_tensor(frame.to_ndarray(format="rgb24"))
...
# after all frames are decoded, pull them out in order and stack into one big tensor
collected = [collected[i] for i in unique_frame_idxs]
tensors = torch.stack(collected)
```

After:
```python
output_slots = {frame_idx: i for i, frame_idx in enumerate(unique_frame_idxs)}
tensors: torch.Tensor | None = None

# inside the decode loop
frame_tensor = torch.as_tensor(frame.to_ndarray(format="rgb24"))
if tensors is None:
    # the first decoded frame tells us the shape; allocate the final-size buffer right here
    tensors = frame_tensor.new_empty((len(unique_frame_idxs), *frame_tensor.shape))
tensors[output_slots[cur_frame_idx]].copy_(frame_tensor)  # copy straight into its final slot
```

No flag, no fallback branch — if an incoming frame's shape doesn't match the preallocated buffer, `copy_()` fails loudly instead of silently doing the wrong thing. A dedicated test, `test_seek_reader_preallocates_without_stack_and_preserves_order`, monkeypatches `torch.stack` to raise if it's ever called, pinning down the fact that `stack` is no longer invoked.

**When this generalizes**: Not limited to video decoding — any time a loop keeps appending tensors to a list/dict and finishes with a single `stack`/`cat` call, the same idea applies: know the final size up front → allocate the buffer once → write directly into place inside the loop, eliminating that last redundant full copy.

---

## 2. Multi-camera lazy loading (lazy initialization)

**Approach**: Each training sample is drawn from a driving clip recorded by 6 cameras, but a given sample only uses about 4.1 cameras on average — a `camera_subsample_weights` config draws a weighted random "camera subset" for every sample ahead of time (e.g. all 6, or just the first 3), rather than always using all 6; this doubles as a form of data augmentation. Before this change, clip admission would **eagerly open all 6 cameras' MP4 files** and build their keyframe indices, regardless of whether a given camera would actually be used. After the change, a `DeferredVideoReader` only records the camera's path (a `video_path` string) and **defers opening the file and constructing the concrete reader** (e.g. `SeekVideoReader`) **until the first time that camera is actually decoded** — a camera that's never selected is never opened.

**Result**: `mean_load_s_per_sample` 1.640 → **1.475** (−10.1%); full-SFT A/B: all-step wall **−5.7%**, severe stalls **−30%**; 90 of 204 camera payload reads eliminated (**−44%**).

**Implementation**: the camera-reader construction logic in the data loading module

Before:
```python
# clip admission eagerly opens all 6 cameras
owned_handles = []
for camera_name in self._camera_names_in_order():
    reader, handle = self._build_camera_reader(camera_name)  # opens the file right away
    clip[camera_name] = reader
    owned_handles.append(handle)
```

After:
```python
class DeferredVideoReader(VideoReader):
    """Path-backed reader that opens its concrete reader on first decode."""

    def __init__(self, video_path, timestamps, reader_cls, thread_count):
        super().__init__(io.BytesIO(), timestamps, thread_count)
        self._video_path = video_path      # only the path is recorded; no file is opened
        self._reader_cls = reader_cls
        self._reader: VideoReader | None = None

    @property
    def initialized(self) -> bool:
        return self._reader is not None

    def _get_reader(self) -> VideoReader:
        """Lazy-init core: construct the real reader only on first use, then reuse it."""
        reader = self._reader
        if reader is None:                     # not initialized yet
            video_handle = open(self._video_path, "rb")           # the file is only opened here
            reader = self._reader_cls(video_handle, self.timestamps, thread_count=self._thread_count)
            self._reader = reader               # cache it so it's never reopened
        return reader                           # whether newly built or cached, this returns it

    # both decode entry points go through _get_reader() to trigger/reuse the lazy init
    def decode_images_from_timestamps(self, requested_timestamps):
        return self._get_reader().decode_images_from_timestamps(requested_timestamps)

    def decode_images_from_frame_indices(self, frame_indices):
        return self._get_reader().decode_images_from_frame_indices(frame_indices)
```

Admission now only puts 6 `DeferredVideoReader` instances (no file handles) into the clip dict; nobody has opened a file yet. The real `SeekVideoReader` is only constructed — and the MP4 only opened — the first time `decode_images_from_frame_indices` is called.

**When this generalizes**: Any time the cost of "initializing / preparing a resource" scales with the **full configured universe** (here, 6 configured cameras) rather than the **subset actually used** (a sample uses 4.1 on average), it's worth deferring initialization until the resource is genuinely needed — lazy loading only pays off when "declared scope > actually-used scope." If the two are always equal (every call really does need the whole resource), there's nothing to gain from deferring it.

---

## 3. Camera decode thread concurrency (`camera_decode_max_concurrency`)

**Approach**:
```
 serial (cap=1):    cam1 ──── cam2 ──── cam3 ──── cam4 ──── cam5 ──── cam6
                    └──────────────── ~6 × t_cam ─────────────────┘

 pair overlap (=2): cam1 ──── cam3 ──── cam5 ────        (worker thread)
                    cam2 ──── cam4 ──── cam6 ────        (helper thread)
                    └──────── ~3 × t_cam ───────┘   latency ÷2, CPU ≈ same

```
The decoder is deliberately single-threaded per worker (`video_decode_thread_count=1`, since 80 worker processes per node can't each afford their own FFmpeg thread pool). A 6-camera, 4-frame sample decodes 30-50 HEVC 1080p frames, and an occasional heavy sample's long-tail latency can stall an entire batch: a worker process delivers samples strictly in FIFO order, so when a heavy sample comes up it must finish decoding before the next sample can be handed off — and distributed training requires every rank to have its batch ready before the step can start, so if even one of the 80 workers is stuck on a heavy sample, every other rank (even ones that finished long ago) has to sit idle waiting. That's exactly how a severe stall gets triggered.

The fix gives each worker process a **reusable helper thread** (`os.register_at_fork` keeps it fork-safe; it's process-private, built once, and reused rather than spun up and torn down every time): cameras are processed in pairs (`camera_decode_max_concurrency=2`) — within each pair, the main thread decodes the first camera while the helper thread decodes the second concurrently, with results landing back in strict input order; the next pair is then processed the same way — the selected cameras aren't all handed to the two threads at once. The concurrency cap is locked to 2 by the config schema (`Literal[1, 2]`); raising it to 4 was measured as a regression (+12% — the SMT hardware-thread budget gets oversubscribed).

**Results**, from three separate test scenarios:
- **S1, paired same-node A/B**: stall gap (extra time per step from stalling) **−0.050 s/step**; severe-stall burden **−33% to −45%**; as a side-effect check — even on a "clean path" that never hits a heavy sample — the cost is only **+1.6%**, which sits inside the ±2% node noise floor, i.e. effectively no added overhead
- **S2, 6 workers (a small worker pool, where a single heavy sample more easily stalls the whole batch)**: stacking "lazy loading + bounded decode concurrency" together drops severe steps from **18.27% → 10.00%** (this is the combined effect of "prefetch=4 + lazy loading + decode concurrency" together, not decode concurrency isolated on its own)
- **⚠️ S2, 10 workers (a large enough worker pool, with a different underlying bottleneck — see file 02)**: severe steps stay flat at **16.5% → 16.5%, no effect**. Reasons: (1) with a big enough worker pool, one stuck worker can be covered by the others, so it's no longer the critical path; (2) S2's severe stalls at this configuration actually stem from the allocator/32MB mmap-threshold issue covered in [02_Dataloader_Preprocessing.md](02_Dataloader_Preprocessing.md) — a different failure mode from "heavy-sample long-tail latency."

**Implementation**: the multi-camera decode scheduling logic

Before:
```python
# cameras decode strictly serially
image_data = {}
for camera_name in cameras_to_load:
    video_cache: VideoReader = clip.cached_data[camera_name]
    images, timestamps = video_cache.decode_images_from_timestamps(t0s)
    image_data[camera_name] = ImageSample(images=images, timestamps=timestamps)
```

After:
```python
class _CameraDecodeHelper:
    """Owns one lazily created, reusable helper thread per DataLoader worker process."""
    def get(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, ...)
            return self._executor

_CAMERA_DECODE_HELPER = _CameraDecodeHelper()
os.register_at_fork(after_in_child=_CAMERA_DECODE_HELPER.reset_after_fork)  # fork-safe

def _decode_camera_pair(first_reader, second_reader, t0s):
    """Hand the second camera to the helper thread; the first stays on the calling thread; results return in input order."""
    second_future = _CAMERA_DECODE_HELPER.get().submit(_decode_camera, second_reader, t0s)
    first_sample = _decode_camera(first_reader, t0s)
    return first_sample, second_future.result()

# process cameras in groups of camera_decode_max_concurrency (1 or 2)
for start in range(0, len(camera_order), group_size):
    camera_names = camera_order[start : start + group_size]
    readers = [clip.cached_data[name] for name in camera_names]
    samples = _decode_camera_pair(readers[0], readers[1], t0s) if len(readers) == 2 \
        else (_decode_camera(readers[0], t0s),)
```

`camera_decode_max_concurrency` is an explicit, validated config switch: it must be 1 or 2, and when set to 2 it strictly requires `video_cache_type="seek"` and `video_decode_thread_count=1` — a misconfiguration fails immediately at config time instead of silently taking effect as broken behavior.
