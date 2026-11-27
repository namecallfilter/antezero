# Future Pipeline Optimizations (1,000-Video Scale-Up Backlog)

> **Status:** Deferred optimization ideas for full 1,000-video (~1h VOD) batch throughput.  
> **Current Baseline:** `main.py` already runs at **`~165.7 frame/s`** (`12.11s` per 2,000 kept frames / `0..100s` of `bu_full.mp4` at `skip_frames=2`) with `100%` action parity (`38/38` raw, `13/13` dedup) and `95/95` template accuracy (see [`notes/template_matching_optimization_README.md`](template_matching_optimization_README.md)).  
> **When to Use:** Defer these until the end-to-end pipeline (`main.py` → `END_ROUND` temporal model → `post_processing.py`) is proven out and ready for full-scale production runs.

---

## Measured Remaining Bottlenecks (`12.11s` / `165.7 fps` Baseline)

- **Video Extraction:** `2.54 ms/frame` (**`41.9%`**) — serial NVDEC pipe + `cv2.cvtColor(..., cv2.COLOR_YUV2BGR_NV12)` on the single producer thread.
- **Template Matching:** `2.75 ms/frame` (**`45.4%`**) — `2x` coarse-to-fine + full-frame gray/half-res prep.
- **Grayscale Convert:** `0.13 ms/frame` (**`2.1%`**)
- **Other (Overhead):** `0.65 ms/frame` (**`10.7%`**) — Python chunk dispatch, queue handoff, and `ScreenStateTracker` suffix re-evaluation.

---

## Ranked Optimization Backlog

### 1. Run 2–3 Videos Concurrently via Separate Processes *(Highest ROI, Zero Code Risk)*
- **Why:** A single `main.py` process is capped by one serial FFmpeg pipe reader (`41.9%`) and Python thread/GIL coordination (`10.7%`). An `RTX 5090` has dual NVDEC hardware decoder engines and a `Ryzen 7 9800X3D` has 8 physical cores / 16 threads.
- **Change:** Launch **2–3 independent `main.py` worker processes** on disjoint videos from the 1,000-video queue (`--max-workers 4` each).
- **Expected Impact:** **~1.6×–1.9× aggregate video throughput** across the batch without touching CV thresholds.

### 2. Move `YUV2BGR_NV12` Conversion Out of the Single Producer Thread (`frame.py` → `main.py`)
- **Why:** Currently [`frame.py`](../frame.py) runs `cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)` serially inside `_prefetch_generator`'s producer thread before pushing frames onto the queue (`~2.54 ms/frame` caps single-stream throughput at ~393 fps even if template matching were free).
- **Change:** Yield raw `nv12` buffers (`2.97 MB` instead of `6.22 MB` BGR) from `VideoReader` and convert `nv12 → BGR` inside `_process_frame` across all 8 worker threads (or lazily convert BGR only for active ROIs / when the grayscale coarse pass succeeds).

### 3. Benchmark `skip_frames=3` (15 fps Kept vs 20 fps Kept)
- **Why:** On 60 fps footage, `skip_frames=2` evaluates every 3rd frame (`20 fps`). Most Balatro button clicks/holds span 6–12+ frames (`100–200+ ms`).
- **Change:** Test `--skip-frames 3` (`15 fps`, every 4th frame) on annotated VOD validation runs.
- **Expected Impact:** **25% fewer kept frames (~+33% effective video speedup)** if action recall remains 100%.

### 4. ROI-Union Crop Before Full-Frame Color/Resize (`main.py`)
- **Why:** [`main.py`](../main.py) currently converts and downsamples all `1920x1080` pixels (`cv2.cvtColor` + `cv2.resize` to `960x540`) on every kept frame, even though `ScreenStateTracker` whitelists only a few localized button ROIs per screen state.
- **Change:** Precompute the union bounding box of active template ROIs per state and convert/resize only that union region (or per-ROI crops).

### 5. Order `active_templates` by Click Frequency + Tune Chunk Size
- **Why:** [`main.py`](../main.py) breaks out of the template loop immediately on the first clicked button (`break`), and re-runs the remaining suffix of a 64-frame chunk whenever `ScreenStateTracker` transitions mid-chunk.
- **Change:** Order templates in each state by empirical click frequency (`Play Hand`, `Discard`, `Cash Out`, `Next Round` before rare buttons like `Options` / `New Run`), and test reducing chunk size from `64` to `32` to cut suffix re-work on transitions.

### 6. Fast Coarse Frame-Diff Skip on Static Idle Frames
- **Why:** During long streamer pauses between actions, consecutive kept frames are nearly identical across the active button ROIs.
- **Change:** Compute a tiny sum-of-absolute-differences check on the active half-res ROIs when the previous frame had no click; if unchanged within a strict noise tolerance, skip `cv2.matchTemplate` for that frame.
