# Template Matching & Full-Pipeline Performance — Optimization Suggestions & Estimates

> **Status:** `main.py`, `frame.py`, `template.py`, `matcher.py`, `eval_template.py`, and `config.py` optimizations are **implemented and benchmarked** (see Section 6); `post_processing.py`, HUD OCR, and `END_ROUND` sections remain planning estimates.  
> **Scope:** Scaling `main.py` (button template scraping), `post_processing.py` (batched CV + HUD OCR + trajectory reconstruction), and large-scale (~1,000-video) batch processing.

---

## 1. Observed Baseline Benchmark (`main.py`)

Measured on laptop hardware (`AMD Ryzen 7 8840HS` 8c/16t, `AMD Radeon 780M` iGPU, ~13.8 GB usable RAM) over 2,000 processed frames (`skip_frames=2`, `1920x1080`):

```text
Processing video: 100%|█████████████████████████████████████████████████████| 2000/2000 [01:08<00:00, 29.11frame/s]
Finished in 68.78 seconds; 0.0344 per frame.

--- Benchmark Breakdown (Effective Wall Time per frame) ---
Video Extraction:   0.01738s (50.5%)
Grayscale Convert:  0.00023s ( 0.7%)
Template Matching:  0.01427s (41.5%)
Other (Overhead):   0.00252s ( 7.3%)
-----------------------------------------------------------
```

### Key Takeaways from the Breakdown
1. **Video Extraction (`17.38 ms`, 50.5%) is the #1 bottleneck**: `VideoReader` (`frame.py:29-119`) wraps a single-threaded `cv2.VideoCapture` decoding 1080p frames on CPU inside `prefetch_generator` (`main.py:121-166`).
2. **Template Matching (`14.27 ms`, 41.5%) is the #2 bottleneck**: `ThreadPoolExecutor` (`main.py:222`) uses `min(32, os.cpu_count() + 4) = 20` threads on 16 logical cores. While OpenCV C++ calls (`cv2.matchTemplate`) release the GIL, pure-Python loop/object overhead in `eval_template.py` and `matcher.py` still contends on the GIL.
3. **Grayscale Conversion (`0.23 ms`, 0.7%) is negligible**: Cropping ROIs before `cv2.cvtColor` would not meaningfully reduce wall time.

---

## 2. Why GPU `cv2.matchTemplate` Is Not Recommended Here

For small UI button ROIs (`px_roi.expanded_bounds` in `eval_template.py:61-63`), uploading frame data/crops to VRAM and launching small CUDA `matchTemplate` kernels is slower than CPU because PCIe transfer and kernel-launch latency exceed compute time. On CPU, those small ROI crops stay hot in L1/L2/L3 cache.

**Optimal hardware split on a desktop (`Ryzen 7 9800X3D + RTX 5090 + 64 GB DDR5-5600`)**:
- **RTX 5090**: Hardware video decoding (**NVDEC**) + batched PyTorch CV/OCR in `post_processing.py`.
- **Ryzen 7 9800X3D**: Cache-resident CPU template matching (96 MB 3D V-Cache + Zen 5 IPC).

---

## 3. Implemented `main.py` Optimizations & Speedups

For a ~30-minute baseline video (~52,400 processed frames at `29.1 fps`, or `36,000` kept frames on 60 fps `bu_full.mp4` at `skip_frames=2`):

### A. Running `main.py` Unoptimized on `9800X3D + RTX 5090 + 64 GB`
- **Measured (`0..100s` scaled to 30 min)**: **~10.6 min (~56.6–59.8 fps on `bu_full.mp4`)**
- *Why*: Without `use_hwaccel` or `coarse_to_fine`, single-threaded CPU `cv2.VideoCapture` decode (`12.10 ms/frame`) and full-resolution masked `cv2.matchTemplate` (`4.46 ms/frame`) dominate wall time.

### B. Implemented Code Changes for `main.py`
1. **Offloaded Video Decoding to RTX 5090 NVDEC ([`frame.py`](../frame.py), `12.10 ms` → `2.54 ms` effective)**:
   - Replaced CPU `cv2.VideoCapture` in `VideoReader` with `ffmpeg -hwaccel cuda` (passing `-c:v av1` for AV1 streams), discarding skipped frames (`skip_frames=2`) in-filter before piping `-pix_fmt nv12` and converting via `cv2.cvtColor(..., cv2.COLOR_YUV2BGR_NV12)` (preserving full `[0..255]` range for shadow validation).
   - Automatically falls back to `cv2.VideoCapture` for short clips (`< 16` kept frames), late seeks (`start_frame > 3600`), or first-frame FFmpeg failure.
2. **Precomputed ROIs, 2× Coarse-to-Fine Matching & Physical-Core Threading ([`template.py`](../template.py), [`matcher.py`](../matcher.py), [`eval_template.py`](../eval_template.py), [`config.py`](../config.py))**:
   - Pre-computes static ROI bounds, even-aligned coordinates, and contiguous template/mask slices once per resolution via `CompiledTemplateSpec` (`Template.get_compiled_specs`).
   - Added `find_element_coarse_to_fine` to pre-screen at 2× downscale and skip full-res masked `cv2.matchTemplate` when `c_max < threshold - coarse_margin` (`0.04`), cutting template matching from `4.46 ms` to `1.95–2.75 ms` with `95/95` (`100%`) accuracy.
   - Kept `ThreadPoolExecutor` capped at physical cores (`max_workers=8` on 16-thread CPUs) rather than switching to `ProcessPoolExecutor`, which was >2.1× slower on Windows due to IPC serialization overhead.
   - **Measured total**: **`~6.1 ms/frame` (`165.7 fps`, `2.92×` faster)** → **~3.6 minutes** per 30-min `bu_full.mp4`-equivalent video.
3. **Evaluated Optional 720p Scaling (`Rejected as Default`)**:
   - Benchmarking an NVDEC `scale_cuda=1280:720` pipeline reached **`3.25 ms/frame` (`307.8 fps`)**, but dropped `testing.test_templates` accuracy from `95/95` (`100%`) to `89/95` (`93.7%`) due to thin button shadow blur, so **1080p** remains the default.

---

## 4. `post_processing.py` & HUD OCR Comparative Estimates

Unlike `main.py` (~52,400 frames), `build_run_trajectory` (`post_processing.py:773-990`) deduplicates clicks and only extracts `needed_frames` (`post_processing.py:812-821`)—typically **~250–500 unique frames** for a full 8-Ante run (~100–200× fewer frames).

| Pipeline Component | Current Laptop (`8840HS` CPU) | `9800X3D + RTX 5090 + 64 GB` |
| --- | --- | --- |
| `extract_video_frames` (~300–500 sparse seeks) | ~3–10 s | ~2–5 s |
| `_run_batched_cv` (YOLO + MobileNetV3 in `classifiers/inference.py`) | ~20–60 s | ~0.5–2 s |
| `on_action` / `GameState` simulation loop | < 0.5 s | < 0.2 s |
| **HUD OCR** (`money`, `hands`, `discards`, `round`, `ante`, `reroll_cost`, `current_chips`) | ~1–4 s *(custom tiny CNN/glyph)* or ~15–60 s *(RapidOCR/PaddleOCR)* | **< 0.5–1 s** *(custom tiny CNN)* or **~2–6 s** *(GPU RapidOCR/PaddleOCR)* |
| **Future Temporal `END_ROUND` Model** (~24 rounds @ 15–20 fps) | ~10–25 s | ~1–3 s |
| **Total `post_processing.py`** | **~35–100 s** | **~5–15 s** |

### HUD OCR Design Note
- Because HUD counters use Balatro's fixed pixel font (`m6x11plus.ttf`) at fixed sidebar coordinates on the **same `needed_frames` already decoded**, OCR adds zero extra video-seeking cost.
- Prefer a **batched tiny PyTorch CNN/CRNN or digit glyph matcher** (or GPU-batched `RapidOCR`/`PaddleOCR`) on cropped HUD boxes over `pytesseract` subprocess calls (which add ~1–3 min of process overhead and struggle with colored drop-shadow pixel fonts).
- Only evaluate specific HUD crops on relevant actions (e.g., `current_chips` around `PLAY_HAND`/`END_ROUND`, `reroll_cost` in `SHOP`).

---

## 5. 1,000-Video Batch Processing & Rolling Download Queue Estimates

| Configuration | `main.py` / video | `post_processing.py` + OCR + `END_ROUND` / video | Estimated Total (1,000 videos) |
| --- | --- | --- | --- |
| Current Laptop (`8840HS`) as-is | ~30 min | ~1.5 min | ~525 hours (~22 days) |
| `9800X3D + RTX 5090` unoptimized baseline (1080p) | ~10.6 min *(measured on `bu_full.mp4`)* | ~10–15 s | ~180 hours (~7.5 days) |
| `9800X3D + RTX 5090` **Implemented Default** (`NVDEC` + `Coarse-to-Fine` + `ThreadPool(8)`, 1080p) | **~3.6 min** *(measured `165.7 fps`)* | ~10–15 s | **~63–65 hours (~2.6–2.7 days)** |
| *(Rejected Alternative)* 720p `scale_cuda` (`89/95` accuracy) | ~1.95 min *(measured `307.8 fps`)* | ~10–15 s | ~35–37 hours (~1.5 days) |

### Rolling 50-Video Download Queue Considerations
Keeping ~50 videos buffered on disk (~50–150 GB) and deleting each after processing would have **negligible (<1–2%) compute impact** on an NVMe SSD, provided:
1. **Use an NVMe SSD** (3,000+ MB/s vs. ~5–15 MB/s sustained download + decode), not a spinning HDD or external USB drive.
2. **Download video-only streams** (e.g., `yt-dlp` H.264/HEVC/AV1 video-only): avoids `ffmpeg` audio+video muxing and prevents accidental CPU re-encoding.
3. **Leave SMT/reader headroom** (`max_workers=8` physical cores on 16 logical cores, which is now the default) or run the downloader at low process priority so NVDEC feeding and downloads never starve workers.
4. **Atomic handoff & rate-limit pacing**: Download to a `.part`/staging path and rename into the ready queue on completion; pace `yt-dlp` requests once the 50-video buffer is full to avoid YouTube/Twitch 403/429 throttling over a multi-day run.

---

## 6. Implemented Optimizations & Measured Benchmarks (`Ryzen 7 9800X3D + RTX 5090`)

All optimizations below are implemented behind default flags in [`config.py`](../config.py) (`use_hwaccel=True`, `coarse_to_fine=True`, `coarse_margin=0.04`, `max_workers=8` via `_default_max_workers()`) and verified on [`assets/sample_inputs/videos/bu_full.mp4`](../assets/sample_inputs/videos/bu_full.mp4) (`1920x1080`, `AV1 Main`, `yuv420p(tv, bt709)`, `60 fps`, `skip_frames=2`).

### 6.1 Step-by-Step Ablation on `bu_full.mp4` (`skip_frames=2`, 1080p)

#### Short Clip (`0..30s`, 600 kept frames / 1,800 raw frames)

| Configuration | Wall Time | Throughput | Extract / Frame | Gray / Frame | Template / Frame | Other / Frame | Actions (Raw / Dedup) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** (`cv2.VideoCapture` CPU + Full-Res + `workers=14`) | `8.95s` | `67.42 frame/s` (`0.0149s/f`) | `10.29 ms` (`69.0%`) | `0.11 ms` (`0.8%`) | `3.64 ms` (`24.4%`) | `0.88 ms` (`5.9%`) | `18` / `4` |
| **+ Compiled Specs & 2x Coarse-to-Fine** (CPU decode, `workers=14`) | `8.06s` | `74.95 frame/s` (`0.0134s/f`) | `10.64 ms` (`79.3%`) | `0.14 ms` (`1.0%`) | `2.02 ms` (`15.0%`) | `0.62 ms` (`4.7%`) | `18` / `4` |
| **+ FFmpeg NVDEC (`-c:v av1`, `nv12`) Only** (`coarse_to_fine=False`, `workers=14`) | `5.18s` | `117.09 frame/s` (`0.0086s/f`) | `3.82 ms` | `0.12 ms` | `3.70 ms` | `0.99 ms` | `18` / `4` |
| **NVDEC + Coarse-to-Fine (`workers=14`)** | `4.17s` | `145.51 frame/s` (`0.0070s/f`) | `3.91 ms` | `0.11 ms` | `2.11 ms` | `0.82 ms` | `18` / `4` |
| **Final Default (`workers=8`, physical cores)** | **`3.83s`** | **`158.92 frame/s` (`0.0064s/f`)** | **`3.22 ms` (`50.4%`)** | **`0.13 ms` (`2.1%`)** | **`2.37 ms` (`37.2%`)** | **`0.66 ms` (`10.3%`)** | **`18` / `4`** |

#### Medium Clip (`0..100s`, 2,000 kept frames / 6,000 raw frames)

| Configuration | Wall Time | Speedup vs Baseline | Throughput | Extract / Frame | Gray / Frame | Template / Frame | Other / Frame | Actions (Raw / Dedup) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** (`cv2.VideoCapture` CPU + Full-Res + `workers=14`) | `35.38s` *(warm: `33.51s`)* | `1.00x` | `56.62 frame/s` *(warm: `59.76`)* | `12.10 ms` (`68.4%`) | `0.12 ms` (`0.7%`) | `4.46 ms` (`25.2%`) | `1.02 ms` (`5.7%`) | `38` / `13` |
| **+ Compiled Specs & 2x Coarse-to-Fine** (CPU decode, `workers=14`) | `30.08s` | `1.18x` | `66.58 frame/s` (`0.0150s/f`) | `12.41 ms` (`82.5%`) | `0.14 ms` (`0.9%`) | `1.95 ms` (`13.0%`) | `0.54 ms` (`3.6%`) | `38` / `13` |
| **+ FFmpeg NVDEC (`-c:v av1`, `nv12`) Only** (`coarse_to_fine=False`, `workers=14`) | `16.10s` | `2.20x` | `124.60 frame/s` (`0.0080s/f`) | `2.15 ms` | `0.11 ms` | `4.67 ms` | `1.13 ms` | `38` / `13` |
| **NVDEC + Coarse-to-Fine (`workers=14`)** | `13.14s` | `2.69x` | `152.78 frame/s` (`0.0066s/f`) | `3.43 ms` (`52.2%`) | `0.12 ms` (`1.9%`) | `2.26 ms` (`34.5%`) | `0.75 ms` (`11.5%`) | `38` / `13` |
| **Final Default (`workers=8`, physical cores)** | **`12.11s`** | **`2.92x`** | **`165.71 frame/s` (`0.0061s/f`)** | **`2.54 ms` (`41.9%`)** | **`0.13 ms` (`2.1%`)** | **`2.75 ms` (`45.4%`)** | **`0.65 ms` (`10.7%`)** | **`38` / `13`** |

Action parity across every `0..100s` run remained **100% identical** (`38` raw action frames -> `13` deduplicated clicks: `[(888, 21), (1146, 12), (1563, 16), (1635, 3), (2172, 16), (2235, 3), (2478, 7), (2850, 7), (3294, 2), (3570, 5), (3855, 11), (3999, 11), (5157, 11)]`).

---

### 6.2 Architecture Notes, Tradeoffs & Discovered Pitfalls

1. **Precomputed ROIs (`CompiledTemplateSpec`) + 2x Coarse-to-Fine (`template.py`, `matcher.py`, `eval_template.py`)**
   - Precomputes contiguous BGR/Gray/Mask arrays, even-aligned `x1, y1`, half-resolution templates/masks, and static bounds once per `(img_w, img_h, expand_roi_factor)` via `Template.get_compiled_specs(...)`.
   - `main._process_frame` computes `img_gray_half` once per frame (`cv2.INTER_NEAREST`) and runs `find_element_coarse_to_fine(...)`, skipping full-resolution `cv2.matchTemplate` (especially expensive masked correlation on wide templates) whenever coarse `c_max < threshold - 0.04`.
   - Cut template matching wall time by **2.29x** (`4.46 ms/f -> 1.95 ms/f` on CPU decode) while scoring **`95/95` (`100%`)** on `testing.test_templates` (zero mismatches vs full-res). Automatically bypasses coarse pre-screening when `debug_plot` or unadjusted `setup_mode` requires full score maps.

2. **FFmpeg NVDEC Decoding Quirks (`frame.py`)**
   - **Explicit `-c:v av1` required**: Passing only `ffmpeg -hwaccel cuda -i bu_full.mp4` silently falls back to CPU `libdav1d` decode (`~37.5 fps`). Probing the stream with `ffprobe` and injecting `-c:v av1` for AV1 streams unlocks hardware NVDEC.
   - **NV12 pipe vs `bgr24`**: Piping `-pix_fmt bgr24` forces single-threaded FFmpeg `swscale` (`~38 fps` bottleneck). Piping raw `-pix_fmt nv12` (1.5 bytes/px) and converting via `cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)` unlocks full speed (`2.20x–2.92x` end-to-end).
   - **Do NOT use raw `nv12[:h, :]` Y-plane as grayscale**: Raw NV12 luma is TV-range `[16..235]`, whereas `cv2.COLOR_YUV2BGR_NV12 -> BGR2GRAY` expands to full-range `[0..255]`. Skipping the full-range conversion compresses Sobel edge contrast and breaks `shadow_threshold=52.0` shadow press validation.
   - `VideoReader` automatically falls back to `cv2.VideoCapture` for tiny clips (`< 16` expected kept frames), late seeks (`start_frame > 3600` to avoid linear frame-skipping from 0), or if FFmpeg fails to emit its first frame.

3. **`ThreadPoolExecutor` Worker Count vs `ProcessPoolExecutor` on Windows**
   - **`0..30s` `ThreadPoolExecutor` worker sweep**:
     - `workers=8`: `3.83s–3.84s` (`158.3–158.9 fps`, Extract `3.20–3.22 ms`, TM `2.37–2.40 ms`)
     - `workers=12`: `3.97s` (`152.9 fps`)
     - `workers=14`: `4.17s–4.20s` (`145.0–145.5 fps`)
     - `workers=16`: `4.11s` (`148.0 fps`)
     - `workers=20`: `4.20s` (`144.5 fps`)
   - **`0..100s` confirmation (`2,000` frames)**: `workers=8` finished in **`12.11s–12.22s` (`164.7–165.7 fps`)** vs `workers=14` at `13.14s–13.24s` (`151.8–152.8 fps`). Capping `_default_max_workers()` at physical cores (`8` on a 16-thread CPU) reduces SMT/GIL contention with the NVDEC pipe reader thread (`Video Extraction` dropped from `3.46 ms` to `2.54–2.61 ms`).
   - **`ProcessPoolExecutor(8)` vs `ThreadPoolExecutor(8)` on Windows (`0..30s`)**: `ProcessPoolExecutor(8)` took **`8.13s` (`73.8 fps`, IPC + eval `5.23s`)** vs `ThreadPoolExecutor(8)` at **`3.83s` (`158.9 fps`)**. Because OpenCV `cv2.matchTemplate` releases the GIL, multithreading avoids ~6 MB/frame pickle serialization overhead and Windows spawn cost while beating multiprocessing by **>2.1x**.

4. **720p Downscaling Speed vs Accuracy Tradeoff (`Rejected as Default`)**
   - Benchmarking an NVDEC `scale_cuda=1280:720` + `nv12` pipeline on `0..30s` reached **`1.95s` (`307.8 frame/s`, `3.25 ms/frame`)**—nearly 2x faster than `3.83s` at 1080p.
   - However, running the annotated `testing/test_templates.py` suite at 720p dropped template accuracy from **`95/95` (`100.0%`)** down to **`89/95` (`93.7%`)** (`6` failures: 3 `Open Pack`, 2 `Sell Card`, and 1 `Use Consumable` caused by thin drop-shadow features blurring under 2/3 downscaling).
   - **Decision**: Keep **1080p** as the default for `100%` action accuracy; 2x coarse-to-fine at 1080p already captures most of the speed gain on full-res frames without corrupting shadow thresholds.

---

### 6.3 Updated 1,000-Video Projection (`bu_full.mp4`-Equivalent 30-Min VODs, `skip_frames=2`)

Assuming `36,000` kept frames per 30-minute 60 fps video (`18x` the measured `0..100s` 2,000-frame clip; `main.py` stage only):

| Configuration (`Ryzen 7 9800X3D + RTX 5090`) | Measured 2,000-Frame Time | Est. `main.py` / 30-Min Video | Est. Total (1,000 Videos) | Template Accuracy (`test_templates`) |
| --- | --- | --- | --- | --- |
| **Unoptimized Baseline** (`cv2` CPU + Full-Res + `workers=14`) | `35.38s` (`56.62 fps`) | `~10.6 min` | `~177 hours` (`~7.4 days`) | `95 / 95` (`100%`) |
| **Final Default** (`NVDEC` + `Coarse-to-Fine` + `CompiledSpecs` + `workers=8`, 1080p) | **`12.11s` (`165.71 fps`)** | **`~3.6 min`** | **`~60.5 hours` (`~2.5 days`)** | **`95 / 95` (`100%`)** |
| *(Rejected Alternative)* 720p `scale_cuda` Pipeline | `1.95s` / `600f` (`307.8 fps`) | `~1.95 min` | `~32.5 hours` (`~1.4 days`) | `89 / 95` (`93.7%`) |

