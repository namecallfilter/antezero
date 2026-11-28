"""
Shared Augmentations for Balatro Detection & Classification
============================================================
Provides:
1. Video compression artifact simulation (downsampling + macroblock quantization + noise)
2. Synthetic Windows cursor occlusion (assets/classifiers/cursor.png)
3. Ultralytics YOLO batch augmentation hook (on_train_batch_start)
4. PyTorch / PIL transforms for Stage 2 crop classifiers
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURSOR_PATH = PROJECT_ROOT / "assets" / "classifiers" / "cursor.png"

_CACHED_CURSOR_RGBA = None
_CACHED_CURSOR_TENSORS = {}
_IMAGENET_MEAN_CACHE = {}
_IMAGENET_STD_CACHE = {}


def _get_cursor_tensor(cursor_rgba: np.ndarray, device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _CACHED_CURSOR_TENSORS.get(key)
    if t is None:
        # Store as (1, 4, H, W) float32 in [0, 255] RGB and [0, 1] A
        t = torch.from_numpy(cursor_rgba).permute(2, 0, 1).unsqueeze(0).contiguous().to(device=device, dtype=torch.float32)
        _CACHED_CURSOR_TENSORS[key] = t
    return t


def _get_imagenet_stats(device: torch.device):
    key = str(device)
    mean = _IMAGENET_MEAN_CACHE.get(key)
    std = _IMAGENET_STD_CACHE.get(key)
    if mean is None or std is None:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        _IMAGENET_MEAN_CACHE[key] = mean
        _IMAGENET_STD_CACHE[key] = std
    return mean, std


def load_cursor_asset(cursor_path=CURSOR_PATH):
    """Loads cursor RGBA image and converts RGB to float32 with alpha in [0, 1]."""
    global _CACHED_CURSOR_RGBA
    if _CACHED_CURSOR_RGBA is not None:
        return _CACHED_CURSOR_RGBA

    p = Path(cursor_path)
    if not p.exists():
        return None

    cursor_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if cursor_img is None or cursor_img.shape[-1] < 4:
        return None

    bgr = cursor_img[..., :3]
    alpha = cursor_img[..., 3:4].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    _CACHED_CURSOR_RGBA = rgba
    return rgba


def _apply_yuv420_chroma_subsampling(norm_rgb: torch.Tensor) -> torch.Tensor:
    """
    Simulates YUV420p 4:2:0 chroma subsampling (H.264/AV1 streaming compression)
    on a float32 tensor in [0, 1] of shape (N, 3, H, W).
    """
    _, _, orig_h, orig_w = norm_rgb.shape
    r, g, b = norm_rgb[:, 0:1], norm_rgb[:, 1:2], norm_rgb[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b

    chroma = torch.cat([cb, cr], dim=1)
    half_h = max(4, orig_h // 2)
    half_w = max(4, orig_w // 2)
    chroma_down = F.interpolate(chroma, size=(half_h, half_w), mode="bilinear", align_corners=False)
    # Coarser chroma quantization typical of low-bitrate 4:2:0 encoders
    chroma_step = 1.0 / 24.0
    chroma_down = torch.round(chroma_down / chroma_step) * chroma_step
    chroma_up = F.interpolate(chroma_down, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    cb_up, cr_up = chroma_up[:, 0:1], chroma_up[:, 1:2]

    y_step = 1.0 / 48.0
    y_q = torch.round(y / y_step) * y_step

    r_out = y_q + 1.402 * cr_up
    g_out = y_q - 0.344136 * cb_up - 0.714136 * cr_up
    b_out = y_q + 1.772 * cb_up
    return torch.clamp(torch.cat([r_out, g_out, b_out], dim=1), 0.0, 1.0)


def apply_video_compression_noise(img_tensor: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Simulates realistic video compression artifacts on a batch (4D) or single (3D) tensor:
    - YUV420p 4:2:0 chroma subsampling & chroma quantization (red/gold smear)
    - Low-pass / bilinear down-sampling (loss of high frequency bitrate)
    - Mild 3x3 box blur + macroblock quantization (discrete color levels)
    - High-frequency dithering / sensor noise along reconstructed edges
    """
    if not isinstance(img_tensor, torch.Tensor):
        return img_tensor

    is_single = False
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
        is_single = True
    elif img_tensor.ndim != 4:
        return img_tensor

    batch_size = img_tensor.shape[0]
    device = img_tensor.device

    mask = torch.rand(batch_size, device=device) < p
    if not mask.any():
        return img_tensor.squeeze(0) if is_single else img_tensor

    out = img_tensor.clone()
    orig_dtype = out.dtype
    selected = out[mask].float()
    orig_h, orig_w = selected.shape[-2:]

    max_val = 255.0 if orig_dtype == torch.uint8 or selected.max() > 1.5 else 1.0
    norm_sel = selected / max_val

    # 1. YUV420p 4:2:0 chroma subsampling on ~80% of compressed samples
    chroma_mask = torch.rand(norm_sel.shape[0], device=device) < 0.80
    if chroma_mask.any():
        norm_sel[chroma_mask] = _apply_yuv420_chroma_subsampling(norm_sel[chroma_mask])

    # 2. Spatial bitrate downscale + level quantization
    scale = torch.empty(1, device=device).uniform_(0.40, 0.85).item()
    down_h = max(8, int(orig_h * scale))
    down_w = max(8, int(orig_w * scale))

    down = F.interpolate(norm_sel, size=(down_h, down_w), mode="bilinear", align_corners=False)
    step = 1.0 / 32.0
    down = torch.round(down / step) * step
    up = F.interpolate(down, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    # 3. Mild 3x3 box blur mix on ~45% of compressed samples
    blur_mask = torch.rand(norm_sel.shape[0], device=device) < 0.45
    if blur_mask.any():
        blurred = F.avg_pool2d(up[blur_mask], kernel_size=3, stride=1, padding=1)
        up[blur_mask] = torch.lerp(up[blur_mask], blurred, 0.45)

    noise = (torch.rand_like(selected) - 0.5) * 0.03 * max_val
    selected_augmented = torch.clamp(up * max_val + noise, 0.0, max_val).to(orig_dtype)
    out[mask] = selected_augmented

    return out.squeeze(0) if is_single else out


def apply_card_crop_jitter(
    img_tensor: torch.Tensor,
    p: float = 0.4,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Applies mild affine crop/scale/rotation jitter on GPU to simulate pack_hand
    framing shifts and YOLO bounding-box tightness variation.

    When `output_size=(target_h, target_w)` is provided and differs from `(H, W)` (e.g. from
    `pad_ratio`-expanded real-frame crops), samples real bordering pixels inside `(H, W)` and
    center-crops the non-jittered batch elements cleanly to `output_size`.
    """
    if not isinstance(img_tensor, torch.Tensor) or img_tensor.ndim != 4 or img_tensor.shape[0] == 0:
        return img_tensor

    batch_size, _, H, W = img_tensor.shape
    target_h, target_w = output_size if output_size is not None else (H, W)
    device = img_tensor.device
    mask = torch.rand(batch_size, device=device) < p
    if not mask.any() and (target_h, target_w) == (H, W):
        return img_tensor

    if (target_h, target_w) == (H, W):
        out = img_tensor.clone()
    else:
        y0 = max(0, (H - target_h) // 2)
        x0 = max(0, (W - target_w) // 2)
        out = img_tensor[:, :, y0 : y0 + target_h, x0 : x0 + target_w].clone()
        if not mask.any():
            return out

    orig_dtype = out.dtype
    selected = img_tensor[mask].float()
    n = selected.shape[0]

    base_sx = float(target_w) / float(max(1, W))
    base_sy = float(target_h) / float(max(1, H))
    scales = torch.empty(n, device=device).uniform_(0.90, 1.08)
    inv_s = 1.0 / scales
    angles = torch.empty(n, device=device).uniform_(-2.5, 2.5) * (3.141592653589793 / 180.0)
    cos_a = torch.cos(angles) * inv_s
    sin_a = torch.sin(angles) * inv_s
    tx = torch.empty(n, device=device).uniform_(-0.06, 0.06)
    ty = torch.empty(n, device=device).uniform_(-0.06, 0.06)

    theta = torch.zeros((n, 2, 3), device=device, dtype=torch.float32)
    theta[:, 0, 0] = cos_a * base_sx
    theta[:, 0, 1] = -sin_a * base_sx
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a * base_sy
    theta[:, 1, 1] = cos_a * base_sy
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=[n, selected.shape[1], target_h, target_w], align_corners=False)
    warped = F.grid_sample(selected, grid, mode="bilinear", padding_mode="border", align_corners=False)
    out[mask] = warped.to(orig_dtype)
    return out


def overlay_cursor_noise(img_tensor: torch.Tensor, cursor_rgba: np.ndarray = None, p: float = 0.3) -> torch.Tensor:
    """
    Randomly overlays a Windows cursor onto ~`p` fraction of images in the batch or single tensor
    to build robustness against mouse occlusions over cards and UI elements.
    """
    if cursor_rgba is None:
        cursor_rgba = load_cursor_asset()
    if cursor_rgba is None or not isinstance(img_tensor, torch.Tensor):
        return img_tensor

    is_single = False
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
        is_single = True
    elif img_tensor.ndim != 4:
        return img_tensor

    batch_size, _, H, W = img_tensor.shape
    device = img_tensor.device

    mask = torch.rand(batch_size, device=device) < p
    if not mask.any():
        return img_tensor.squeeze(0) if is_single else img_tensor

    out = img_tensor.clone()
    orig_dtype = out.dtype
    c_h, c_w = cursor_rgba.shape[:2]
    cursor_tensor = _get_cursor_tensor(cursor_rgba, device)
    is_unit_scale = (orig_dtype != torch.uint8 and out.max().item() <= 1.5)

    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        # Scale cursor to realistic screen proportion (around 20px - 44px)
        scale = torch.empty(1).uniform_(0.08, 0.22).item()
        target_h = max(14, min(H, int(c_h * scale)))
        target_w = max(10, min(W, int(c_w * scale)))

        max_y = max(1, H - target_h)
        max_x = max(1, W - target_w)
        y = int(torch.randint(0, max_y, (1,)).item())
        x = int(torch.randint(0, max_x, (1,)).item())

        resized = F.interpolate(cursor_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
        c_rgb = resized[:3]
        c_alpha = resized[3:4]
        if is_unit_scale:
            c_rgb = c_rgb / 255.0

        bg_patch = out[idx, :, y:y+target_h, x:x+target_w].float()
        blended = bg_patch * (1.0 - c_alpha) + c_rgb * c_alpha
        out[idx, :, y:y+target_h, x:x+target_w] = blended.to(orig_dtype)

    return out.squeeze(0) if is_single else out


def apply_gpu_batch_augmentations(
    images: torch.Tensor,
    is_train: bool = True,
    compression_prob: float = 0.5,
    cursor_prob: float = 0.3,
    color_jitter: bool = True,
    geom_prob: float = 0.0,
    pad_crop_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Converts uint8/float (B, 3, H, W) tensor batch on GPU to normalized float32,
    applying vectorized ColorJitter, optional affine crop jitter, video compression,
    and cursor overlays on-device.
    """
    if images.dtype == torch.uint8:
        x = images.float().div_(255.0)
    else:
        x = images.float()

    if is_train:
        b = x.shape[0]
        device = x.device
        if (geom_prob > 0.0 or (pad_crop_size is not None and tuple(x.shape[-2:]) != tuple(pad_crop_size))) and b > 0:
            x = apply_card_crop_jitter(x, p=geom_prob, output_size=pad_crop_size)
        if color_jitter and b > 0:
            # Vectorized brightness, contrast, and saturation jitter on GPU
            bright = torch.empty((b, 1, 1, 1), device=device).uniform_(0.85, 1.15)
            x = x * bright

            contrast = torch.empty((b, 1, 1, 1), device=device).uniform_(0.85, 1.15)
            mean_gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]).mean(dim=(-2, -1), keepdim=True)
            x = torch.lerp(mean_gray, x, contrast)

            sat = torch.empty((b, 1, 1, 1), device=device).uniform_(0.85, 1.15)
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
            x = torch.lerp(gray, x, sat)
            x = torch.clamp_(x, 0.0, 1.0)

        if compression_prob > 0.0:
            x = apply_video_compression_noise(x, p=compression_prob)
        if cursor_prob > 0.0:
            x = overlay_cursor_noise(x, p=cursor_prob)
    elif pad_crop_size is not None and tuple(x.shape[-2:]) != tuple(pad_crop_size):
        th, tw = pad_crop_size
        H, W = x.shape[-2:]
        y0 = max(0, (H - th) // 2)
        x0 = max(0, (W - tw) // 2)
        x = x[:, :, y0 : y0 + th, x0 : x0 + tw]

    mean, std = _get_imagenet_stats(x.device)
    return (x - mean) / std


def attach_batch_augmentations(model, compression_prob: float = 0.5, cursor_prob: float = 0.3):
    """
    Hooks into the Ultralytics training loop via 'on_train_batch_start' callback.
    """
    cursor_rgba = load_cursor_asset()
    if cursor_rgba is not None:
        print(f"[Augmentations] Loaded cursor asset from {CURSOR_PATH}")
    else:
        print(f"[Augmentations] Notice: Cursor asset not found at {CURSOR_PATH}. Cursor augmentation skipped.")

    def on_train_batch_start(trainer):
        if hasattr(trainer, "batch") and isinstance(trainer.batch, dict) and "img" in trainer.batch:
            if compression_prob > 0.0:
                trainer.batch["img"] = apply_video_compression_noise(trainer.batch["img"], p=compression_prob)
            if cursor_prob > 0.0 and cursor_rgba is not None:
                trainer.batch["img"] = overlay_cursor_noise(trainer.batch["img"], cursor_rgba, p=cursor_prob)

    model.add_callback("on_train_batch_start", on_train_batch_start)
    print(f"[Augmentations] Attached YOLO batch hooks: compression_p={compression_prob}, cursor_p={cursor_prob}")


class BalatroStreamAugmentation:
    """
    PyTorch transform applied after ToTensor() (values in [0, 1])
    to inject realistic video compression artifacts and mouse cursor occlusions.
    """
    def __init__(self, compression_prob: float = 0.5, cursor_prob: float = 0.3):
        self.compression_prob = compression_prob
        self.cursor_prob = cursor_prob
        self.cursor_rgba = load_cursor_asset()

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if self.compression_prob > 0.0:
            img_tensor = apply_video_compression_noise(img_tensor, p=self.compression_prob)
        if self.cursor_prob > 0.0 and self.cursor_rgba is not None:
            img_tensor = overlay_cursor_noise(img_tensor, self.cursor_rgba, p=self.cursor_prob)
        return img_tensor
