"""Place isolated pottery onto the earth-tone set's wall-and-wood studio.

Pottery pixels are not filtered or retouched. Only the surroundings are replaced
with textures sampled from collection-earthtones.jpg.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent
PHOTO_DIR = ROOT / "photos"
RAW_DIR = PHOTO_DIR / "raw"
COLLECTION = RAW_DIR / "collection-earthtones.jpg"

# Finished ware that should sit on the set. Process shot stays on the wheel.
KEEP_ORIGINAL = {
    "collection-earthtones.jpg",
    "process-carved-spiral.jpg",
}

# Top-down views: full tabletop, no wall.
OVERHEAD = {
    "mug-plate-in-use.jpg",
    "plate-reactive.jpg",
    "dish-marbled.jpg",
    "yunomi-interior.jpg",
    "bowl-midnight-interior.jpg",
    "bowl-sculptural-interior.jpg",
    "dish-spiral-black.jpg",
    "grater-garlic.jpg",
    "dish-lidded-heart.jpg",
    "bowl-folded-rim.jpg",
}

# Circular top-down dishes: optional rim cleanup. Left empty because
# aggressive circle-clipping cut into bowls.
ROUND_CLIP: set[str] = set()

MULTI = {
    "mug-plate-in-use.jpg",
    "pour-over-set.jpg",
}

TABLE_WALL_SIZE = (1400, 2100)
OVERHEAD_SIZE = (1600, 2000)

_OBJ_SESSION = None
_HUM_SESSION = None


def _sessions():
    global _OBJ_SESSION, _HUM_SESSION
    if _OBJ_SESSION is None:
        _OBJ_SESSION = new_session("isnet-general-use")
    return _OBJ_SESSION


def _load_textures() -> tuple[Image.Image, Image.Image]:
    src = Image.open(COLLECTION).convert("RGB")
    w, h = src.size
    wall = src.crop((0, 0, int(w * 0.72), int(h * 0.20)))
    wood = src.crop((int(w * 0.02), int(h * 0.78), int(w * 0.55), int(h * 0.88)))
    return wall, wood


def _tile_fill(patch: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Fill a rectangle with a texture, overlapping tiles so seams stay quiet."""
    W, H = size
    pw, ph = patch.size
    # Keep grain readable: scale so the patch's long side is at least half the target.
    scale = max(0.9, min(W / pw, H / ph, 2.4))
    tiled = patch.resize((max(1, int(pw * scale)), max(1, int(ph * scale))), Image.Resampling.LANCZOS)
    tw, th = tiled.size
    canvas = Image.new("RGB", (W + tw, H + th))
    overlap = 36
    step_x = max(1, tw - overlap)
    step_y = max(1, th - overlap)
    arr = np.array(canvas)
    src = np.array(tiled).astype(np.float32)
    weight = np.ones((th, tw, 1), dtype=np.float32)
    # edge fade for blending
    fade = min(overlap, tw // 3, th // 3)
    if fade > 0:
        ramp = np.linspace(0.15, 1.0, fade, dtype=np.float32)
        weight[:, :fade, 0] *= ramp
        weight[:, -fade:, 0] *= ramp[::-1]
        weight[:fade, :, 0] *= ramp[:, None]
        weight[-fade:, :, 0] *= ramp[::-1, None]
    acc = np.zeros((H + th, W + tw, 3), dtype=np.float32)
    wsum = np.zeros((H + th, W + tw, 1), dtype=np.float32)
    for y in range(0, H + 1, step_y):
        for x in range(0, W + 1, step_x):
            acc[y : y + th, x : x + tw] += src * weight
            wsum[y : y + th, x : x + tw] += weight
    acc /= np.clip(wsum, 1e-4, None)
    out = np.clip(acc[:H, :W], 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def make_studio(mode: str) -> Image.Image:
    wall, wood = _load_textures()
    if mode == "overhead":
        W, H = OVERHEAD_SIZE
        return _tile_fill(wood, (W, H))
    W, H = TABLE_WALL_SIZE
    studio = Image.new("RGB", (W, H))
    wall_h = int(H * 0.36)
    wood_y = int(H * 0.30)
    studio.paste(ImageOps.fit(wall, (W, wall_h), Image.Resampling.LANCZOS), (0, 0))
    wood_fill = _tile_fill(wood, (W, H - wood_y))
    studio.paste(wood_fill, (0, wood_y))
    arr = np.array(studio).astype(np.float32)
    blend_h = 55
    for i in range(blend_h):
        t = min(1.0, i / blend_h)
        y = min(H - 1, wood_y + i)
        shade = 0.90 + 0.10 * t
        arr[y] *= shade
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _keep_components(alpha: np.ndarray, multi: bool) -> np.ndarray:
    mask = (alpha > 20).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    keep = set()
    if multi:
        total = mask.size
        for i in order:
            if areas[i] >= total * 0.012:
                keep.add(i + 1)
            if len(keep) >= 3:
                break
        if not keep:
            keep.add(int(order[0]) + 1)
    else:
        keep.add(int(order[0]) + 1)
        # a second piece only if it is large and close in size (lid, dripper)
        if len(order) > 1 and areas[order[1]] > areas[order[0]] * 0.18:
            keep.add(int(order[1]) + 1)
    kept = np.isin(labels, list(keep))
    out = alpha.copy()
    out[~kept] = 0
    return out


def isolate(im: Image.Image, *, multi: bool, name: str = "") -> Image.Image:
    obj_sess = _sessions()
    rgba = remove(im, session=obj_sess).convert("RGBA")
    a = np.array(rgba)
    obj_m = a[:, :, 3]

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = (obj_m > 30).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    if name not in OVERHEAD:
        binary = _split_hand_from_border(binary)
    binary = _keep_components(binary, multi=multi)
    binary = _drop_small_protrusions(binary)
    if name in ROUND_CLIP:
        binary = _clip_round_rim(binary)
    obj_m = np.minimum(obj_m, binary)
    erode = cv2.erode(obj_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    obj_m = cv2.GaussianBlur(erode, (0, 0), 0.9)
    a[:, :, 3] = obj_m
    return Image.fromarray(a)


def _split_hand_from_border(binary: np.ndarray) -> np.ndarray:
    """If a hand enters from the frame edge, keep the central pot and drop the arm."""
    fg = binary > 0
    if fg.mean() < 0.01:
        return binary
    h, w = fg.shape
    border = np.zeros_like(fg)
    margin = max(4, min(h, w) // 80)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    if not np.any(fg & border):
        return binary

    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    x0b, x1b = int(xs.min()), int(xs.max())
    # Hands usually enter from below; seed the pot in the upper part of the blob.
    cy = int(y0 + 0.22 * (y1 - y0))
    cx = int((x0b + x1b) / 2)
    markers = np.zeros((h, w), np.int32)
    markers[~fg] = 3  # sure background
    seed = max(8, min(h, w) // 40)
    y0, y1 = max(0, cy - seed), min(h, cy + seed)
    x0, x1 = max(0, cx - seed), min(w, cx + seed)
    border_fg = (fg & border).astype(np.uint8) * 255
    border_fg = cv2.dilate(border_fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    markers[border_fg > 0] = 2
    markers[y0:y1, x0:x1] = 1

    bgr = cv2.cvtColor(np.array(Image.fromarray(binary).convert("RGB")), cv2.COLOR_RGB2BGR)
    ws = cv2.watershed(bgr, markers)
    keep = ws == 1
    if keep.mean() < 0.004:
        return binary
    orig_ys, orig_xs = np.where(fg)
    ks, xs = np.where(keep)
    if len(xs) == 0:
        return binary
    orig_w = orig_xs.max() - orig_xs.min()
    keep_w = xs.max() - xs.min()
    # If the split chopped the pot, keep the full silhouette (hand and all).
    if keep_w < orig_w * 0.84 or keep.sum() < fg.sum() * 0.58:
        return binary
    out = np.zeros_like(binary)
    out[keep] = 255
    return out


def _drop_small_protrusions(binary: np.ndarray) -> np.ndarray:
    """Knock off leftover fingertips on round dishes without clipping a spout."""
    fg = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return binary
    out = np.zeros_like(binary)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        x, y, bw, bh, area = stats[i]
        if area < 80:
            continue
        # Round-ish blobs: stronger opening removes a thumb on the rim.
        aspect = max(bw, bh) / max(1, min(bw, bh))
        solidity = area / max(1, bw * bh)
        ksz = 29 if aspect < 1.45 and solidity > 0.52 else 5
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        cleaned = cv2.morphologyEx(comp * 255, cv2.MORPH_OPEN, k)
        if cleaned.mean() == 0:
            cleaned = comp * 255
        out = np.maximum(out, cleaned)
    return out


def _clip_round_rim(binary: np.ndarray) -> np.ndarray:
    """Drop a fingertip on the rim by fitting a circle to the inner mass."""
    fg = (binary > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return binary
    cnt = max(cnts, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 20:
        return binary
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    inner = pts[dist <= np.percentile(dist, 90)]
    (ccx, ccy), r = cv2.minEnclosingCircle(inner.astype(np.float32))
    yy, xx = np.ogrid[: binary.shape[0], : binary.shape[1]]
    circle = (xx - ccx) ** 2 + (yy - ccy) ** 2 <= (r * 1.04) ** 2
    out = np.zeros_like(binary)
    out[circle & (fg > 0)] = 255
    if out.mean() < 0.004:
        return binary
    return out


def _crop_subject(rgba: Image.Image, pad: int = 24) -> Image.Image:
    a = np.array(rgba.split()[-1])
    ys, xs = np.where(a > 12)
    if len(xs) == 0:
        return rgba
    x0, x1 = max(0, xs.min() - pad), min(rgba.width, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(rgba.height, ys.max() + pad)
    return rgba.crop((x0, y0, x1, y1))


def _shadow(size: tuple[int, int], ellipse: tuple[int, int, int, int], opacity: float = 0.32) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    # Draw via numpy ellipse
    x0, y0, x1, y1 = ellipse
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return layer
    yy, xx = np.ogrid[: size[1], : size[0]]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = w / 2, h / 2
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[mask] = (20, 16, 12, int(255 * opacity))
    img = Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(radius=18))
    return img


def composite(name: str, src: Image.Image) -> Image.Image:
    multi = name in MULTI
    cut = isolate(src, multi=multi, name=name)
    cut = _crop_subject(cut)
    mode = "overhead" if name in OVERHEAD else "table"
    studio = make_studio(mode)
    W, H = studio.size
    cw, ch = cut.size

    if mode == "overhead":
        target_w = int(W * 0.72)
        scale = target_w / max(cw, 1)
        # keep it from overflowing vertically
        if ch * scale > H * 0.78:
            scale = (H * 0.78) / ch
    else:
        target_h = int(H * 0.58)
        scale = target_h / max(ch, 1)
        if cw * scale > W * 0.78:
            scale = (W * 0.78) / cw

    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    cut_r = cut.resize((nw, nh), Image.Resampling.LANCZOS)

    if mode == "overhead":
        x = (W - nw) // 2
        y = (H - nh) // 2
        shadow_box = (x + int(nw * 0.08), y + int(nh * 0.78), x + int(nw * 0.92), y + int(nh * 0.98))
    else:
        x = (W - nw) // 2
        # Sit on the table plane (wood starts ~38% down).
        table_y = int(H * 0.32)
        y = table_y + int((H - table_y) * 0.18)
        foot = y + nh
        if foot > H - int(H * 0.10):
            y = H - int(H * 0.10) - nh
        if y < int(H * 0.06):
            y = int(H * 0.06)
        shadow_box = (
            x + int(nw * 0.16) + 8,
            y + nh - int(nh * 0.07),
            x + int(nw * 0.86) + 14,
            y + nh + int(nh * 0.025),
        )

    base = studio.convert("RGBA")
    shade = _shadow((W, H), shadow_box)
    base = Image.alpha_composite(base, shade)
    base.paste(cut_r, (x, y), cut_r)
    return base.convert("RGB")


def process_all() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(RAW_DIR.glob("*.jpg"))
    if not sources:
        sources = [p for p in sorted(PHOTO_DIR.glob("*.jpg"))]
        raw_fallback = True
    else:
        raw_fallback = False
    for path in sources:
        name = path.name
        dest = PHOTO_DIR / name
        if name in KEEP_ORIGINAL:
            if path.resolve() != dest.resolve():
                Image.open(path).convert("RGB").save(dest, format="JPEG", quality=90, optimize=True)
            print(f"keep {name}")
            continue
        src = Image.open(path).convert("RGB")
        out = composite(name, src)
        out.save(dest, format="JPEG", quality=90, optimize=True)
        print(f"studio {name} -> {out.size}")
    if raw_fallback:
        print("studio: used photos/ as source (no photos/raw/)")
