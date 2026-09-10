#!/usr/bin/env python3
"""Build a scrapbook-style ceramics portfolio PDF.

New same-setup photos replace earlier shots of those pieces. Pieces that
were not re-photographed stay on their original background. Every frame is
cropped in on the ware so the piece reads clearly; pottery pixels are not
retouched, and the crop does not cut into the pot.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from rembg import new_session, remove
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
PHOTO_DIR = ROOT / "photos"
RAW_DIR = PHOTO_DIR / "raw"
FONT_DIR = ROOT / "fonts"
PAPER_PATH = ROOT / "paper" / "kraft.jpg"
OUTPUT = ROOT / "Mennatallah_Rihan_Ceramics_Portfolio.pdf"
SOURCE_DIR = Path("/home/ubuntu/.cursor/projects/workspace/assets")

PAGE = letter
PAGE_W, PAGE_H = PAGE

INK = (0.22, 0.16, 0.13)
MUTED = (0.42, 0.34, 0.28)
TEAL = (0.27, 0.38, 0.34)
POLAROID = (0.99, 0.97, 0.94)
SHADOW = (0.70, 0.62, 0.52)

MAX_PHOTO_EDGE = 2000

_REMBG = None


def _rembg_session():
    global _REMBG
    if _REMBG is None:
        _REMBG = new_session("isnet-general-use")
    return _REMBG


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Cormorant", str(FONT_DIR / "CormorantGaramond-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("CormorantSemibold", str(FONT_DIR / "CormorantGaramond-SemiBold.ttf")))
    pdfmetrics.registerFont(TTFont("CormorantItalic", str(FONT_DIR / "CormorantGaramond-Italic.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSans", str(FONT_DIR / "SourceSans3-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSansLight", str(FONT_DIR / "SourceSans3-Light.ttf")))


def seed_for(name: str) -> np.random.Generator:
    n = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(n)


def _expand_to_aspect(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    lo: float,
    hi: float,
) -> tuple[int, int, int, int]:
    """Grow a crop window so the polaroid is not a skinny strip."""
    cw, ch = right - left, bottom - top
    if cw < 8 or ch < 8:
        return left, top, right, bottom
    aspect = cw / ch
    if aspect > hi:
        target_h = int(cw / hi)
        extra = max(0, target_h - ch)
        up = int(extra * 0.38)
        down = extra - up
        if top - up < 0:
            down += up - top
            up = top
        if bottom + down > height:
            up += bottom + down - height
            down = height - bottom
            up = min(up, top)
        top -= up
        bottom += down
    elif aspect < lo:
        target_w = int(ch * lo)
        extra = max(0, target_w - cw)
        grow_l = extra // 2
        grow_r = extra - grow_l
        if left - grow_l < 0:
            grow_r += grow_l - left
            grow_l = left
        if right + grow_r > width:
            grow_l += right + grow_r - width
            grow_r = width - right
            grow_l = min(grow_l, left)
        left -= grow_l
        right += grow_r
    return left, top, right, bottom


def tight_crop(im: Image.Image, name: str) -> Image.Image:
    """Crop to the pottery plus a little table/wall. Does not cut the ware."""
    W, H = im.size
    if name == "process-carved-spiral.jpg":
        # rembg treats the whole wheel as the subject; crop in on the bowl.
        side = int(0.66 * min(W, H))
        cx, cy = W // 2, int(H * 0.53)
        left = max(0, cx - side // 2)
        top = max(0, cy - side // 2)
        return im.crop((left, top, min(W, left + side), min(H, top + side)))

    rgba = remove(im, session=_rembg_session())
    alpha = np.array(rgba.split()[-1].filter(ImageFilter.MaxFilter(7)))
    ys, xs = np.where(alpha > 40)
    if len(xs) < 400:
        return im
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    ow, oh = max(1, x1 - x0), max(1, y1 - y0)
    left = max(0, x0 - int(0.12 * ow))
    top = max(0, y0 - int(0.14 * oh))
    right = min(W, x1 + int(0.12 * ow))
    bottom = min(H, y1 + int(0.18 * oh))
    left, top, right, bottom = _expand_to_aspect(left, top, right, bottom, W, H, 0.78, 1.08)
    if right - left < 32 or bottom - top < 32:
        return im
    return im.crop((left, top, right, bottom))


def prepare_photo(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im = tight_crop(im, dest.name)
    w, h = im.size
    scale = min(1.0, MAX_PHOTO_EDGE / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    im.save(dest, format="JPEG", quality=90, optimize=True, subsampling=0)


def photo_size(name: str) -> tuple[int, int]:
    with Image.open(PHOTO_DIR / name) as im:
        return im.size


def make_paper(path: Path, size: tuple[int, int] = (1650, 2200)) -> None:
    """Warm kraft album paper with terracotta sprigs and sage pluses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = size
    rng = np.random.default_rng(21)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    t = 0.35 * yy + 0.20 * xx
    c1 = np.array([236, 220, 194], dtype=np.float32)
    c2 = np.array([214, 190, 156], dtype=np.float32)
    base = c1 * (1 - t[..., None]) + c2 * t[..., None]
    base += rng.normal(0, 4.5, (H, W, 3)).astype(np.float32)
    for y in range(0, H, 6):
        base[y, :, :] *= 0.988
    speck = rng.random((H, W))
    base[speck > 0.994] *= 0.70
    base[speck < 0.006] = np.minimum(base[speck < 0.006] + 20, 255)

    yy_i, xx_i = np.ogrid[:H, :W]
    terracotta = np.array([176, 118, 92], dtype=np.float32)
    sage = np.array([130, 148, 126], dtype=np.float32)
    alpha = np.zeros((H, W), dtype=np.float32)
    step = 52
    row = ((yy_i // step) % 2) * (step // 2)
    dx = (xx_i + row) % step - step // 2
    dy = yy_i % step - step // 2
    dist = np.sqrt(dx * dx + dy * dy)
    alpha[dist < 2.2] = 0.38
    for k in range(6):
        ang = k * np.pi / 3.0
        px = dx - 7.2 * np.cos(ang)
        py = dy - 7.2 * np.sin(ang)
        petal = np.sqrt(px * px + py * py) < 2.4
        alpha[petal] = np.maximum(alpha[petal], 0.28)
    out = base * (1 - alpha[..., None]) + terracotta * alpha[..., None]

    step2 = 52
    row2 = ((yy_i // step2) % 2) * (step2 // 2) + step2 // 2
    dx2 = (xx_i + row2) % step2 - step2 // 2
    dy2 = (yy_i + step2 // 2) % step2 - step2 // 2
    plus = ((np.abs(dx2) < 1.2) & (np.abs(dy2) < 5.5)) | ((np.abs(dy2) < 1.2) & (np.abs(dx2) < 5.5))
    sage_a = np.zeros((H, W), dtype=np.float32)
    sage_a[plus] = 0.22
    out = out * (1 - sage_a[..., None]) + sage * sage_a[..., None]

    cy, cx = (H - 1) / 2, (W - 1) / 2
    vr = np.sqrt(((yy_i - cy) / H) ** 2 + ((xx_i - cx) / W) ** 2)
    out *= (1.0 - 0.06 * np.clip(vr, 0, 1))[..., None]
    img = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(img, "RGB").save(path, format="JPEG", quality=88, optimize=True)


def opaque(c: canvas.Canvas) -> None:
    """ReportLab keeps fill alpha in ExtGState; RGB color changes do not clear it."""
    c.setFillAlpha(1)
    c.setStrokeAlpha(1)


def paint_page(c: canvas.Canvas) -> None:
    opaque(c)
    c.drawImage(str(PAPER_PATH), 0, 0, width=PAGE_W, height=PAGE_H)
    opaque(c)


def footer_number(c: canvas.Canvas, n: int) -> None:
    opaque(c)
    c.setFillColorRGB(*MUTED)
    c.setFont("CormorantItalic", 10)
    c.drawCentredString(PAGE_W / 2, 0.32 * inch, str(n))


def _aabb(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    cw, sw = abs(math.cos(a)), abs(math.sin(a))
    return w * cw + h * sw, w * sw + h * cw


def _tape(c: canvas.Canvas, x: float, y: float, w: float, h: float, angle: float, color: Color) -> None:
    c.saveState()
    c.translate(x, y)
    c.rotate(angle)
    c.setFillColor(color)
    c.rect(-w / 2, -h / 2, w, h, fill=1, stroke=0)
    c.restoreState()


def draw_polaroid(
    c: canvas.Canvas,
    name: str,
    cx: float,
    cy: float,
    max_w: float,
    max_h: float,
    *,
    angle: float | None = None,
    caption: str = "",
    tape: bool = True,
) -> None:
    """Place the photograph in a tilted polaroid. The JPEG is already tight-cropped."""
    im_w, im_h = photo_size(name)
    rng = seed_for(name)
    if angle is None:
        angle = float(rng.uniform(-2.6, 2.8))

    side = 0.07
    bottom = 0.18
    photo_aspect = im_w / im_h

    def polaroid_size(photo_w: float) -> tuple[float, float, float, float]:
        photo_h = photo_w / photo_aspect
        outer_w = photo_w * (1 + 2 * side)
        outer_h = photo_h + photo_w * (side + bottom)
        return photo_w, photo_h, outer_w, outer_h

    lo, hi = 4.0, max(max_w, max_h)
    for _ in range(22):
        mid = (lo + hi) / 2
        _, _, ow, oh = polaroid_size(mid)
        rw, rh = _aabb(ow, oh, angle)
        if rw <= max_w and rh <= max_h:
            lo = mid
        else:
            hi = mid
    photo_w, photo_h, outer_w, outer_h = polaroid_size(lo)

    c.saveState()
    c.translate(cx, cy)
    c.rotate(angle)

    opaque(c)
    c.setFillColorRGB(*SHADOW)
    c.roundRect(-outer_w / 2 + 5, -outer_h / 2 - 6, outer_w, outer_h, 4, fill=1, stroke=0)
    opaque(c)
    c.setFillColorRGB(*POLAROID)
    c.setStrokeColorRGB(0.86, 0.80, 0.72)
    c.setLineWidth(0.4)
    c.roundRect(-outer_w / 2, -outer_h / 2, outer_w, outer_h, 3, fill=1, stroke=1)

    pad_x = photo_w * side
    pad_top = photo_w * side
    img_x = -outer_w / 2 + pad_x
    img_y = outer_h / 2 - pad_top - photo_h
    opaque(c)
    c.drawImage(
        str(PHOTO_DIR / name),
        img_x,
        img_y,
        width=photo_w,
        height=photo_h,
        preserveAspectRatio=True,
        anchor="c",
        mask=None,
    )

    if caption:
        opaque(c)
        c.setFillColorRGB(*INK)
        c.setFont("CormorantItalic", 11)
        c.drawCentredString(0, -outer_h / 2 + photo_w * 0.06, caption)

    if tape:
        tones = [
            Color(0.78, 0.62, 0.46, alpha=0.55),
            Color(0.62, 0.70, 0.62, alpha=0.50),
            Color(0.82, 0.52, 0.42, alpha=0.48),
            Color(0.93, 0.88, 0.72, alpha=0.62),
        ]
        color = tones[int(rng.integers(0, len(tones)))]
        tw = outer_w * float(rng.uniform(0.22, 0.32))
        th = 11
        _tape(c, 0, outer_h / 2 - 2, tw, th, float(rng.uniform(-8, 8)), color)
        if rng.random() > 0.45:
            color2 = tones[int(rng.integers(0, len(tones)))]
            _tape(
                c,
                -outer_w / 2 + 16,
                outer_h / 2 - 18,
                outer_w * 0.14,
                9,
                float(rng.uniform(35, 55)),
                color2,
            )
        opaque(c)

    c.restoreState()


def page_title(c: canvas.Canvas, title: str, subtitle: str) -> None:
    opaque(c)
    c.setFillColorRGB(*INK)
    c.setFont("CormorantSemibold", 15)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 0.52 * inch, title)
    c.setFillColorRGB(*MUTED)
    c.setFont("SourceSansLight", 8.5)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 0.70 * inch, subtitle)


def work_single(
    c: canvas.Canvas,
    photo: str,
    title: str,
    subtitle: str,
    page_no: int,
    polaroid_caption: str = "",
    angle: float | None = None,
) -> None:
    paint_page(c)
    page_title(c, title, subtitle)
    draw_polaroid(
        c,
        photo,
        PAGE_W / 2,
        PAGE_H / 2 - 0.02 * inch,
        PAGE_W - 1.05 * inch,
        PAGE_H - 2.15 * inch,
        angle=angle if angle is not None else 1.2,
        caption=polaroid_caption or title,
    )
    footer_number(c, page_no)
    c.showPage()


def work_pair(
    c: canvas.Canvas,
    left: str,
    right: str,
    title: str,
    subtitle: str,
    page_no: int,
    left_caption: str = "",
    right_caption: str = "",
) -> None:
    paint_page(c)
    page_title(c, title, subtitle)
    box_w = PAGE_W * 0.50
    box_h = PAGE_H - 1.75 * inch
    cy = PAGE_H / 2 - 0.05 * inch
    draw_polaroid(
        c,
        left,
        PAGE_W * 0.28,
        cy + 0.12 * inch,
        box_w,
        box_h,
        angle=-2.4,
        caption=left_caption,
    )
    draw_polaroid(
        c,
        right,
        PAGE_W * 0.72,
        cy - 0.08 * inch,
        box_w,
        box_h,
        angle=2.1,
        caption=right_caption,
    )
    footer_number(c, page_no)
    c.showPage()


def cover_page(c: canvas.Canvas, hero: str) -> None:
    paint_page(c)
    opaque(c)
    c.setFillColorRGB(*TEAL)
    c.setFont("SourceSans", 9)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 0.62 * inch, "MENNATALLAH RIHAN")
    c.setFillColorRGB(*INK)
    c.setFont("Cormorant", 40)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.18 * inch, "Ceramics")
    c.setFillColorRGB(*MUTED)
    c.setFont("CormorantItalic", 13)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.50 * inch, "a scrapbook of recent work")

    draw_polaroid(
        c,
        hero,
        PAGE_W / 2 - 0.18 * inch,
        PAGE_H / 2 + 0.12 * inch,
        PAGE_W - 1.55 * inch,
        PAGE_H - 3.55 * inch,
        angle=-1.8,
        caption="earth-tone collection",
    )
    draw_polaroid(
        c,
        "pour-over-stacked.jpg",
        PAGE_W * 0.695,
        2.55 * inch,
        2.45 * inch,
        3.15 * inch,
        angle=6.4,
        caption="pour-over",
    )

    opaque(c)
    c.setFillColorRGB(*MUTED)
    c.setFont("SourceSansLight", 8.5)
    c.drawString(0.62 * inch, 0.42 * inch, "Hand-thrown and hand-built stoneware")
    c.showPage()


def colophon_page(c: canvas.Canvas, page_no: int) -> None:
    paint_page(c)
    # note card
    card_w, card_h = 6.3 * inch, 7.4 * inch
    cx, cy = PAGE_W / 2, PAGE_H / 2 + 0.1 * inch
    c.saveState()
    c.translate(cx, cy)
    c.rotate(-0.8)
    opaque(c)
    c.setFillColorRGB(*SHADOW)
    c.roundRect(-card_w / 2 + 5, -card_h / 2 - 6, card_w, card_h, 6, fill=1, stroke=0)
    opaque(c)
    c.setFillColorRGB(*POLAROID)
    c.setStrokeColorRGB(0.86, 0.80, 0.72)
    c.setLineWidth(0.5)
    c.roundRect(-card_w / 2, -card_h / 2, card_w, card_h, 5, fill=1, stroke=1)
    _tape(c, 0, card_h / 2 - 4, 1.4 * inch, 12, 3, Color(0.62, 0.70, 0.62, alpha=0.5))
    opaque(c)

    c.setFillColorRGB(*TEAL)
    c.setFont("SourceSans", 8.5)
    c.drawCentredString(0, card_h / 2 - 0.55 * inch, "NOTES")
    c.setFillColorRGB(*INK)
    c.setFont("Cormorant", 26)
    c.drawCentredString(0, card_h / 2 - 1.05 * inch, "About this scrapbook")

    body = [
        "Most pieces were photographed on the same studio set —",
        "a wood table against a white wall — so the book reads as one sitting.",
        "Where a piece has not been re-shot yet, the earlier photograph is kept.",
        "Frames are cropped in on the ware so glaze and form read clearly;",
        "nothing is cut from the pot, and the pottery itself is unretouched.",
        "",
        "Near-duplicate frames of the same view were reduced to the",
        "clearest view.",
        "",
        "Wheel-thrown and hand-built stoneware:",
        "reactive glazes, carved surfaces, functional ware.",
        "",
        "Mennatallah Rihan",
        "menna.rihan@outlook.com",
    ]
    c.setFont("SourceSansLight", 10)
    opaque(c)
    c.setFillColorRGB(*INK)
    ty = card_h / 2 - 1.55 * inch
    for line in body:
        c.drawCentredString(0, ty, line)
        ty -= 15
    c.restoreState()
    footer_number(c, page_no)
    c.showPage()


# New same-setup photographs (unique compositions; filesize duplicates omitted).
NEW_SHOTS = [
    ("01a08bd1-6216-7e26-b0ee-2f3f08b558d0.jpg", "pour-over-stacked.jpg"),
    ("01a08bd1-61ae-716a-9f49-ef8fb126c8dd.jpg", "pour-over-pair.jpg"),
    ("01a08bd1-6b34-7a5f-8e3d-3b96ea6319be.jpg", "vase-swirl.jpg"),
    ("01a08bd1-735c-758f-8b12-001c13744919.jpg", "bowl-oxblood.jpg"),
    ("01a08bd1-7c2b-71f1-ade4-0edc9ffc15b3.jpg", "dish-marbled.jpg"),
    ("01a08bd1-852a-742f-833f-5425cecc6fc0.jpg", "bowl-and-dish.jpg"),
    ("01a08bd1-8ded-70e2-a003-1898cd3e0dbb.jpg", "cup-moss.jpg"),
    ("01a08bd1-9a6b-797f-97ad-4560a300df34.jpg", "bowl-midnight-exterior.jpg"),
    ("01a08bd1-a35d-7ce2-ae06-942e9e0beb59.jpg", "pitcher-green.jpg"),
    ("1BE249BA-1C01-4403-8315-792701EF06D3_L0_001.jpg", "cellar-frog-overhead.jpg"),
    ("67A3EA6C-1C10-42BF-99CE-51E189E28829_L0_001.jpg", "cellar-frog.jpg"),
    ("D3551C23-8EBF-40A6-9861-349A4B27B257_L0_001.jpg", "wheat-pair.jpg"),
    ("DB9E5187-A6FE-4FF2-975A-EDBC5F5C7631_L0_001.jpg", "tumbler-carved-dark.jpg"),
    ("EEE14675-9BF5-4AE5-862F-CECAC228A6FE_L0_001.jpg", "vase-low-teal.jpg"),
]

# Pieces with no new photograph — original background, tight-cropped to the ware.
LEFTOVER = [
    ("01a07eda-fd53-70c0-94aa-a4495d9181cf.jpg", "collection-earthtones.jpg"),
    ("01a07eda-fd0e-7f48-9626-60511b116192.jpg", "jar-lidded-amber.jpg"),
    ("01a07eda-fcd7-7d54-9129-d8fe87ddc3cd.jpg", "mug-plate-in-use.jpg"),
    ("01a07edc-a16a-73c5-86af-b0cbb4abcdf0.jpg", "plate-reactive.jpg"),
    ("01a07eda-fd97-7b70-900f-4a3d32dcd43f.jpg", "yunomi-exterior.jpg"),
    ("01a07eda-fd6a-7fe2-85e9-9e08a8a8b74e.jpg", "yunomi-interior.jpg"),
    ("01a07edc-9d41-7de5-8e35-c7f89c6637d2.jpg", "bowl-midnight-interior.jpg"),
    ("01a07edc-9ea3-7d94-929d-ae44ee034ccf.jpg", "bowl-sculptural-interior.jpg"),
    ("01a07edc-9ebb-7504-bfc5-0b1b6b98a51e.jpg", "bowl-sculptural-side.jpg"),
    ("01a07edc-9f71-775b-927e-8139bf7f4578.jpg", "dish-spiral-black.jpg"),
    ("01a07edc-a017-7efc-9c2b-43ffd6965e02.jpg", "grater-garlic.jpg"),
    ("01a07edc-a1ee-74c9-b95a-68ebd10caf50.jpg", "process-carved-spiral.jpg"),
    ("01a07edc-a086-7a9b-a407-c29e3040c04c.jpg", "dish-lidded-heart.jpg"),
    ("01a07edc-a0f9-7f55-8a29-73c726bc5e5a.jpg", "bowl-folded-rim.jpg"),
]

# Old photos replaced by new studio shots — do not keep in photos/.
REPLACED = {
    "cylinder-carved-wheat.jpg",
    "pour-over-set.jpg",
    "vase-blue-tan.jpg",
}


def prepare_all() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {dest for _, dest in NEW_SHOTS} | {dest for _, dest in LEFTOVER}

    for src_name, dest_name in NEW_SHOTS + LEFTOVER:
        src = SOURCE_DIR / src_name
        if src.exists():
            prepare_photo(src, PHOTO_DIR / dest_name)
            print(f"prepared {dest_name}")
            continue
        raw = RAW_DIR / dest_name
        if raw.exists():
            prepare_photo(raw, PHOTO_DIR / dest_name)
            print(f"prepared {dest_name} from raw")
            continue
        existing = PHOTO_DIR / dest_name
        if existing.exists():
            print(f"using existing {dest_name}")
            continue
        raise FileNotFoundError(src)

    for old in list(PHOTO_DIR.glob("*.jpg")):
        if old.name not in wanted:
            old.unlink()
            print(f"removed replaced {old.name}")


def build_pdf() -> None:
    make_paper(PAPER_PATH)
    register_fonts()
    c = canvas.Canvas(str(OUTPUT), pagesize=PAGE)
    c.setTitle("Ceramics Portfolio — Mennatallah Rihan")
    c.setAuthor("Mennatallah Rihan")
    c.setSubject("A scrapbook of recent pottery work")
    c.setCreator("ceramics-portfolio/build_portfolio.py")

    cover_page(c, "collection-earthtones.jpg")

    n = 1
    work_single(
        c,
        "wheat-pair.jpg",
        "Wheat pair",
        "Carved cylinder and matching letter holder; tan rim over pooled chocolate",
        n,
        "cylinder and holder",
        angle=1.0,
    )
    n += 1
    work_single(
        c,
        "tumbler-carved-dark.jpg",
        "Carved tumbler",
        "Dark iron glaze over wheat-stalk relief; unglazed foot",
        n,
        "tumbler",
        angle=-1.3,
    )
    n += 1
    work_pair(
        c,
        "jar-lidded-amber.jpg",
        "dish-lidded-heart.jpg",
        "Lidded ware",
        "Amber jar with knob lid; speckled cream dish with a heart finial",
        n,
        "amber jar",
        "heart dish",
    )
    n += 1
    work_pair(
        c,
        "pour-over-stacked.jpg",
        "pour-over-pair.jpg",
        "Pour-over set",
        "Conical dripper and mug; reactive teal, cobalt, and copper",
        n,
        "stacked",
        "side by side",
    )
    n += 1
    work_pair(
        c,
        "mug-plate-in-use.jpg",
        "plate-reactive.jpg",
        "Mug and plate",
        "Matching set in use, and the plate’s cream, indigo, and bronze well",
        n,
        "in use",
        "the plate",
    )
    n += 1
    work_single(
        c,
        "bowl-and-dish.jpg",
        "Marbled set",
        "Small bowl and shallow plate together; oxblood, cyan, and cream",
        n,
        "bowl and plate",
        angle=1.1,
    )
    n += 1
    work_pair(
        c,
        "dish-marbled.jpg",
        "bowl-oxblood.jpg",
        "Marbled wells",
        "The same plate and bowl from above",
        n,
        "plate",
        "bowl",
    )
    n += 1
    work_pair(
        c,
        "vase-swirl.jpg",
        "vase-low-teal.jpg",
        "Bud vases",
        "Swirled blue-and-tan marble; low form in teal with rust flashing",
        n,
        "swirl",
        "low vase",
    )
    n += 1
    work_pair(
        c,
        "yunomi-exterior.jpg",
        "yunomi-interior.jpg",
        "Yunomi",
        "Olive-tan drip over cerulean; interior pooling at the well",
        n,
        "exterior",
        "interior",
    )
    n += 1
    work_pair(
        c,
        "bowl-midnight-exterior.jpg",
        "bowl-midnight-interior.jpg",
        "Midnight bowl",
        "Pale-blue wave over a dark body; glossy navy well",
        n,
        "exterior",
        "interior",
    )
    n += 1
    work_pair(
        c,
        "pitcher-green.jpg",
        "cup-moss.jpg",
        "Green ware",
        "Handled pitcher with iron streaks, and a small moss-green cup",
        n,
        "pitcher",
        "cup",
    )
    n += 1
    work_pair(
        c,
        "bowl-sculptural-interior.jpg",
        "bowl-sculptural-side.jpg",
        "Sculptural bowl",
        "Leaf-like rim attachments; spiral well and metallic body",
        n,
        "interior",
        "side",
    )
    n += 1
    work_pair(
        c,
        "cellar-frog-overhead.jpg",
        "cellar-frog.jpg",
        "Frog salt cellar",
        "Pinched cellar with painted eyes, salt, and a small spoon",
        n,
        "from above",
        "three-quarter",
    )
    n += 1
    work_pair(
        c,
        "dish-spiral-black.jpg",
        "grater-garlic.jpg",
        "Spiral black",
        "Palm dish with a continuous spiral; garlic grater with rim holes",
        n,
        "dish",
        "grater",
    )
    n += 1
    work_single(
        c,
        "bowl-folded-rim.jpg",
        "Folded-rim bowl",
        "Pinched tri-lobe rim; cobalt, periwinkle, and sage interior",
        n,
        "folded rim",
        angle=1.4,
    )
    n += 1
    work_single(
        c,
        "process-carved-spiral.jpg",
        "On the wheel",
        "Leather-hard carving: radiating teardrop texture before firing",
        n,
        "leather-hard",
        angle=-0.8,
    )
    n += 1
    colophon_page(c, n)

    c.save()
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    prepare_all()
    build_pdf()
