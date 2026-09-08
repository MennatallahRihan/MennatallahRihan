#!/usr/bin/env python3
"""Build a print-ready ceramics portfolio PDF from unaltered photographs.

Photographs are orientation-corrected (EXIF) and scaled to fit the page.
Pixels of the pottery itself are not retouched, filtered, or composited.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
PHOTO_DIR = ROOT / "photos"
FONT_DIR = ROOT / "fonts"
OUTPUT = ROOT / "Mennatallah_Rihan_Ceramics_Portfolio.pdf"

# Original uploads (outside the repo). Copied into photos/ at build time.
SOURCE_DIR = Path("/home/ubuntu/.cursor/projects/workspace/assets")

PAGE = letter  # 8.5 x 11 in
PAGE_W, PAGE_H = PAGE

# Gallery palette
CREAM = (0.965, 0.945, 0.910)  # #F6F1E8
INK = (0.165, 0.149, 0.133)  # #2A2622
MUTED = (0.478, 0.447, 0.408)  # #7A7268
RULE = (0.788, 0.733, 0.659)  # #C9BBA8
TEAL = (0.239, 0.353, 0.329)  # #3D5A54

MAX_PHOTO_EDGE = 2200  # px, enough for letter at ~250 dpi


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Cormorant", str(FONT_DIR / "CormorantGaramond-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("CormorantSemibold", str(FONT_DIR / "CormorantGaramond-SemiBold.ttf")))
    pdfmetrics.registerFont(TTFont("CormorantItalic", str(FONT_DIR / "CormorantGaramond-Italic.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSans", str(FONT_DIR / "SourceSans3-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("SourceSansLight", str(FONT_DIR / "SourceSans3-Light.ttf")))


def prepare_photo(src: Path, dest: Path) -> None:
    """Copy a photo with EXIF orientation applied. No other pixel edits."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = im.size
    scale = min(1.0, MAX_PHOTO_EDGE / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    im.save(dest, format="JPEG", quality=90, optimize=True, subsampling=0)


def photo_size(name: str) -> tuple[int, int]:
    with Image.open(PHOTO_DIR / name) as im:
        return im.size


def paint_page(c: canvas.Canvas) -> None:
    c.setFillColorRGB(*CREAM)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)


def draw_rule(c: canvas.Canvas, x: float, y: float, w: float) -> None:
    c.setStrokeColorRGB(*RULE)
    c.setLineWidth(0.6)
    c.line(x, y, x + w, y)


def fit_box(im_w: int, im_h: int, box_w: float, box_h: float) -> tuple[float, float]:
    scale = min(box_w / im_w, box_h / im_h)
    return im_w * scale, im_h * scale


def draw_photo(
    c: canvas.Canvas,
    name: str,
    x: float,
    y: float,
    box_w: float,
    box_h: float,
    *,
    valign: str = "center",
) -> tuple[float, float, float, float]:
    """Place a photo inside a box, preserving aspect ratio. Returns the drawn rect."""
    im_w, im_h = photo_size(name)
    dw, dh = fit_box(im_w, im_h, box_w, box_h)
    dx = x + (box_w - dw) / 2
    if valign == "top":
        dy = y + box_h - dh
    elif valign == "bottom":
        dy = y
    else:
        dy = y + (box_h - dh) / 2
    path = str(PHOTO_DIR / name)
    c.drawImage(
        path,
        dx,
        dy,
        width=dw,
        height=dh,
        preserveAspectRatio=True,
    )
    return dx, dy, dw, dh


def caption_block(
    c: canvas.Canvas,
    title: str,
    subtitle: str,
    y: float,
    *,
    page_label: str = "",
) -> None:
    left = 0.7 * inch
    c.setFillColorRGB(*INK)
    c.setFont("CormorantSemibold", 16)
    c.drawString(left, y, title)
    c.setFillColorRGB(*MUTED)
    c.setFont("SourceSansLight", 9)
    c.drawString(left, y - 14, subtitle)
    if page_label:
        c.setFont("SourceSansLight", 8)
        c.setFillColorRGB(*MUTED)
        c.drawRightString(PAGE_W - 0.7 * inch, 0.42 * inch, page_label)


def footer_number(c: canvas.Canvas, n: int) -> None:
    c.setFillColorRGB(*MUTED)
    c.setFont("SourceSansLight", 8)
    c.drawCentredString(PAGE_W / 2, 0.38 * inch, str(n))


# --- page builders -----------------------------------------------------------

MARGIN = 0.65 * inch
CAPTION_H = 0.72 * inch


def work_page_single(
    c: canvas.Canvas,
    photo: str,
    title: str,
    subtitle: str,
    page_no: int,
) -> None:
    paint_page(c)
    box_x = MARGIN
    box_y = MARGIN + CAPTION_H
    box_w = PAGE_W - 2 * MARGIN
    box_h = PAGE_H - MARGIN - CAPTION_H - 0.55 * inch
    draw_photo(c, photo, box_x, box_y, box_w, box_h, valign="center")
    draw_rule(c, MARGIN, MARGIN + 0.52 * inch, PAGE_W - 2 * MARGIN)
    caption_block(c, title, subtitle, MARGIN + 0.28 * inch)
    footer_number(c, page_no)
    c.showPage()


def work_page_pair(
    c: canvas.Canvas,
    left: str,
    right: str,
    title: str,
    subtitle: str,
    page_no: int,
    left_note: str = "",
    right_note: str = "",
) -> None:
    paint_page(c)
    gap = 0.18 * inch
    box_y = MARGIN + CAPTION_H + 0.18 * inch
    box_h = PAGE_H - MARGIN - CAPTION_H - 0.7 * inch
    col_w = (PAGE_W - 2 * MARGIN - gap) / 2
    draw_photo(c, left, MARGIN, box_y, col_w, box_h, valign="center")
    draw_photo(c, right, MARGIN + col_w + gap, box_y, col_w, box_h, valign="center")
    if left_note or right_note:
        c.setFillColorRGB(*MUTED)
        c.setFont("SourceSansLight", 8)
        c.drawString(MARGIN, MARGIN + CAPTION_H + 0.02 * inch, left_note)
        c.drawRightString(PAGE_W - MARGIN, MARGIN + CAPTION_H + 0.02 * inch, right_note)
    draw_rule(c, MARGIN, MARGIN + 0.52 * inch, PAGE_W - 2 * MARGIN)
    caption_block(c, title, subtitle, MARGIN + 0.28 * inch)
    footer_number(c, page_no)
    c.showPage()


def cover_page(c: canvas.Canvas, hero: str) -> None:
    paint_page(c)
    # Header
    c.setFillColorRGB(*TEAL)
    c.setFont("SourceSans", 8.5)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 0.72 * inch, "MENNATALLAH RIHAN")
    c.setFillColorRGB(*INK)
    c.setFont("Cormorant", 42)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.32 * inch, "Ceramics")
    c.setFillColorRGB(*MUTED)
    c.setFont("CormorantItalic", 13)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 1.62 * inch, "A portfolio of recent work")
    draw_rule(c, PAGE_W / 2 - 0.9 * inch, PAGE_H - 1.82 * inch, 1.8 * inch)

    box_x = MARGIN
    box_y = 1.15 * inch
    box_w = PAGE_W - 2 * MARGIN
    box_h = PAGE_H - 3.15 * inch
    draw_photo(c, hero, box_x, box_y, box_w, box_h, valign="center")

    c.setFillColorRGB(*MUTED)
    c.setFont("SourceSansLight", 8.5)
    c.drawCentredString(PAGE_W / 2, 0.72 * inch, "Hand-thrown and hand-built stoneware")
    c.drawCentredString(PAGE_W / 2, 0.56 * inch, "Photographs unaltered  ·  objects shown as fired")
    c.showPage()


def colophon_page(c: canvas.Canvas, page_no: int) -> None:
    paint_page(c)
    y = PAGE_H - 2.1 * inch
    c.setFillColorRGB(*TEAL)
    c.setFont("SourceSans", 8.5)
    c.drawCentredString(PAGE_W / 2, y, "NOTES")
    c.setFillColorRGB(*INK)
    c.setFont("Cormorant", 28)
    c.drawCentredString(PAGE_W / 2, y - 0.5 * inch, "About this portfolio")

    body = [
        "These photographs are the artist's own records of finished and in-process work.",
        "Each object is shown unaltered: no filters, retouching, or background removal.",
        "Where several frames showed the same piece, the clearest detail views were kept",
        "and weaker duplicates were set aside.",
        "",
        "Work includes wheel-thrown and hand-built forms, reactive and layered glazes,",
        "carved and sculptural surfaces, and functional ware.",
        "",
        "Mennatallah Rihan",
        "menna.rihan@outlook.com",
    ]
    c.setFillColorRGB(*INK)
    c.setFont("SourceSansLight", 11)
    ty = y - 1.15 * inch
    for line in body:
        c.drawCentredString(PAGE_W / 2, ty, line)
        ty -= 16

    footer_number(c, page_no)
    c.showPage()


# Selected frames. Duplicates omitted:
#   01a07eda-fcf5-…  same reactive plate as plate-reactive.jpg, cluttered kitchen
#   01a07edc-9d59-…  same midnight bowl exterior as bowl-midnight-exterior.jpg
#   01a07edc-9ed3-…  same sculptural bowl side as bowl-sculptural-side.jpg
SELECTION = [
    ("01a07eda-fd53-70c0-94aa-a4495d9181cf.jpg", "collection-earthtones.jpg"),
    ("01a07eda-fd24-7299-9e2d-262a19f48d3e.jpg", "cylinder-carved-wheat.jpg"),
    ("01a07eda-fd0e-7f48-9626-60511b116192.jpg", "jar-lidded-amber.jpg"),
    ("01a07eda-fd3c-706c-82ec-1eb5da58d16f.jpg", "pour-over-set.jpg"),
    ("01a07eda-fcd7-7d54-9129-d8fe87ddc3cd.jpg", "mug-plate-in-use.jpg"),
    ("01a07edc-a16a-73c5-86af-b0cbb4abcdf0.jpg", "plate-reactive.jpg"),
    ("01a07eda-fd81-7c0a-832d-0f16787949b6.jpg", "dish-marbled.jpg"),
    ("01a07eda-fe48-7c51-a961-90ec059f5ddc.jpg", "vase-blue-tan.jpg"),
    ("01a07eda-fd97-7b70-900f-4a3d32dcd43f.jpg", "yunomi-exterior.jpg"),
    ("01a07eda-fd6a-7fe2-85e9-9e08a8a8b74e.jpg", "yunomi-interior.jpg"),
    ("01a07edc-9d41-7de5-8e35-c7f89c6637d2.jpg", "bowl-midnight-interior.jpg"),
    ("01a07edc-9dea-7418-878c-02266cc97cda.jpg", "bowl-midnight-exterior.jpg"),
    ("01a07edc-9e88-749d-b0a4-5953bdc1998d.jpg", "pitcher-green.jpg"),
    ("01a07edc-9ea3-7d94-929d-ae44ee034ccf.jpg", "bowl-sculptural-interior.jpg"),
    ("01a07edc-9ebb-7504-bfc5-0b1b6b98a51e.jpg", "bowl-sculptural-side.jpg"),
    ("01a07edc-9f71-775b-927e-8139bf7f4578.jpg", "dish-spiral-black.jpg"),
    ("01a07edc-a017-7efc-9c2b-43ffd6965e02.jpg", "grater-garlic.jpg"),
    ("01a07edc-a1ee-74c9-b95a-68ebd10caf50.jpg", "process-carved-spiral.jpg"),
    ("01a07edc-a086-7a9b-a407-c29e3040c04c.jpg", "dish-lidded-heart.jpg"),
    ("01a07edc-a0f9-7f55-8a29-73c726bc5e5a.jpg", "bowl-folded-rim.jpg"),
]


def prepare_all() -> None:
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_DIR.exists():
        missing = [dest for _, dest in SELECTION if not (PHOTO_DIR / dest).exists()]
        if missing:
            raise FileNotFoundError(
                f"Source photos not found and missing from photos/: {missing}"
            )
        print("using existing photos/")
        return
    for src_name, dest_name in SELECTION:
        src = SOURCE_DIR / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        prepare_photo(src, PHOTO_DIR / dest_name)
        print(f"prepared {dest_name}")


def build_pdf() -> None:
    register_fonts()
    c = canvas.Canvas(str(OUTPUT), pagesize=PAGE)
    c.setTitle("Ceramics Portfolio — Mennatallah Rihan")
    c.setAuthor("Mennatallah Rihan")
    c.setSubject("A portfolio of recent pottery work")
    c.setCreator("ceramics-portfolio/build_portfolio.py")

    cover_page(c, "mug-plate-in-use.jpg")

    n = 1
    work_page_single(
        c,
        "collection-earthtones.jpg",
        "Earth-tone collection",
        "Six related forms in tan, amber, and chocolate glazes",
        n,
    )
    n += 1
    work_page_single(
        c,
        "cylinder-carved-wheat.jpg",
        "Carved cylinder",
        "Tall cup or vase with wheat-stalk carving; tan rim over pooled chocolate glaze",
        n,
    )
    n += 1
    work_page_single(
        c,
        "jar-lidded-amber.jpg",
        "Lidded jar",
        "Rounded jar with knob lid; speckled amber shoulder over dark brown",
        n,
    )
    n += 1
    work_page_single(
        c,
        "pour-over-set.jpg",
        "Pour-over set",
        "Conical dripper on a matching mug; reactive teal, cobalt, and copper",
        n,
    )
    n += 1
    work_page_single(
        c,
        "mug-plate-in-use.jpg",
        "Mug and plate",
        "Matching set in use; sage, teal, and cream reactive glaze",
        n,
    )
    n += 1
    work_page_single(
        c,
        "plate-reactive.jpg",
        "Reactive glaze plate",
        "Palm-sized dish; cream field meeting indigo, teal, and bronze",
        n,
    )
    n += 1
    work_page_single(
        c,
        "dish-marbled.jpg",
        "Marbled dish",
        "Shallow plate; oxblood, cyan, and cream layered in the firing",
        n,
    )
    n += 1
    work_page_single(
        c,
        "vase-blue-tan.jpg",
        "Bud vase",
        "Small bulbous form; sky-blue and tan marble with rust movement",
        n,
    )
    n += 1
    work_page_pair(
        c,
        "yunomi-exterior.jpg",
        "yunomi-interior.jpg",
        "Yunomi",
        "Olive-tan drip over cerulean; interior pooling at the well",
        n,
        "Exterior",
        "Interior",
    )
    n += 1
    work_page_pair(
        c,
        "bowl-midnight-interior.jpg",
        "bowl-midnight-exterior.jpg",
        "Midnight bowl",
        "Glossy navy interior; exterior wave of pale blue over an unglazed foot",
        n,
        "Interior",
        "Exterior",
    )
    n += 1
    work_page_single(
        c,
        "pitcher-green.jpg",
        "Pitcher",
        "Handled pouring form; moss and forest-green glaze with iron streaks",
        n,
    )
    n += 1
    work_page_pair(
        c,
        "bowl-sculptural-interior.jpg",
        "bowl-sculptural-side.jpg",
        "Sculptural bowl",
        "Leaf-like rim attachments; spiral well, metallic body, dark green glaze",
        n,
        "Interior",
        "Side",
    )
    n += 1
    work_page_single(
        c,
        "dish-spiral-black.jpg",
        "Spiral dish",
        "Palm-sized black glaze with a continuous interior spiral",
        n,
    )
    n += 1
    work_page_single(
        c,
        "grater-garlic.jpg",
        "Garlic grater",
        "Black glazed grater with raised spiral teeth and three rim holes",
        n,
    )
    n += 1
    work_page_single(
        c,
        "process-carved-spiral.jpg",
        "On the wheel",
        "Leather-hard carving: radiating teardrop texture before firing",
        n,
    )
    n += 1
    work_page_single(
        c,
        "dish-lidded-heart.jpg",
        "Lidded dish",
        "Speckled cream glaze with a red heart finial",
        n,
    )
    n += 1
    work_page_single(
        c,
        "bowl-folded-rim.jpg",
        "Folded-rim bowl",
        "Pinched tri-lobe rim; cobalt, periwinkle, and sage interior",
        n,
    )
    n += 1
    colophon_page(c, n)

    c.save()
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    prepare_all()
    build_pdf()
