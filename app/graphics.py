"""Render intro plate, titles, logo bug, promo lower third, and end card at any frame size.

Reused from the FreeHoops Promo Tool v1.3 (same look and defaults), plus the
two-line highlights title and the corner logo used by the Highlights Tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ORANGE_BAR = (232, 90, 18, 255)
ORANGE_TEXT = (255, 140, 40, 255)
INTRO_BG = (8, 10, 16, 255)
END_BG = (12, 14, 22, 255)
BANNER_BG = (10, 12, 18, 200)
TITLE_PLAQUE = (8, 10, 16, 190)
WHITE = (255, 255, 255, 255)
CAPTION_GRAY = (200, 200, 200, 255)
SHADOW = (0, 0, 0, 220)

LT_TITLE = "FreeHoops.net"
LT_SUB = "Get FreeHoops Mobile on Google Play"
END_TITLE = "Visit FreeHoops.net"
END_SUB = "Get the FreeHoops app on Google Play"
QR_CAPTION = "Scan for Google Play"
DEFAULT_QR_URL = "https://play.google.com/store/apps/details?id=com.rjpablo.freehoops"


def _load_font(fonts_dir: Path, bold: bool, size: int):
    from PIL import ImageFont

    size = max(8, int(round(size)))
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = fonts_dir / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    fallbacks = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for fb in fallbacks:
        if fb.exists():
            return ImageFont.truetype(str(fb), size)
    return ImageFont.load_default()


def _text_ink_size(font, text: str) -> tuple[int, int]:
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (4000, 800), 0)
    ImageDraw.Draw(canvas).text((20, 20), text, font=font, fill=255)
    bbox = canvas.getbbox()
    if not bbox:
        return (0, 0)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_text(im, xy, text, font, fill, anchor: str = "lt"):
    from PIL import Image, ImageDraw

    x, y = xy
    layer = Image.new("L", (4000, 800), 0)
    ImageDraw.Draw(layer).text((40, 40), text, font=font, fill=255)
    bbox = layer.getbbox()
    if not bbox:
        return
    crop = layer.crop(bbox)
    tw, th = crop.size
    if "m" in anchor:
        x = int(round(x - tw / 2))
    elif "r" in anchor:
        x = int(round(x - tw))
    if anchor.endswith("b"):
        y = int(round(y - th))
    colored = Image.new("RGBA", crop.size, fill)
    colored.putalpha(crop)
    im.alpha_composite(colored, (int(x), int(y)))


def _scale_logo(logo, width: int, height: int):
    from PIL import Image

    width = max(1, int(width))
    height = max(1, int(height))
    return logo.resize((width, height), Image.Resampling.LANCZOS)


def _scale_qr(qr, size: int):
    from PIL import Image

    size = max(32, int(size))
    return qr.resize((size, size), Image.Resampling.NEAREST)


def make_qr_image(url: str, size: int):
    """Render a crisp QR code for any URL (bundled Nayuki generator, no extra deps)."""
    from PIL import Image
    from qrcodegen import QrCode

    text = (url or DEFAULT_QR_URL).strip() or DEFAULT_QR_URL
    qr = QrCode.encode_text(text, QrCode.Ecc.MEDIUM)
    n = qr.get_size()
    border = 3
    dim = n + 2 * border
    img = Image.new("RGBA", (dim, dim), (255, 255, 255, 255))
    px = img.load()
    for y in range(n):
        for x in range(n):
            if qr.get_module(x, y):
                px[x + border, y + border] = (0, 0, 0, 255)
    size = max(32, int(size))
    return img.resize((size, size), Image.Resampling.NEAREST)


@dataclass
class Layout:
    width: int
    height: int
    scale: float
    margin: int
    margin_top: int
    margin_right: int
    margin_bottom: int
    bar: int
    is_portrait: bool
    is_square: bool


def make_layout(width: int, height: int) -> Layout:
    width, height = int(width), int(height)
    short = min(width, height)
    scale = short / 1080.0
    is_portrait = height >= width * 1.2
    is_square = (not is_portrait) and width < height * 1.2
    margin = max(16, int(round(60 * scale)))
    margin_top = max(12, int(round(30 * scale)))
    margin_right = max(12, int(round(40 * scale)))
    margin_bottom = max(16, int(round(70 * scale)))
    if is_portrait:
        margin_bottom = max(margin_bottom, int(round(0.07 * height)))
    bar = max(4, int(round(8 * height / 1080.0)))
    return Layout(
        width=width,
        height=height,
        scale=scale,
        margin=margin,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
        bar=bar,
        is_portrait=is_portrait,
        is_square=is_square,
    )


def wrap_text(text: str, font, max_width: int) -> list[str]:
    """Wrap on spaces, honoring existing newlines. Long words stay on their own line."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    lines: list[str] = []
    for para in text.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            if _text_ink_size(font, trial)[0] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def render_intro_plate(width: int, height: int):
    from PIL import Image, ImageDraw

    bar = max(4, int(round(7 * height / 1080.0)))
    im = Image.new("RGBA", (width, height), INTRO_BG)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, width, bar - 1), fill=ORANGE_BAR)
    draw.rectangle((0, height - bar, width, height), fill=ORANGE_BAR)
    return im


def render_title(width: int, height: int, text: str, fonts_dir: Path):
    """Opening title over the start of the source footage, sized for any frame."""
    from PIL import Image, ImageDraw

    layout = make_layout(width, height)
    s = layout.scale
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    text = (text or "").strip()
    if not text:
        return im

    # Keep a generous safe area, a bit more on portrait so wrapped lines stay readable.
    # Also leave room for the top-right logo if a promo card is on at the same time.
    side = max(layout.margin, int(round((0.08 if layout.is_portrait else 0.06) * width)))
    logo_h = max(48, int(round(110 * s)))
    logo_w = int(round(logo_h * 572 / 500))  # matches the corner logo aspect
    logo_clear = logo_w + layout.margin_right + max(8, int(round(16 * s)))
    side = max(side, logo_clear)
    max_w = max(64, width - 2 * side)
    max_h = int(height * (0.42 if layout.is_portrait else 0.36))

    font_size = 72 * s
    if layout.is_portrait or layout.is_square:
        font_size = 64 * s

    font = None
    lines: list[str] = []
    line_sizes: list[tuple[int, int]] = []
    gap = 0
    block_w = block_h = 0
    for _ in range(12):
        font = _load_font(fonts_dir, True, font_size)
        lines = wrap_text(text, font, max_w)
        if not lines:
            return im
        # If a single word is wider than the safe area, shrink and try again.
        too_wide = any(_text_ink_size(font, ln)[0] > max_w for ln in lines if ln)
        line_sizes = [_text_ink_size(font, ln or " ") for ln in lines]
        gap = max(6, int(round(font_size * 0.22)))
        block_w = max((w for w, h in line_sizes), default=0)
        block_h = sum(h for w, h in line_sizes) + gap * (len(lines) - 1)
        if (not too_wide) and block_h <= max_h:
            break
        font_size = max(16, font_size * 0.88)
    assert font is not None

    pad_x = max(18, int(round(36 * s)))
    pad_y = max(12, int(round(22 * s)))
    radius = max(8, int(round(16 * s)))
    plaque_w = min(width - 2 * side, block_w + 2 * pad_x)
    plaque_h = block_h + 2 * pad_y
    # Sit in the upper-middle so it stays clear of a bottom promo card.
    plaque_x = (width - plaque_w) // 2
    plaque_y = max(layout.margin_top + int(round(40 * s)), int(height * 0.18) - plaque_h // 2)
    plaque_y = min(plaque_y, height - layout.margin_bottom - plaque_h)

    draw = ImageDraw.Draw(im)
    accent = max(4, int(round(7 * s)))
    draw.rounded_rectangle(
        (plaque_x, plaque_y, plaque_x + plaque_w - 1, plaque_y + plaque_h - 1),
        radius=radius,
        fill=TITLE_PLAQUE,
    )
    # Orange bar along the top of the plaque, inset for the rounded corners.
    draw.rectangle(
        (plaque_x + radius, plaque_y, plaque_x + plaque_w - 1 - radius, plaque_y + accent - 1),
        fill=ORANGE_BAR,
    )

    shadow_off = max(1, int(round(3 * s)))
    y = plaque_y + pad_y
    for line, (lw, lh) in zip(lines, line_sizes):
        cx = plaque_x + plaque_w // 2
        _draw_text(im, (cx + shadow_off, y + shadow_off), line, font, SHADOW, anchor="mt")
        _draw_text(im, (cx, y), line, font, WHITE, anchor="mt")
        y += lh + gap
    return im


def render_lower_third(
    width: int,
    height: int,
    logo,
    fonts_dir: Path,
    line1: str = LT_TITLE,
    line2: str = LT_SUB,
    include_logo: bool = True,
):
    from PIL import Image, ImageDraw

    layout = make_layout(width, height)
    s = layout.scale
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    line1 = (line1 or "").strip() or LT_TITLE
    line2 = (line2 or "").strip() or LT_SUB

    logo_h = max(48, int(round(110 * s)))
    logo_w = int(round(logo_h * (logo.width / logo.height)))
    max_logo_w = max(32, width - layout.margin - layout.margin_right)
    if logo_w > max_logo_w:
        logo_w = max_logo_w
        logo_h = int(round(logo_w * (logo.height / logo.width)))
    logo_img = _scale_logo(logo, logo_w, logo_h)
    logo_x = width - layout.margin_right - logo_w
    logo_y = layout.margin_top
    if include_logo:
        im.alpha_composite(logo_img, (logo_x, logo_y))

    title_font = _load_font(fonts_dir, True, 32 * s)
    sub_font = _load_font(fonts_dir, False, 22 * s)
    title_w, title_h = _text_ink_size(title_font, line1)
    sub_w, sub_h = _text_ink_size(sub_font, line2)
    accent = max(4, int(round(7 * s)))
    pad_left = max(12, int(round(20 * s)))
    pad_right = max(16, int(round(28 * s)))
    text_w = max(title_w, sub_w)
    needed = accent + pad_left + text_w + pad_right
    avail = width - layout.margin - layout.margin_right
    if needed > avail:
        shrink = avail / needed
        title_font = _load_font(fonts_dir, True, max(10, 32 * s * shrink))
        sub_font = _load_font(fonts_dir, False, max(8, 22 * s * shrink))
        title_w, title_h = _text_ink_size(title_font, line1)
        sub_w, sub_h = _text_ink_size(sub_font, line2)
        text_w = max(title_w, sub_w)
        needed = accent + pad_left + text_w + pad_right

    banner_h = max(title_h + sub_h + max(16, int(round(24 * s))), int(round(91 * s)))
    banner_w = min(avail, max(needed, int(round(721 * s))))
    radius = max(6, int(round(12 * s)))
    banner_x = layout.margin
    banner_y = height - layout.margin_bottom - banner_h
    logo_bottom = logo_y + logo_h + max(8, int(round(12 * s)))
    if banner_y < logo_bottom:
        banner_y = logo_bottom

    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle(
        (banner_x, banner_y, banner_x + banner_w - 1, banner_y + banner_h - 1),
        radius=radius,
        fill=BANNER_BG,
    )
    inset = min(radius, max(6, int(round(8 * s))))
    ax = banner_x
    draw.rectangle(
        (ax, banner_y + inset, ax + accent - 1, banner_y + banner_h - 1 - inset),
        fill=ORANGE_BAR,
    )

    text_x = banner_x + accent + pad_left
    gap = max(4, int(round(5 * s)))
    block_h = title_h + gap + sub_h
    text_y = banner_y + (banner_h - block_h) // 2
    _draw_text(im, (text_x, text_y), line1, title_font, ORANGE_TEXT)
    _draw_text(im, (text_x, text_y + title_h + gap), line2, sub_font, WHITE)
    return im


def render_endcard(
    width: int,
    height: int,
    logo,
    qr,
    fonts_dir: Path,
    title: str = END_TITLE,
    sub: str = END_SUB,
    caption: str = QR_CAPTION,
):
    from PIL import Image, ImageDraw

    layout = make_layout(width, height)
    s = layout.scale
    im = Image.new("RGBA", (width, height), END_BG)
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, width, layout.bar - 1), fill=ORANGE_BAR)
    draw.rectangle((0, height - layout.bar, width, height), fill=ORANGE_BAR)
    title = (title or "").strip() or END_TITLE
    sub = (sub or "").strip() or END_SUB
    caption = (caption or "").strip() or QR_CAPTION

    logo_w = max(80, int(round(480 * s)))
    logo_h = int(round(logo_w * (logo.height / logo.width)))
    max_logo_h = int(height * (0.32 if layout.is_portrait else 0.42))
    if logo_h > max_logo_h:
        logo_h = max_logo_h
        logo_w = int(round(logo_h * (logo.width / logo.height)))
    logo_img = _scale_logo(logo, logo_w, logo_h)

    title_font = _load_font(fonts_dir, True, 64 * s)
    sub_font = _load_font(fonts_dir, False, 36 * s)
    cap_font = _load_font(fonts_dir, False, 16 * s)
    title_w, title_h = _text_ink_size(title_font, title)
    sub_w, sub_h = _text_ink_size(sub_font, sub)
    cap_w, cap_h = _text_ink_size(cap_font, caption)

    max_text = width - 2 * layout.margin
    if title_w > max_text or sub_w > max_text:
        shrink = max_text / max(title_w, sub_w)
        title_font = _load_font(fonts_dir, True, max(12, 64 * s * shrink))
        sub_font = _load_font(fonts_dir, False, max(10, 36 * s * shrink))
        cap_font = _load_font(fonts_dir, False, max(8, 16 * s * shrink))
        title_w, title_h = _text_ink_size(title_font, title)
        sub_w, sub_h = _text_ink_size(sub_font, sub)
        cap_w, cap_h = _text_ink_size(cap_font, caption)

    qr_size = max(80, int(round(300 * s)))
    qr_max = min(
        int(width * 0.4),
        int(height * 0.28),
        width - 2 * layout.margin,
    )
    if qr_size > qr_max:
        qr_size = max(64, qr_max)
    qr_img = _scale_qr(qr, qr_size)
    cap_gap = max(6, int(round(13 * s)))
    qr_block_h = qr_size + cap_gap + cap_h

    gap_logo_title = max(12, int(round(39 * s)))
    gap_title_sub = max(8, int(round(22 * s)))
    gap_sub_qr = max(16, int(round(40 * s)))

    inner_top = layout.bar + layout.margin_top
    inner_bot = height - layout.bar - max(layout.margin_bottom // 2, int(round(20 * s)))

    stack = layout.is_portrait or layout.is_square or (width < logo_w + qr_size + 3 * layout.margin)

    if stack:
        block_h = logo_h + gap_logo_title + title_h + gap_title_sub + sub_h + gap_sub_qr + qr_block_h
        y = inner_top + max(0, (inner_bot - inner_top - block_h) // 2)
        x_logo = (width - logo_w) // 2
        im.alpha_composite(logo_img, (x_logo, y))
        y += logo_h + gap_logo_title
        _draw_text(im, (width // 2, y), title, title_font, ORANGE_TEXT, anchor="mt")
        y += title_h + gap_title_sub
        _draw_text(im, (width // 2, y), sub, sub_font, WHITE, anchor="mt")
        y += sub_h + gap_sub_qr
        qx = (width - qr_size) // 2
        im.alpha_composite(qr_img, (qx, y))
        _draw_text(
            im,
            (width // 2, y + qr_size + cap_gap),
            caption,
            cap_font,
            CAPTION_GRAY,
            anchor="mt",
        )
    else:
        text_block_h = title_h + gap_title_sub + sub_h
        group_h = logo_h + gap_logo_title + text_block_h
        y = layout.bar + max(layout.margin_top, int(round(90 * s)) - layout.bar)
        if y + group_h > inner_bot:
            y = max(inner_top, inner_bot - group_h)
        x_logo = (width - logo_w) // 2
        im.alpha_composite(logo_img, (x_logo, y))
        y_title = y + logo_h + gap_logo_title
        _draw_text(im, (width // 2, y_title), title, title_font, ORANGE_TEXT, anchor="mt")
        y_sub = y_title + title_h + gap_title_sub
        _draw_text(im, (width // 2, y_sub), sub, sub_font, WHITE, anchor="mt")

        qx = width - layout.margin_right - qr_size
        qy = inner_bot - qr_block_h
        if qy < y_sub + sub_h + max(12, int(round(16 * s))):
            qy = y_sub + sub_h + max(12, int(round(16 * s)))
        im.alpha_composite(qr_img, (qx, qy))
        _draw_text(
            im,
            (qx + qr_size // 2, qy + qr_size + cap_gap),
            caption,
            cap_font,
            CAPTION_GRAY,
            anchor="mt",
        )
    return im


HL_SUB = "Next Possession Open Run"


def render_logo_bug(width: int, height: int, logo, opacity: float = 0.9):
    """Corner logo (same size and spot as the promo card's logo), full-frame RGBA."""
    from PIL import Image

    layout = make_layout(width, height)
    s = layout.scale
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    logo_h = max(48, int(round(110 * s)))
    logo_w = int(round(logo_h * (logo.width / logo.height)))
    img = _scale_logo(logo, logo_w, logo_h)
    if opacity < 1:
        a = img.getchannel("A").point(lambda v: int(v * opacity))
        img.putalpha(a)
    im.alpha_composite(img, (width - layout.margin_right - logo_w, layout.margin_top))
    return im


def render_highlight_title(width: int, height: int, main: str, sub: str, fonts_dir: Path):
    """'<date> Highlights' with 'Next Possession Open Run' underneath, on a plaque."""
    from PIL import Image, ImageDraw

    layout = make_layout(width, height)
    s = layout.scale
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    main = (main or "").strip()
    sub = (sub or "").strip()
    if not main and not sub:
        return im
    side = max(layout.margin, int(round((0.07 if layout.is_portrait else 0.08) * width)))
    max_w = max(64, width - 2 * side)
    size_main = (70 if layout.is_portrait else 80) * s
    size_sub = size_main * 0.5
    for _ in range(14):
        f_main = _load_font(fonts_dir, True, size_main)
        f_sub = _load_font(fonts_dir, False, size_sub)
        lines_main = wrap_text(main, f_main, max_w) if main else []
        lines_sub = wrap_text(sub, f_sub, max_w) if sub else []
        widths = [_text_ink_size(f_main, l)[0] for l in lines_main] + [_text_ink_size(f_sub, l)[0] for l in lines_sub]
        if all(w <= max_w for w in widths) and len(lines_main) <= 3:
            break
        size_main *= 0.9
        size_sub = size_main * 0.5
    sizes_m = [_text_ink_size(f_main, l or " ") for l in lines_main]
    sizes_s = [_text_ink_size(f_sub, l or " ") for l in lines_sub]
    gap_m = max(6, int(round(size_main * 0.22)))
    gap_s = max(4, int(round(size_sub * 0.3)))
    gap_between = max(10, int(round(size_main * 0.45))) if lines_main and lines_sub else 0
    block_w = max([w for w, _ in sizes_m + sizes_s] or [0])
    block_h = (sum(h for _, h in sizes_m) + gap_m * max(0, len(sizes_m) - 1)
               + gap_between + sum(h for _, h in sizes_s) + gap_s * max(0, len(sizes_s) - 1))
    pad_x = max(20, int(round(44 * s)))
    pad_y = max(14, int(round(28 * s)))
    radius = max(8, int(round(18 * s)))
    pw = min(width - 2 * layout.margin, block_w + 2 * pad_x)
    ph = block_h + 2 * pad_y
    px = (width - pw) // 2
    py = (height - ph) // 2 - int(height * (0.05 if layout.is_portrait else 0.0))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((px, py, px + pw - 1, py + ph - 1), radius=radius, fill=TITLE_PLAQUE)
    accent = max(4, int(round(8 * s)))
    draw.rectangle((px + radius, py, px + pw - 1 - radius, py + accent - 1), fill=ORANGE_BAR)
    shadow = max(1, int(round(3 * s)))
    y = py + pad_y
    cx = px + pw // 2
    for line, (_, lh) in zip(lines_main, sizes_m):
        _draw_text(im, (cx + shadow, y + shadow), line, f_main, SHADOW, anchor="mt")
        _draw_text(im, (cx, y), line, f_main, WHITE, anchor="mt")
        y += lh + gap_m
    if lines_main:
        y += gap_between - gap_m
    for line, (_, lh) in zip(lines_sub, sizes_s):
        _draw_text(im, (cx, y), line, f_sub, ORANGE_TEXT, anchor="mt")
        y += lh + gap_s
    return im


def render_highlight_graphics(
    out_dir: Path, width: int, height: int, assets: Path, fonts_dir: Path,
    title_main: str = "", title_sub: str = HL_SUB,
    lt_line1: str = LT_TITLE, lt_line2: str = LT_SUB, lt_logo: bool = True,
    end_title: str = END_TITLE, end_sub: str = END_SUB,
    qr_caption: str = QR_CAPTION, qr_url: str = DEFAULT_QR_URL,
    logo_opacity: float = 0.9,
) -> dict:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    logo = Image.open(assets / "logo.png").convert("RGBA")
    qr = make_qr_image(qr_url or DEFAULT_QR_URL, 1200)
    paths = {k: out_dir / f"{k}.png" for k in ("intro_plate", "title", "logo_bug", "lower_third", "endcard")}
    render_intro_plate(width, height).convert("RGB").save(paths["intro_plate"])
    render_highlight_title(width, height, title_main, title_sub, fonts_dir).save(paths["title"])
    render_logo_bug(width, height, logo, logo_opacity).save(paths["logo_bug"])
    render_lower_third(width, height, logo, fonts_dir, line1=lt_line1, line2=lt_line2,
                       include_logo=lt_logo).save(paths["lower_third"])
    render_endcard(width, height, logo, qr, fonts_dir, title=end_title, sub=end_sub,
                   caption=qr_caption).convert("RGB").save(paths["endcard"])
    return paths
