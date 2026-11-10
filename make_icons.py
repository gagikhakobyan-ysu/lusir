"""
Draws the app icons into ./icons (PNG 256 px + multi-size Windows .ico):
    clahe.png / clahe.ico              tiles of locally equalized grey + a CDF curve
    base_detail.png / base_detail.ico  smooth base layer + a detail wiggle on top
    local_contrast.png / .ico          blocks whose waves are stretched to one contrast

    pip install pillow
    python make_icons.py
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

S = 1024                      # draw big, then downsample (anti-aliasing)
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")


def rounded_background(top, bottom):
    """Rounded square with a vertical gradient and a soft inner highlight."""
    grad = Image.new("RGB", (S, S))
    d = ImageDraw.Draw(grad)
    for y in range(S):
        t = y / (S - 1)
        d.line([(0, y), (S, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([24, 24, S - 24, S - 24], radius=210, fill=255)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    return img, mask


def polyline(draw, pts, color, width):
    draw.line(pts, fill=color, width=width, joint="curve")
    r = width / 2
    for x, y in (pts[0], pts[-1]):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def clahe_icon():
    img, mask = rounded_background((18, 52, 74), (10, 28, 42))
    d = ImageDraw.Draw(img)
    # 4x4 tiles in different greys: each tile "equalized on its own"
    greys = [[ 70, 150, 105, 200],
             [130,  95, 215, 120],
             [185, 225,  80, 160],
             [100, 170, 140, 235]]
    x0, y0, cell, gap = 170, 170, 165, 12
    for i in range(4):
        for j in range(4):
            g = greys[i][j]
            x, y = x0 + j * cell, y0 + i * cell
            d.rounded_rectangle([x + gap, y + gap, x + cell - gap, y + cell - gap],
                                radius=26, fill=(g, g, g, 255))
    # CDF / mapping curve across the tiles, with a glow
    pts = []
    for k in range(101):
        t = k / 100
        v = 1 / (1 + math.exp(-10 * (t - 0.45)))        # S-shaped equalization curve
        pts.append((150 + t * 724, 874 - v * 724))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    polyline(ImageDraw.Draw(glow), pts, (255, 170, 40, 200), 70)
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(26)))
    polyline(d, pts, (255, 190, 70, 255), 44)
    polyline(d, pts, (255, 236, 190, 255), 14)
    return img, mask


def base_detail_icon():
    img, mask = rounded_background((48, 30, 86), (20, 14, 40))
    d = ImageDraw.Draw(img)
    # base layer: a smooth, edge-preserving step (filled area)
    def base(x):
        t = (x - 150) / 724
        return 640 - 260 / (1 + math.exp(-28 * (t - 0.5)))
    xs = [150 + k * 724 / 200 for k in range(201)]
    area = [(150, 874)] + [(x, base(x)) for x in xs] + [(874, 874)]
    d.polygon(area, fill=(90, 150, 230, 255))
    polyline(d, [(x, base(x)) for x in xs], (170, 215, 255, 255), 26)
    # detail layer: small wiggles riding above the base, with a glow
    pts = [(x, base(x) - 120 + 34 * math.sin((x - 150) / 30) * math.cos((x - 150) / 140))
           for x in xs]
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    polyline(ImageDraw.Draw(glow), pts, (255, 120, 60, 200), 60)
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(22)))
    polyline(d, pts, (255, 140, 70, 255), 30)
    polyline(d, pts, (255, 225, 190, 255), 10)
    # "+" sign: base + detail
    cx, cy, a, w = 270, 270, 62, 26
    d.rounded_rectangle([cx - a, cy - w / 2, cx + a, cy + w / 2], radius=12, fill=(255, 255, 255, 235))
    d.rounded_rectangle([cx - w / 2, cy - a, cx + w / 2, cy + a], radius=12, fill=(255, 255, 255, 235))
    return img, mask


def local_contrast_icon():
    img, mask = rounded_background((16, 70, 58), (8, 34, 30))
    d = ImageDraw.Draw(img)
    # 3x3 blocks; each has a wave of different amplitude in the input (faint)
    # and the same, "optimal" amplitude after local stretching (bright)
    x0, y0, cell, gap = 150, 150, 241, 14
    levels = [[60, 120, 90], [140, 80, 170], [100, 190, 130]]
    amps = [[0.25, 0.9, 0.5], [0.4, 0.15, 0.7], [1.0, 0.3, 0.6]]
    for i in range(3):
        for j in range(3):
            x, y = x0 + j * cell, y0 + i * cell
            g = levels[i][j]
            d.rounded_rectangle([x + gap, y + gap, x + cell - gap, y + cell - gap],
                                radius=30, fill=(g // 2, g, g * 3 // 4, 255))
            cy = y + cell / 2
            xs = [x + 40 + k * (cell - 80) / 60 for k in range(61)]
            faint = [(u, cy + 55 * amps[i][j] * math.sin((u - x) / 11)) for u in xs]
            polyline(d, faint, (255, 255, 255, 90), 10)
            strong = [(u, cy + 55 * math.sin((u - x) / 11)) for u in xs]
            polyline(d, strong, (255, 214, 90, 255), 16)
    return img, mask


def save(name, drawing):
    img, mask = drawing
    img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", (S, S), 0), mask))
    os.makedirs(OUT, exist_ok=True)
    small = img.resize((256, 256), Image.LANCZOS)
    small.save(os.path.join(OUT, f"{name}.png"))
    small.save(os.path.join(OUT, f"{name}.ico"), sizes=ICO_SIZES)
    print("wrote", os.path.join(OUT, f"{name}.png / .ico"))


if __name__ == "__main__":
    save("clahe", clahe_icon())
    save("base_detail", base_detail_icon())
    save("local_contrast", local_contrast_icon())
