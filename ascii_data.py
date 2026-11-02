r"""
Labelled ASCII-art data, generated on the fly (numpy + Pillow only).

Sources
  shapes : procedurally drawn shapes, no download needed (8 classes)
  mnist  : MNIST digits via torchvision (10 classes, downloads once)

Styles (how pixels become characters)
  edge   : edge direction -> - / | \   (line-art look, readable)
  tone   : brightness ramp ' .:-=+*#%@' (shading look)
  mixed  : random per sample

Canvas: W x H cells (default 32 x 16; cells are ~2x taller than wide, so a
square image maps to a 2:1 grid).
"""
import math
import random

import numpy as np
from PIL import Image, ImageDraw

TONE_RAMP = " .:-=+*#%@"                 # dark-on-light: 0 = blank, 1 = densest
SHAPE_CLASSES = ["circle", "square", "triangle", "diamond", "plus", "cross", "star", "arrow"]


# ------------------------------------------------------------ pixels -> chars
def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0


def to_tone(img: np.ndarray, W: int, H: int) -> list[str]:
    """img: float array in [0,1], 1 = ink. Cell value = mean ink -> ramp char."""
    s = _resize(img, W, H)
    idx = np.clip((s * len(TONE_RAMP)).astype(int), 0, len(TONE_RAMP) - 1)
    return ["".join(TONE_RAMP[i] for i in row) for row in idx]


def to_edge(img: np.ndarray, W: int, H: int, k: int = 4, thresh: float = 0.15) -> list[str]:
    """
    Edge-direction art. Work at k x k sub-pixels per cell, take the strongest
    gradient in each cell, and pick a char from the edge (tangent) direction:
        ~0 deg -> '-'   ~45 -> '/'   ~90 -> '|'   ~135 -> '\\'
    Bottom-of-shape horizontal edges use '_' so outlines close nicely.
    """
    s = _resize(img, W * k, H * k)
    gy, gx = np.gradient(s)                       # gy: downward rows
    mag = np.hypot(gx, gy)
    out = []
    for r in range(H):
        row = []
        for c in range(W):
            m = mag[r * k:(r + 1) * k, c * k:(c + 1) * k]
            if m.max() < thresh:
                row.append(" ")
                continue
            i, j = np.unravel_index(m.argmax(), m.shape)
            yy, xx = r * k + i, c * k + j
            theta = math.degrees(math.atan2(-gy[yy, xx], gx[yy, xx]))   # y-up convention
            tangent = (theta + 90) % 180
            if tangent < 22.5 or tangent >= 157.5:
                row.append("_" if gy[yy, xx] < 0 else "-")              # ink above -> bottom edge
            elif tangent < 67.5:
                row.append("/")
            elif tangent < 112.5:
                row.append("|")
            else:
                row.append("\\")
        out.append("".join(row))
    return out


def to_ascii(img, W, H, style="edge"):
    if style == "mixed":
        style = random.choice(["edge", "tone"])
    return to_edge(img, W, H) if style == "edge" else to_tone(img, W, H)


# ------------------------------------------------------------ procedural shapes
def draw_shape(name: str, size: int = 128, rng: random.Random = random) -> np.ndarray:
    """Random scale / offset / rotation / outline-vs-filled. Returns float [0,1], 1 = ink."""
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    r = size * rng.uniform(0.25, 0.42)
    cx = size / 2 + rng.uniform(-0.1, 0.1) * size
    cy = size / 2 + rng.uniform(-0.1, 0.1) * size
    rot = rng.uniform(-15, 15)
    width = max(2, int(size * rng.uniform(0.04, 0.08)))
    filled = rng.random() < 0.5

    def poly(pts):
        a = math.radians(rot)
        pts = [(cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)) for x, y in pts]
        if filled:
            d.polygon(pts, fill=255)
        else:
            d.line(pts + [pts[0]], fill=255, width=width, joint="curve")

    if name == "circle":
        box = [cx - r, cy - r, cx + r, cy + r]
        d.ellipse(box, fill=255) if filled else d.ellipse(box, outline=255, width=width)
    elif name == "square":
        poly([(-r, -r), (r, -r), (r, r), (-r, r)])
    elif name == "triangle":
        poly([(0, -r), (r * 0.95, r * 0.8), (-r * 0.95, r * 0.8)])
    elif name == "diamond":
        poly([(0, -r), (r * 0.75, 0), (0, r), (-r * 0.75, 0)])
    elif name == "plus":
        t = r * 0.3
        poly([(-t, -r), (t, -r), (t, -t), (r, -t), (r, t), (t, t), (t, r), (-t, r), (-t, t), (-r, t), (-r, -t), (-t, -t)])
    elif name == "cross":
        a = r * 0.75
        lw = width * 2
        d.line([(cx - a, cy - a), (cx + a, cy + a)], fill=255, width=lw)
        d.line([(cx - a, cy + a), (cx + a, cy - a)], fill=255, width=lw)
    elif name == "star":
        pts = []
        for i in range(10):
            rr = r if i % 2 == 0 else r * 0.45
            ang = math.radians(-90 + 36 * i)
            pts.append((rr * math.cos(ang), rr * math.sin(ang)))
        poly(pts)
    elif name == "arrow":
        t = r * 0.25
        poly([(-r, -t), (r * 0.2, -t), (r * 0.2, -r * 0.6), (r, 0), (r * 0.2, r * 0.6), (r * 0.2, t), (-r, t)])
    else:
        raise ValueError(name)
    return np.asarray(im, dtype=np.float32) / 255.0


class ShapeSource:
    classes = SHAPE_CLASSES

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def sample(self):
        y = self.rng.randrange(len(self.classes))
        return draw_shape(self.classes[y], rng=self.rng), y


class MnistSource:
    classes = [str(i) for i in range(10)]

    def __init__(self, train=True, root="./data", seed=0):
        from torchvision.datasets import MNIST          # only needed for this source
        self.ds = MNIST(root, train=train, download=True)
        self.rng = random.Random(seed)

    def sample(self):
        img, y = self.ds[self.rng.randrange(len(self.ds))]
        return np.asarray(img, dtype=np.float32) / 255.0, y


def make_source(name, train=True, seed=0):
    if name == "shapes":
        return ShapeSource(seed=seed if train else seed + 10_000)
    if name == "mnist":
        return MnistSource(train=train, seed=seed)
    raise ValueError(name)


if __name__ == "__main__":
    src = ShapeSource(seed=1)
    W, H = 32, 16
    for name in ["circle", "triangle", "star", "arrow"]:
        img = draw_shape(name, rng=random.Random(hash(name) % 100))
        for style in ("edge", "tone"):
            print(f"--- {name} / {style} ---")
            print("\n".join(to_ascii(img, W, H, style)))
