"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) step by step,
implemented from scratch in NumPy, with a PyQt5 viewer that shows each step.
Shared loading/UI code lives in viewer_common.py (keep it in the same folder).

Install:
    pip install PyQt5 numpy opencv-python matplotlib tifffile imagecodecs
Run:
    python clahe_demo.py                 # synthetic thermal-like test image
    python clahe_demo.py image.tif       # TIFF: single/multi-page, 8/16-bit, float
    python clahe_demo.py image.png       # any image OpenCV reads
    python clahe_demo.py frames.raw      # raw 16-bit little-endian 640x512 frames
Settings (parameters, selected tile, last folder, window) are kept in
clahe_demo.ini next to this script.

Steps (one tab each):
    0. Bring the input to 8 bit (raw 14/16-bit data is stretched by percentiles)
    1. Split the image into a grid of tiles
    2. Compute a histogram per tile
    3. Clip each histogram at the clip limit, redistribute the excess
    4. Turn each clipped histogram into a mapping (CDF -> LUT)
    5. Map every pixel by bilinear interpolation between the 4 nearest tile LUTs
    6. Compare with OpenCV's cv2.createCLAHE
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QCheckBox, QSpinBox

from viewer_common import (  # noqa: F401  (re-exported for scripts/tests)
    ImageView, Plot, StepViewer, captioned, dspin, group, load_image, run, step_tab,
    synthetic_image, to_8bit,
)


# ----------------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------------

def pad_to_tiles(img, tiles_y, tiles_x):
    """Step 1: pad so the image splits into equal tiles.
    Matches OpenCV exactly: if either axis does not divide evenly, OpenCV pads
    BOTH axes by (tiles - size % tiles), even an axis that already divides
    (it then gets a whole extra tile's worth of reflected pixels)."""
    h, w = img.shape
    if h % tiles_y == 0 and w % tiles_x == 0:
        return img, h // tiles_y, w // tiles_x
    pad_y = tiles_y - h % tiles_y
    pad_x = tiles_x - w % tiles_x
    padded = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101)
    return padded, padded.shape[0] // tiles_y, padded.shape[1] // tiles_x


def tile_histograms(padded, tiles_y, tiles_x, th, tw):
    """Step 2: one 256-bin histogram per tile -> array (tiles_y, tiles_x, 256)."""
    tiles = padded.reshape(tiles_y, th, tiles_x, tw).swapaxes(1, 2)
    hists = np.zeros((tiles_y, tiles_x, 256), np.int64)
    for i in range(tiles_y):
        for j in range(tiles_x):
            hists[i, j] = np.bincount(tiles[i, j].ravel(), minlength=256)
    return hists


def clip_histogram(hist, limit):
    """Step 3: cut every bin at `limit`, spread the excess over all bins.
    This caps the slope of the mapping, so noise is not over-amplified."""
    excess = int(np.maximum(hist - limit, 0).sum())
    clipped = np.minimum(hist, limit)
    clipped += excess // 256                       # spread evenly
    residual = excess % 256                        # spread the remainder
    if residual:
        step = max(256 // residual, 1)
        clipped[np.arange(0, 256, step)[:residual]] += 1
    return clipped, excess


def histogram_to_lut(hist, tile_area):
    """Step 4: cumulative histogram (CDF) scaled to 0..255 = the tile's mapping."""
    cdf = np.cumsum(hist)
    return np.clip(np.round(cdf * 255.0 / tile_area), 0, 255).astype(np.uint8)


def interpolate(img, luts, th, tw):
    """Step 5: each pixel is mapped with the LUTs of the 4 nearest tile
    centres and blended bilinearly, so tile borders do not show."""
    h, w = img.shape
    tiles_y, tiles_x = luts.shape[:2]

    yf = np.arange(h) / th - 0.5            # position in tile units
    xf = np.arange(w) / tw - 0.5
    y1 = np.floor(yf).astype(int)
    x1 = np.floor(xf).astype(int)
    ya = (yf - y1)[:, None]                 # vertical weight
    xa = (xf - x1)[None, :]                 # horizontal weight
    y2 = np.clip(y1 + 1, 0, tiles_y - 1)[:, None]
    x2 = np.clip(x1 + 1, 0, tiles_x - 1)[None, :]
    y1 = np.clip(y1, 0, tiles_y - 1)[:, None]
    x1 = np.clip(x1, 0, tiles_x - 1)[None, :]

    l11 = luts[y1, x1, img].astype(np.float32)   # top-left tile LUT
    l12 = luts[y1, x2, img].astype(np.float32)   # top-right
    l21 = luts[y2, x1, img].astype(np.float32)   # bottom-left
    l22 = luts[y2, x2, img].astype(np.float32)   # bottom-right

    out = (1 - ya) * ((1 - xa) * l11 + xa * l12) + ya * ((1 - xa) * l21 + xa * l22)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def clahe_steps(img8, clip_limit=2.0, tiles_y=8, tiles_x=8):
    """Run all steps and keep every intermediate result for display."""
    padded, th, tw = pad_to_tiles(img8, tiles_y, tiles_x)
    area = th * tw
    hists = tile_histograms(padded, tiles_y, tiles_x, th, tw)

    # Same definition as OpenCV: clip_limit is relative to a flat histogram.
    limit = max(int(clip_limit * area / 256), 1)

    clipped = np.zeros_like(hists)
    excess = np.zeros((tiles_y, tiles_x), np.int64)
    luts = np.zeros((tiles_y, tiles_x, 256), np.uint8)
    for i in range(tiles_y):
        for j in range(tiles_x):
            clipped[i, j], excess[i, j] = clip_histogram(hists[i, j], limit)
            luts[i, j] = histogram_to_lut(clipped[i, j], area)

    out = interpolate(img8, luts, th, tw)
    return dict(img=img8, th=th, tw=tw, area=area, limit=limit, hists=hists,
                clipped=clipped, excess=excess, luts=luts, out=out)


# ----------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------

class ClaheDemo(StepViewer):
    TITLE = "CLAHE step by step"
    ICON = "clahe"

    def build_ui(self):
        self.clip = self.persist("params/clip_limit", dspin(0.1, 40.0, 2.0, 0.5))
        self.tiles_x = self.persist("params/tiles_x", QSpinBox(minimum=1, maximum=32, value=8))
        self.tiles_y = self.persist("params/tiles_y", QSpinBox(minimum=1, maximum=32, value=8))
        self.sel_col = self.persist("view/tile_x", QSpinBox(minimum=0, maximum=99, value=1))
        self.sel_row = self.persist("view/tile_y", QSpinBox(minimum=0, maximum=99, value=3))
        self.show_grid = self.persist("view/show_grid", QCheckBox("Show tile grid", checked=True))

        for w in (self.clip, self.tiles_x, self.tiles_y):
            w.valueChanged.connect(self.recompute)
        for w in (self.sel_row, self.sel_col):
            w.valueChanged.connect(self.refresh)
        self.show_grid.toggled.connect(self.refresh)

        side = [
            group("Parameters", [("Clip limit", self.clip), ("Tiles X", self.tiles_x),
                                 ("Tiles Y", self.tiles_y)]),
            group("Inspect tile (or click an image)", [
                ("Tile x (column)", self.sel_col), ("Tile y (row)", self.sel_row),
                self.show_grid]),
        ]

        self.v_input, self.p_input = ImageView(), Plot()
        self.v_tiles = ImageView()
        self.v_tile_zoom, self.p_hist = ImageView(), Plot()
        self.p_lut = Plot()
        self.v_in2, self.v_he, self.v_clahe = ImageView(), ImageView(), ImageView()
        self.v_cv, self.v_diff = ImageView(), ImageView()
        for v in (self.v_input, self.v_tiles, self.v_in2, self.v_he,
                  self.v_clahe, self.v_cv, self.v_diff):
            v.clicked.connect(self.select_tile_at)

        tabs = [
            (step_tab(
                "Step 0 — Input. Raw 14/16-bit data is stretched to 8 bit (0.5–99.5 "
                "percentile). Note how most pixels sit in a narrow band of the histogram: "
                "faint details are barely visible.",
                captioned(self.v_input, "Input (8 bit)"), self.p_input), "0 Input"),
            (step_tab(
                "Step 1 — Tiles. The image is split into a grid of tiles. Each tile gets "
                "its own contrast mapping, so dark and bright regions are equalized "
                "separately. The red tile is the one inspected in the next tabs: click any "
                "tile (here or on any image) to select it.",
                captioned(self.v_tiles, "Tile grid")), "1 Tiles"),
            (step_tab(
                "Steps 2 and 3 — Histogram and clipping. Each tile's histogram is cut at "
                "the clip limit and the cut-off counts are spread evenly over all bins. "
                "A higher clip limit means stronger contrast but more amplified noise.",
                captioned(self.v_tile_zoom, "Selected tile"), self.p_hist), "2–3 Histogram"),
            (step_tab(
                "Step 4 — Mapping. The cumulative sum of the clipped histogram, scaled to "
                "0..255, is the tile's lookup table. Steep parts stretch contrast. "
                "Clipping caps the steepness compared with plain equalization.",
                self.p_lut), "4 Mapping"),
            (step_tab(
                "Step 5 — Result. Every pixel is mapped with the LUTs of the 4 nearest "
                "tile centres and blended bilinearly, so tile borders do not show. "
                "Compare with global histogram equalization, which uses one LUT for all.",
                captioned(self.v_in2, "Input"), captioned(self.v_he, "Global HE"),
                captioned(self.v_clahe, "CLAHE (ours)")), "5 Result"),
            (step_tab(
                "Step 6 — Check. The same parameters in cv2.createCLAHE. Differences "
                "should be at most a level or two (rounding).",
                captioned(self.v_cv, "OpenCV CLAHE"),
                captioned(self.v_diff, "|ours − OpenCV| × 50")), "6 vs OpenCV"),
        ]
        self.res = None
        return side, tabs

    def compute(self, raw):
        img8 = to_8bit(raw)
        ty, tx = self.tiles_y.value(), self.tiles_x.value()
        self.res = clahe_steps(img8, self.clip.value(), ty, tx)
        for box, top in ((self.sel_row, ty - 1), (self.sel_col, tx - 1)):
            box.blockSignals(True)
            box.setMaximum(top)
            box.blockSignals(False)

    def select_tile_at(self, x, y):
        """Clicked image pixel (x, y) -> tile (x = column, y = row)."""
        r = self.res
        if r is None:
            return
        tx = min(x // r["tw"], self.tiles_x.value() - 1)
        ty = min(y // r["th"], self.tiles_y.value() - 1)
        for box, val in ((self.sel_col, tx), (self.sel_row, ty)):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.save_settings()
        self.refresh()

    def refresh(self, *_):
        r = self.res
        if r is None:
            return
        img, th, tw = r["img"], r["th"], r["tw"]
        i, j = self.sel_row.value(), self.sel_col.value()
        h, w = img.shape
        bins = np.arange(256)

        # Step 0
        self.v_input.set_array(img)
        ax = self.p_input.axes[0]
        ax.clear()
        ax.fill_between(bins, np.bincount(img.ravel(), minlength=256), step="mid")
        ax.set_title("Global histogram")
        ax.set_xlim(0, 255)
        self.p_input.draw()

        # Step 1
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.show_grid.isChecked():
            for k in range(1, self.tiles_y.value()):
                cv2.line(rgb, (0, k * th), (w, k * th), (0, 200, 255), 1)
            for k in range(1, self.tiles_x.value()):
                cv2.line(rgb, (k * tw, 0), (k * tw, h), (0, 200, 255), 1)
        cv2.rectangle(rgb, (j * tw, i * th), ((j + 1) * tw - 1, (i + 1) * th - 1),
                      (255, 40, 40), 2)
        self.v_tiles.set_array(rgb)

        # Steps 2-3
        self.v_tile_zoom.set_array(img[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
        ax = self.p_hist.axes[0]
        ax.clear()
        ax.plot(bins, r["hists"][i, j], label="original", lw=1)
        ax.plot(bins, r["clipped"][i, j], label="clipped + redistributed", lw=1)
        ax.axhline(r["limit"], color="red", ls="--", lw=1, label=f"clip limit = {r['limit']}")
        ax.set_title(f"Tile (x={j}, y={i}): excess moved = {r['excess'][i, j]} px")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=8)
        self.p_hist.draw()

        # Step 4
        ax = self.p_lut.axes[0]
        ax.clear()
        for lut in r["luts"].reshape(-1, 256):
            ax.plot(bins, lut, color="0.85", lw=0.6)
        ax.plot(bins, r["luts"][i, j], lw=2, label=f"tile (x={j}, y={i}) CLAHE LUT")
        unclipped = histogram_to_lut(r["hists"][i, j], r["area"])
        ax.plot(bins, unclipped, ls=":", lw=1.5, label="same tile, no clipping (AHE)")
        ax.plot(bins, bins, color="k", ls="--", lw=0.8, label="identity")
        ax.set_xlabel("input level")
        ax.set_ylabel("output level")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Grey: LUTs of all tiles")
        self.p_lut.draw()

        # Step 5
        self.v_in2.set_array(img)
        self.v_he.set_array(cv2.equalizeHist(img))
        self.v_clahe.set_array(r["out"])

        # Step 6
        cv_clahe = cv2.createCLAHE(clipLimit=self.clip.value(),
                                   tileGridSize=(self.tiles_x.value(), self.tiles_y.value()))
        cv_out = cv_clahe.apply(img)
        diff = np.abs(cv_out.astype(np.int16) - r["out"].astype(np.int16))
        self.v_cv.set_array(cv_out)
        self.v_diff.set_array(np.clip(diff * 50, 0, 255).astype(np.uint8))

        self.info.setText(
            f"Image {w}×{h}, tile {tw}×{th} px\n"
            f"Clip limit {self.clip.value():g} → {r['limit']} counts per bin\n"
            f"vs OpenCV: max diff {diff.max()}, mean {diff.mean():.3f}")


if __name__ == "__main__":
    run(ClaheDemo)
