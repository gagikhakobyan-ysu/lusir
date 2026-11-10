"""
Local optimal contrast dynamic range compression, step by step, with a PyQt5
viewer that shows each step. Shared loading/UI code lives in viewer_common.py
and the guided filter in base_detail_demo.py (keep them in the same folder).

Follows the processing structure of DRCE-LOC:
    Y. Zhu, Y. Zhou, W. Jin, L. Zhang, G. Wu, Y. Shao, "A Low-Delay Dynamic
    Range Compression and Contrast Enhancement Algorithm Based on an Uncooled
    Infrared Sensor with Local Optimal Contrast", Sensors 23(21), 8860, 2023.
    doi:10.3390/s23218860
Related earlier work on IR dynamic-range compression and contrast enhancement:
    F. Branchitta, M. Diani, G. Corsini, A. Porta, "Dynamic-range compression
    and contrast enhancement in infrared imaging systems", Optical Engineering
    47(7), 076401, 2008. doi:10.1117/1.2956655

NOTE: the step structure follows the DRCE-LOC paper; the formulas marked
"(our choice)" below are a plausible implementation written without the
paper's full text, and should be checked against its equations.

Steps (one tab each):
    0. Input: raw 14/16-bit -> float [0, 1], full precision
    1. Blocks: split the image into blocks, local mean and std per block
    2. Local stretching coefficient per block, from its std, upsampled to a
       smooth per-pixel map with Gaussian weighting
    3. Guided-filter layering: base B and detail D = input - B
    4. Base compression: local stretch around a bias, blended with a global
       linear guide map that keeps the light/dark order of the scene
    5. Detail: noise mask from local detail activity (suppress detail in flat,
       noise-only areas), then gain
    6. Result = compressed base + masked detail, vs linear / CLAHE / base+detail

    pip install PyQt5 numpy opencv-python matplotlib tifffile imagecodecs
    python local_contrast_demo.py [frame.tiff]
"""
import time

import cv2
import numpy as np
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QCheckBox, QSpinBox

from base_detail_demo import box_mean, detail_to_8bit, guided_filter, map_base
from viewer_common import (
    ImageView, Plot, StepViewer, captioned, dspin, group, run, step_tab, to_8bit,
    to_unit, unit_to_8bit,
)


# ----------------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------------

def block_stats(img, blocks_y, blocks_x):
    """Step 1: mean and standard deviation of every block (area averaging,
    so the image size need not divide evenly)."""
    size = (blocks_x, blocks_y)
    mean = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    mean_sq = cv2.resize(img * img, size, interpolation=cv2.INTER_AREA)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    return mean, std


def stretch_coefficient(std, target, k_min, k_max):
    """Step 2 (our choice): the gain that brings a block's contrast (std) to
    the target contrast, limited to [k_min, k_max]. Flat blocks get stretched
    (up to k_max, so noise is not blown up), very contrasted blocks get
    compressed (down to k_min)."""
    return np.clip(target / (std + 1e-6), k_min, k_max).astype(np.float32)


def upsample_gaussian(block_map, shape, smooth):
    """Step 2: block values -> per-pixel map. Each pixel gets a Gaussian-
    weighted average of the nearby blocks' values (nearest upsampling, then a
    Gaussian blur of `smooth` x block size), so block borders do not show."""
    h, w = shape
    up = cv2.resize(block_map, (w, h), interpolation=cv2.INTER_NEAREST)
    by, bx = block_map.shape
    sy, sx = max(smooth * h / by, 0.5), max(smooth * w / bx, 0.5)
    return cv2.GaussianBlur(up, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_REFLECT)


def global_guide(base):
    """Step 4: global linear brightness map of the base (keeps the light/dark
    order of the whole scene; a min/max mapping is cheap on an FPGA)."""
    lo, hi = float(base.min()), float(base.max())
    return (base - lo) / max(hi - lo, 1e-6)


def compress_base(base, k_map, mean_map, guide, bias, k2):
    """Step 4 (our choice): local stretch L = bias + k(x) (B - mu(x)) gives every
    region the target contrast around the same mid level; blending with the
    global guide (weight k2) restores which regions are brighter overall."""
    local = bias + k_map * (base - mean_map)
    return np.clip(k2 * guide + (1 - k2) * local, 0, 1), local


def detail_noise_mask(detail, window, noise_floor):
    """Step 5 (our choice, HVS-inspired): local RMS of the detail layer. Where
    it is at the noise level (flat areas, where the eye sees noise most) the
    mask goes to 0; where there is real texture it goes to 1."""
    rms2 = box_mean(detail * detail, window)
    t2 = noise_floor * noise_floor
    return rms2 / (rms2 + t2 + 1e-12)


def heat(x, lo=None, hi=None):
    """Float map -> RGB colour-mapped image for display."""
    lo = float(x.min()) if lo is None else lo
    hi = float(x.max()) if hi is None else hi
    u8 = unit_to_8bit((x - lo) / max(hi - lo, 1e-9))
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------

class LocalContrastDemo(StepViewer):
    TITLE = "Local optimal contrast DRC step by step"
    ICON = "local_contrast"

    def build_ui(self):
        P = self.persist
        self.blocks_x = P("params/blocks_x", QSpinBox(minimum=1, maximum=64, value=8))
        self.blocks_y = P("params/blocks_y", QSpinBox(minimum=1, maximum=64, value=8))
        self.target = P("params/target_contrast", dspin(0.005, 0.5, 0.08, 0.01, 3))
        self.k_min = P("params/k_min", dspin(0.05, 2.0, 0.3, 0.05))
        self.k_max = P("params/k_max", dspin(1.0, 30.0, 6.0, 0.5))
        self.smooth = P("params/smooth", dspin(0.1, 3.0, 0.6, 0.1))
        self.radius = P("params/radius", QSpinBox(minimum=1, maximum=40, value=8))
        self.sigma = P("params/edge_sigma", dspin(0.001, 0.5, 0.03, 0.005, 3))
        self.k2 = P("params/k2_global_weight", dspin(0.0, 1.0, 0.4, 0.05))
        self.bias = P("params/bias", dspin(0.0, 1.0, 0.5, 0.05))
        self.edge_mean = P("params/edge_aware_mean",
                           QCheckBox("Edge-aware local mean μ(x)", checked=True))
        self.gain = P("params/detail_gain", dspin(0.0, 30.0, 2.5, 0.5))
        self.noise = P("params/noise_floor", dspin(0.0, 0.05, 0.002, 0.0005, 4))
        self.window = P("params/mask_window", QSpinBox(minimum=1, maximum=20, value=3))
        self.sel_x = P("view/block_x", QSpinBox(minimum=0, maximum=99, value=2))
        self.sel_y = P("view/block_y", QSpinBox(minimum=0, maximum=99, value=2))
        self.show_grid = P("view/show_grid", QCheckBox("Show block grid", checked=True))

        for w in (self.blocks_x, self.blocks_y, self.target, self.k_min, self.k_max,
                  self.smooth, self.radius, self.sigma, self.k2, self.bias, self.gain,
                  self.noise, self.window):
            w.valueChanged.connect(self.recompute)
        for w in (self.sel_x, self.sel_y):
            w.valueChanged.connect(self.refresh)
        self.show_grid.toggled.connect(self.refresh)
        self.edge_mean.toggled.connect(self.recompute)

        side = [
            group("1 · Blocks", [("Blocks X", self.blocks_x), ("Blocks Y", self.blocks_y)]),
            group("2 · Local stretch", [("Target contrast", self.target),
                                        ("Min gain", self.k_min), ("Max gain", self.k_max),
                                        ("Smoothing (× block)", self.smooth)]),
            group("3 · Guided filter", [("Radius r (px)", self.radius),
                                        ("Edge σ (of range)", self.sigma)]),
            group("4 · Base compression", [("Global weight k2", self.k2),
                                           ("Bias (mid level)", self.bias),
                                           self.edge_mean]),
            group("5 · Detail", [("Gain", self.gain), ("Noise floor", self.noise),
                                 ("Mask window (px)", self.window)]),
            group("Inspect block (or click an image)", [("Block x", self.sel_x),
                                                        ("Block y", self.sel_y),
                                                        self.show_grid]),
        ]

        self.v_in, self.p_in = ImageView(), Plot()
        self.v_blocks, self.p_stats = ImageView(), Plot(ncols=2)
        self.v_kmap, self.p_k = ImageView(), Plot()
        self.v_base, self.v_detail = ImageView(), ImageView()
        self.v_guide, self.v_local, self.v_cbase = ImageView(), ImageView(), ImageView()
        self.v_mask, self.v_denh = ImageView(), ImageView()
        self.v_lin, self.v_clahe, self.v_bd, self.v_out = (ImageView() for _ in range(4))
        self.c_detail = captioned(self.v_detail, "Detail D")
        for v in (self.v_in, self.v_blocks, self.v_kmap, self.v_base, self.v_detail,
                  self.v_guide, self.v_local, self.v_cbase, self.v_mask, self.v_denh,
                  self.v_lin, self.v_clahe, self.v_bd, self.v_out):
            v.clicked.connect(self.select_block_at)

        note = " (Formula: our implementation of the paper's step.)"
        tabs = [
            (step_tab(
                "Step 0 — Input. The raw frame is normalized to float [0, 1] (0.01–99.99 "
                "percentile) at full precision.",
                captioned(self.v_in, "Input (linear stretch)"), self.p_in), "0 Input"),
            (step_tab(
                "Step 1 — Blocks. The image is split into blocks and each block's mean "
                "(brightness) and standard deviation (local contrast) are measured. "
                "Click a block to inspect it.",
                captioned(self.v_blocks, "Blocks"), self.p_stats), "1 Blocks"),
            (step_tab(
                "Step 2 — Local stretching coefficient. Each block gets the gain that "
                "brings its contrast to the target: low-contrast blocks are stretched "
                "(up to the max gain, so noise is not blown up), high-contrast ones are "
                "compressed. The block gains are upsampled with Gaussian weighting into "
                "a smooth per-pixel map." + note,
                captioned(self.v_kmap, "Per-pixel gain k(x)"), self.p_k), "2 Local gain"),
            (step_tab(
                "Step 3 — Guided-filter layering. The base layer B carries the large "
                "dynamic range, the detail layer D = input − B the fine texture (shown "
                "around mid-grey, contrast-stretched).",
                captioned(self.v_base, "Base B"), self.c_detail), "3 Layers"),
            (step_tab(
                "Step 4 — Base compression. Local stretch L = bias + k(x)·(B − μ(x)) gives "
                "every region the target contrast around the same mid level; the global "
                "linear guide map G keeps the overall light/dark order. B′ = k2·G + "
                "(1 − k2)·L. μ(x) is the local mean: with 'edge-aware' it is a guided "
                "filter of B with a block-sized radius, so it follows strong edges; "
                "without it, the Gaussian-upsampled block mean mixes both sides of an "
                "edge and creates halos (switch it off to see)." + note,
                captioned(self.v_guide, "Global guide G"),
                captioned(self.v_local, "Local stretch L"),
                captioned(self.v_cbase, "Compressed base B′")), "4 Base compression"),
            (step_tab(
                "Step 5 — Detail noise mask. The local RMS of D is compared with the noise "
                "floor: where it is at noise level (flat areas, where the eye sees noise "
                "most) the mask suppresses the detail; where there is texture, the detail "
                "is kept and amplified by the gain." + note,
                captioned(self.v_mask, "Noise mask (white = keep)"),
                captioned(self.v_denh, "Masked, amplified detail D′")), "5 Detail mask"),
            (step_tab(
                "Step 6 — Result = B′ + D′, converted to 8 bit only at the end. Compared "
                "with a linear stretch, CLAHE (OpenCV, clip 2, 8×8) and the plain "
                "base + detail method (guided filter, plateau equalization, same radius, "
                "edge σ, gain and noise floor).",
                captioned(self.v_lin, "Linear stretch"), captioned(self.v_clahe, "CLAHE"),
                captioned(self.v_bd, "Base + detail"),
                captioned(self.v_out, "Local optimal contrast")), "6 Result"),
        ]
        self.res = None
        return side, tabs

    def compute(self, raw):
        t0 = time.perf_counter()
        img = to_unit(raw)
        by, bx = self.blocks_y.value(), self.blocks_x.value()
        mean_b, std_b = block_stats(img, by, bx)
        k_b = stretch_coefficient(std_b, self.target.value(), self.k_min.value(),
                                  self.k_max.value())
        k_map = upsample_gaussian(k_b, img.shape, self.smooth.value())
        base = guided_filter(img, img, self.radius.value(), self.sigma.value() ** 2)
        if self.edge_mean.isChecked():
            # (our choice) edge-preserving local mean over about one block
            block_r = max(img.shape[0] // by, img.shape[1] // bx) // 2
            mean_map = guided_filter(base, base, max(block_r, 1), self.sigma.value() ** 2)
        else:
            mean_map = upsample_gaussian(mean_b, img.shape, self.smooth.value())
        detail = img - base
        guide = global_guide(base)
        cbase, local = compress_base(base, k_map, mean_map, guide,
                                     self.bias.value(), self.k2.value())
        mask = detail_noise_mask(detail, self.window.value(), self.noise.value())
        denh = self.gain.value() * mask * detail
        out = np.clip(cbase + denh, 0, 1)
        ms = (time.perf_counter() - t0) * 1000

        # reference: plain base + detail with the same filter, gain and noise floor
        bd_base, *_ = map_base(base, "Plateau equalization", 3.0)
        d2 = detail * detail
        bd = np.clip(bd_base + self.gain.value() * detail * d2
                     / (d2 + self.noise.value() ** 2 + 1e-12), 0, 1)
        img8 = to_8bit(raw)
        self.res = dict(raw=raw, img=img, mean_b=mean_b, std_b=std_b, k_b=k_b,
                        k_map=k_map, base=base, detail=detail, guide=guide, local=local,
                        cbase=cbase, mask=mask, denh=denh, out=out, bd=bd, ms=ms,
                        clahe=cv2.createCLAHE(2.0, (8, 8)).apply(img8))
        for box, top in ((self.sel_x, bx - 1), (self.sel_y, by - 1)):
            box.blockSignals(True)
            box.setMaximum(top)
            box.blockSignals(False)

    def select_block_at(self, x, y):
        r = self.res
        if r is None:
            return
        h, w = r["img"].shape
        bx = min(x * self.blocks_x.value() // w, self.blocks_x.value() - 1)
        by = min(y * self.blocks_y.value() // h, self.blocks_y.value() - 1)
        for box, val in ((self.sel_x, bx), (self.sel_y, by)):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.save_settings()
        self.refresh()

    def block_rect(self, i, j):
        h, w = self.res["img"].shape
        bx, by = self.blocks_x.value(), self.blocks_y.value()
        return (j * w // bx, i * h // by, (j + 1) * w // bx - 1, (i + 1) * h // by - 1)

    def with_grid(self, rgb):
        if self.show_grid.isChecked():
            h, w = rgb.shape[:2]
            bx, by = self.blocks_x.value(), self.blocks_y.value()
            for k in range(1, by):
                cv2.line(rgb, (0, k * h // by), (w, k * h // by), (0, 200, 255), 1)
            for k in range(1, bx):
                cv2.line(rgb, (k * w // bx, 0), (k * w // bx, h), (0, 200, 255), 1)
        x0, y0, x1, y1 = self.block_rect(self.sel_y.value(), self.sel_x.value())
        cv2.rectangle(rgb, (x0, y0), (x1, y1), (255, 40, 40), 2)
        return rgb

    def refresh(self, *_):
        r = self.res
        if r is None:
            return
        img = r["img"]
        i, j = self.sel_y.value(), self.sel_x.value()
        gray = lambda a: cv2.cvtColor(unit_to_8bit(a), cv2.COLOR_GRAY2RGB)

        # Step 0
        self.v_in.set_array(unit_to_8bit(img))
        ax = self.p_in.axes[0]
        ax.clear()
        ax.hist(r["raw"].ravel(), bins=256, histtype="stepfilled")
        ax.set_title("Raw value histogram")
        ax.set_xlabel("raw value")
        self.p_in.draw()

        # Step 1
        self.v_blocks.set_array(self.with_grid(gray(img)))
        for ax, data, title in zip(self.p_stats.axes, (r["mean_b"], r["std_b"]),
                                   ("Block mean μ", "Block std σ")):
            ax.clear()
            ax.imshow(data, cmap="inferno")
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="cyan", lw=2))
            ax.set_title(f"{title}\nselected: {data[i, j]:.4f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        self.p_stats.draw()

        # Step 2
        self.v_kmap.set_array(self.with_grid(heat(r["k_map"])))
        ax = self.p_k.axes[0]
        ax.clear()
        s = np.linspace(1e-4, max(float(r["std_b"].max()) * 1.2, self.target.value() * 2), 300)
        ax.plot(s, stretch_coefficient(s, self.target.value(), self.k_min.value(),
                                       self.k_max.value()), lw=2, label="gain k(σ)")
        ax.scatter(r["std_b"].ravel(), r["k_b"].ravel(), s=12, color="0.4", label="blocks")
        ax.scatter([r["std_b"][i, j]], [r["k_b"][i, j]], s=60, color="red",
                   label=f"block ({j}, {i}): k = {r['k_b'][i, j]:.2f}")
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_xlabel("block std σ")
        ax.set_ylabel("stretching coefficient k")
        ax.set_title("Local stretching coefficient (k > 1 stretch, k < 1 compress)")
        ax.legend(fontsize=8)
        self.p_k.draw()

        # Step 3
        d = r["detail"]
        std = float(d.std()) + 1e-9
        scale = 0.5 / (4 * std)
        self.v_base.set_array(unit_to_8bit(r["base"]))
        self.v_detail.set_array(detail_to_8bit(d, scale))
        self.c_detail.caption.setText(f"Detail D (display ±{4 * std:.4f} = full range)")

        # Step 4
        self.v_guide.set_array(unit_to_8bit(r["guide"]))
        self.v_local.set_array(unit_to_8bit(np.clip(r["local"], 0, 1)))
        self.v_cbase.set_array(unit_to_8bit(r["cbase"]))

        # Step 5
        self.v_mask.set_array(unit_to_8bit(r["mask"]))
        self.v_denh.set_array(detail_to_8bit(r["denh"], scale / max(self.gain.value(), 1e-6)))

        # Step 6
        self.v_lin.set_array(unit_to_8bit(img))
        self.v_clahe.set_array(r["clahe"])
        self.v_bd.set_array(unit_to_8bit(r["bd"]))
        self.v_out.set_array(unit_to_8bit(r["out"]))

        h, w = img.shape
        self.info.setText(
            f"Image {w}×{h}, {self.blocks_x.value()}×{self.blocks_y.value()} blocks\n"
            f"Block ({j}, {i}): μ = {r['mean_b'][i, j]:.4f}, σ = {r['std_b'][i, j]:.4f}, "
            f"k = {r['k_b'][i, j]:.2f}\n"
            f"Pipeline time: {r['ms']:.1f} ms (this PC's CPU)")


if __name__ == "__main__":
    run(LocalContrastDemo)
