"""
Base / detail layer decomposition step by step (the idea behind thermal-camera
"digital detail enhancement"), with a PyQt5 viewer that shows each step.
Shared loading/UI code lives in viewer_common.py (keep it in the same folder).

Install:
    pip install PyQt5 numpy opencv-python matplotlib tifffile imagecodecs
Run:
    python base_detail_demo.py               # synthetic thermal-like test image
    python base_detail_demo.py frame.tiff    # your data (Prev/Next walks the folder)
Settings are kept in base_detail_demo.ini next to this script.

Steps (one tab each):
    0. Input: raw 14/16-bit -> float [0, 1], full precision (no 8-bit step)
    1. Base layer B = edge-preserving smoothing of the input
         Gaussian  : reference only, blurs across edges -> halos
         Bilateral : weights neighbours by distance AND intensity difference
         Guided    : local linear model (He et al.), O(1) per pixel, no halos
    2. Detail layer D = input - B (small-scale texture and noise)
    3. Base mapping: compress the large dynamic range of B into the display
       range (plateau histogram equalization, or linear)
    4. Detail enhancement: D' = gain * D, with a soft noise floor (coring)
    5. Result = mapped base + enhanced detail, vs linear / global HE / CLAHE
    6. The three filters side by side, with their run times

The edge-strength parameter sigma is in units of the normalized input range:
guided filter eps = sigma^2, bilateral sigmaColor = sigma.
"""
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import QComboBox, QSpinBox

from viewer_common import (
    ImageView, Plot, StepViewer, captioned, dspin, group, run, step_tab, to_8bit,
    to_unit, unit_to_8bit,
)

FILTERS = ("Guided", "Bilateral", "Gaussian")
MAPPINGS = ("Plateau equalization", "Linear")


# ----------------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------------

def box_mean(x, r):
    """Mean over a (2r+1) x (2r+1) window; cost does not depend on r."""
    return cv2.boxFilter(x, cv2.CV_32F, (2 * r + 1, 2 * r + 1),
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """Guided filter (He, Sun, Tang 2010). In each window the output is a
    linear function of the guide, q = a*I + b. Where the window is flat
    (variance << eps) a -> 0 and q is the local mean (smoothing); across a
    strong edge (variance >> eps) a -> 1 and the edge is kept."""
    mean_i = box_mean(guide, r)
    mean_p = box_mean(src, r)
    var_i = box_mean(guide * guide, r) - mean_i * mean_i
    cov_ip = box_mean(guide * src, r) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_mean(a, r) * guide + box_mean(b, r)


def base_layer(img, kind, r, sigma):
    """Step 1: smooth large structures, keep (or not) the strong edges."""
    if kind == "Guided":
        return guided_filter(img, img, r, sigma * sigma)
    if kind == "Bilateral":
        return cv2.bilateralFilter(img, 2 * r + 1, sigma, max(r / 2, 0.5),
                                   borderType=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(img, (0, 0), max(r / 2, 0.5), borderType=cv2.BORDER_REFLECT)


def map_base(base, mode, plateau, bins=1024):
    """Step 3: compress the base layer into [0, 1].
    Plateau equalization = histogram equalization whose bins are capped at
    `plateau` x the mean bin count, so a big uniform area (sky, hot object)
    cannot take over the output range."""
    hist, edges = np.histogram(base, bins=bins, range=(0, 1))
    if mode == "Linear":
        return base.copy(), edges, hist, None, edges[1:]
    limit = plateau * hist.mean()
    clipped = np.minimum(hist, limit)
    cdf = np.cumsum(clipped)
    curve = cdf / max(cdf[-1], 1e-9)
    mapped = np.interp(base, edges[1:], curve).astype(np.float32)
    return mapped, edges, hist, limit, curve


def enhance_detail(detail, gain, noise_floor):
    """Step 4: D' = gain * D * D^2 / (D^2 + t^2). Soft coring: amplitudes well
    above the noise floor t get the full gain, those below it are suppressed."""
    d2 = detail * detail
    return gain * detail * d2 / (d2 + noise_floor * noise_floor + 1e-12)


def detail_to_8bit(detail, scale):
    """Show a signed detail layer around mid-grey."""
    return unit_to_8bit(0.5 + detail * scale)


# ----------------------------------------------------------------------------
# Viewer
# ----------------------------------------------------------------------------

class BaseDetailDemo(StepViewer):
    TITLE = "Base / detail decomposition step by step"
    ICON = "base_detail"

    def build_ui(self):
        self.filter = self.persist("params/filter", QComboBox())
        self.filter.addItems(FILTERS)
        self.radius = self.persist("params/radius", QSpinBox(minimum=1, maximum=40, value=8))
        self.sigma = self.persist("params/edge_sigma", dspin(0.001, 0.5, 0.03, 0.005, 3))
        self.mapping = self.persist("params/base_mapping", QComboBox())
        self.mapping.addItems(MAPPINGS)
        self.plateau = self.persist("params/plateau", dspin(0.1, 50.0, 3.0, 0.5))
        self.gain = self.persist("params/detail_gain", dspin(0.0, 30.0, 3.0, 0.5))
        self.noise = self.persist("params/noise_floor", dspin(0.0, 0.05, 0.002, 0.0005, 4))
        self.row = self.persist("view/profile_row", QSpinBox(minimum=0, maximum=9999, value=395))

        for w in (self.radius, self.sigma, self.plateau, self.gain, self.noise):
            w.valueChanged.connect(self.recompute)
        for w in (self.filter, self.mapping):
            w.currentIndexChanged.connect(self.recompute)
        self.row.valueChanged.connect(self.refresh)

        side = [
            group("1 · Base filter", [("Filter", self.filter), ("Radius r (px)", self.radius),
                                      ("Edge σ (of range)", self.sigma)]),
            group("3 · Base mapping", [("Mapping", self.mapping),
                                       ("Plateau (× mean bin)", self.plateau)]),
            group("4 · Detail", [("Gain", self.gain), ("Noise floor", self.noise)]),
            group("Profile (or click an image)", [("Row y", self.row)]),
        ]

        self.v_in, self.p_in = ImageView(), Plot()
        self.v_base, self.p_profile = ImageView(), Plot()
        self.v_detail, self.p_detail = ImageView(), Plot()
        self.v_mapped, self.p_map = ImageView(), Plot()
        self.v_enh, self.p_transfer = ImageView(), Plot()
        self.v_lin, self.v_he, self.v_clahe, self.v_out = (ImageView() for _ in range(4))
        self.v_cmp = {k: ImageView() for k in FILTERS}
        self.c_cmp = {k: captioned(self.v_cmp[k], k) for k in FILTERS}
        self.c_detail = captioned(self.v_detail, "Detail layer D")
        clickable = [self.v_in, self.v_base, self.v_detail, self.v_mapped, self.v_enh,
                     self.v_lin, self.v_he, self.v_clahe, self.v_out, *self.v_cmp.values()]
        for v in clickable:
            v.clicked.connect(lambda x, y: self.row.setValue(y))

        tabs = [
            (step_tab(
                "Step 0 — Input. The raw frame is normalized to float [0, 1] (0.01–99.99 "
                "percentile) and kept at full precision: all steps run before any 8-bit "
                "conversion. The red line is the profile row used in the next tab — click "
                "an image to move it.",
                captioned(self.v_in, "Input (linear stretch)"), self.p_in), "0 Input"),
            (step_tab(
                "Step 1 — Base layer. Smoothing keeps the large structures. The profile "
                "shows the difference: Gaussian blurs across edges (it overshoots and "
                "undershoots next to them, which become halos after enhancement); the "
                "bilateral and guided filters follow the edge and only flatten the "
                "small texture. Radius r sets the size of what counts as 'detail'; "
                "edge σ sets how strong a step must be to be kept as an edge.",
                captioned(self.v_base, "Base layer B (selected filter)"), self.p_profile),
             "1 Base"),
            (step_tab(
                "Step 2 — Detail layer D = input − B: texture, small objects and noise, "
                "centred on zero (shown around mid-grey, contrast-stretched). With an "
                "edge-preserving base, strong edges stay in B and do not leak into D.",
                self.c_detail, self.p_detail), "2 Detail"),
            (step_tab(
                "Step 3 — Base mapping. The base holds the large dynamic range (e.g. a hot "
                "object next to a cool background). Plateau equalization caps each "
                "histogram bin at plateau × the mean bin count, so large uniform areas "
                "cannot take over the output range. A lower plateau gives a more linear "
                "mapping, a higher one a stronger equalization.",
                captioned(self.v_mapped, "Mapped base B′"), self.p_map), "3 Base mapping"),
            (step_tab(
                "Step 4 — Detail enhancement. D′ = gain · D, with a soft noise floor: "
                "amplitudes well above it get the full gain, those below it (mostly "
                "sensor noise) are suppressed instead of amplified.",
                captioned(self.v_enh, "Enhanced detail D′"), self.p_transfer),
             "4 Detail gain"),
            (step_tab(
                "Step 5 — Result = B′ + D′, clipped to [0, 1] and converted to 8 bit "
                "only at the very end. Compare with a plain linear stretch, global "
                "histogram equalization and CLAHE (OpenCV, clip 2, 8×8 tiles).",
                captioned(self.v_lin, "Linear stretch"), captioned(self.v_he, "Global HE"),
                captioned(self.v_clahe, "CLAHE"), captioned(self.v_out, "Base + detail")),
             "5 Result"),
            (step_tab(
                "Step 6 — Filters compared: the full pipeline with each base filter and "
                "the same parameters. Look for halos around the hot object with Gaussian. "
                "Times are this PC's CPU, for comparison only (the guided filter's cost "
                "does not grow with the radius; the bilateral filter's does).",
                *self.c_cmp.values()), "6 Filters compared"),
        ]
        self.res = None
        return side, tabs

    def pipeline(self, img, kind):
        t0 = time.perf_counter()
        base = base_layer(img, kind, self.radius.value(), self.sigma.value())
        ms = (time.perf_counter() - t0) * 1000
        detail = img - base
        mapped, edges, hist, limit, curve = map_base(
            base, self.mapping.currentText(), self.plateau.value())
        enh = enhance_detail(detail, self.gain.value(), self.noise.value())
        out = np.clip(mapped + enh, 0, 1)
        return dict(base=base, detail=detail, mapped=mapped, edges=edges, hist=hist,
                    limit=limit, curve=curve, enh=enh, out=out, ms=ms)

    def compute(self, raw):
        img = to_unit(raw)
        runs = {k: self.pipeline(img, k) for k in FILTERS}
        img8 = to_8bit(raw)
        self.res = dict(raw=raw, img=img, runs=runs, sel=runs[self.filter.currentText()],
                        he=cv2.equalizeHist(img8),
                        clahe=cv2.createCLAHE(2.0, (8, 8)).apply(img8))
        self.row.blockSignals(True)
        self.row.setMaximum(img.shape[0] - 1)
        self.row.blockSignals(False)

    def with_row(self, gray8):
        rgb = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)
        y = self.row.value()
        cv2.line(rgb, (0, y), (rgb.shape[1] - 1, y), (255, 40, 40), 1)
        return rgb

    def refresh(self, *_):
        r = self.res
        if r is None:
            return
        img, sel, runs = r["img"], r["sel"], r["runs"]
        y = self.row.value()
        kind = self.filter.currentText()

        # Step 0
        self.v_in.set_array(self.with_row(unit_to_8bit(img)))
        ax = self.p_in.axes[0]
        ax.clear()
        ax.hist(r["raw"].ravel(), bins=256, histtype="stepfilled")
        ax.set_title("Raw value histogram")
        ax.set_xlabel("raw value")
        self.p_in.draw()

        # Step 1
        self.v_base.set_array(self.with_row(unit_to_8bit(sel["base"])))
        ax = self.p_profile.axes[0]
        ax.clear()
        ax.plot(img[y], color="0.6", lw=0.8, label="input")
        styles = {"Gaussian": dict(ls="--"), "Bilateral": dict(), "Guided": dict()}
        for k in FILTERS:
            ax.plot(runs[k]["base"][y], lw=2 if k == kind else 1.2, label=f"{k} base",
                    **styles[k])
        ax.set_title(f"Profile along row y = {y}")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("normalized value")
        ax.legend(fontsize=8)
        self.p_profile.draw()

        # Step 2
        d = sel["detail"]
        std = float(d.std()) + 1e-9
        scale = 0.5 / (4 * std)
        self.v_detail.set_array(detail_to_8bit(d, scale))
        self.c_detail.caption.setText(f"Detail layer D (display ±{4 * std:.4f} = full range)")
        ax = self.p_detail.axes[0]
        ax.clear()
        ax.hist(d.ravel(), bins=200, range=(-6 * std, 6 * std), histtype="stepfilled")
        t = self.noise.value()
        if t > 0:
            for s in (-t, t):
                ax.axvline(s, color="red", ls="--", lw=1)
        ax.set_title("Detail histogram (red: noise floor ±t)")
        ax.set_xlabel("D")
        self.p_detail.draw()

        # Step 3
        self.v_mapped.set_array(unit_to_8bit(sel["mapped"]))
        ax = self.p_map.axes[0]
        ax.clear()
        centers = (sel["edges"][:-1] + sel["edges"][1:]) / 2
        ax.fill_between(centers, sel["hist"], step="mid", alpha=0.5, label="base histogram")
        if sel["limit"] is not None:
            ax.axhline(sel["limit"], color="red", ls="--", lw=1, label="plateau")
        ax.set_xlabel("base value B")
        ax.set_ylabel("pixel count")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax2 = ax.twinx() if not hasattr(self, "_map_ax2") else self._map_ax2
        self._map_ax2 = ax2
        ax2.clear()
        ax2.plot(sel["edges"][1:], sel["curve"], color="k", lw=2, label="mapping B → B′")
        ax2.plot([0, 1], [0, 1], color="k", ls=":", lw=0.8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("B′")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.legend(fontsize=8, loc="lower right")
        ax.set_title("Base histogram and mapping curve")
        self.p_map.draw()

        # Step 4
        self.v_enh.set_array(detail_to_8bit(sel["enh"], scale / max(self.gain.value(), 1e-6)))
        ax = self.p_transfer.axes[0]
        ax.clear()
        x = np.linspace(-6 * std, 6 * std, 400)
        ax.plot(x, enhance_detail(x, self.gain.value(), t), lw=2, label="D′ (with noise floor)")
        ax.plot(x, self.gain.value() * x, ls="--", lw=1, label="gain · D (no noise floor)")
        ax.plot(x, x, color="k", ls=":", lw=0.8, label="identity")
        ax.set_xlabel("D")
        ax.set_ylabel("D′")
        ax.set_title("Detail transfer curve")
        ax.legend(fontsize=8)
        self.p_transfer.draw()

        # Step 5
        self.v_lin.set_array(unit_to_8bit(img))
        self.v_he.set_array(r["he"])
        self.v_clahe.set_array(r["clahe"])
        self.v_out.set_array(unit_to_8bit(sel["out"]))

        # Step 6
        for k in FILTERS:
            self.v_cmp[k].set_array(unit_to_8bit(runs[k]["out"]))
            self.c_cmp[k].caption.setText(f"{k} base — {runs[k]['ms']:.1f} ms")

        h, w = img.shape
        self.info.setText(
            f"Image {w}×{h}\n"
            f"Base filter time: " + ", ".join(f"{k} {runs[k]['ms']:.1f} ms" for k in FILTERS)
            + f"\nGuided eps = σ² = {self.sigma.value() ** 2:.2e}\n"
            f"Detail std = {std:.4f}, noise floor = {t:g}")


if __name__ == "__main__":
    run(BaseDetailDemo)
