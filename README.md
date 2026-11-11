# imx8-video-pipeline

Step-by-step demos of contrast enhancement for thermal (LWIR) video. They are used to prepare
the image-processing pipeline for an embedded board (NXP i.MX 8M Plus or Rockchip).

Each demo is a small PyQt5 app. It runs one algorithm on a frame and shows every
intermediate step in its own tab, so you can see what each parameter changes.

## Demos

| App | What it shows |
|---|---|
| `clahe_demo.py` | CLAHE implemented from scratch in NumPy: tiling, per-tile histograms, clipping, CDF → LUT, bilinear interpolation, and a comparison with OpenCV `cv2.createCLAHE` (matches within 1 grey level). Click a tile on the image to inspect it. |
| `base_detail_demo.py` | Base/detail layer decomposition with a guided, bilateral or Gaussian filter. Plateau equalization of the base layer, detail gain with soft coring, and recombination to 8 bit. |
| `local_contrast_demo.py` | Local optimal contrast dynamic range compression: block statistics, stretch coefficients, base compression and a detail noise mask. Follows the structure of Zhu et al., *Sensors* 2023 ([doi:10.3390/s23218860](https://doi.org/10.3390/s23218860)). Some formulas are our own until they are checked against the paper. |

Shared code is in `viewer_common.py`: image loaders, clickable image views, plots, step tabs,
folder navigation and settings persistence.

## Install

Python 3.10 or newer:

```
pip install PyQt5 numpy opencv-python matplotlib tifffile imagecodecs
```

`imagecodecs` is needed for LZW-compressed TIFFs. Without it the loader falls back to OpenCV.

## Run

```
python clahe_demo.py                  # synthetic thermal-like test image
python clahe_demo.py data/<frame>.tiff
python base_detail_demo.py image.png
python local_contrast_demo.py frames.raw
```

Supported inputs:
- **TIFF:** single or multi-page; 8, 12, 14 or 16-bit, or float.
- **Any image OpenCV reads.**
- **`.raw` / `.bin`:** 16-bit little-endian frames, back to back. The frame size is
  `RAW_SHAPE` in `viewer_common.py`, currently 512 × 640. The PICO640S sensor is 640 × 480,
  so change it to `(480, 640)` for raw frames from that sensor.

In the app:
- Open a file, or step through the frames of a folder with Prev/Next.
- Pick a page of a multi-page file with the frame spin box.
- Parameters, the selected tile, the last folder and the window geometry are saved to
  `<app>.ini` next to the script. The `.ini` files are git-ignored.

## Files

```
viewer_common.py          shared UI and loaders
clahe_demo.py             CLAHE demo
base_detail_demo.py       base/detail demo (also used by local_contrast_demo.py)
local_contrast_demo.py    local optimal contrast DRC demo
make_icons.py             generates icons/*.ico and *.png
icons/                    app icons
data/                     sample thermal frames (TIFF)
docs/                     discussion notes for the project
fig1_*.png, fig2_*.png    camera/processing board block diagrams
```

## Adding a new demo

Subclass `StepViewer` from `viewer_common.py` and implement:
- `build_ui()`: add the parameter widgets, and call `self.persist(key, widget)` for each one
  so it is saved to the `.ini`.
- `compute()`: run the algorithm on the current frame.
- `refresh()`: update the step tabs.

Set `ICON = "<name>"` to use `icons/<name>.ico`, and start the app with `run(MyDemo)`.
`clahe_demo.py` is the shortest example.

## Target platform (for reference)

- **i.MX 8M Plus:** 4× Cortex-A53 at 1.8 GHz, Vivante GC7000UL GPU (OpenCL/OpenVX), NPU 2.3 TOPS,
  LPDDR4. Cameras connect only over MIPI CSI-2.
- **Requirements:** 50 fps, end-to-end latency under 50 ms.

Priorities, in order:
1. Hot-air (turbulence) correction
2. Stabilization
3. Focus
4. HDR/AGC (these demos)
