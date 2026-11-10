"""
Shared pieces for the step-by-step image-processing demos
(clahe_demo.py, base_detail_demo.py):

  - loading: TIFF (single/multi-page, 8/16-bit, float, LZW...), raw frames, PNG/JPG
  - test data: a synthetic 14-bit thermal-like scene
  - conversions: raw -> 8 bit, raw -> float [0, 1]
  - widgets: clickable ImageView, matplotlib Plot, captioned(), step_tab()
  - StepViewer: main window with the common left panel (open / synthetic image,
    frame and file-in-folder navigation), a tab per step, and QSettings
    persistence in an .ini file next to the demo script.

Install:
    pip install PyQt5 numpy opencv-python matplotlib tifffile imagecodecs
"""
import os
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

RAW_SHAPE = (512, 640)   # height, width of the sensor matrix
RAW_DTYPE = "<u2"        # 16-bit little-endian; adjust to the manufacturer's doc
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".raw", ".bin")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _to_gray_stack(img, bgr=False):
    """Normalize any loaded array to a grayscale stack (frames, h, w).
    Accepts (h, w), (h, w, 3/4), (n, h, w) and (n, h, w, 3/4)."""
    img = np.squeeze(np.asarray(img))
    if img.ndim in (3, 4) and img.shape[-1] in (3, 4):        # colour, drop alpha
        rgb = img[..., :3].astype(np.float32)
        if bgr:
            rgb = rgb[..., ::-1]
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        img = gray if img.dtype.kind == "f" else np.round(gray).astype(img.dtype)
    if img.ndim == 2:
        img = img[None]
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape {img.shape}")
    if img.dtype.kind == "f":
        img = np.nan_to_num(img)
    elif img.dtype == bool:
        img = img.astype(np.uint8) * 255
    return img


def _read_tiff(path):
    """TIFF: tifffile handles multi-page stacks, float, 12/14/16-bit and most
    compressions. Falls back to OpenCV (all pages) if tifffile is missing or
    lacks a codec (LZW/JPEG need the 'imagecodecs' package)."""
    try:
        import tifffile
    except ImportError:
        tifffile = None
    if tifffile is not None:
        try:
            with tifffile.TiffFile(path) as tif:
                if len(tif.series) == 1:        # one image or a described stack
                    return _to_gray_stack(tif.series[0].asarray())
                # several series (pages appended one by one): stack every page
                # with the first page's shape, skipping thumbnails/previews
                first = tif.pages[0].shape
                pages = [p.asarray() for p in tif.pages if p.shape == first]
                return np.concatenate([_to_gray_stack(p) for p in pages])
        except ValueError as e:
            if "imagecodecs" not in str(e):
                raise
    if hasattr(cv2, "imdecodemulti"):          # OpenCV >= 4.7, non-Latin paths OK
        ok, pages = cv2.imdecodemulti(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        ok, pages = cv2.imreadmulti(path, flags=cv2.IMREAD_UNCHANGED)
    if not ok or not pages:
        raise IOError(f"Cannot read {path} (try: pip install tifffile imagecodecs)")
    return np.concatenate([_to_gray_stack(p, bgr=True) for p in pages])


def load_image(path):
    """Load an image file as a grayscale stack (frames, h, w).
    .tif/.tiff: single or multi-page, 8/16-bit, float, RGB(A)
    .raw/.bin:  16-bit frames of RAW_SHAPE, back to back
    others:     anything OpenCV reads (png, jpg, bmp, ...)"""
    p = path.lower()
    if p.endswith((".raw", ".bin")):
        data = np.fromfile(path, dtype=RAW_DTYPE)
        size = RAW_SHAPE[0] * RAW_SHAPE[1]
        n = data.size // size
        if n == 0:
            raise IOError(f"{path} is smaller than one {RAW_SHAPE} frame")
        return data[:n * size].reshape(n, *RAW_SHAPE)
    if p.endswith((".tif", ".tiff")):
        return _read_tiff(path)
    # np.fromfile + imdecode also works with non-Latin paths on Windows
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return _to_gray_stack(img, bgr=True)


def folder_images(path):
    """All image files in the folder of `path`, sorted, with the same extension."""
    folder, ext = os.path.dirname(path) or ".", os.path.splitext(path)[1].lower()
    names = sorted(n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() == ext)
    return [os.path.join(folder, n) for n in names]


def synthetic_image(h=512, w=640):
    """Low-contrast, thermal-like 14-bit scene: smooth background, a faint
    object and a hot object. A global stretch hides the faint details."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 8000 + 600 * (x / w) + 300 * (y / h)            # background gradient
    img[100:220, 80:260] += 40                              # faint object
    img[130:190, 120:220] += 25 * np.sin(x[130:190, 120:220] / 3)  # faint texture
    img[330:460, 380:600] += 2500                           # hot object
    img[330:460, 380:600] += 60 * np.sin(y[330:460, 380:600] / 4)  # detail on it
    cv2.circle(img, (520, 120), 40, 8700, -1)               # warm disc
    img += rng.normal(0, 6, img.shape).astype(np.float32)   # sensor noise
    return np.clip(img, 0, 16383).astype(np.uint16)


# ----------------------------------------------------------------------------
# Conversions
# ----------------------------------------------------------------------------

def to_unit(img, low_pct=0.01, high_pct=99.99):
    """Raw (any bit depth) -> float32 in [0, 1], percentile-clipped, keeping
    the full precision of 14/16-bit data."""
    img = img.astype(np.float32)
    lo, hi = (float(v) for v in np.percentile(img, [low_pct, high_pct]))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def to_8bit(img, low_pct=0.5, high_pct=99.5):
    """Raw -> uint8 by a percentile stretch (8-bit input is returned as is)."""
    if img.dtype == np.uint8:
        return img
    lo, hi = np.percentile(img, [low_pct, high_pct])
    out = (img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def unit_to_8bit(img):
    """Float [0, 1] -> uint8."""
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Widgets
# ----------------------------------------------------------------------------

class ImageView(QLabel):
    """Shows a uint8 grayscale or RGB array, scaled to fit, aspect kept.
    A left click emits `clicked(x, y)` in image pixel coordinates."""

    clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(160, 128)
        self.setCursor(Qt.CrossCursor)
        self._arr = None
        self._scale = 1.0
        self._offset = (0.0, 0.0)

    def set_array(self, arr):
        self._arr = np.ascontiguousarray(arr)
        self._render()

    def resizeEvent(self, event):
        self._render()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self._arr is None or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        # widget position -> position inside the centred, scaled pixmap -> image pixel
        x = (event.pos().x() - self._offset[0]) / self._scale
        y = (event.pos().y() - self._offset[1]) / self._scale
        h, w = self._arr.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.clicked.emit(int(x), int(y))

    def _render(self):
        if self._arr is None:
            return
        a = self._arr
        fmt = QImage.Format_Grayscale8 if a.ndim == 2 else QImage.Format_RGB888
        qimg = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], fmt).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)
        # remember where the pixmap sits, to map clicks back to image pixels
        self._scale = pix.width() / a.shape[1]
        self._offset = ((self.width() - pix.width()) / 2,
                        (self.height() - pix.height()) / 2)


class Plot(FigureCanvasQTAgg):
    def __init__(self, ncols=1):
        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(self.fig)
        self.axes = list(np.atleast_1d(self.fig.subplots(1, ncols)))


def captioned(widget, caption):
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    label = QLabel(caption)
    label.setAlignment(Qt.AlignCenter)
    lay.addWidget(label)
    lay.addWidget(widget, 1)
    box.caption = label          # so a demo can update the text later
    return box


def step_tab(explanation, *widgets):
    tab = QWidget()
    lay = QVBoxLayout(tab)
    text = QLabel(explanation)
    text.setWordWrap(True)
    lay.addWidget(text)
    row = QHBoxLayout()
    for w in widgets:
        row.addWidget(w, 1)
    lay.addLayout(row, 1)
    return tab


def dspin(minimum, maximum, value, step, decimals=2):
    """QDoubleSpinBox with decimals set BEFORE the value (otherwise Qt rounds
    the value to the default 2 decimals, e.g. 0.002 -> 0)."""
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def group(title, rows):
    """QGroupBox with a form: rows are (label, widget) or a lone widget."""
    box = QGroupBox(title)
    form = QFormLayout(box)
    for row in rows:
        if isinstance(row, tuple):
            form.addRow(*row)
        else:
            form.addRow(row)
    return box


# ----------------------------------------------------------------------------
# Main window base class
# ----------------------------------------------------------------------------

class StepViewer(QMainWindow):
    """Common main window. A demo subclass implements:

        build_ui()   create its widgets; return (list of side-panel widgets,
                     list of (tab widget, tab title))
        compute(raw) process one raw frame (any dtype) and store the results
        refresh()    redraw all views from the stored results

    and calls self.persist("section/key", widget) for every widget whose value
    should be saved in the .ini file. Changing a persisted widget saves the
    settings; connect it to self.recompute or self.refresh as needed.
    """

    TITLE = "Step viewer"
    ICON = None              # e.g. "clahe" -> icons/clahe.ico (or .png) next to this file
    SETTINGS_FILE = None     # default: <demo script>.ini next to the script

    def __init__(self, stack=None, source="file", path=None):
        super().__init__()
        self.setWindowTitle(self.TITLE)
        icon = icon_path(self.ICON)
        if icon:
            self.setWindowIcon(QIcon(icon))
        self._persisted = []
        self.last_dir = ""
        self.files = []          # images in the folder of the loaded file
        self.file_index = 0

        # --- common side panel ---
        open_btn = QPushButton("Open image / raw frame…")
        open_btn.clicked.connect(self.open_image)
        synth_btn = QPushButton("Synthetic test image")
        synth_btn.clicked.connect(
            lambda: self.set_stack(synthetic_image()[None], "synthetic image"))

        self.frame = QSpinBox(minimum=0, maximum=0)
        self.frame_label = QLabel()
        self.frame_label.setWordWrap(True)
        self.prev_btn, self.next_btn = QPushButton("◀ Prev"), QPushButton("Next ▶")
        self.prev_btn.clicked.connect(lambda: self.step_file(-1))
        self.next_btn.clicked.connect(lambda: self.step_file(+1))
        nav = QWidget()
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.addWidget(self.prev_btn)
        nav_lay.addWidget(self.next_btn)
        frames = group("Frame / file", [("Frame in file", self.frame),
                                        ("File in folder", nav), self.frame_label])

        self.info = QLabel()
        self.info.setWordWrap(True)

        side_widgets, tabs = self.build_ui()

        side = QVBoxLayout()
        for w in [open_btn, synth_btn, frames] + list(side_widgets) + [self.info]:
            side.addWidget(w)
        side.addStretch(1)
        side_box = QWidget()
        side_box.setLayout(side)
        side_scroll = QScrollArea()             # the panel scrolls on small screens
        side_scroll.setWidget(side_box)
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_box.setFixedWidth(300)
        side_scroll.setFixedWidth(300 + side_scroll.verticalScrollBar().sizeHint().width() + 4)

        self.tabs = QTabWidget()
        for widget, title in tabs:
            self.tabs.addTab(widget, title)

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.addWidget(side_scroll)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.resize(1300, 760)

        # --- settings ---
        ini = self.SETTINGS_FILE or self.default_settings_file()
        self.settings = QSettings(ini, QSettings.IniFormat)
        self.load_settings()
        for _, w in self._persisted:
            self._changed_signal(w).connect(self.save_settings)
        self.tabs.currentChanged.connect(self.save_settings)
        self.frame.valueChanged.connect(self.recompute)

        if stack is None:
            stack, source = synthetic_image()[None], "synthetic image"
        if path:
            self._remember_file(path)
        self.set_stack(stack, source)

    # --- to implement in a demo ---

    def build_ui(self):
        raise NotImplementedError

    def compute(self, raw):
        raise NotImplementedError

    def refresh(self):
        raise NotImplementedError

    # --- settings (<script>.ini) ---

    def default_settings_file(self):
        module = sys.modules.get(type(self).__module__)
        script = getattr(module, "__file__", None) or sys.argv[0] or "viewer"
        return os.path.splitext(os.path.abspath(script))[0] + ".ini"

    def persist(self, key, widget):
        self._persisted.append((key, widget))
        return widget

    @staticmethod
    def _changed_signal(w):
        if isinstance(w, QAbstractButton):
            return w.toggled
        if isinstance(w, QComboBox):
            return w.currentIndexChanged
        return w.valueChanged

    def load_settings(self):
        s = self.settings
        for key, w in self._persisted:   # restore without triggering recomputes
            w.blockSignals(True)
            if isinstance(w, QDoubleSpinBox):
                w.setValue(s.value(key, w.value(), type=float))
            elif isinstance(w, QSpinBox):
                w.setValue(s.value(key, w.value(), type=int))
            elif isinstance(w, QAbstractButton):
                w.setChecked(s.value(key, w.isChecked(), type=bool))
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(s.value(key, w.currentIndex(), type=int))
            w.blockSignals(False)
        self.tabs.setCurrentIndex(s.value("view/tab", 0, type=int))
        self.last_dir = s.value("files/last_dir", "", type=str)
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def save_settings(self, *_):
        s = self.settings
        for key, w in self._persisted:
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                s.setValue(key, w.value())
            elif isinstance(w, QAbstractButton):
                s.setValue(key, w.isChecked())
            elif isinstance(w, QComboBox):
                s.setValue(key, w.currentIndex())
        s.setValue("view/tab", self.tabs.currentIndex())
        s.setValue("files/last_dir", self.last_dir)
        s.setValue("window/geometry", self.saveGeometry())
        s.sync()

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    # --- loading ---

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", self.last_dir,
            "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.raw *.bin);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        self._remember_file(path)
        try:
            stack = load_image(path)
        except Exception as e:  # show the reason instead of crashing the app
            QMessageBox.warning(self, "Cannot open file", f"{path}\n\n{e}")
            return
        self.set_stack(stack, os.path.basename(path))

    def _remember_file(self, path):
        self.last_dir = os.path.dirname(os.path.abspath(path))
        self.save_settings()
        try:
            self.files = folder_images(path)
            self.file_index = [os.path.normcase(os.path.abspath(f)) for f in self.files] \
                .index(os.path.normcase(os.path.abspath(path)))
        except (OSError, ValueError):
            self.files, self.file_index = [path], 0

    def step_file(self, delta):
        if len(self.files) > 1:
            i = (self.file_index + delta) % len(self.files)
            self.load_path(self.files[i])

    def set_stack(self, stack, source):
        self.stack, self.source = stack, source
        n = len(stack)
        self.frame.blockSignals(True)
        self.frame.setMaximum(n - 1)
        self.frame.setValue(0)
        self.frame.blockSignals(False)
        self.frame.setEnabled(n > 1)
        in_folder = source != "synthetic image" and len(self.files) > 1
        self.prev_btn.setEnabled(in_folder)
        self.next_btn.setEnabled(in_folder)
        where = f"file {self.file_index + 1} of {len(self.files)} in folder\n" if in_folder else ""
        self.frame_label.setText(
            f"{source}\n{where}{n} frame(s), {stack.shape[2]}×{stack.shape[1]}, "
            f"{stack.dtype}, values {stack.min()}…{stack.max()}")
        self.recompute()

    def recompute(self, *_):
        self.compute(self.stack[self.frame.value()])
        self.refresh()


def icon_path(name):
    """icons/<name>.ico (Windows) or .png, next to this file; None if missing."""
    if not name:
        return None
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    for ext in (".ico", ".png"):
        path = os.path.join(folder, name + ext)
        if os.path.exists(path):
            return path
    return None


def run(viewer_class):
    """Entry point: python demo.py [image]"""
    if sys.platform == "win32" and viewer_class.ICON:
        # own taskbar entry/icon instead of the generic python.exe one
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f"qulab.imx8.{viewer_class.ICON}")
        except Exception:
            pass
    app = QApplication(sys.argv)
    icon = icon_path(viewer_class.ICON)
    if icon:
        app.setWindowIcon(QIcon(icon))
    win = viewer_class()
    if len(sys.argv) > 1:
        win.load_path(sys.argv[1])
    win.show()
    sys.exit(app.exec_())
