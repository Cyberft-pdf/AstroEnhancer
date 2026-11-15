"""Pixel RAW Enhancer – desktop app for enhancing DNG photos from Pixel phones.

Usage:  python app.py [file.dng]
"""
from __future__ import annotations

import os
import sys
import traceback
from dataclasses import asdict, replace

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QScrollArea, QSlider, QSplitter,
    QVBoxLayout, QWidget,
)

import pipeline
from pipeline import Params

PREVIEW_MAX = 1400          # delší strana zmenšeného náhledu
CROP_SIZE = (1400, 1000)    # velikost výřezu 100 %

RAW_FILTER = "RAW files (*.dng *.DNG *.nef *.cr2 *.cr3 *.arw *.raf *.rw2 *.orf);;All files (*)"

# (atribut, popisek, min, max, dělitel)
SLIDERS = {
    "Tone & Color": [
        ("exposure", "Exposure (EV)", -300, 800, 100),
        ("temperature", "Temperature", -100, 100, 1),
        ("tint", "Tint", -100, 100, 1),
        ("highlights", "Highlights", -100, 100, 1),
        ("shadows", "Shadows", -100, 100, 1),
        ("contrast", "Contrast", -100, 100, 1),
        ("vibrance", "Vibrance", -100, 100, 1),
        ("saturation", "Saturation", -100, 100, 1),
    ],
    "Night Sky": [
        ("bg_removal", "Background flattening", 0, 100, 1),
        ("black_point", "Black point (% of sky)", 0, 1000, 10),
        ("structure", "Sky structure", 0, 100, 1),
        ("stars", "Star brightness", 0, 100, 1),
    ],
    "Noise Reduction": [
        ("denoise_luma", "Luminance noise", 0, 100, 1),
        ("denoise_chroma", "Color noise", 0, 100, 1),
        ("smooth", "Smoothing", 0, 100, 1),
    ],
    "Sharpness & Detail": [
        ("clarity", "Clarity", -100, 100, 1),
        ("sharpen_amount", "Sharpening", 0, 100, 1),
        ("sharpen_radius", "Radius (px)", 3, 30, 10),
        ("sharpen_masking", "Masking", 0, 100, 1),
    ],
}


# Jednoduché funkce: (klíč, popisek, nápověda). Každá posouvá několik
# podrobných posuvníků najednou – viz apply_macros().
MACROS = [
    ("stars", "Boost stars", "Right: brighter stars and more faint stars.\nLeft: smaller, dimmer stars."),
    ("milky", "Boost Milky Way", "Strengthens the glow and dark lanes of the Milky Way."),
    ("dark", "Darker sky", "Right: blacker background and more contrast.\nLeft: lighter, greyer sky."),
    ("smooth", "Smoother sky", "Right: smoother glow without grain (fewer faint stars).\nLeft: more detail, but more grain."),
    ("color", "More vivid colors", "Brings out the colors of the stars and the Milky Way."),
    ("warm", "Warmer tone", "Right: warmer (more yellow). Left: cooler (more blue)."),
]


def apply_macros(base: Params, m: dict) -> Params:
    """Výchozí nastavení (z Auto) + jednoduché funkce -> podrobné parametry."""
    def c(v, lo, hi):
        return float(min(hi, max(lo, v)))

    p = replace(base)
    s = m.get("stars", 0.0)
    p.stars = c(base.stars + 45 * s, 0, 100)
    k = m.get("milky", 0.0)
    p.structure = c(base.structure + 50 * k, 0, 100)
    p.exposure = c(base.exposure + 0.8 * k, -3, 8)
    p.clarity = c(base.clarity + 15 * k, -100, 100)
    d = m.get("dark", 0.0)
    if base.black_point > 0:
        p.black_point = c(base.black_point + 1.2 * d, 0, 99.5)
    else:
        p.shadows = c(base.shadows - 40 * d, -100, 100)
    p.contrast = c(base.contrast + 20 * d, -100, 100)
    h = m.get("smooth", 0.0)
    p.smooth = c(base.smooth + 40 * h - 15 * max(s, 0), 0, 100)
    if base.black_point <= 0:                       # běžná fotka: hladší = víc odšumění
        p.denoise_luma = c(base.denoise_luma + 30 * h, 0, 100)
    col = m.get("color", 0.0)
    p.saturation = c(base.saturation + 60 * col, -100, 100)
    p.vibrance = c(base.vibrance + 20 * col, -100, 100)
    p.temperature = c(base.temperature + 40 * m.get("warm", 0.0), -100, 100)
    return p


# ---------------------------------------------------------------- práce na pozadí

class _Signals(QObject):
    done = Signal(object, object)   # (štítek, výsledek)
    failed = Signal(object, str)


class Job(QRunnable):
    """Spustí funkci ve vlákně a výsledek pošle signálem do GUI."""

    def __init__(self, tag, fn, *args):
        super().__init__()
        self.tag, self.fn, self.args = tag, fn, args
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.tag, self.fn(*self.args))
        except Exception as exc:
            traceback.print_exc()
            self.signals.failed.emit(self.tag, str(exc))


# ---------------------------------------------------------------- widgety

class ImageView(QWidget):
    clicked = Signal(float, float)  # relativní pozice kliknutí v obrázku (0..1)

    def __init__(self):
        super().__init__()
        self._pix: QPixmap | None = None
        self._rect = None
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background:#1e1e1e;")
        self.text = "Open a RAW file (Ctrl+O) or drag it here"

    def set_image(self, rgb8: np.ndarray):
        h, w = rgb8.shape[:2]
        qimg = QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pix = QPixmap.fromImage(qimg.copy())
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._pix is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text)
            return
        scaled = self._pix.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        if scaled.width() > self._pix.width():   # nezvětšovat nad 100 %
            scaled = self._pix.size()
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._rect = (x, y, scaled.width(), scaled.height())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(x, y, scaled.width(), scaled.height(), self._pix)

    def mousePressEvent(self, ev):
        if self._rect and ev.button() == Qt.MouseButton.LeftButton:
            x, y, w, h = self._rect
            rx = (ev.position().x() - x) / w
            ry = (ev.position().y() - y) / h
            if 0 <= rx <= 1 and 0 <= ry <= 1:
                self.clicked.emit(rx, ry)


class ParamSlider(QWidget):
    changed = Signal()

    def __init__(self, lo: int, hi: int, div: int):
        super().__init__()
        self.div = div
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.label = QLabel("0")
        self.label.setFixedWidth(44)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, _):
        self._refresh()
        self.changed.emit()

    def _refresh(self):
        v = self.value()
        self.label.setText(f"{v:+.2f}" if self.div == 100 else (f"{v:.1f}" if self.div == 10 else f"{v:.0f}"))

    def value(self) -> float:
        return self.slider.value() / self.div

    def set_value(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self.div)))
        self.slider.blockSignals(False)
        self._refresh()


# ---------------------------------------------------------------- hlavní okno

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel RAW Enhancer")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.path: str | None = None
        self.lin: np.ndarray | None = None       # plné rozlišení, lineární uint16
        self.small: np.ndarray | None = None     # zmenšený náhled
        self.small_scale = 1.0
        self.bg = None                           # model pozadí oblohy
        self.crop_center = (0.5, 0.5)
        self.base_ev = 0.0
        self.render_id = 0
        self.rendering = False
        self.pending = False
        self.sliders: dict[str, ParamSlider] = {}
        self.base = Params()

        self.view = ImageView()
        self.view.clicked.connect(self._on_view_click)

        panel = QWidget()
        pl = QVBoxLayout(panel)

        # --- jednoduché funkce
        quick = QGroupBox("Quick adjustments")
        ql = QVBoxLayout(quick)
        row = QHBoxLayout()
        self.btn_astro = QPushButton("Night sky")
        self.btn_astro.setToolTip("Automatically sets everything up for stars and the Milky Way.")
        self.btn_astro.clicked.connect(self.apply_astro)
        self.btn_auto = QPushButton("Regular photo")
        self.btn_auto.setToolTip("Automatically sets everything up for a regular (daytime) shot.")
        self.btn_auto.clicked.connect(self.apply_auto)
        row.addWidget(self.btn_astro)
        row.addWidget(self.btn_auto)
        ql.addLayout(row)
        self.macros: dict[str, ParamSlider] = {}
        for key, label, tip in MACROS:
            lab = QLabel(label)
            lab.setToolTip(tip)
            ms = ParamSlider(-100, 100, 1)
            ms.setToolTip(tip)
            ms.set_value(0)
            ms.changed.connect(self._on_macro_changed)
            self.macros[key] = ms
            ql.addWidget(lab)
            ql.addWidget(ms)
        self.btn_zero = QPushButton("Reset quick adjustments to zero")
        self.btn_zero.clicked.connect(self._zero_macros)
        ql.addWidget(self.btn_zero)
        pl.addWidget(quick)

        # --- podrobné posuvníky (schované, dokud je uživatel nechce)
        self.adv_toggle = QCheckBox("Show detailed settings (linked to quick adjustments)")
        self.adv_toggle.toggled.connect(lambda on: self.advanced.setVisible(on))
        pl.addWidget(self.adv_toggle)
        self.advanced = QWidget()
        al = QVBoxLayout(self.advanced)
        al.setContentsMargins(0, 0, 0, 0)
        for group, items in SLIDERS.items():
            box = QGroupBox(group)
            form = QFormLayout(box)
            for attr, label, lo, hi, div in items:
                s = ParamSlider(lo, hi, div)
                s.changed.connect(self._on_advanced_changed)
                self.sliders[attr] = s
                form.addRow(label, s)
            al.addWidget(box)
        pl.addWidget(self.advanced)
        self.adv_toggle.setChecked(True)
        box = QGroupBox("Crop (% from each side)")
        form = QFormLayout(box)
        self.crop_sliders: dict[str, ParamSlider] = {}
        for key, label in (("top", "Top"), ("bottom", "Bottom"), ("left", "Left"), ("right", "Right")):
            cs = ParamSlider(0, 45, 1)
            cs.set_value(0)
            cs.changed.connect(self.schedule_render)
            self.crop_sliders[key] = cs
            form.addRow(label, cs)
        pl.addWidget(box)
        self.lens_check = QCheckBox("Lens shading correction (GainMap from DNG)")
        self.lens_check.setChecked(True)
        self.lens_check.setToolTip("Applied when the file is loaded – changing it reloads the file.")
        self.lens_check.toggled.connect(lambda _: self.path and self.open_file(self.path))
        pl.addWidget(self.lens_check)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        pl.addWidget(self.info)
        pl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)

        split = QSplitter()
        split.addWidget(self.view)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        self.setCentralWidget(split)

        tb = self.addToolBar("Main")
        tb.setMovable(False)
        self.act_open = QAction("Open…", self, shortcut=QKeySequence.StandardKey.Open, triggered=self.open_dialog)
        self.act_auto = QAction("Auto", self, shortcut="Ctrl+U", triggered=self.apply_auto)
        self.act_astro = QAction("Auto night sky", self, shortcut="Ctrl+Shift+U", triggered=self.apply_astro)
        self.act_astro.setToolTip("Settings for stars and the Milky Way: removes the bright background,\n"
                                  "darkens the sky and enhances structure (Ctrl+Shift+U)")
        self.act_reset = QAction("Reset", self, shortcut="Ctrl+R", triggered=self.reset_params)
        self.act_before = QAction("Before / After", self, shortcut="B", checkable=True, triggered=self.schedule_render)
        self.act_crop = QAction("100% crop", self, shortcut="Z", checkable=True, triggered=self.schedule_render)
        self.act_exact = QAction("Exact preview", self, shortcut="P", checkable=True, triggered=self.schedule_render)
        self.act_exact.setToolTip("The preview is computed from full resolution, so it matches the export exactly\n"
                                  "(especially noise and smoothing). It is slower (P).")
        self.act_export = QAction("Export…", self, shortcut="Ctrl+E", triggered=self.export_dialog)
        for a in (self.act_open, self.act_auto, self.act_astro, self.act_reset, self.act_before, self.act_crop,
                  self.act_exact, self.act_export):
            tb.addAction(a)
        self.act_auto.setToolTip("Automatic settings based on the image content (Ctrl+U)")
        self.act_before.setToolTip("Show the image without adjustments (B)")
        self.act_crop.setToolTip("Full-resolution crop for judging noise and sharpness (Z).\n"
                                 "Click in the preview to choose where to crop.")

        self.timer = QTimer(self, singleShot=True, interval=80, timeout=self._start_render)
        self._set_enabled(False)
        self.set_params(Params())
        self.statusBar().showMessage("Ready")

    # ---- parametry

    def params(self) -> Params:
        return Params(**{k: s.value() for k, s in self.sliders.items()})

    def _show_params(self, p: Params):
        for k, v in asdict(p).items():
            self.sliders[k].set_value(v)

    def set_params(self, p: Params):
        """Nové výchozí nastavení (Auto, Reset): rychlé úpravy se vynulují."""
        self.base = replace(p)
        for ms in self.macros.values():
            ms.set_value(0)
        self._show_params(p)
        self.schedule_render()

    def _on_macro_changed(self):
        self._show_params(apply_macros(self.base, self._macro_values()))
        self.schedule_render()

    def _macro_values(self) -> dict:
        return {k: ms.value() / 100.0 for k, ms in self.macros.items()}

    def _on_advanced_changed(self):
        # Ruční změna podrobného posuvníku: rychlé úpravy zůstanou, kde jsou.
        # Posune se výchozí stav tak, aby "výchozí + rychlé úpravy" dalo právě
        # to, co je teď nastaveno – další pohyb rychlou úpravou na to naváže.
        cur = asdict(self.params())
        calc = asdict(apply_macros(self.base, self._macro_values()))
        base = asdict(self.base)
        for k in cur:
            if abs(cur[k] - calc[k]) > 1e-9:
                base[k] += cur[k] - calc[k]
        self.base = Params(**base)
        self.schedule_render()

    def _zero_macros(self):
        self.set_params(self.base)      # zpět na výchozí stav včetně ručních změn v podrobnostech

    def reset_params(self):
        self.set_params(Params(exposure=self.base_ev))

    def apply_auto(self):
        if self.lin is not None:
            self.set_params(pipeline.auto_params(self.lin))

    def apply_astro(self):
        if self.lin is not None:
            self.set_params(pipeline.auto_astro_params(self.lin, self.small, self.bg))

    def crop_box(self):
        """Ořez jako podíly (top, bottom, left, right)."""
        return tuple(self.crop_sliders[k].value() / 100.0 for k in ("top", "bottom", "left", "right"))

    @staticmethod
    def apply_crop(img: np.ndarray, box) -> np.ndarray:
        t, b, l, r = box
        h, w = img.shape[:2]
        return img[int(round(t * h)):h - int(round(b * h)), int(round(l * w)):w - int(round(r * w))]

    def _set_enabled(self, on: bool):
        for a in (self.act_auto, self.act_astro, self.act_reset, self.act_before, self.act_crop,
                  self.act_exact, self.act_export, self.btn_astro, self.btn_auto):
            a.setEnabled(on)

    # ---- načtení

    def open_dialog(self):
        start = os.path.dirname(self.path) if self.path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open RAW", start, RAW_FILTER)
        if path:
            self.open_file(path)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = ev.mimeData().urls()
        if urls:
            self.open_file(urls[0].toLocalFile())

    def open_file(self, path: str):
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
        self._set_enabled(False)
        job = Job(path, pipeline.load_raw, path, self.lens_check.isChecked())
        job.signals.done.connect(self._on_loaded)
        job.signals.failed.connect(self._on_failed)
        self.pool.start(job)

    def _on_loaded(self, path, result):
        lin, info = result
        self.path, self.lin = path, lin
        h, w = lin.shape[:2]
        self.small_scale = min(1.0, PREVIEW_MAX / max(h, w))
        self.small = cv2.resize(lin, None, fx=self.small_scale, fy=self.small_scale,
                                interpolation=cv2.INTER_AREA) if self.small_scale < 1 else lin
        self.bg = pipeline.fit_background(self.small)
        self.crop_center = (0.5, 0.5)
        self.base_ev = pipeline.auto_exposure(lin)
        self.setWindowTitle(f"Pixel RAW Enhancer – {os.path.basename(path)}")
        lens = (f"applied ({info['gain_maps']} maps)" if info.get("gain_maps")
                else "not found in file" if self.lens_check.isChecked() else "off")
        self.info.setText(f"{w} × {h} px ({w * h / 1e6:.1f} MP)\nLens correction: {lens}")
        self._set_enabled(True)
        self.apply_auto()

    def _on_failed(self, _tag, msg):
        self.statusBar().showMessage("Error")
        self._set_enabled(self.lin is not None)
        QMessageBox.critical(self, "Error", msg)

    # ---- náhled

    def _on_view_click(self, rx, ry):
        if self.lin is None:
            return
        if not self.act_crop.isChecked():
            t, b, l, r = self.crop_box()
            self.crop_center = (l + rx * (1 - l - r), t + ry * (1 - t - b))
            self.act_crop.setChecked(True)
        else:
            self.act_crop.setChecked(False)
        self.schedule_render()

    def _preview_source(self):
        if not self.act_crop.isChecked():
            return self.small, self.small_scale, (0.0, 0.0, 1.0, 1.0)
        h, w = self.lin.shape[:2]
        cw, ch = min(CROP_SIZE[0], w), min(CROP_SIZE[1], h)
        x0 = int(np.clip(self.crop_center[0] * w - cw / 2, 0, w - cw))
        y0 = int(np.clip(self.crop_center[1] * h - ch / 2, 0, h - ch))
        region = (x0 / w, y0 / h, (x0 + cw) / w, (y0 + ch) / h)
        return self.lin[y0:y0 + ch, x0:x0 + cw], 1.0, region

    def schedule_render(self, *_):
        if self.lin is not None:
            self.timer.start()

    def _start_render(self):
        if self.lin is None:
            return
        if self.rendering:          # jeden výpočet najednou, další až po dokončení
            self.pending = True
            return
        self.rendering = True
        self.render_id += 1
        src, scale, region = self._preview_source()
        p = Params(exposure=self.base_ev) if self.act_before.isChecked() else self.params()
        if self.act_exact.isChecked() and not self.act_crop.isChecked():
            fit = self.small_scale

            def exact(lin, p, bg):          # celé plné rozlišení, pak zmenšit
                out = pipeline.process(lin, p, 1.0, bg)
                return cv2.resize(out, None, fx=fit, fy=fit, interpolation=cv2.INTER_AREA) if fit < 1 else out

            job = Job(self.render_id, exact, self.lin, p, self.bg)
        else:
            job = Job(self.render_id, pipeline.process, src, p, scale, self.bg, region)
        job.signals.done.connect(self._on_rendered)
        job.signals.failed.connect(self._on_render_failed)
        self.statusBar().showMessage("Rendering preview…")
        self.pool.start(job)

    def _on_rendered(self, _rid, rgb16):
        self.rendering = False
        if not self.act_crop.isChecked():
            rgb16 = self.apply_crop(rgb16, self.crop_box())
        self.view.set_image(np.ascontiguousarray((rgb16 >> 8).astype(np.uint8)))
        mode = ("100% crop" if self.act_crop.isChecked()
                else "full image (exact)" if self.act_exact.isChecked() else "full image (fast)")
        state = "without adjustments" if self.act_before.isChecked() else "with adjustments"
        self.statusBar().showMessage(f"Preview: {mode}, {state}")
        if self.pending:
            self.pending = False
            self._start_render()

    def _on_render_failed(self, _rid, msg):
        self.rendering = False
        self.statusBar().showMessage(f"Preview error: {msg}")

    # ---- export

    def export_dialog(self):
        if self.lin is None:
            return
        base = os.path.splitext(self.path)[0] + "_edited.jpg"
        path, flt = QFileDialog.getSaveFileName(
            self, "Export", base, "JPEG (*.jpg);;TIFF 16-bit (*.tif);;PNG 16-bit (*.png)")
        if not path:
            return
        if "." not in os.path.basename(path):
            path += {"J": ".jpg", "T": ".tif", "P": ".png"}[flt[0]]

        bg = self.bg

        box = self.crop_box()

        def work(lin, p, out):
            img = self.apply_crop(pipeline.process(lin, p, 1.0, bg), box)
            pipeline.save_image(out, np.ascontiguousarray(img))
            return out

        self.progress = QProgressDialog("Exporting at full resolution…", "", 0, 0, self)
        self.progress.setCancelButton(None)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        job = Job("export", work, self.lin, replace(self.params()), path)
        job.signals.done.connect(self._on_exported)
        job.signals.failed.connect(self._on_export_failed)
        self.pool.start(job)

    def _on_exported(self, _tag, out):
        self.progress.close()
        self.statusBar().showMessage(f"Saved: {out}")

    def _on_export_failed(self, _tag, msg):
        self.progress.close()
        QMessageBox.critical(self, "Export failed", msg)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win.open_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
