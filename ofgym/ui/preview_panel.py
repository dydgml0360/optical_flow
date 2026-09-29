"""가운데 2D 프리뷰: 소스(캡처/재구성) x 뷰 x 레이어를 골라 그린다."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import VIEWS
from ofgym.data import Sample
from ofgym.recon import ReconOutput
from ofgym.ui.imageio import load_layer

SOURCE_CAPTURE = "캡처"
SOURCE_RECON = "재구성 3D"


class PreviewPanel(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._sample: Optional[Sample] = None
        self._recon: Optional[ReconOutput] = None
        self._pixmap: Optional[QPixmap] = None
        # 콤보를 코드로 채우는 동안 신호가 재진입하는 걸 막는다.
        self._loading = False

        self._source_combo = QComboBox()
        self._source_combo.currentTextChanged.connect(self._on_source_changed)

        self._view_combo = QComboBox()
        self._view_combo.currentTextChanged.connect(self._on_view_changed)

        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)

        self._info = QLabel("-")
        self._info.setStyleSheet("color: palette(mid);")

        bar = QHBoxLayout()
        bar.addWidget(QLabel("소스"))
        bar.addWidget(self._source_combo)
        bar.addSpacing(10)
        bar.addWidget(QLabel("뷰"))
        bar.addWidget(self._view_combo)
        bar.addSpacing(10)
        bar.addWidget(QLabel("레이어"))
        bar.addWidget(self._layer_combo, 1)

        self._canvas = QLabel("샘플을 고르세요")
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(320, 240)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._canvas.setStyleSheet("background: #1b1b1b; border-radius: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(bar)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._info)

    def set_sample(self, sample: Sample, recon: Optional[ReconOutput]) -> None:
        keep_source = self._source_combo.currentText()
        self._sample = sample
        self._recon = recon

        self._loading = True
        try:
            self._source_combo.clear()
            self._source_combo.addItem(SOURCE_CAPTURE)
            if recon is not None:
                self._source_combo.addItem(SOURCE_RECON)
            if keep_source and self._source_combo.findText(keep_source) >= 0:
                self._source_combo.setCurrentText(keep_source)
            self._refill_views()
            self._refill_layers()
        finally:
            self._loading = False
        self._render()

    # ── 콤보 채우기 ────────────────────────────────────────────────────────
    def _available_views(self) -> List[str]:
        if self._sample is None:
            return []
        if self._source_combo.currentText() == SOURCE_RECON and self._recon is not None:
            return self._recon.views
        return [v for v in VIEWS if self._sample.views.get(v)]

    def _available_layers(self) -> Dict[str, object]:
        view = self._view_combo.currentText()
        if not view or self._sample is None:
            return {}
        if self._source_combo.currentText() == SOURCE_RECON and self._recon is not None:
            return self._recon.layers(view)
        holder = self._sample.views.get(view)
        return dict(holder.layers) if holder else {}

    def _refill_views(self) -> None:
        previous = self._view_combo.currentText()
        self._view_combo.clear()
        self._view_combo.addItems(self._available_views())
        if previous and self._view_combo.findText(previous) >= 0:
            self._view_combo.setCurrentText(previous)

    def _refill_layers(self) -> None:
        previous = self._layer_combo.currentText()
        self._layer_combo.clear()
        self._layer_combo.addItems(list(self._available_layers()))
        if previous and self._layer_combo.findText(previous) >= 0:
            self._layer_combo.setCurrentText(previous)

    # ── 신호 ──────────────────────────────────────────────────────────────
    def _on_source_changed(self, _text: str) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            self._refill_views()
            self._refill_layers()
        finally:
            self._loading = False
        self._render()

    def _on_view_changed(self, _text: str) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            self._refill_layers()
        finally:
            self._loading = False
        self._render()

    def _on_layer_changed(self, _text: str) -> None:
        if not self._loading:
            self._render()

    # ── 그리기 ────────────────────────────────────────────────────────────
    def _render(self) -> None:
        layers = self._available_layers()
        layer = self._layer_combo.currentText()
        if not layer or layer not in layers:
            self._pixmap = None
            self._canvas.setText("샘플을 고르세요")
            self._info.setText("-")
            return

        path = layers[layer]
        try:
            image, info = load_layer(path)
        except Exception as exc:  # 파일 하나가 깨져도 UI 는 살아 있어야 한다
            self._pixmap = None
            self._canvas.setText(f"불러오기 실패\n{exc}")
            self._info.setText(str(path))
            self.statusMessage.emit(f"{path.name}: {exc}")
            return

        self._pixmap = QPixmap.fromImage(image)
        self._info.setText(f"{path.name}   {info}")
        self._update_canvas()

    def _update_canvas(self) -> None:
        if self._pixmap is None:
            return
        self._canvas.setPixmap(
            self._pixmap.scaled(
                self._canvas.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt 이름 규칙)
        super().resizeEvent(event)
        self._update_canvas()
