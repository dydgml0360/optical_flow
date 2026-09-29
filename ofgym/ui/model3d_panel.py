"""3D 뷰 탭 — 재구성 결과 포인트 클라우드를 돌려 본다."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import VIEWS
from ofgym.recon import ReconOutput
from ofgym.ui.pointcloud_view import PointCloudView

VIEW_COLORS = {  # 뷰별 색 — 어느 뷰가 어디를 채웠는지 보려고
    "left": (232, 106, 106),
    "center": (120, 200, 120),
    "right": (110, 150, 235),
}

# 표시용 상한. 6x 업샘플이면 뷰당 300만 점이 넘는데, 화면 900px 에 그걸 다 찍어도
# 보이는 게 달라지지 않고 프레임당 정렬만 몇 배로 늘어난다. 저장본은 그대로다.
MAX_DISPLAY_PER_VIEW = 700_000


class Model3DPanel(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._recon: Optional[ReconOutput] = None
        self._clouds: List = []
        self._valid_counts: dict = {}

        self._view = PointCloudView()

        self._checks = {}
        bar = QHBoxLayout()
        bar.addWidget(QLabel("뷰"))
        for name in VIEWS:
            box = QCheckBox(name)
            box.setChecked(True)
            box.toggled.connect(self._apply)
            bar.addWidget(box)
            self._checks[name] = box

        self._by_view = QCheckBox("뷰별 색")
        self._by_view.setToolTip("실제 색 대신 뷰마다 다른 색으로 칠해 커버리지를 본다")
        self._by_view.toggled.connect(self._apply)
        bar.addSpacing(12)
        bar.addWidget(self._by_view)

        bar.addSpacing(12)
        bar.addWidget(QLabel("점 크기"))
        self._size = QSlider(Qt.Orientation.Horizontal)
        self._size.setRange(1, 6)
        self._size.setValue(2)
        self._size.setFixedWidth(90)
        self._size.valueChanged.connect(self._view.set_point_size)
        bar.addWidget(self._size)

        reset = QPushButton("시점 초기화")
        reset.clicked.connect(self._view.reset_view)
        bar.addStretch(1)
        bar.addWidget(reset)

        self._info = QLabel("재구성 결과가 없습니다")
        self._info.setStyleSheet("color: palette(mid);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(bar)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._info)

    def set_recon(self, recon: Optional[ReconOutput]) -> None:
        self._recon = recon
        if recon is None:
            self._clouds = []
            self._view.clear()
            self._info.setText(
                "재구성 결과가 없습니다 — 오른쪽 '3D 생성' 탭에서 만드세요"
            )
            return
        try:
            self._valid_counts = recon.valid_counts()
            self._clouds = list(
                zip(recon.views, recon.load_clouds(MAX_DISPLAY_PER_VIEW))
            )
        except Exception as exc:
            self._clouds = []
            self._view.clear()
            self._info.setText(f"불러오기 실패: {exc}")
            self.statusMessage.emit(f"3D 불러오기 실패: {exc}")
            return
        self._apply()

    def _apply(self) -> None:
        if not self._clouds:
            return
        import numpy as np

        selected = []
        counts = []
        for name, (points, colors) in self._clouds:
            if not self._checks[name].isChecked():
                continue
            if self._by_view.isChecked():
                tint = np.array(VIEW_COLORS.get(name, (200, 200, 200)), np.uint8)
                colors = np.repeat(tint[None, :], len(points), axis=0)
            selected.append((points, colors))
            counts.append(f"{name} {self._valid_counts.get(name, len(points)):,}")

        self._view.set_clouds(selected)
        shown = self._view.point_count
        actual = sum(
            n for name, n in self._valid_counts.items() if self._checks[name].isChecked()
        )

        head = " · ".join(counts) if counts else "표시할 뷰 없음"
        grid = self._recon.grid_shape if self._recon else None
        bits = [head, f"합계 {actual:,}점"]
        if grid:
            bits.append(f"{grid[1]}x{grid[0]} 격자")
        if self._recon and self._recon.suffix:
            bits.append(self._recon.suffix.lstrip("_"))
        if shown < actual:
            bits.append(f"화면 표시 {shown:,}점으로 솎음")
        self._info.setText(
            "   ".join(bits)
            + "   —  드래그=회전 · 휠=줌 · 우클릭 드래그=이동 · 더블클릭=초기화"
        )
