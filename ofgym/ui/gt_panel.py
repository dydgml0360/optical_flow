"""Disparity GT 생성 패널.

정점맵(3D)을 스테레오 페어 카메라로 재투영해 disparity ground truth 를 만든다.
지금은 설정만 잡아 두고, 실제 렌더러는 아직 붙지 않았다.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import PAIRS, PATHS, depth_available


class GtPanel(QWidget):
    logMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._pair = QComboBox()
        for base, target in PAIRS:
            self._pair.addItem(f"{base} → {target}", (base, target))

        self._max_disp = QDoubleSpinBox()
        self._max_disp.setRange(16, 1024)
        self._max_disp.setValue(192)
        self._max_disp.setSuffix(" px")

        self._z_near = QDoubleSpinBox()
        self._z_near.setRange(1, 100000)
        self._z_near.setValue(150)
        self._z_near.setSuffix(" mm")

        self._z_far = QDoubleSpinBox()
        self._z_far.setRange(1, 100000)
        self._z_far.setValue(1200)
        self._z_far.setSuffix(" mm")

        form = QFormLayout()
        form.addRow("스테레오 페어", self._pair)
        form.addRow("최대 disparity", self._max_disp)
        form.addRow("Z 최소", self._z_near)
        form.addRow("Z 최대", self._z_far)

        out = QLabel(str(PATHS.gt))
        out.setWordWrap(True)
        out.setStyleSheet("color: palette(mid);")
        form.addRow("출력", out)

        self._run = QPushButton("GT 생성")
        self._run.clicked.connect(self._on_run)

        note = QLabel(
            "입력은 '3D 생성' 탭이 만든 병합 3D(model/*_vertex.npy)입니다. "
            "그 정점들을 페어 카메라 좌표계로 옮겨 재투영 → z-buffer 로 가려짐 처리 → "
            "disparity 맵과 유효 마스크를 저장하는 순서로 붙일 예정입니다.\n"
            "카메라 내·외부 파라미터는 thirdparty/depth 의 config/intrinsic.yaml · "
            "data/pose_calibration.json 을 씁니다."
            + ("" if depth_available() else "\n⚠ depth 서브모듈이 아직 체크아웃되지 않았습니다.")
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(form)
        layout.addWidget(self._run)
        layout.addWidget(note)
        layout.addStretch(1)

    def _on_run(self) -> None:
        base, target = self._pair.currentData()
        self.logMessage.emit(
            f"[GT] {base}→{target}, max_disp={self._max_disp.value():.0f}px, "
            f"Z={self._z_near.value():.0f}~{self._z_far.value():.0f}mm "
            "— 렌더러 미구현. 다음 단계에서 붙입니다."
        )
