"""파인튜닝 패널.

CREStereo / RAFT-Stereo 체크포인트를 우리 GT 로 이어 학습시킨다.
지금은 설정 UI 만 있고 학습 루프는 아직 붙지 않았다.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import PATHS

MODELS = ("CREStereo", "RAFT-Stereo", "IGEV-Stereo")


class TrainPanel(QWidget):
    logMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._model = QComboBox()
        self._model.addItems(MODELS)

        self._checkpoint = QLineEdit()
        self._checkpoint.setPlaceholderText("사전학습 체크포인트 경로 (.pth)")

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 1000)
        self._epochs.setValue(20)

        self._batch = QSpinBox()
        self._batch.setRange(1, 64)
        self._batch.setValue(2)

        self._lr = QDoubleSpinBox()
        self._lr.setDecimals(6)
        self._lr.setRange(1e-6, 1.0)
        self._lr.setSingleStep(1e-5)
        self._lr.setValue(1e-4)

        form = QFormLayout()
        form.addRow("모델", self._model)
        form.addRow("체크포인트", self._checkpoint)
        form.addRow("epoch", self._epochs)
        form.addRow("batch", self._batch)
        form.addRow("learning rate", self._lr)

        gt = QLabel(str(PATHS.gt))
        gt.setWordWrap(True)
        gt.setStyleSheet("color: palette(mid);")
        form.addRow("학습 데이터", gt)

        self._start = QPushButton("학습 시작")
        self._start.clicked.connect(self._on_start)
        self._stop = QPushButton("중지")
        self._stop.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addWidget(self._start, 1)
        buttons.addWidget(self._stop)

        note = QLabel(
            "모델 구현은 thirdparty/depth 의 external/stereo_models 를 씁니다.\n"
            "⚠ 서브모듈 클론에는 CREStereo/RAFT-Stereo 소스가 따라오지 않습니다 "
            "(upstream 에서 무시된 디렉터리). 별도로 받아야 합니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(note)
        layout.addStretch(1)

    def _on_start(self) -> None:
        self.logMessage.emit(
            f"[학습] {self._model.currentText()}  epoch={self._epochs.value()} "
            f"batch={self._batch.value()} lr={self._lr.value():g} "
            "— 학습 루프 미구현. 다음 단계에서 붙입니다."
        )
