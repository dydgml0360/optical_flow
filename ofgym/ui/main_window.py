"""메인 윈도우. 위 탭으로 두 작업대를 오간다, 아래는 로그.

    Flow Gym   record(아틀라스 3D) → 합성 촬영 쌍 + optical flow GT
    재구성     데이터셋 트리 | (2D 프리뷰 · 3D 뷰) | (3D 생성 · GT · 학습)
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
)

from ofgym.config import PATHS, depth_available
from ofgym.data import Sample
from ofgym.recon import find_output
from ofgym.ui.dataset_panel import DatasetPanel
from ofgym.ui.flow_gym_panel import FlowGymPanel
from ofgym.ui.gt_panel import GtPanel
from ofgym.ui.model3d_panel import Model3DPanel
from ofgym.ui.preview_panel import PreviewPanel
from ofgym.ui.recon_panel import ReconPanel
from ofgym.ui.train_panel import TrainPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("optical_flow_gym")
        self.resize(1560, 940)

        self._sample: Optional[Sample] = None

        self._dataset = DatasetPanel()
        self._preview = PreviewPanel()
        self._model3d = Model3DPanel()
        self._recon = ReconPanel()
        self._gt = GtPanel()
        self._train = TrainPanel()

        center = QTabWidget()
        center.addTab(self._preview, "2D 프리뷰")
        center.addTab(self._model3d, "3D 뷰")
        self._center_tabs = center

        right = QTabWidget()
        right.addTab(self._recon, "3D 생성")
        right.addTab(self._gt, "Disparity GT")
        right.addTab(self._train, "파인튜닝")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._dataset)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 860, 400])

        self._flow_gym = FlowGymPanel()
        pages = QTabWidget()
        pages.setDocumentMode(True)
        pages.addTab(self._flow_gym, "Flow Gym")
        pages.addTab(splitter, "재구성")
        self.setCentralWidget(pages)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        dock = QDockWidget("로그", self)
        dock.setWidget(self._log)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [170], Qt.Orientation.Vertical)

        self._dataset.sampleSelected.connect(self._on_sample)
        self._dataset.statusMessage.connect(self.log)
        self._preview.statusMessage.connect(self.log)
        self._model3d.statusMessage.connect(self.log)
        self._recon.logMessage.connect(self.log)
        self._recon.reconFinished.connect(self._on_recon_finished)
        self._gt.logMessage.connect(self.log)
        self._train.logMessage.connect(self.log)
        self._flow_gym.logMessage.connect(self.log)

        self._dataset.reload()
        self._recon.set_all_samples(self._dataset.all_samples())

        self._flow_gym.reload()

        self.log(f"데이터셋 루트: {PATHS.dataset}")
        self.log(
            f"depth 서브모듈: {PATHS.depth} "
            f"({'준비됨' if depth_available() else '미체크아웃'})"
        )

    def log(self, message: str) -> None:
        self._log.appendPlainText(message)
        first = message.strip().splitlines()[0] if message.strip() else ""
        if first:
            self.statusBar().showMessage(first, 5000)

    def _on_sample(self, sample: Sample) -> None:
        self._sample = sample
        recon = find_output(sample)
        self._preview.set_sample(sample, recon)
        self._model3d.set_recon(recon)
        self._recon.set_sample(sample)
        self._recon.set_all_samples(self._dataset.all_samples())
        state = "3D 있음" if recon is not None else "3D 없음"
        self.log(f"샘플 {sample.sample_id}  [{state}]  ({sample.path})")

    def _on_recon_finished(self, sample: Sample) -> None:
        """방금 만든 결과를 바로 볼 수 있게 갱신하고 3D 탭으로 옮겨 준다."""
        self._dataset.mark_recon(sample)
        if self._sample is not None and self._sample.sample_id == sample.sample_id:
            recon = find_output(sample)
            self._preview.set_sample(sample, recon)
            self._model3d.set_recon(recon)
            self._center_tabs.setCurrentWidget(self._model3d)
