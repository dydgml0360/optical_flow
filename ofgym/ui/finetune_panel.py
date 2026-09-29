"""파인튜닝 탭 — 일괄 생성한 데이터로 RAFT-Stereo / CREStereo 를 이어 학습시키고,
사전학습 그대로(기존)와 나란히 놓고 본다.

    왼쪽    데이터셋 · 모델 · 학습 설정, 시작/중지, 지난 실행 목록
    가운데  학습 곡선 / 비교 (지표 표 + 쌍별 그림 6칸)

학습은 depth 서브모듈의 venv 에서 `ofgym/train/worker.py` 가 한다. 여기서는 그
프로세스를 띄우고, 한 줄에 하나씩 나오는 JSON 이벤트를 받아 그린다.

비교 그림 6칸:

    왼쪽 사진     기존 예측     튜닝 예측
    정답 시차     기존 오차     튜닝 오차

시차 세 장은 같은 색 범위를 쓴다. 실제 촬영 쌍은 정답이 없어 아래 줄이 빈다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QProcessEnvironment, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import PATHS
from ofgym.train import runs as train_runs
from ofgym.train.runs import Dataset, Run, Sample, TrainConfig
from ofgym.ui.flow_gym_panel import ImagePane

ITEM_ROLE = Qt.ItemDataRole.UserRole

PANES = ("왼쪽 사진", "기존 예측", "튜닝 예측", "정답 시차", "기존 오차", "튜닝 오차")
ERROR_RANGE = 3.0  # 오차 그림에서 가장 밝은 색이 되는 픽셀 오차

REGIONS = (("all", "전체"), ("face", "얼굴"), ("background", "배경"))
MEASURES = (("epe", "EPE (px)", "{:.3f}"), ("bad1", "1px 초과 (%)", "{:.2f}"),
            ("bad3", "3px 초과 (%)", "{:.2f}"))


def _colorize(values: np.ndarray, low: float, high: float,
              mask: Optional[np.ndarray] = None) -> np.ndarray:
    span = max(high - low, 1e-6)
    norm = np.clip((values - low) / span * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    if mask is not None:
        rgb[~mask] = 0
    return rgb


class CurveView(QWidget):
    """학습 곡선. 위는 학습 EPE(크롭, 증강된 입력), 아래는 검증 EPE(전체 화면)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 200)
        self.clear()

    def clear(self) -> None:
        self._train: List[tuple] = []  # (step, epe)
        self._val: Dict[str, List[tuple]] = {key: [] for key, _ in REGIONS}
        self._steps = 1
        self.update()

    def set_steps(self, steps: int) -> None:
        self._steps = max(1, steps)
        self.update()

    def add_train(self, step: int, epe: float) -> None:
        self._train.append((step, epe))
        self.update()

    def add_val(self, step: int, metrics: dict) -> None:
        for key, _ in REGIONS:
            if key in metrics:
                self._val[key].append((step, metrics[key]["epe"]))
        self.update()

    def _plot(self, painter: QPainter, area: QRectF, title: str, series: list,
              baseline: Optional[float] = None) -> None:
        text = self.palette().text().color()
        faint = QColor(text)
        faint.setAlpha(60)
        painter.setPen(text)
        painter.drawText(area.adjusted(0, -18, 0, 0).topLeft() + QRectF(0, 0, 0, 14).bottomLeft(),
                         title)
        painter.setPen(QPen(faint, 1))
        painter.drawRect(area)

        values = [v for _, points, _ in series for _, v in points]
        if not values:
            return
        top = max(values) * 1.1 or 1.0
        for i in range(1, 4):
            y = area.bottom() - area.height() * i / 4
            painter.setPen(QPen(faint, 1, Qt.PenStyle.DotLine))
            painter.drawLine(area.left(), y, area.right(), y)
            painter.setPen(text)
            painter.drawText(area.left() + 4, y - 2, f"{top * i / 4:.2f}")

        def place(step: float, value: float):
            return (area.left() + area.width() * step / self._steps,
                    area.bottom() - area.height() * min(value, top) / top)

        if baseline is not None:
            y = place(0, baseline)[1]
            painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
            painter.drawLine(area.left(), y, area.right(), y)

        legend_x = area.right() - 8
        for label, points, color in series:
            if not points:
                continue
            painter.setPen(QPen(color, 2))
            spots = [place(s, v) for s, v in points]
            for (x0, y0), (x1, y1) in zip(spots, spots[1:]):
                painter.drawLine(x0, y0, x1, y1)
            if len(points) < 40:
                for x, y in spots:
                    painter.drawEllipse(QRectF(x - 2.5, y - 2.5, 5, 5))
            note = f"{label} {points[-1][1]:.3f}"
            width = painter.fontMetrics().horizontalAdvance(note)
            legend_x -= width + 14
            painter.drawText(legend_x, area.top() + 14, note)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        half = (height - 60) / 2
        upper = QRectF(12, 26, width - 24, half)
        lower = QRectF(12, 26 + half + 30, width - 24, half)

        self._plot(painter, upper, "학습 EPE (px) — 크롭, 증강된 입력",
                   [("학습", self._train, QColor(95, 160, 235))])
        colors = dict(all=QColor(90, 200, 120), face=QColor(240, 150, 70),
                      background=QColor(150, 130, 230))
        series = [(label, self._val[key], colors[key]) for key, label in REGIONS]
        first = self._val["all"][0][1] if self._val["all"] else None
        self._plot(painter, lower,
                   "검증 EPE (px) — 0 스텝이 기존 모델, 점선은 그 값(전체)", series, first)


class FinetunePanel(QWidget):
    logMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._run_dir: Optional[Path] = None
        self._buffer = b""
        self._log_file = None
        self._run: Optional[Run] = None
        self._samples: List[Sample] = []
        self._stopping = False

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1200])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ── 화면 구성 ─────────────────────────────────────────────────────────
    def _build_controls(self) -> QWidget:
        defaults = TrainConfig()

        self._dataset = QComboBox()
        self._dataset.currentIndexChanged.connect(self._on_dataset_changed)
        refresh = QPushButton("다시 읽기")
        refresh.clicked.connect(self.reload)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self._dataset, 1)
        dataset_row.addWidget(refresh)
        self._dataset_info = QLabel("-")
        self._dataset_info.setWordWrap(True)
        self._dataset_info.setStyleSheet("color: palette(mid);")

        self._model = QComboBox()
        for key, spec in train_runs.MODELS.items():
            self._model.addItem(spec["label"], key)
        self._model.currentIndexChanged.connect(self._sync_start)
        self._model_info = QLabel("")
        self._model_info.setWordWrap(True)
        self._model_info.setStyleSheet("color: palette(mid);")
        self._model_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        data_form = QFormLayout()
        data_form.addRow("데이터셋", dataset_row)
        data_form.addRow(self._dataset_info)
        data_form.addRow("모델", self._model)
        data_form.addRow(self._model_info)
        data_box = QGroupBox("무엇을")
        data_box.setLayout(data_form)

        def spin(low, high, value, step=1, suffix=""):
            box = QSpinBox()
            box.setRange(low, high)
            box.setSingleStep(step)
            box.setValue(value)
            box.setSuffix(suffix)
            box.setGroupSeparatorShown(True)
            return box

        self._steps = spin(10, 1_000_000, defaults.steps, 100, " 스텝")
        self._batch = spin(1, 64, defaults.batch_size)
        self._lr = QDoubleSpinBox()
        self._lr.setDecimals(7)
        self._lr.setRange(1e-7, 1e-2)
        self._lr.setSingleStep(1e-5)
        self._lr.setValue(defaults.lr)
        self._crop_h = spin(64, 2048, defaults.crop_height, 32, " px")
        self._crop_w = spin(64, 2048, defaults.crop_width, 32, " px")
        self._train_iters = spin(2, 64, defaults.train_iters)
        self._valid_iters = spin(2, 64, defaults.valid_iters)
        self._val_every = spin(10, 100_000, defaults.val_every, 50, " 스텝")
        self._val_samples = spin(1, 10_000, defaults.val_samples, 4, " 쌍")
        self._compare_samples = spin(1, 10_000, defaults.compare_samples, 10, " 쌍")

        crop_row = QHBoxLayout()
        crop_row.addWidget(self._crop_h)
        crop_row.addWidget(QLabel("×"))
        crop_row.addWidget(self._crop_w)

        train_form = QFormLayout()
        train_form.addRow("학습 길이", self._steps)
        train_form.addRow("배치", self._batch)
        train_form.addRow("learning rate", self._lr)
        train_form.addRow("크롭 (세로×가로)", crop_row)
        train_form.addRow("반복 (학습)", self._train_iters)
        train_form.addRow("반복 (검증)", self._valid_iters)
        train_form.addRow("검증 간격", self._val_every)
        train_form.addRow("검증 쌍 수", self._val_samples)
        train_form.addRow("비교 쌍 수", self._compare_samples)
        hint = QLabel(
            "기본값은 메모리 16GB 맥에서 돌아가는 크기입니다 (스텝당 1.5~3초, 메모리 5GB). "
            "배치 2 는 10GB 를 쓰고, 크롭까지 키우면 메모리를 넘겨 열 배 가까이 느려집니다."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        train_form.addRow(hint)
        train_box = QGroupBox("어떻게")
        train_box.setLayout(train_form)

        self._start = QPushButton("파인튜닝 시작")
        self._start.clicked.connect(self._toggle)
        self._bar = QProgressBar()
        self._bar.setVisible(False)
        self._status = QLabel("-")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: palette(mid);")

        self._runs = QListWidget()
        self._runs.setUniformItemSizes(True)
        self._runs.currentItemChanged.connect(self._on_run_changed)
        self._recompare = QPushButton("이 실행 다시 비교")
        self._recompare.setToolTip("저장된 체크포인트로 비교만 다시 한다 (학습은 하지 않는다)")
        self._recompare.clicked.connect(self._compare_again)
        runs_layout = QVBoxLayout()
        runs_layout.addWidget(self._runs)
        runs_layout.addWidget(self._recompare)
        runs_box = QGroupBox("지난 실행")
        runs_box.setLayout(runs_layout)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(data_box)
        layout.addWidget(train_box)
        layout.addWidget(self._start)
        layout.addWidget(self._bar)
        layout.addWidget(self._status)
        layout.addWidget(runs_box, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        return scroll

    def _build_views(self) -> QWidget:
        self._curve = CurveView()

        self._table = QTableWidget(len(MEASURES) * len(REGIONS), 5)
        self._table.setHorizontalHeaderLabels(["영역", "지표", "기존", "튜닝", "변화"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setFixedHeight(
            self._table.verticalHeader().defaultSectionSize() * (len(MEASURES) * len(REGIONS))
            + self._table.horizontalHeader().height() + 6)
        self._summary = QLabel("비교 결과가 없습니다 — 파인튜닝이 끝나면 여기에 나옵니다")
        self._summary.setWordWrap(True)

        self._sample = QComboBox()
        self._sample.currentIndexChanged.connect(self._show_sample)
        previous = QPushButton("◀")
        following = QPushButton("▶")
        previous.setFixedWidth(40)
        following.setFixedWidth(40)
        previous.clicked.connect(lambda: self._step_sample(-1))
        following.clicked.connect(lambda: self._step_sample(1))
        self._sample_info = QLabel("")
        self._sample_info.setStyleSheet("color: palette(mid);")
        sample_bar = QHBoxLayout()
        sample_bar.addWidget(QLabel("쌍"))
        sample_bar.addWidget(previous)
        sample_bar.addWidget(self._sample, 1)
        sample_bar.addWidget(following)
        sample_bar.addWidget(self._sample_info, 2)

        self._panes: Dict[str, ImagePane] = {}
        grid = QGridLayout()
        grid.setSpacing(6)
        for index, title in enumerate(PANES):
            pane = ImagePane(title)
            self._panes[title] = pane
            grid.addWidget(pane, index // 3, index % 3)

        compare = QWidget()
        compare_layout = QVBoxLayout(compare)
        compare_layout.setContentsMargins(6, 6, 6, 6)
        compare_layout.addWidget(self._summary)
        compare_layout.addWidget(self._table)
        compare_layout.addLayout(sample_bar)
        compare_layout.addLayout(grid, 1)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._curve, "학습 곡선")
        self._compare_tab = compare
        self._tabs.addTab(compare, "비교")

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._tabs)
        return box

    # ── 목록 ──────────────────────────────────────────────────────────────
    def reload(self) -> None:
        kept = self._dataset.currentData()
        self._dataset.blockSignals(True)
        self._dataset.clear()
        for dataset in train_runs.find_datasets():
            self._dataset.addItem(dataset.label, dataset)
        if kept is not None:
            for i in range(self._dataset.count()):
                if self._dataset.itemData(i).path == kept.path:
                    self._dataset.setCurrentIndex(i)
        self._dataset.blockSignals(False)
        self._on_dataset_changed()
        self._reload_runs()

    def _reload_runs(self, select: Optional[Path] = None) -> None:
        self._runs.blockSignals(True)
        self._runs.clear()
        chosen = None
        for run in train_runs.find_runs():
            item = QListWidgetItem(run.label)
            item.setData(ITEM_ROLE, run)
            item.setToolTip(str(run.path))
            self._runs.addItem(item)
            if select is not None and run.path == select:
                chosen = item
        self._runs.blockSignals(False)
        if chosen is not None:
            self._runs.setCurrentItem(chosen)
        self._recompare.setEnabled(self._runs.currentItem() is not None
                                   and self._process is None)

    def _on_dataset_changed(self, *_args) -> None:
        dataset: Optional[Dataset] = self._dataset.currentData()
        if dataset is None:
            self._dataset_info.setText(
                f"데이터셋이 없습니다 — Flow Gym 의 '일괄 생성' 으로 먼저 만드세요\n{PATHS.flow}")
        else:
            self._dataset_info.setText(dataset.summary)
        self._sync_start()

    def _sync_start(self, *_args) -> None:
        problem = train_runs.model_problem(self._model.currentData())
        self._model_info.setText(problem or "")
        ready = self._dataset.currentData() is not None and problem is None
        self._start.setEnabled(ready or self._process is not None)

    # ── 실행 ──────────────────────────────────────────────────────────────
    def config(self) -> TrainConfig:
        dataset: Dataset = self._dataset.currentData()
        return TrainConfig(
            model=self._model.currentData(),
            dataset=str(dataset.path),
            models_root=str(train_runs.models_root()),
            steps=self._steps.value(),
            batch_size=self._batch.value(),
            lr=self._lr.value(),
            crop_height=self._crop_h.value(),
            crop_width=self._crop_w.value(),
            train_iters=self._train_iters.value(),
            valid_iters=self._valid_iters.value(),
            val_every=min(self._val_every.value(), self._steps.value()),
            val_samples=self._val_samples.value(),
            compare_samples=self._compare_samples.value(),
        )

    def _toggle(self) -> None:
        if self._process is not None:
            self._stopping = True
            self._start.setEnabled(False)
            self._start.setText("중지하는 중…")
            self._process.terminate()  # worker 가 SIGTERM 을 받아 체크포인트를 쓰고 끝낸다
            return
        python = PATHS.depth / ".venv" / "bin" / "python"
        if not python.is_file():
            self.logMessage.emit(
                f"[튜닝] depth venv 가 없습니다: {python}\n  uv sync --project {PATHS.depth}")
            return
        run = train_runs.create_run(self.config())
        self._curve.clear()
        self._curve.set_steps(self._steps.value())
        self._tabs.setCurrentWidget(self._curve)
        self._launch(run, "train")

    def _compare_again(self) -> None:
        item = self._runs.currentItem()
        if item is None or self._process is not None:
            return
        self._launch(item.data(ITEM_ROLE).path, "compare")

    def _launch(self, run: Path, command: str) -> None:
        python = PATHS.depth / ".venv" / "bin" / "python"
        self._run_dir = run
        self._buffer = b""
        self._stopping = False
        self._log_file = (run / "log.jsonl").open("a", encoding="utf-8")

        env = QProcessEnvironment.systemEnvironment()
        env.remove("VIRTUAL_ENV")  # UI 의 venv 가 worker 로 새지 않게
        env.insert("PYTHONUNBUFFERED", "1")
        process = QProcess(self)
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(PATHS.depth))
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(
            lambda error: self.logMessage.emit(f"[튜닝] 프로세스 오류: {error}"))
        self._process = process

        self._bar.setRange(0, 0)  # 첫 이벤트가 올 때까지는 바쁨 표시
        self._bar.setVisible(True)
        self._start.setText("중지")
        self._start.setEnabled(True)
        self._recompare.setEnabled(False)
        self._status.setText("모델을 올리는 중…")
        self.logMessage.emit(f"[튜닝] {command} → {run}")
        process.start(str(python), [str(train_runs.WORKER), command, "--run", str(run)])

    def _on_output(self) -> None:
        if self._process is None:
            return
        self._buffer += bytes(self._process.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            event = None
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except ValueError:
                    event = None
            if event is None or "event" not in event:
                if "Warning" not in line and not line.startswith("  "):
                    self.logMessage.emit(f"[튜닝] {line}")
                continue
            if self._log_file is not None:
                self._log_file.write(line + "\n")
                self._log_file.flush()
            self._on_event(event)

    def _on_event(self, event: dict) -> None:
        kind = event["event"]
        if kind == "start":
            self._bar.setRange(0, int(event["steps"]))
            self._curve.set_steps(int(event["steps"]))
            self.logMessage.emit(
                f"[튜닝] {event['model']} on {event['device']} — 학습 {event['train']:,}쌍, "
                f"검증 {event['val']}쌍, {event['width']}x{event['height']}"
                + (" (세로 기선 → 90° 돌림)" if event.get("vertical") else ""))
            self._status.setText("기존 모델을 검증하는 중…")
        elif kind == "step":
            self._bar.setRange(0, int(event["steps"]))
            self._bar.setValue(int(event["step"]))
            self._curve.add_train(int(event["step"]), float(event["epe"]))
            left = (event["steps"] - event["step"]) / max(event["rate"], 1e-6)
            self._status.setText(
                f"{event['step']:,} / {event['steps']:,} 스텝   손실 {event['loss']:.3f}   "
                f"EPE {event['epe']:.3f}px   {event['rate']:.2f} 스텝/초   "
                f"학습 남은 시간 약 {left / 60:.0f}분")
        elif kind == "val":
            self._curve.add_val(int(event["step"]), event["metrics"])
            bits = [f"{label} {event['metrics'][key]['epe']:.3f}"
                    for key, label in REGIONS if key in event["metrics"]]
            head = "기존 모델" if event["step"] == 0 else f"{event['step']:,} 스텝"
            self.logMessage.emit(f"[튜닝] 검증 EPE ({head})  " + "  ".join(bits))
        elif kind == "trained":
            self._bar.setRange(0, 0)
            state = "중지됨" if event.get("stopped") else "끝"
            self.logMessage.emit(
                f"[튜닝] 학습 {state} — {event['steps']:,} 스텝, 검증 EPE "
                f"{event['baseline_epe']:.3f} → {event['best_epe']:.3f} (가장 좋았던 것)")
            self._status.setText("기존 모델과 비교하는 중…")
        elif kind == "compare":
            which = "기존" if event["stage"] == "base" else "튜닝"
            self._status.setText(f"비교하는 중 — {which} 모델, {event['samples']}쌍")
        elif kind == "compared":
            self.logMessage.emit(f"[튜닝] 비교 끝 — {event['samples']}쌍")
        elif kind == "error":
            self.logMessage.emit(f"[튜닝] 실패: {event['message']}")

    def _on_finished(self, code: int, _status) -> None:
        self._on_output()
        process, self._process = self._process, None
        if process is not None:
            process.deleteLater()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._bar.setVisible(False)
        self._start.setText("파인튜닝 시작")
        self._sync_start()
        if code != 0 and not self._stopping:
            self._status.setText(f"실패 (종료 코드 {code}) — 아래 로그를 보세요")
        else:
            self._status.setText("중지됨" if self._stopping else "끝")
        self._reload_runs(select=self._run_dir)

    def shutdown(self) -> None:
        """창을 닫을 때. 돌고 있는 학습을 멈춘다."""
        if self._process is not None:
            self._process.terminate()
            if not self._process.waitForFinished(15000):
                self._process.kill()
                self._process.waitForFinished(3000)

    # ── 비교 보기 ─────────────────────────────────────────────────────────
    def _on_run_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        self._recompare.setEnabled(current is not None and self._process is None)
        run: Optional[Run] = current.data(ITEM_ROLE) if current is not None else None
        self._run = run
        self._show_metrics(run)
        self._samples = run.samples() if run is not None else []
        self._sample.blockSignals(True)
        self._sample.clear()
        for sample in self._samples:
            self._sample.addItem(sample.label)
        self._sample.blockSignals(False)
        self._show_sample()
        if run is None:
            return
        if self._process is None:  # 돌고 있는 학습의 곡선을 덮지 않는다
            self._curve.clear()
            self._curve.set_steps(int(run.config.get("steps", 1)))
            for event in run.events():
                if event.get("event") == "step":
                    self._curve.add_train(int(event["step"]), float(event["epe"]))
                elif event.get("event") == "val":
                    self._curve.add_val(int(event["step"]), event["metrics"])
        if run.metrics is not None:
            self._tabs.setCurrentWidget(self._compare_tab)

    def _show_metrics(self, run: Optional[Run]) -> None:
        self._table.clearContents()
        if run is None or run.metrics is None:
            self._summary.setText("비교 결과가 없습니다 — 파인튜닝이 끝나면 여기에 나옵니다")
            return
        metrics = run.metrics["metrics"]
        row = 0
        for key, label in REGIONS:
            for measure, title, fmt in MEASURES:
                base = metrics.get("base", {}).get(key, {}).get(measure)
                tuned = metrics.get("tuned", {}).get(key, {}).get(measure)
                cells = [label, title,
                         "-" if base is None else fmt.format(base),
                         "-" if tuned is None else fmt.format(tuned), "-"]
                better = None
                if base is not None and tuned is not None:
                    change = (tuned - base) / base * 100 if base else 0.0
                    cells[4] = f"{change:+.1f}%"
                    better = tuned < base
                for column, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if column >= 2:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                              | Qt.AlignmentFlag.AlignVCenter)
                    if column == 4 and better is not None:
                        item.setForeground(QColor(70, 170, 100) if better
                                           else QColor(215, 90, 80))
                    self._table.setItem(row, column, item)
                row += 1
        model = train_runs.MODELS.get(run.metrics.get("model"), {}).get("label", "?")
        self._summary.setText(
            f"{model} — 검증 {run.metrics['samples']}쌍, 반복 {run.metrics['iters']}회. "
            "낮을수록 좋다. 기존 = 사전학습 가중치 그대로, 튜닝 = 검증 EPE 가 가장 낮았던 "
            "체크포인트. 합성 쌍으로 잰 값이라 실제 촬영에서의 향상을 보장하지는 않는다."
        )

    def _step_sample(self, delta: int) -> None:
        if self._sample.count():
            self._sample.setCurrentIndex(
                (self._sample.currentIndex() + delta) % self._sample.count())

    def _show_sample(self, *_args) -> None:
        index = self._sample.currentIndex()
        if index < 0 or index >= len(self._samples):
            for pane in self._panes.values():
                pane.set_image(None)
            self._sample_info.setText("")
            return
        sample = self._samples[index]
        try:
            data = sample.load()
        except Exception as exc:
            self._sample_info.setText(f"불러오기 실패: {exc}")
            return

        self._panes["왼쪽 사진"].set_image(data["left"])
        if "gt" in data:
            valid = data["valid"]
            low, high = np.percentile(data["gt"][valid], [1, 99]) if valid.any() else (0, 1)
        else:
            valid = None
            low, high = np.percentile(data["base"], [1, 99])
        self._panes["기존 예측"].set_image(_colorize(data["base"], low, high))
        self._panes["튜닝 예측"].set_image(_colorize(data["tuned"], low, high))
        for title in ("기존 예측", "튜닝 예측"):
            self._panes[title].set_caption(f"{title}  ({low:.0f} ~ {high:.0f}px)")

        if "gt" not in data:
            self._panes["정답 시차"].set_image(
                _colorize(np.abs(data["tuned"] - data["base"]), 0, ERROR_RANGE))
            self._panes["정답 시차"].set_caption(
                f"두 예측의 차이  (0 ~ {ERROR_RANGE:.0f}px) — 정답 없음")
            self._panes["기존 오차"].set_image(None)
            self._panes["튜닝 오차"].set_image(None)
            self._panes["기존 오차"].set_caption("기존 오차  (정답 없음)")
            self._panes["튜닝 오차"].set_caption("튜닝 오차  (정답 없음)")
            self._sample_info.setText(
                "실제 촬영 — 정답이 없어 눈으로만 본다. 보정(rectify) 없이 크기만 맞춘 쌍이다")
            return

        self._panes["정답 시차"].set_image(_colorize(data["gt"], low, high, valid))
        self._panes["정답 시차"].set_caption(f"정답 시차  ({low:.0f} ~ {high:.0f}px)")
        notes = []
        for key, title in (("base", "기존"), ("tuned", "튜닝")):
            error = np.abs(data[key] - data["gt"])
            self._panes[f"{title} 오차"].set_image(_colorize(error, 0, ERROR_RANGE, valid))
            face = valid & data["face"]
            bits = [f"전체 {error[valid].mean():.3f}"] if valid.any() else []
            if face.any():
                bits.append(f"얼굴 {error[face].mean():.3f}")
            self._panes[f"{title} 오차"].set_caption(
                f"{title} 오차  (0 ~ {ERROR_RANGE:.0f}px)   EPE " + " · ".join(bits))
            notes.append(f"{title} {' · '.join(bits)}")
        self._sample_info.setText("EPE(px)   " + "     ".join(notes))
