"""3D 생성 패널 — depth 의 make3D 파이프라인(3뷰 병합 포함)을 돌린다."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import PATHS, depth_available
from ofgym.data import Sample
from ofgym.recon import ReconOptions, ReconWorker
from ofgym.recon.runner import (
    ReconError,
    default_upsample,
    default_upsample_method,
    depth_python,
    stereo_config,
)


class ReconPanel(QWidget):
    logMessage = Signal(str)
    reconFinished = Signal(object)  # Sample — 결과가 새로 생긴 샘플

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._sample: Optional[Sample] = None
        self._all: List[Sample] = []
        self._worker: Optional[ReconWorker] = None

        self._target = QComboBox()
        self._target.addItem("선택한 샘플", "one")
        self._target.addItem("모든 샘플", "all")

        self._model = QComboBox()
        self._model.addItem("u2net (기본)", "u2net")
        self._model.addItem("disk", "disk")

        # make3D CLI 기본값은 1 이지만 stereo.yaml 이 선언한 배율(=6)을 기본으로 쓴다.
        # 1 로 두면 1/decimation 격자에서 끝나 업샘플 단계가 통째로 빠진다.
        self._upsample = QComboBox()
        for scale in (1, 2, 3, 6):
            label = f"{scale} x"
            if scale == default_upsample():
                label += "  (stereo.yaml)"
            elif scale == 1:
                label += "  (업샘플 없음)"
            self._upsample.addItem(label, scale)
        self._upsample.setCurrentIndex(
            max(0, self._upsample.findData(default_upsample()))
        )

        self._render = QCheckBox("Open3D 정면 렌더(render.png)도 저장")
        self._render.setChecked(True)

        self._debug = QCheckBox("병합·후처리 로그 보기 (PYTR_DEBUG)")
        self._debug.setChecked(True)
        self._debug.setToolTip(
            "pytoningreacher 는 print 를 dprint 로 가려 두었다 — 끄면 어떤 후처리가 "
            "돌았는지 한 줄도 안 보인다. 켜면 depth/logs/<시각>/ 도 생긴다."
        )

        form = QFormLayout()
        form.addRow("대상", self._target)
        form.addRow("마스크 소스", self._model)
        form.addRow("업샘플", self._upsample)
        form.addRow("", self._render)
        form.addRow("", self._debug)

        out = QLabel(str(PATHS.recon))
        out.setWordWrap(True)
        out.setStyleSheet("color: palette(mid);")
        form.addRow("출력", out)

        self._run = QPushButton("3D 생성")
        self._run.clicked.connect(self._on_run)
        self._stop = QPushButton("중지")
        self._stop.setEnabled(False)
        self._stop.clicked.connect(self._on_stop)

        buttons = QHBoxLayout()
        buttons.addWidget(self._run, 1)
        buttons.addWidget(self._stop)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setVisible(False)

        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: palette(mid);")
        self._note.setText(self._describe())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self._progress)
        layout.addWidget(self._note)
        layout.addStretch(1)

    def _describe(self) -> str:
        merge = (stereo_config().get("merge") or {}).get("method", "?")
        lines = [
            "캡처를 make3D 규약으로 스테이징한 뒤 depth 의 make3D 를 돌립니다. "
            f"병합은 stereo.yaml `merge.method={merge}` — 스파이크 제거 · 정중선 컷 · "
            "centerAlign · 합의/색합의 필터 · 입사각 컬링 · center 방위각 게이트 · "
            "색 이득 보정 · 아웃라이어 제거까지 이 안에서 끝납니다. "
            "(ROI 하드 컷과 시임 블렌딩은 이 경로에서 의도적으로 빠집니다.)",
            "업샘플만 별개입니다 — stereo.yaml 의 upsample 섹션은 "
            "setUpsampleCallback 을 등록한 쪽에서만 도는데 make3D 는 등록하지 않습니다. "
            f"그래서 여기서 `--upsample {default_upsample()} "
            f"--upsample-method {default_upsample_method()}` 로 직접 넘깁니다. "
            "⚠ 이 인라인 경로는 보간(interpolate)만 되고 rematch(풀해상도 재매칭)는 "
            "쓸 수 없습니다.",
            "출력: model/{left,center,right}_vertex.npy · _blend.npy/.png · "
            "_facemask.png · _mask.png · render.png",
        ]
        if not depth_available():
            lines.append("⚠ depth 서브모듈이 체크아웃되지 않았습니다.")
        else:
            try:
                depth_python()
            except ReconError as exc:
                lines.append(f"⚠ {exc}")
        return "\n\n".join(lines)

    def set_sample(self, sample: Sample) -> None:
        self._sample = sample

    def set_all_samples(self, samples: List[Sample]) -> None:
        self._all = list(samples)

    def _targets(self) -> List[Sample]:
        if self._target.currentData() == "all":
            return self._all
        return [self._sample] if self._sample is not None else []

    def _on_run(self) -> None:
        if self._worker is not None:
            return
        targets = self._targets()
        if not targets:
            self.logMessage.emit("[3D] 대상 샘플이 없습니다")
            return

        options = ReconOptions(
            model=self._model.currentData(),
            upsample=self._upsample.currentData(),
            upsample_method=default_upsample_method(),
            render=self._render.isChecked(),
            debug=self._debug.isChecked(),
        )
        self.logMessage.emit(
            f"[3D] {len(targets)}개 샘플 · model={options.model} "
            f"upsample={options.upsample}x({options.upsample_method}) "
            f"render={'on' if options.render else 'off'}"
        )

        self._progress.setRange(0, len(targets))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._run.setEnabled(False)
        self._stop.setEnabled(True)

        worker = ReconWorker(targets, options, parent=self)
        worker.line.connect(self.logMessage)
        worker.sampleDone.connect(self._on_sample_done)
        worker.finishedAll.connect(self._on_all_done)
        self._worker = worker
        worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self.logMessage.emit("[3D] 중지 요청…")
            self._worker.cancel()
        self._stop.setEnabled(False)

    def _on_sample_done(self, sample: Sample, ok: bool) -> None:
        self._progress.setValue(self._progress.value() + 1)
        if ok:
            self.reconFinished.emit(sample)

    def _on_all_done(self, ok_count: int, total: int) -> None:
        self.logMessage.emit(f"[3D] 완료 {ok_count}/{total}")
        self._progress.setVisible(False)
        self._run.setEnabled(True)
        self._stop.setEnabled(False)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
