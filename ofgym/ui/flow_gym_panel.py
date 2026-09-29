"""Flow Gym — record 를 3D 로 올리고, 스테레오처럼 두 장 찍어 flow GT 를 눈으로 확인한다.

    왼쪽    디렉터리 선택 + record 목록
    가운데  3D 씬 (카메라 두 대의 시야 포함) / 촬영 결과 6칸
    오른쪽  카메라(촬영 위치·흔들림)·얼굴 움직임·무작위 범위·배경 조절, 저장

촬영 결과 6칸:

    사진 1      사진 2          flow (방향=색, 크기=채도)
    마스크      되돌린 사진 2   색 오차

'되돌린 사진 2' 는 사진 2 를 GT flow 로 끌어와 사진 1 자리에 맞춘 것이다. GT 가 맞으면
사진 1 과 겹치고 색 오차가 0 에 가깝다 — 가려진 곳(빨강)만 예외다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import PATHS
from ofgym.flow import gt as flow_gt
from ofgym.flow.camera import device_baseline
from ofgym.flow.mesh import build_mesh
from ofgym.flow.record import Record, find_records
from ofgym.flow.renderer import Renderer
from ofgym.flow.scene import (
    AXIS_HORIZONTAL,
    AXIS_VERTICAL,
    BACKGROUND_IMAGE,
    BACKGROUND_NOISE,
    BACKGROUND_NONE,
    FACE_ID,
    VIEW_ANGLES,
    RandomRanges,
    Scene,
    SceneParams,
    Shot,
    randomized,
)
from ofgym.ui.scene_view import SceneView

RECORD_ROLE = Qt.ItemDataRole.UserRole

PANES = ("사진 1", "사진 2", "flow", "마스크", "되돌린 사진 2", "색 오차")
ERROR_RANGE = 5.0  # 색 오차 그림에서 가장 밝은 색이 되는 계조 차

# flow 그림. 스테레오식 촬영은 flow 방향이 한쪽뿐이라 색상환으로는 전부 같은 색이 된다.
# 그래서 기본은 크기만 색으로 펴서 보여 주고, 범위는 얼굴에 맞춘다.
FLOW_FACE = "크기 — 얼굴 범위"
FLOW_ALL = "크기 — 전체 범위"
FLOW_WHEEL = "방향·크기 (색상환)"

FREE_VIEW = "자유 시점 (3D 씬)"


class ImagePane(QWidget):
    """제목 + 창 크기에 맞춰 줄여 그리는 그림 한 칸."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None

        self._title = QLabel(title)
        self._title.setStyleSheet("color: palette(mid);")
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(80, 80)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._canvas.setStyleSheet("background: #1b1b1b; border-radius: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._title)
        layout.addWidget(self._canvas, 1)

    def set_caption(self, text: str) -> None:
        self._title.setText(text)

    def set_image(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None:
            self._pixmap = None
            self._canvas.clear()
            return
        rgb = np.ascontiguousarray(rgb)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self._fit()

    def _fit(self) -> None:
        if self._pixmap is None:
            return
        self._canvas.setPixmap(
            self._pixmap.scaled(
                self._canvas.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()


def _spin(low: float, high: float, value: float, step: float, suffix: str,
          decimals: int = 1) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    box.setKeyboardTracking(False)  # 숫자를 치는 도중에는 다시 찍지 않는다
    return box


class FlowGymPanel(QWidget):
    logMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._renderer: Optional[Renderer] = None
        self._scene: Optional[Scene] = None
        self._record: Optional[Record] = None
        self._shot: Optional[Shot] = None
        self._root: Path = PATHS.shared
        self._background_image: Optional[str] = None
        self._background_seed = 0
        self._free_view: Optional[np.ndarray] = None
        self._station_angle = 0.0  # 자유 시점일 때도 기억해 두는 마지막 촬영 위치
        self._rng = np.random.default_rng()
        self._loading = False

        self._pending = QTimer(self)
        self._pending.setSingleShot(True)
        self._pending.setInterval(30)
        self._pending.timeout.connect(self._shoot)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_browser())
        splitter.addWidget(self._build_views())
        splitter.addWidget(self._build_controls())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900, 340])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # ── 화면 구성 ─────────────────────────────────────────────────────────
    def _build_browser(self) -> QWidget:
        self._root_label = QLabel()
        self._root_label.setWordWrap(True)
        self._root_label.setStyleSheet("color: palette(mid);")

        choose = QPushButton("디렉터리 선택…")
        choose.clicked.connect(self._choose_root)

        self._list = QListWidget()
        self._list.setUniformItemSizes(True)
        self._list.currentItemChanged.connect(self._on_record_changed)

        self._record_info = QLabel("-")
        self._record_info.setWordWrap(True)
        self._record_info.setStyleSheet("color: palette(mid);")

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(choose)
        layout.addWidget(self._root_label)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._record_info)
        return box

    def _build_views(self) -> QWidget:
        self._scene_view = SceneView()

        self._panes: Dict[str, ImagePane] = {}
        grid = QGridLayout()
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)
        for index, title in enumerate(PANES):
            pane = ImagePane(title)
            self._panes[title] = pane
            grid.addWidget(pane, index // 3, index % 3)
        self._flow_mode = QComboBox()
        self._flow_mode.addItems([FLOW_FACE, FLOW_ALL, FLOW_WHEEL])
        self._flow_mode.currentIndexChanged.connect(self._redraw)
        shots_bar = QHBoxLayout()
        shots_bar.setContentsMargins(6, 6, 6, 0)
        shots_bar.addWidget(QLabel("flow 표시"))
        shots_bar.addWidget(self._flow_mode)
        shots_bar.addStretch(1)
        shots = QWidget()
        shots_layout = QVBoxLayout(shots)
        shots_layout.setContentsMargins(0, 0, 0, 0)
        shots_layout.setSpacing(0)
        shots_layout.addLayout(shots_bar)
        shots_layout.addLayout(grid, 1)

        self._shoot_here = QPushButton("현재 위치에서 촬영")
        self._shoot_here.setToolTip(
            "지금 3D 씬을 보고 있는 자리에 카메라를 놓고 찍는다 (카메라 흔들림은 0 으로).\n"
            "사진은 세로로 길다 — 이 화면의 높이만큼이 사진의 폭에 담긴다."
        )
        self._shoot_here.setEnabled(False)
        self._shoot_here.clicked.connect(self._shoot_from_view)
        scene_tab = QWidget()
        scene_layout = QVBoxLayout(scene_tab)
        scene_layout.setContentsMargins(6, 6, 6, 6)
        scene_layout.addWidget(self._shoot_here)
        scene_layout.addWidget(self._scene_view, 1)

        self._stats = QLabel("-")
        self._stats.setWordWrap(True)
        self._stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._tabs = QTabWidget()
        self._tabs.addTab(scene_tab, "3D 씬")
        self._shots_tab = shots
        self._tabs.addTab(shots, "촬영 결과")

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._stats)
        return box

    def _build_controls(self) -> QWidget:
        defaults = SceneParams(baseline_mm=device_baseline())

        # 카메라
        self._scale = QComboBox()
        for label, value in (("1/8", 0.125), ("1/6", 1 / 6), ("1/4", 0.25),
                             ("1/2", 0.5), ("1/1 (3040x4032)", 1.0)):
            self._scale.addItem(label, value)
        self._scale.setCurrentIndex(2)
        self._baseline = _spin(0.0, 200.0, defaults.baseline_mm, 1.0, " mm")
        self._axis = QComboBox()
        self._axis.addItems([AXIS_VERTICAL, AXIS_HORIZONTAL])
        self._view = QComboBox()
        for label, angle in VIEW_ANGLES.items():
            self._view.addItem(label, angle)
        self._view.addItem(FREE_VIEW, None)
        self._view.setCurrentIndex(list(VIEW_ANGLES.values()).index(0.0))
        self._camera_info = QLabel("-")
        self._camera_info.setStyleSheet("color: palette(mid);")

        camera_form = QFormLayout()
        camera_form.addRow("해상도", self._scale)
        camera_form.addRow("기선 길이", self._baseline)
        camera_form.addRow("기선 방향", self._axis)
        camera_form.addRow("촬영 위치", self._view)
        camera_form.addRow(self._camera_info)
        camera_box = QGroupBox("카메라")
        camera_box.setLayout(camera_form)

        # 카메라 흔들림 · 얼굴 움직임 — 같은 모양의 tvec / rvec 여섯 칸
        self._camera_pose = self._pose_spins(100.0, 45.0)
        self._face_pose = self._pose_spins(500.0, 180.0)
        jitter_box = QGroupBox("카메라 흔들림 (촬영 위치의 카메라 좌표계)")
        jitter_box.setLayout(self._pose_form(self._camera_pose))
        pose_box = QGroupBox("얼굴 움직임 (캡처 자세 기준, 월드 좌표계)")
        pose_box.setLayout(self._pose_form(self._face_pose))

        # 무작위 범위
        ranges = RandomRanges()
        self._range_camera_t = _spin(0, 100, ranges.camera_t, 1, " mm")
        self._range_camera_r = _spin(0, 45, ranges.camera_r, 0.5, " °")
        self._range_face_t = _spin(0, 200, ranges.face_t, 1, " mm")
        self._range_face_r = _spin(0, 90, ranges.face_r, 0.5, " °")
        self._random_view = QCheckBox("촬영 위치도 뽑기 (+50° / 0° / -50°)")
        self._random_view.setChecked(ranges.random_view)

        random_button = QPushButton("무작위 배치")
        random_button.setToolTip(
            "카메라 흔들림·얼굴 움직임을 성분마다 ±범위 안에서 새로 뽑는다 (배경 무늬도)"
        )
        random_button.clicked.connect(self._randomize)
        reset_button = QPushButton("캡처 자세로")
        reset_button.setToolTip("흔들림과 움직임을 0 으로 — 디바이스가 실제로 찍던 배치")
        reset_button.clicked.connect(self._reset_pose)
        pose_buttons = QHBoxLayout()
        pose_buttons.addWidget(random_button)
        pose_buttons.addWidget(reset_button)

        random_form = QFormLayout()
        random_form.addRow("카메라 tvec ±", self._range_camera_t)
        random_form.addRow("카메라 rvec ±", self._range_camera_r)
        random_form.addRow("얼굴 tvec ±", self._range_face_t)
        random_form.addRow("얼굴 rvec ±", self._range_face_r)
        random_form.addRow(self._random_view)
        random_form.addRow(pose_buttons)
        random_box = QGroupBox("무작위 범위")
        random_box.setLayout(random_form)

        # 배경
        self._background = QComboBox()
        self._background.addItems([BACKGROUND_NOISE, BACKGROUND_IMAGE, BACKGROUND_NONE])
        self._background_z = _spin(100, 10000, defaults.background_z, 50, " mm", 0)
        self._image_button = QPushButton("이미지 고르기…")
        self._image_button.clicked.connect(self._choose_background)
        self._image_label = QLabel("선택 안 됨")
        self._image_label.setStyleSheet("color: palette(mid);")
        self._image_label.setWordWrap(True)

        background_form = QFormLayout()
        background_form.addRow("종류", self._background)
        background_form.addRow("거리 Z", self._background_z)
        background_form.addRow(self._image_button)
        background_form.addRow(self._image_label)
        background_box = QGroupBox("배경판")
        background_box.setLayout(background_form)

        # 저장
        self._save = QPushButton("이 쌍 저장")
        self._save.clicked.connect(self._save_shot)
        self._save.setEnabled(False)
        out = QLabel(str(PATHS.flow))
        out.setWordWrap(True)
        out.setStyleSheet("color: palette(mid);")
        save_form = QFormLayout()
        save_form.addRow(self._save)
        save_form.addRow("출력", out)
        save_box = QGroupBox("저장")
        save_box.setLayout(save_form)

        for box in (self._baseline, self._background_z,
                    *self._camera_pose.values(), *self._face_pose.values()):
            box.valueChanged.connect(self._on_params_changed)
        for combo in (self._scale, self._axis, self._view, self._background):
            combo.currentIndexChanged.connect(self._on_params_changed)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        for box in (camera_box, jitter_box, pose_box, random_box, background_box,
                    save_box):
            layout.addWidget(box)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(inner)
        self._sync_background_widgets()
        return scroll

    @staticmethod
    def _pose_spins(limit_t: float, limit_r: float) -> Dict[str, QDoubleSpinBox]:
        spins = {}
        for axis in "xyz":
            spins[f"t{axis}"] = _spin(-limit_t, limit_t, 0, 1, " mm")
        for axis in "xyz":
            spins[f"r{axis}"] = _spin(-limit_r, limit_r, 0, 1, " °")
        return spins

    @staticmethod
    def _pose_form(spins: Dict[str, QDoubleSpinBox]) -> QFormLayout:
        # 작은 각에서 rvec 의 성분은 그 축 둘레의 회전이다.
        hints = dict(tx="오른쪽", ty="아래", tz="앞", rx="끄덕임", ry="좌우 돌림", rz="갸웃")
        form = QFormLayout()
        for key, box in spins.items():
            kind = "tvec" if key[0] == "t" else "rvec"
            form.addRow(f"{kind} {key[1].upper()}  ({hints[key]})", box)
        return form

    # ── record 목록 ───────────────────────────────────────────────────────
    def reload(self, root: Optional[Path] = None) -> None:
        if root is not None:
            self._root = Path(root)
        self._root_label.setText(str(self._root))
        self._list.clear()

        records = find_records(self._root)
        if not records:
            item = QListWidgetItem("아틀라스가 있는 record 가 없습니다")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(item)
            self.logMessage.emit(f"[FlowGym] record 없음: {self._root}")
            return

        for record in records:
            item = QListWidgetItem(record.label)
            item.setData(RECORD_ROLE, record)
            item.setToolTip(str(record.path))
            self._list.addItem(item)
        self.logMessage.emit(f"[FlowGym] record {len(records)}개  ({self._root})")
        self._list.setCurrentRow(0)

    def _choose_root(self) -> None:
        start = self._root if self._root.is_dir() else Path.home()
        chosen = QFileDialog.getExistingDirectory(
            self, "record 가 있는 디렉터리 (record 하나를 바로 골라도 됩니다)", str(start)
        )
        if chosen:
            self.reload(Path(chosen))

    def _on_record_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        record = current.data(RECORD_ROLE) if current is not None else None
        if record is not None:
            self._load_record(record)

    def _ensure_scene(self) -> Optional[Scene]:
        if self._scene is not None:
            return self._scene
        try:
            self._renderer = Renderer()
        except Exception as exc:
            self._scene_view.set_message(f"렌더러를 만들지 못했습니다\n{exc}")
            self.logMessage.emit(f"[FlowGym] 렌더러 실패: {exc}")
            return None
        self._scene = Scene(self._renderer)
        self.logMessage.emit(f"[FlowGym] 렌더러: {self._renderer.description}")
        return self._scene

    def _load_record(self, record: Record) -> None:
        scene = self._ensure_scene()
        if scene is None:
            return
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mesh = build_mesh(record)
            scene.set_mesh(mesh)
        except Exception as exc:
            scene.set_mesh(None)
            self._record = None
            self._shot = None
            self._save.setEnabled(False)
            self._shoot_here.setEnabled(False)
            self._scene_view.set_render(None)
            self._scene_view.set_message(f"불러오기 실패\n{exc}")
            self._record_info.setText(str(exc))
            self.logMessage.emit(f"[FlowGym] {record.record_id}: {exc}")
            return
        finally:
            QGuiApplication.restoreOverrideCursor()

        self._record = record
        meta = record.meta
        self._record_info.setText(
            f"{record.group} / {record.record_id}\n"
            f"아틀라스 {meta['width']}x{meta['height']}  텍셀 {meta['z_rate'] * 1000:.0f}µm\n"
            f"메쉬 정점 {len(mesh.vertices):,} · 면 {len(mesh.faces):,} · "
            f"격자 {mesh.step_mm:.2f}mm"
        )
        self.logMessage.emit(
            f"[FlowGym] {record.record_id} 메쉬 {len(mesh.faces):,}면  ({record.path})"
        )
        self._shoot_here.setEnabled(True)
        self._scene_view.set_render(self._render_overview)
        self._scene_view.frame(mesh.center, mesh.radius)
        self._shoot()

    # ── 조절값 ────────────────────────────────────────────────────────────
    def params(self) -> SceneParams:
        pose = {f"camera_{key}": box.value() for key, box in self._camera_pose.items()}
        pose.update({f"face_{key}": box.value() for key, box in self._face_pose.items()})
        return SceneParams(
            scale=float(self._scale.currentData()),
            baseline_mm=self._baseline.value(),
            baseline_axis=self._axis.currentText(),
            view_angle=self._station_angle,
            free_view=self._current_free_view(),
            background=self._background.currentText(),
            background_z=self._background_z.value(),
            background_seed=self._background_seed,
            background_image=self._background_image,
            **pose,
        )

    def ranges(self) -> RandomRanges:
        return RandomRanges(
            camera_t=self._range_camera_t.value(),
            camera_r=self._range_camera_r.value(),
            face_t=self._range_face_t.value(),
            face_r=self._range_face_r.value(),
            random_view=self._random_view.isChecked(),
        )

    def _set_pose(self, params: SceneParams) -> None:
        values = params.as_dict()
        self._loading = True
        try:
            for key, box in self._camera_pose.items():
                box.setValue(values[f"camera_{key}"])
            for key, box in self._face_pose.items():
                box.setValue(values[f"face_{key}"])
            if params.free_view is None:
                index = self._view.findData(params.view_angle)
            else:
                index = self._view.findText(FREE_VIEW)
                self._free_view = np.array(params.free_view, np.float64)
            if index >= 0:
                self._view.setCurrentIndex(index)
            self._sync_view()
            self._background_seed = params.background_seed
        finally:
            self._loading = False
        self._on_params_changed()

    def _randomize(self) -> None:
        self._set_pose(randomized(self.params(), self.ranges(), self._rng))

    def _reset_pose(self) -> None:
        self._set_pose(SceneParams(view_angle=self._station_angle,
                                   free_view=self._current_free_view(),
                                   background_seed=self._background_seed))

    # ── 촬영 위치 / 자유 시점 ──────────────────────────────────────────────
    def _sync_view(self) -> None:
        """콤보가 가리키는 것을 상태에 옮긴다. 자유 시점을 처음 고르면 지금 보는 자리를 잡는다."""
        angle = self._view.currentData()
        if angle is not None:
            self._station_angle = float(angle)
        elif self._free_view is None:
            self._free_view = self._scene_view.pose()

    def _current_free_view(self):
        if self._view.currentData() is not None or self._free_view is None:
            return None
        return self._free_view.tolist()

    def _shoot_from_view(self) -> None:
        if self._scene is None or self._scene.mesh is None:
            return
        self._free_view = self._scene_view.pose()
        self._loading = True
        try:
            for box in self._camera_pose.values():
                box.setValue(0.0)
            self._view.setCurrentIndex(self._view.findText(FREE_VIEW))
        finally:
            self._loading = False
        self._pending.stop()
        self._shoot()
        self._tabs.setCurrentWidget(self._shots_tab)

    def _sync_background_widgets(self) -> None:
        uses_image = self._background.currentText() == BACKGROUND_IMAGE
        self._image_button.setVisible(uses_image)
        self._image_label.setVisible(uses_image)
        self._background_z.setEnabled(self._background.currentText() != BACKGROUND_NONE)

    def _choose_background(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "배경 이미지", str(Path.home()), "이미지 (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if chosen:
            self._background_image = chosen
            self._image_label.setText(Path(chosen).name)
            self._on_params_changed()

    def _on_params_changed(self, *_args) -> None:
        self._sync_background_widgets()
        self._sync_view()
        if not self._loading:
            self._pending.start()

    # ── 촬영 ──────────────────────────────────────────────────────────────
    def _render_overview(self, viewer, extrinsic):
        if self._scene is None or self._scene.mesh is None:
            return None
        return self._scene.overview(self.params(), viewer, extrinsic)

    def _shoot(self) -> None:
        if self._scene is None or self._scene.mesh is None:
            return
        params = self.params()
        try:
            shot = self._scene.shoot(params)
        except Exception as exc:
            self._shot = None
            self._save.setEnabled(False)
            self._stats.setText(f"촬영 실패: {exc}")
            self.logMessage.emit(f"[FlowGym] 촬영 실패: {exc}")
            return
        if self._scene.background_error:
            self.logMessage.emit(f"[FlowGym] {self._scene.background_error}")

        self._shot = shot
        self._save.setEnabled(True)
        self._show(shot)
        self._scene_view.refresh()

    def _show(self, shot: Shot) -> None:
        gt = shot.gt
        camera = shot.first.camera
        stats = gt.stats()

        warped = flow_gt.warp_back(shot.second.color, gt.flow)
        error = flow_gt.photometric_error(shot.first.color, warped, gt.valid)

        self._panes["사진 1"].set_image(shot.first.color)
        self._panes["사진 2"].set_image(shot.second.color)
        self._show_flow(shot)
        self._panes["마스크"].set_image(flow_gt.mask_picture(gt))
        self._panes["마스크"].set_caption("마스크  (초록 = 유효, 빨강 = 가려짐)")
        self._panes["되돌린 사진 2"].set_image(warped)
        self._panes["색 오차"].set_image(
            flow_gt.scalar_to_color(error, gt.valid, 0.0, ERROR_RANGE)
        )
        self._panes["색 오차"].set_caption(f"색 오차  (0 ~ {ERROR_RANGE:.0f} 계조)")

        self._camera_info.setText(
            f"{camera.width}x{camera.height}  f={camera.fx:.1f}px"
            + ("  · 자유 시점" if shot.params.free_view is not None else "")
        )
        if "flow_mean" not in stats:
            self._stats.setText("유효한 픽셀이 없습니다 — 얼굴과 배경이 모두 화면 밖입니다")
            return
        mean_error = float(error[gt.valid].mean())
        self._stats.setText(
            f"유효 {stats['valid'] * 100:.1f}%   가려짐 {stats['occluded'] * 100:.1f}%   "
            f"표면 없음 {(1 - stats['surface']) * 100:.1f}%      "
            f"flow 크기 {stats['flow_min']:.1f} ~ {stats['flow_max']:.1f}px "
            f"(평균 {stats['flow_mean']:.1f},  u {stats['u_mean']:+.1f} / "
            f"v {stats['v_mean']:+.1f})      "
            f"깊이 {stats['depth_min']:.0f} ~ {stats['depth_max']:.0f}mm      "
            f"되돌림 색 오차 평균 {mean_error:.2f} 계조"
        )

    def _show_flow(self, shot: Shot) -> None:
        gt = shot.gt
        mode = self._flow_mode.currentText()
        if mode == FLOW_WHEEL:
            top = flow_gt.magnitude_range(gt.flow, gt.surface)[1]
            picture = flow_gt.flow_to_color(gt.flow, gt.surface, top)
            caption = f"flow  (채도 최대 = {top:.1f}px)"
        else:
            face = shot.first.ids == FACE_ID
            on_face = mode == FLOW_FACE and face.any()
            low, high = flow_gt.magnitude_range(gt.flow, face if on_face else gt.surface)
            picture = flow_gt.scalar_to_color(
                np.linalg.norm(gt.flow, axis=2), gt.surface, low, high
            )
            where = "얼굴 기준" if on_face else "전체 기준"
            caption = f"flow 크기  ({low:.1f} ~ {high:.1f}px, {where})"
        self._panes["flow"].set_image(picture)
        self._panes["flow"].set_caption(caption)

    def _redraw(self, *_args) -> None:
        if self._shot is not None:
            self._show_flow(self._shot)

    # ── 저장 ──────────────────────────────────────────────────────────────
    def _save_shot(self) -> None:
        if self._shot is None or self._record is None:
            return
        parent = PATHS.flow / self._record.record_id
        index = 0
        while (parent / f"{index:04d}").exists():
            index += 1
        target = parent / f"{index:04d}"
        try:
            flow_gt.save_sample(
                target, self._shot.first, self._shot.second, self._shot.gt,
                dict(record=self._record.record_id, source=str(self._record.path),
                     params=self._shot.params.as_dict()),
            )
        except OSError as exc:
            self.logMessage.emit(f"[FlowGym] 저장 실패: {exc}")
            return
        self.logMessage.emit(f"[FlowGym] 저장: {target}")
