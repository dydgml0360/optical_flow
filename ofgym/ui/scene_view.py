"""씬을 바깥에서 돌려 보는 3D 뷰.

그리기는 `Scene.overview` (오프스크린 렌더러)가 하고, 여기서는 받은 그림을 그대로
붙인다. QOpenGLWidget 을 쓰지 않는 이유는 촬영용 렌더러와 **같은 코드 경로**로 그려야
보이는 것과 찍히는 것이 어긋나지 않기 때문이다.

좌표계는 캡처 그대로다 — +Y 가 아래라 화면 아래로 간다.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from ofgym.flow import camera as cam
from ofgym.flow.camera import Camera

# (관찰 카메라, world→camera) → RGB 그림
RenderFn = Callable[[Camera, np.ndarray], Optional[np.ndarray]]

_FOCAL_RATIO = 1.4  # 초점거리 / 화면 짧은 변 — 화각 약 40°


class SceneView(QWidget):
    """드래그=회전, 휠=줌, 우클릭 드래그=이동, 더블클릭=초기화."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 240)

        self._render_fn: Optional[RenderFn] = None
        self._image: Optional[QImage] = None
        self._message = "왼쪽에서 record 를 고르세요"

        self._target = np.zeros(3)
        self._home_distance = 500.0
        self._yaw = 0.0
        self._pitch = 0.0
        self._distance = self._home_distance
        self._pan = np.zeros(2)

        self._last_pos: Optional[QPoint] = None
        self._dragging = 0

    def set_render(self, render_fn: Optional[RenderFn]) -> None:
        self._render_fn = render_fn
        self.refresh()

    def set_message(self, message: str) -> None:
        self._message = message
        self._image = None
        self.update()

    def frame(self, target: np.ndarray, radius: float) -> None:
        """보는 중심과 기본 거리를 잡는다. 카메라 두 대가 같이 보이도록 넉넉히 물러선다."""
        self._target = np.asarray(target, np.float64)
        self._home_distance = max(radius * 4.5, 50.0)
        self.reset_view()

    def reset_view(self) -> None:
        self._yaw = 35.0
        self._pitch = -15.0
        self._distance = self._home_distance
        self._pan[:] = 0.0
        self.refresh()

    # ── 그리기 ────────────────────────────────────────────────────────────
    def _viewer(self):
        width, height = max(2, self.width()), max(2, self.height())
        # 고해상도 화면에서 흐릿하지 않게 장치 픽셀로 그린다.
        ratio = self.devicePixelRatioF()
        width, height = int(width * ratio), int(height * ratio)
        focal = _FOCAL_RATIO * min(width, height)
        viewer = Camera(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2)

        rot = cam.rotation(self._yaw, self._pitch, 0.0)
        # yaw/pitch 가 0 이면 캡처 카메라와 같은 쪽(-Z)에서 본다.
        eye = self._target + rot.T @ np.array([0.0, 0.0, -self._distance])
        extrinsic = cam.look_at(eye, self._target)
        extrinsic[:3, 3] += np.array([self._pan[0], self._pan[1], 0.0])
        return viewer, extrinsic

    def refresh(self) -> None:
        if self._render_fn is None or not self.isVisible():
            self._image = None
            self.update()
            return
        viewer, extrinsic = self._viewer()
        try:
            rgb = self._render_fn(viewer, extrinsic)
        except Exception as exc:  # 렌더 실패가 위젯을 죽이면 안 된다
            self._message = f"렌더 실패: {exc}"
            rgb = None
        if rgb is None:
            self._image = None
        else:
            rgb = np.ascontiguousarray(rgb)
            height, width, _ = rgb.shape
            image = QImage(rgb.data, width, height, 3 * width,
                           QImage.Format.Format_RGB888).copy()
            image.setDevicePixelRatio(self.devicePixelRatioF())
            self._image = image
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._image is not None:
            painter.drawImage(0, 0, self._image)
            return
        painter.fillRect(self.rect(), QColor(23, 22, 26))
        painter.setPen(QColor(150, 150, 155))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.refresh()

    # ── 입력 ──────────────────────────────────────────────────────────────
    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._last_pos = event.position().toPoint()
        self._dragging = 1 if event.button() == Qt.MouseButton.LeftButton else 2

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._last_pos is None or not self._dragging:
            return
        pos = event.position().toPoint()
        delta = pos - self._last_pos
        self._last_pos = pos
        if self._dragging == 1:
            self._yaw -= delta.x() * 0.4
            self._pitch = max(-89.0, min(89.0, self._pitch + delta.y() * 0.4))
        else:
            # 화면 1픽셀이 중심 거리에서 몇 mm 인지로 환산한다.
            per_pixel = self._distance / (_FOCAL_RATIO * max(2, min(self.width(), self.height())))
            self._pan += np.array([delta.x(), delta.y()]) * per_pixel
        self.refresh()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = 0
        self._last_pos = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self._distance = float(np.clip(self._distance / (1.12**steps), 20.0, 8000.0))
        self.refresh()
