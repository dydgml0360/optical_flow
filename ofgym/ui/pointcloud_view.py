"""포인트 클라우드 3D 뷰어.

Open3D 0.18 은 최신 macOS 에서 대화형 창을 못 연다 (docs/mac-reconstruction.md §2A —
번들 GLFW 가 디스플레이 서비스 포트를 못 잡는다). 그래서 여기서는 Open3D 도 OpenGL 도
쓰지 않고 **numpy 로 직접 투영해 QImage 에 스플랫**한다. 30만 점 규모에서 회전이
충분히 부드럽고, 의존성이 늘지 않으며, 오프스크린에서도 똑같이 동작한다.

좌표계는 재구성 좌표계 그대로다 — +Y 아래(턱 방향), +Z 가 카메라에서 멀어지는 방향.
정면 뷰는 −Z 에서 보고 up 이 −Y 다.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

Cloud = Tuple[np.ndarray, np.ndarray]  # (points Nx3 float32, colors Nx3 uint8 RGB)

# 드래그 중에는 이만큼만 그린다. 손을 떼면 전체를 다시 그린다.
DRAG_BUDGET = 90_000
BACKGROUND = (23, 22, 26)  # 디바이스 UI 와 같은 배경


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    return rx @ ry


class PointCloudView(QWidget):
    """드래그=회전, 휠=줌, 가운데/우클릭 드래그=이동, 더블클릭=초기화."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._points = np.zeros((0, 3), np.float32)
        self._colors = np.zeros((0, 3), np.uint8)
        self._center = np.zeros(3, np.float32)
        self._extent = 1.0

        self._yaw = 0.0
        self._pitch = 0.0
        self._zoom = 1.0
        self._pan = np.zeros(2, np.float32)
        self._point_size = 2

        self._last_pos: Optional[QPoint] = None
        self._dragging = 0  # 0=없음 1=회전 2=이동
        self._cache: Optional[QPixmap] = None

    # ── 데이터 ────────────────────────────────────────────────────────────
    def set_clouds(self, clouds: Sequence[Cloud]) -> None:
        parts_p: List[np.ndarray] = []
        parts_c: List[np.ndarray] = []
        for points, colors in clouds:
            if len(points):
                parts_p.append(np.asarray(points, np.float32))
                parts_c.append(np.asarray(colors, np.uint8))

        if not parts_p:
            self.clear()
            return

        self._points = np.concatenate(parts_p)
        self._colors = np.concatenate(parts_c)
        # 중앙값 중심 + 로버스트 범위 — 아웃라이어 한 점에 시야가 끌려가지 않게.
        self._center = np.median(self._points, axis=0).astype(np.float32)
        spread = np.percentile(np.abs(self._points - self._center), 98, axis=0)
        self._extent = float(max(spread.max(), 1e-3))
        self.reset_view()

    def clear(self) -> None:
        self._points = np.zeros((0, 3), np.float32)
        self._colors = np.zeros((0, 3), np.uint8)
        self._cache = None
        self.update()

    @property
    def point_count(self) -> int:
        return len(self._points)

    # ── 카메라 ────────────────────────────────────────────────────────────
    def reset_view(self) -> None:
        self._yaw = 0.0
        self._pitch = 0.0
        self._zoom = 1.0
        self._pan[:] = 0.0
        self._cache = None
        self.update()

    def set_point_size(self, size: int) -> None:
        self._point_size = max(1, min(6, int(size)))
        self._cache = None
        self.update()

    # ── 렌더 ──────────────────────────────────────────────────────────────
    def _render(self, width: int, height: int, budget: Optional[int]) -> QImage:
        canvas = np.empty((height, width, 3), np.uint8)
        canvas[:] = BACKGROUND

        if not len(self._points):
            return QImage(
                canvas.data, width, height, 3 * width, QImage.Format.Format_RGB888
            ).copy()

        points, colors = self._points, self._colors
        if budget is not None and len(points) > budget:
            step = len(points) // budget + 1
            points, colors = points[::step], colors[::step]

        rotated = (points - self._center) @ _rotation(self._yaw, self._pitch).T

        # 정사영. 얼굴 하나를 보는 용도라 원근을 넣어도 형상 판단에 보태는 게 없고,
        # 정사영은 회전해도 스케일이 안 흔들려 비교에 유리하다.
        scale = 0.5 * min(width, height) / self._extent * self._zoom
        xs = rotated[:, 0] * scale + width * 0.5 + self._pan[0]
        # 재구성 좌표계는 +Y 가 아래 → 화면 y 와 방향이 같다.
        ys = rotated[:, 1] * scale + height * 0.5 + self._pan[1]

        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        if not inside.any():
            return QImage(
                canvas.data, width, height, 3 * width, QImage.Format.Format_RGB888
            ).copy()

        xi, yi = xi[inside], yi[inside]
        zi = rotated[inside, 2]
        ci = colors[inside]

        # 화가 알고리즘 — 먼 것부터 그려서 가까운 것이 덮게 한다.
        # +Z 가 카메라에서 멀어지는 방향이므로 z 내림차순.
        order = np.argsort(-zi, kind="stable")
        xi, yi, ci = xi[order], yi[order], ci[order]

        radius = self._point_size - 1
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                px = xi + dx
                py = yi + dy
                ok = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                canvas[py[ok], px[ok]] = ci[ok]

        return QImage(
            canvas.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._cache is None or self._cache.size() != self.size():
            budget = DRAG_BUDGET if self._dragging else None
            self._cache = QPixmap.fromImage(
                self._render(max(1, self.width()), max(1, self.height()), budget)
            )
        painter.drawPixmap(0, 0, self._cache)

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
            self._yaw += delta.x() * 0.4
            self._pitch += delta.y() * 0.4
            self._pitch = max(-89.0, min(89.0, self._pitch))
        else:
            self._pan += np.array([delta.x(), delta.y()], np.float32)
        self._cache = None
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = 0
        self._last_pos = None
        self._cache = None  # 전체 점으로 다시 그린다
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self._zoom = float(np.clip(self._zoom * (1.12**steps), 0.05, 40.0))
        self._cache = None
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._cache = None
