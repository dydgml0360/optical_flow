"""핀홀 카메라와 자세.

규약은 OpenCV 와 같다 — 카메라 좌표계에서 +X 오른쪽, +Y 아래, +Z 앞. 픽셀 (u, v) 는
픽셀 **중심**이 정수다::

    u = fx * X / Z + cx        v = fy * Y / Z + cy

자세는 4x4 `world → camera` 행렬로 들고 다닌다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import yaml

from ofgym.config import PATHS

# depth 의 설정을 못 읽을 때 쓰는 값. config/intrinsic.yaml · stereo.yaml 의 값 그대로다.
_FALLBACK_FOCAL = 423.7465 * 6
_FALLBACK_SIZE = (3040, 4032)
_FALLBACK_PRINCIPAL = (237.0 * 6, 337.71 * 6)
_FALLBACK_BASELINE = 15.0


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, scale: float) -> "Camera":
        """해상도를 `scale` 배로. 픽셀 중심 규약이라 주점은 0.5 만큼 보정한다."""
        width = max(1, int(round(self.width * scale)))
        height = max(1, int(round(self.height * scale)))
        sx, sy = width / self.width, height / self.height
        return replace(
            self,
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    def project(self, points: np.ndarray):
        """카메라 좌표 (...,3) → (u, v, Z). Z<=0 은 호출한 쪽이 걸러야 한다."""
        z = points[..., 2]
        safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
        return (
            self.fx * points[..., 0] / safe + self.cx,
            self.fy * points[..., 1] / safe + self.cy,
            z,
        )

    def gl_projection(self, near: float, far: float) -> np.ndarray:
        """클립 공간 행렬. 프레임버퍼를 읽은 그대로가 위→아래 이미지가 되도록 y 를 뒤집지 않는다.

        OpenGL 은 y_ndc=-1 이 프레임버퍼 첫 행이다. 여기에 v=-0.5(이미지 맨 위)를 보내면
        읽어 온 버퍼를 뒤집을 필요가 없다. 대신 삼각형 감김이 반대가 되므로 컬링은 끈다.
        """
        w, h = self.width, self.height
        return np.array(
            [
                [2 * self.fx / w, 0, (2 * self.cx + 1) / w - 1, 0],
                [0, 2 * self.fy / h, (2 * self.cy + 1) / h - 1, 0],
                [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, 1, 0],
            ],
            np.float64,
        )


def device_camera() -> Camera:
    """디바이스 카메라 (전체 해상도, 세로). 읽기 실패하면 알려진 값으로.

    `intrinsic.yaml` 은 센서 가로 방향 1/6 해상도(672x506) 기준이다. 디바이스는 센서를
    90° 돌려 달아 저장 이미지는 세로(3040x4032)이고, 주점의 x/y 도 서로 바뀐다 —
    center 정점맵을 투영해 마스크와 맞춰 본 결과 (cx, cy) ≈ (1424, 2004) 로 이 값과 맞는다.
    """
    try:
        with (PATHS.depth / "config" / "intrinsic.yaml").open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        scale = float(cfg["width"]) / float(cfg["calibImageSize"][0])
        px, py = cfg["principalPoint"]
        return Camera(
            width=int(cfg["height"]),
            height=int(cfg["width"]),
            fx=float(cfg["focalLengthY"]) * scale,
            fy=float(cfg["focalLengthX"]) * scale,
            cx=float(py) * scale,
            cy=float(px) * scale,
        )
    except (OSError, KeyError, TypeError, ValueError, IndexError, yaml.YAMLError):
        return Camera(
            width=_FALLBACK_SIZE[0],
            height=_FALLBACK_SIZE[1],
            fx=_FALLBACK_FOCAL,
            fy=_FALLBACK_FOCAL,
            cx=_FALLBACK_PRINCIPAL[0],
            cy=_FALLBACK_PRINCIPAL[1],
        )


def device_baseline() -> float:
    """스테레오 기선 길이 (mm)."""
    try:
        with (PATHS.depth / "config" / "stereo.yaml").open(encoding="utf-8") as fh:
            return float((yaml.safe_load(fh) or {})["baseline"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return _FALLBACK_BASELINE


# ── 자세 ──────────────────────────────────────────────────────────────────
def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """도 단위. yaw 는 Y(세로축), pitch 는 X, roll 은 Z(시선축) 회전."""
    y, p, r = np.radians([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def place(center: np.ndarray, offset, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """`center` 를 축으로 돌린 뒤 `offset` 만큼 옮기는 4x4 모델 행렬."""
    center = np.asarray(center, np.float64)
    rot = rotation(yaw, pitch, roll)
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = center + np.asarray(offset, np.float64) - rot @ center
    return matrix


def shifted(extrinsic: np.ndarray, baseline) -> np.ndarray:
    """카메라를 **자기 좌표계 기준** `baseline` 만큼 옮긴 자세. 방향은 그대로다."""
    out = np.array(extrinsic, np.float64)
    out[:3, 3] -= np.asarray(baseline, np.float64)
    return out


def look_at(eye, target, down=(0.0, 1.0, 0.0)) -> np.ndarray:
    """`eye` 에서 `target` 을 보는 world→camera 행렬. 월드의 +Y(아래)가 화면 아래로 간다."""
    eye = np.asarray(eye, np.float64)
    forward = np.asarray(target, np.float64) - eye
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(np.asarray(down, np.float64), forward)
    norm = np.linalg.norm(right)
    right = right / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    rot = np.stack([right, np.cross(forward, right), forward])
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = -rot @ eye
    return matrix


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    return -extrinsic[:3, :3].T @ extrinsic[:3, 3]


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def frustum_lines(camera: Camera, extrinsic: np.ndarray, depth: float) -> np.ndarray:
    """카메라 시야를 그릴 선분들 (K,2,3), 월드 좌표."""
    u = np.array([-0.5, camera.width - 0.5, camera.width - 0.5, -0.5])
    v = np.array([-0.5, -0.5, camera.height - 0.5, camera.height - 0.5])
    corners = np.stack(
        [(u - camera.cx) / camera.fx * depth, (v - camera.cy) / camera.fy * depth,
         np.full(4, depth)], axis=1
    )
    inverse = np.linalg.inv(extrinsic)
    corners = transform(inverse, corners)
    origin = camera_center(extrinsic)
    lines = [(origin, c) for c in corners]
    lines += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return np.array(lines, np.float32)


def fit_distance(camera: Camera, radius: float) -> float:
    """반지름 `radius` 인 공이 화면에 다 들어오는 거리."""
    half = min(camera.width / camera.fx, camera.height / camera.fy) * 0.5
    return radius / max(half, 1e-6) * 1.05

