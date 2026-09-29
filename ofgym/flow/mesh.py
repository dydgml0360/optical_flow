"""record → 아틀라스 텍스처를 입힌 삼각 메쉬 한 장.

아틀라스는 원통 전개도다 (`ui/shared/atlas_builder.py` 와 같은 규약)::

    theta = degrees(arctan2(x, RR - z))     l = y
    col   = (theta - theta0) / theta_rate   row = (l - l0) / z_rate

아틀라스에는 색만 있고 반경은 없다. 형상은 뷰별 정점맵에 있으므로, 세 뷰의 정점을
(θ, l) 격자에 뿌려 반경 맵 r(θ, l) 을 만들고 그 격자를 그대로 메쉬로 쓴다::

    x = r sin(theta)     y = l     z = RR - r cos(theta)

이렇게 하면 뷰 셋이 겹치는 곳에서도 면이 한 겹이고, 격자 좌표가 곧 UV 다.
얼굴은 회전축에서 보면 거의 별 모양(한 θ, l 에 표면이 하나)이라 이 표현이 맞는다 —
콧방울 안쪽처럼 접히는 곳은 바깥 면 하나로 뭉개진다.

좌표계는 캡처 그대로다: center 카메라가 원점, +Z 가 카메라에서 멀어지는 방향,
+Y 가 아래(턱), 단위 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ofgym.flow.record import Record

# 메쉬 격자 간격. 정점맵이 1/6 데시메이션이라 실제 형상 정보는 ~0.42mm 간격이다.
# 그보다 촘촘히 잡아도 보간일 뿐이라 비슷한 값으로 둔다.
DEFAULT_STEP_MM = 0.4

# 격자 한 칸 사이 반경 차가 이보다 크면 면을 잇지 않는다 (실루엣에서 늘어난 치마).
MAX_R_JUMP_MM = 4.0

# 뿌린 가중치를 번지게 하는 정도(격자 칸). 정점 간격 ≈ 격자 간격이라 1칸이면 빈칸이 메워진다.
_SPREAD_SIGMA = 1.0
_MIN_SUPPORT = 0.12


@dataclass
class Mesh:
    vertices: np.ndarray  # (N,3) float32, mm
    uvs: np.ndarray  # (N,2) float32, 0~1. v 는 아틀라스 행 방향(위→아래)
    faces: np.ndarray  # (M,3) int32
    texture: np.ndarray  # (H,W,3) uint8 RGB, 빈 텍셀은 주변 색으로 메움
    center: np.ndarray  # (3,) float32 — 회전·배치의 기준점
    radius: float  # center 에서 가장 먼 정점까지, mm
    step_mm: float
    pivot_radius: float  # 카메라 회전축까지의 거리 (meta 의 rr), mm

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def _cylinder(points: np.ndarray, rr: float):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    depth = rr - z
    return np.degrees(np.arctan2(x, depth)), y, np.hypot(x, depth)


def _radius_grid(record: Record, step: int):
    """세 뷰의 정점을 (θ, l) 격자에 쌍선형으로 뿌린 반경 맵. 격자점 (i,j) = 텍셀 (i*step, j*step)."""
    meta = record.meta
    rows = meta["height"] // step + 1
    cols = meta["width"] // step + 1
    acc = np.zeros(rows * cols, np.float64)
    weight = np.zeros(rows * cols, np.float64)

    for path in record.vertex_paths().values():
        grid = np.load(path)
        if grid.ndim != 3 or grid.shape[2] != 3:
            continue
        points = grid.reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)]
        if not len(points):
            continue
        theta, l, r = _cylinder(points, meta["rr"])
        gx = (theta - meta["theta0"]) / meta["theta_rate"] / step
        gy = (l - meta["l0"]) / meta["z_rate"] / step
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        fx, fy = gx - x0, gy - y0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                xi, yi, w = x0 + dx, y0 + dy, wx * wy
                ok = (xi >= 0) & (xi < cols) & (yi >= 0) & (yi < rows)
                index = yi[ok] * cols + xi[ok]
                acc += np.bincount(index, w[ok] * r[ok], rows * cols)
                weight += np.bincount(index, w[ok], rows * cols)

    acc = cv2.GaussianBlur(acc.reshape(rows, cols), (0, 0), _SPREAD_SIGMA)
    weight = cv2.GaussianBlur(weight.reshape(rows, cols), (0, 0), _SPREAD_SIGMA)
    valid = weight > _MIN_SUPPORT
    radius = np.zeros_like(acc)
    radius[valid] = acc[valid] / weight[valid]
    return radius, valid


def _fill_uncovered(image: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """빈 텍셀을 주변 색으로 채운다. 경계에서 필터링이 검은색을 끌어오지 않게 하려는 것."""
    if covered.all():
        return image
    scale = 8
    small_size = (max(1, image.shape[1] // scale), max(1, image.shape[0] // scale))
    # 전체 해상도에서는 uint8 로만 다룬다 — 1100만 텍셀을 float 로 바꾸면 그것만 0.3초다.
    masked = image.copy()
    masked[~covered] = 0
    num = cv2.resize(masked, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    den = cv2.resize(covered.astype(np.uint8) * 255, small_size,
                     interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    filled = np.zeros_like(num)
    known = np.zeros_like(den)
    for sigma in (1.5, 4, 12, 40, 120):  # 가까운 색이 먼저 자리를 잡는다
        blur_num = cv2.GaussianBlur(num, (0, 0), sigma)
        blur_den = cv2.GaussianBlur(den, (0, 0), sigma)
        take = (known < 0.5) & (blur_den > 1e-3)
        filled[take] = blur_num[take] / blur_den[take][:, None]
        known[take] = 1.0

    big = cv2.resize(np.clip(filled, 0, 255).astype(np.uint8),
                     (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    np.copyto(masked, big, where=~covered[..., None])
    return masked


def load_texture(record: Record) -> Tuple[np.ndarray, np.ndarray]:
    """(RGB 텍스처, covered 마스크). 크기가 meta 와 다르면 예외."""
    meta = record.meta
    bgr = cv2.imread(str(record.atlas_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"아틀라스를 읽지 못했습니다: {record.atlas_path}")
    if bgr.shape[:2] != (meta["height"], meta["width"]):
        raise ValueError(
            f"아틀라스 크기 {bgr.shape[1]}x{bgr.shape[0]} 가 meta "
            f"{meta['width']}x{meta['height']} 와 다릅니다"
        )
    mask = cv2.imread(str(record.covered_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != bgr.shape[:2]:
        covered = np.any(bgr > 0, axis=2)
    else:
        covered = mask >= 128
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _fill_uncovered(rgb, covered), covered


def build_mesh(record: Record, step_mm: float = DEFAULT_STEP_MM) -> Mesh:
    meta = record.meta
    step = max(1, int(round(step_mm / meta["z_rate"])))
    texture, covered = load_texture(record)

    radius, valid = _radius_grid(record, step)
    rows, cols = radius.shape

    # 아틀라스가 비어 있는 곳은 색이 없으니 면도 만들지 않는다.
    sample_r = np.minimum(np.arange(rows) * step, meta["height"] - 1)
    sample_c = np.minimum(np.arange(cols) * step, meta["width"] - 1)
    valid &= covered[np.ix_(sample_r, sample_c)]
    if not valid.any():
        raise ValueError("메쉬로 만들 정점이 없습니다 (정점맵과 아틀라스가 겹치지 않음)")

    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    theta = np.radians(meta["theta0"] + jj * step * meta["theta_rate"])
    l = meta["l0"] + ii * step * meta["z_rate"]
    xyz = np.stack(
        [radius * np.sin(theta), l, meta["rr"] - radius * np.cos(theta)], axis=-1
    )
    # 텍셀 k 의 중심이 텍스처 좌표 (k + 0.5) / N 이다.
    uv = np.stack(
        [(jj * step + 0.5) / meta["width"], (ii * step + 0.5) / meta["height"]], axis=-1
    )

    corners = (valid[:-1, :-1], valid[:-1, 1:], valid[1:, :-1], valid[1:, 1:])
    quad = corners[0] & corners[1] & corners[2] & corners[3]
    r4 = np.stack([radius[:-1, :-1], radius[:-1, 1:], radius[1:, :-1], radius[1:, 1:]])
    quad &= (r4.max(axis=0) - r4.min(axis=0)) < MAX_R_JUMP_MM

    qi, qj = np.nonzero(quad)
    a = qi * cols + qj
    b, c, d = a + 1, a + cols, a + cols + 1
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])

    used = np.zeros(rows * cols, bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    vertices = xyz.reshape(-1, 3)[used].astype(np.float32)

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    return Mesh(
        vertices=vertices,
        uvs=uv.reshape(-1, 2)[used].astype(np.float32),
        faces=remap[faces].astype(np.int32),
        texture=texture,
        center=center,
        radius=float(np.linalg.norm(vertices - center, axis=1).max()),
        step_mm=step * meta["z_rate"],
        pivot_radius=float(meta["rr"]),
    )
