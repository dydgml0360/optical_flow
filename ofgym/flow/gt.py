"""두 프레임 사이의 optical flow ground truth 와 그 검증.

flow 는 **계산**한다 — 첫 프레임의 픽셀이 본 표면점을 두 번째 프레임의 카메라로 다시
투영한 자리에서 원래 자리를 뺀다. 렌더 두 장을 비교해 추정하는 단계가 없어서 값 자체는
부동소수 오차 수준으로 정확하다.

마스크 셋을 같이 낸다.

- `surface`  첫 프레임에서 표면이 찍힌 픽셀. flow 값이 정의되는 곳.
- `occluded` 두 번째 프레임에서는 다른 것에 가려졌거나 화면 밖으로 나간 픽셀.
             flow 값은 있지만(FlyingThings 와 같다) 두 사진에서 대응을 찾을 수는 없다.
- `valid`    surface 이고 occluded 가 아닌 곳. 학습 손실에 쓸 마스크.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from ofgym.flow.renderer import Frame

# 가려짐 판정 여유. 두 번째 프레임에서 다시 본 깊이가 그 자리 깊이보다 이만큼 넘게 멀면 가려진 것.
OCCLUSION_TOLERANCE_MM = 0.05

_FLO_MAGIC = 202021.25


@dataclass
class FlowGT:
    flow: np.ndarray  # (H,W,2) float32, 픽셀. surface 밖은 0
    surface: np.ndarray  # (H,W) bool
    occluded: np.ndarray  # (H,W) bool
    valid: np.ndarray  # (H,W) bool
    depth: np.ndarray  # (H,W) float32, mm. surface 밖은 0

    def stats(self) -> Dict[str, float]:
        out = {
            "surface": float(self.surface.mean()),
            "valid": float(self.valid.mean()),
            "occluded": float((self.occluded & self.surface).mean()),
        }
        if self.valid.any():
            magnitude = np.linalg.norm(self.flow[self.valid], axis=1)
            out.update(
                flow_min=float(magnitude.min()),
                flow_mean=float(magnitude.mean()),
                flow_max=float(magnitude.max()),
                u_mean=float(self.flow[self.valid][:, 0].mean()),
                v_mean=float(self.flow[self.valid][:, 1].mean()),
                depth_min=float(self.depth[self.valid].min()),
                depth_max=float(self.depth[self.valid].max()),
            )
        return out


def compute_flow(first: Frame, second: Frame) -> FlowGT:
    """`first` 의 각 픽셀이 `second` 에서 어디로 갔는지."""
    height, width = first.ids.shape
    surface = first.ids > 0

    here = first.camera_points()
    there = first.points_in(second.extrinsic, second.models)
    u2, v2, z2 = second.camera.project(there)

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64),
                         np.arange(height, dtype=np.float64))
    flow = np.zeros((height, width, 2), np.float32)
    flow[..., 0] = np.where(surface, u2 - uu, 0.0)
    flow[..., 1] = np.where(surface, v2 - vv, 0.0)

    # 두 번째 프레임에서 그 자리에 실제로 찍힌 깊이. 픽셀 하나 안에서도 표면이 기울면
    # 깊이가 달라지므로 3x3 이웃의 가장 먼 값과 비교한다 — 보이는 점은 이웃 범위 안에 든다.
    # 대가로 가려짐 경계에서 1픽셀 폭은 '보임' 으로 남는다.
    depth2 = second.depth()
    finite = np.where(np.isfinite(depth2), depth2, np.float64(1e12))
    farthest = cv2.dilate(finite, np.ones((3, 3), np.uint8))

    ui = np.rint(u2).astype(np.int64)
    vi = np.rint(v2).astype(np.int64)
    inside = surface & (z2 > 0) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    seen = np.zeros((height, width), bool)
    seen[inside] = z2[inside] <= farthest[vi[inside], ui[inside]] + OCCLUSION_TOLERANCE_MM

    occluded = surface & ~seen
    return FlowGT(
        flow=flow,
        surface=surface,
        occluded=occluded,
        valid=surface & seen,
        depth=np.where(surface, here[..., 2], 0.0).astype(np.float32),
    )


# ── 검증 ──────────────────────────────────────────────────────────────────
def warp_back(second_color: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """두 번째 사진을 flow 로 끌어와 첫 사진 자리에 맞춘다. GT 가 맞으면 첫 사진과 겹친다."""
    height, width = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    return cv2.remap(
        second_color, uu + flow[..., 0], vv + flow[..., 1],
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def photometric_error(first_color: np.ndarray, warped: np.ndarray,
                      valid: np.ndarray) -> np.ndarray:
    """픽셀별 색 차이 (H,W) float32, 0~255. valid 밖은 0."""
    diff = np.abs(first_color.astype(np.float32) - warped.astype(np.float32)).mean(axis=2)
    return np.where(valid, diff, 0.0).astype(np.float32)


# ── 그림 ──────────────────────────────────────────────────────────────────
def _color_wheel() -> np.ndarray:
    """Middlebury flow 색상환 (Baker et al.). KITTI·Sintel·RAFT 시각화와 같은 것."""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    wheel = np.zeros((ry + yg + gc + cb + bm + mr, 3))
    col = 0
    wheel[0:ry, 0] = 255
    wheel[0:ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col += ry
    wheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    wheel[col:col + yg, 1] = 255
    col += yg
    wheel[col:col + gc, 1] = 255
    wheel[col:col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    wheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    wheel[col:col + cb, 2] = 255
    col += cb
    wheel[col:col + bm, 2] = 255
    wheel[col:col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    wheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    wheel[col:col + mr, 0] = 255
    return wheel


_WHEEL = _color_wheel()


def flow_to_color(flow: np.ndarray, mask: np.ndarray, max_magnitude: float) -> np.ndarray:
    """방향은 색상, 크기는 채도. `max_magnitude` 픽셀에서 채도가 꽉 찬다. mask 밖은 검정."""
    scale = max(float(max_magnitude), 1e-6)
    u = flow[..., 0] / scale
    v = flow[..., 1] / scale
    radius = np.sqrt(u * u + v * v)
    angle = np.arctan2(-v, -u) / np.pi
    position = (angle + 1) / 2 * (len(_WHEEL) - 1)
    k0 = np.floor(position).astype(np.int32)
    k1 = (k0 + 1) % len(_WHEEL)
    frac = (position - k0)[..., None]

    color = ((1 - frac) * _WHEEL[k0] + frac * _WHEEL[k1]) / 255.0
    inside = (radius <= 1)[..., None]
    r = radius[..., None]
    color = np.where(inside, 1 - r * (1 - color), color * 0.75)
    out = (color * 255).astype(np.uint8)
    out[~mask] = 0
    return out


def scalar_to_color(values: np.ndarray, mask: np.ndarray, low: float, high: float,
                    colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    span = max(high - low, 1e-9)
    norm = np.clip((values - low) / span * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(norm, colormap), cv2.COLOR_BGR2RGB)
    rgb[~mask] = 0
    return rgb


def mask_picture(gt: FlowGT) -> np.ndarray:
    """valid=초록, 가려짐=빨강, 표면 없음=검정."""
    out = np.zeros(gt.surface.shape + (3,), np.uint8)
    out[gt.valid] = (70, 190, 110)
    out[gt.occluded] = (225, 80, 70)
    return out


# ── 저장 ──────────────────────────────────────────────────────────────────
def write_flo(path: Path, flow: np.ndarray) -> None:
    """Middlebury .flo — RAFT 의 `frame_utils.readFlow` 가 읽는 형식."""
    height, width = flow.shape[:2]
    with open(path, "wb") as fh:
        np.array([_FLO_MAGIC], np.float32).tofile(fh)
        np.array([width, height], np.int32).tofile(fh)
        flow.astype(np.float32).tofile(fh)


def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        magic = np.fromfile(fh, np.float32, 1)
        if magic.size != 1 or magic[0] != np.float32(_FLO_MAGIC):
            raise ValueError(f".flo 매직 넘버가 아닙니다: {path}")
        width, height = np.fromfile(fh, np.int32, 2)
        return np.fromfile(fh, np.float32, 2 * width * height).reshape(height, width, 2)


def save_sample(directory: Path, first: Frame, second: Frame, gt: FlowGT,
                meta: dict) -> Path:
    """한 쌍을 `directory` 에 쓴다.

        img1.png img2.png   사진
        flow.flo            img1 → img2, 픽셀
        valid.png           255 = 손실에 쓸 픽셀
        occluded.png        255 = img2 에서 안 보이는 픽셀
        depth.npy           img1 의 카메라 Z (mm), float32
        meta.json           카메라·자세·기선
    """
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / "img1.png"), cv2.cvtColor(first.color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(directory / "img2.png"), cv2.cvtColor(second.color, cv2.COLOR_RGB2BGR))
    write_flo(directory / "flow.flo", gt.flow)
    cv2.imwrite(str(directory / "valid.png"), gt.valid.astype(np.uint8) * 255)
    cv2.imwrite(str(directory / "occluded.png"), gt.occluded.astype(np.uint8) * 255)
    np.save(directory / "depth.npy", gt.depth)

    camera = first.camera
    full = dict(meta)
    full.update(
        camera=dict(width=camera.width, height=camera.height, fx=camera.fx,
                    fy=camera.fy, cx=camera.cx, cy=camera.cy),
        extrinsic1=first.extrinsic.tolist(),
        extrinsic2=second.extrinsic.tolist(),
        models1={str(k): v.tolist() for k, v in first.models.items()},
        models2={str(k): v.tolist() for k, v in second.models.items()},
        stats=gt.stats(),
    )
    (directory / "meta.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return directory
