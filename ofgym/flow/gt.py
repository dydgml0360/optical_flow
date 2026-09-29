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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from ofgym.flow.renderer import Frame, PairFrame

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
    _stats: Optional[Dict[str, float]] = field(default=None, repr=False, compare=False)

    def stats(self) -> Dict[str, float]:
        """한 번 계산해 기억해 둔다 — 저장과 목록이 같은 값을 쓴다."""
        if self._stats is not None:
            return self._stats
        out = {
            "surface": float(np.count_nonzero(self.surface)) / self.surface.size,
            "valid": float(np.count_nonzero(self.valid)) / self.valid.size,
            "occluded": float(np.count_nonzero(self.occluded)) / self.occluded.size,
        }
        if out["valid"] > 0:
            u = self.flow[..., 0][self.valid]
            v = self.flow[..., 1][self.valid]
            depth = self.depth[self.valid]
            magnitude = np.hypot(u, v)
            out.update(
                flow_min=float(magnitude.min()),
                flow_mean=float(magnitude.mean(dtype=np.float64)),
                flow_max=float(magnitude.max()),
                u_mean=float(u.mean(dtype=np.float64)),
                v_mean=float(v.mean(dtype=np.float64)),
                depth_min=float(depth.min()),
                depth_max=float(depth.max()),
            )
        self._stats = out
        return out


def compute_flow(first: Frame, second: Frame) -> FlowGT:
    """`first` 의 각 픽셀이 `second` 에서 어디로 갔는지."""
    height, width = first.ids.shape
    surface = first.ids > 0

    here = first.camera_points()
    there = first.points_in(second.extrinsic, second.models)
    u2, v2, z2 = second.camera.project(there)

    # 빈 픽셀은 points_in 이 0 으로 두므로 투영이 (cx, cy) 가 된다 — 마스크로 0 을 채운다.
    flow = np.zeros((height, width, 2), np.float32)
    np.subtract(u2, np.arange(width, dtype=np.float64)[None, :], out=u2)
    np.subtract(v2, np.arange(height, dtype=np.float64)[:, None], out=v2)
    np.copyto(flow[..., 0], u2, where=surface, casting="same_kind")
    np.copyto(flow[..., 1], v2, where=surface, casting="same_kind")

    # 두 번째 프레임에서 그 자리에 실제로 찍힌 깊이. 픽셀 하나 안에서도 표면이 기울면
    # 깊이가 달라지므로 3x3 이웃의 가장 먼 값과 비교한다 — 보이는 점은 이웃 범위 안에 든다.
    # 대가로 가려짐 경계에서 1픽셀 폭은 '보임' 으로 남는다.
    depth2 = second.depth()
    finite = np.where(np.isfinite(depth2), depth2, np.float64(1e12))
    farthest = cv2.dilate(finite, np.ones((3, 3), np.uint8))

    # u2, v2 는 이제 flow 다. 도착한 픽셀은 제자리 + flow 를 반올림한 곳.
    ui = np.rint(u2).astype(np.int32) + np.arange(width, dtype=np.int32)[None, :]
    vi = np.rint(v2).astype(np.int32) + np.arange(height, dtype=np.int32)[:, None]
    inside = surface & (z2 > 0) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    np.clip(ui, 0, width - 1, out=ui)
    np.clip(vi, 0, height - 1, out=vi)
    seen = inside & (z2 <= farthest[vi, ui] + OCCLUSION_TOLERANCE_MM)

    occluded = surface & ~seen
    return FlowGT(
        flow=flow,
        surface=surface,
        occluded=occluded,
        valid=surface & seen,
        depth=np.where(surface, here[..., 2], 0.0).astype(np.float32),
    )


def flow_from_pair(pair: PairFrame) -> FlowGT:
    """GPU 가 계산해 온 것을 `compute_flow` 와 같은 모양으로 푼다. 배열은 복사하지 않는다."""
    state = pair.raw[..., 3]
    surface = state >= 0.75  # 번호는 1 부터다
    occluded = surface & ((state - np.floor(state)) > 0.25)
    return FlowGT(
        flow=pair.raw[..., :2],
        surface=surface,
        occluded=occluded,
        valid=surface & ~occluded,
        depth=pair.raw[..., 2],
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


def magnitude_range(flow: np.ndarray, mask: np.ndarray) -> tuple:
    """`mask` 안 flow 크기의 1~99 백분위 (low, high). 비어 있으면 (0, 1)."""
    if not mask.any():
        return 0.0, 1.0
    low, high = np.percentile(np.linalg.norm(flow[mask], axis=1), [1, 99])
    return float(low), float(max(high, low + 1e-3))


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
        np.ascontiguousarray(flow, np.float32).tofile(fh)


def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        magic = np.fromfile(fh, np.float32, 1)
        if magic.size != 1 or magic[0] != np.float32(_FLO_MAGIC):
            raise ValueError(f".flo 매직 넘버가 아닙니다: {path}")
        width, height = np.fromfile(fh, np.int32, 2)
        return np.fromfile(fh, np.float32, 2 * width * height).reshape(height, width, 2)


def save_sample(directory: Path, first: Frame, second: Frame, gt: FlowGT,
                meta: dict, depth: bool = True) -> Path:
    """CPU 경로로 찍은 한 쌍을 쓴다. 파일은 `save_pair` 참고."""
    return save_pair(
        directory, first.color, second.color, gt, first.camera,
        first.extrinsic, second.extrinsic, first.models, second.models, meta, depth,
    )


def save_pair(directory: Path, color1: np.ndarray, color2: np.ndarray, gt: FlowGT,
              camera, extrinsic1, extrinsic2, models1: dict, models2: dict,
              meta: dict, depth: bool = True) -> Path:
    """한 쌍을 `directory` 에 쓴다.

        img1.png img2.png   사진
        flow.flo            img1 → img2, 픽셀
        valid.png           255 = 손실에 쓸 픽셀
        occluded.png        255 = img2 에서 안 보이는 픽셀
        depth.npy           img1 의 카메라 Z (mm), float32  (`depth=False` 면 생략)
        meta.json           카메라·자세·기선
    """
    directory.mkdir(parents=True, exist_ok=True)
    # PNG 압축 단계는 OpenCV 기본값을 둔다 — 재 보니 단계 1 이 오히려 느렸다 (22ms 대 15ms).
    cv2.imwrite(str(directory / "img1.png"), cv2.cvtColor(color1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(directory / "img2.png"), cv2.cvtColor(color2, cv2.COLOR_RGB2BGR))
    write_flo(directory / "flow.flo", gt.flow)
    cv2.imwrite(str(directory / "valid.png"), gt.valid.view(np.uint8) * 255)
    cv2.imwrite(str(directory / "occluded.png"), gt.occluded.view(np.uint8) * 255)
    if depth:
        np.save(directory / "depth.npy", np.ascontiguousarray(gt.depth))

    full = dict(meta)
    full.update(
        camera=dict(width=camera.width, height=camera.height, fx=camera.fx,
                    fy=camera.fy, cx=camera.cx, cy=camera.cy),
        extrinsic1=np.asarray(extrinsic1).tolist(),
        extrinsic2=np.asarray(extrinsic2).tolist(),
        models1={str(k): np.asarray(v).tolist() for k, v in models1.items()},
        models2={str(k): np.asarray(v).tolist() for k, v in models2.items()},
        stats=gt.stats(),
    )
    (directory / "meta.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return directory
