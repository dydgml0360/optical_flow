"""씬 — 얼굴 메쉬 하나 + 배경판 + 카메라 한 쌍.

월드 좌표계는 캡처 좌표계 그대로다 (center 카메라가 원점, +Z 앞, +Y 아래, mm).
모든 값이 0 인 기본 씬은 디바이스가 실제로 찍던 배치를 재현한다.

카메라는 디바이스처럼 회전축 둘레의 세 자리(left +50° / center 0° / right -50°) 중
하나에 서고, 거기서 `tvec`/`rvec` 만큼 흔들린다. 얼굴도 제자리에서 `tvec`/`rvec` 만큼
움직인다. 둘 다 OpenCV 와 같은 뜻이다 (`X' = R(rvec) X + tvec`, rvec 은 도 단위).

    카메라   world→camera = [rvec|tvec] · orbit(촬영 위치)     ← 카메라 좌표계에서 흔든다
    얼굴     object→world = 메쉬 중심을 축으로 rvec, 그 뒤 tvec  ← 월드 좌표계에서 움직인다

촬영은 스테레오처럼 한다: 첫 카메라를 놓고, **기선만큼 평행 이동**한 자리에서 한 장 더.
씬은 그사이 움직이지 않으므로 flow 는 시차(disparity)와 같고 기선 방향으로만 생긴다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ofgym.flow import camera as cam
from ofgym.flow.camera import Camera
from ofgym.flow.gt import FlowGT, compute_flow
from ofgym.flow.mesh import Mesh
from ofgym.flow.renderer import BACKGROUND_COLOR, Frame, Renderer

FACE_ID = 1
BACKGROUND_ID = 2

BACKGROUND_NONE = "없음"
BACKGROUND_NOISE = "절차적 무늬"
BACKGROUND_IMAGE = "이미지 파일"

# 기선 방향. 디바이스는 센서를 90° 돌려 달아 짝 카메라가 **세로로** 떨어져 있다 —
# center.jpg → center_pair.jpg 가 +y 로 ~239px(전체 해상도) 움직인다.
AXIS_VERTICAL = "세로 (디바이스)"
AXIS_HORIZONTAL = "가로"
_AXES = {AXIS_VERTICAL: (0.0, -1.0, 0.0), AXIS_HORIZONTAL: (1.0, 0.0, 0.0)}


# 촬영 위치. 디바이스 펌웨어의 angles=[50, 0, -50] 과 같다.
VIEW_ANGLES = {"left (+50°)": 50.0, "center (0°)": 0.0, "right (-50°)": -50.0}


@dataclass
class SceneParams:
    # 카메라
    scale: float = 0.25  # 디바이스 해상도 대비
    baseline_mm: float = 15.0
    baseline_axis: str = AXIS_VERTICAL
    view_angle: float = 0.0  # 촬영 위치, 도
    # 카메라 흔들림 — 촬영 위치의 카메라 좌표계 기준
    camera_tx: float = 0.0
    camera_ty: float = 0.0
    camera_tz: float = 0.0
    camera_rx: float = 0.0
    camera_ry: float = 0.0
    camera_rz: float = 0.0
    # 얼굴 움직임 — 캡처된 자리 기준, 월드 좌표계
    face_tx: float = 0.0
    face_ty: float = 0.0
    face_tz: float = 0.0
    face_rx: float = 0.0
    face_ry: float = 0.0
    face_rz: float = 0.0
    # 배경
    background: str = BACKGROUND_NOISE
    background_z: float = 600.0
    background_seed: int = 0
    background_image: Optional[str] = None

    def baseline_vector(self) -> np.ndarray:
        axis = np.array(_AXES.get(self.baseline_axis, _AXES[AXIS_VERTICAL]))
        return axis * self.baseline_mm

    @property
    def camera_tvec(self) -> Tuple[float, float, float]:
        return (self.camera_tx, self.camera_ty, self.camera_tz)

    @property
    def camera_rvec(self) -> Tuple[float, float, float]:
        return (self.camera_rx, self.camera_ry, self.camera_rz)

    @property
    def face_tvec(self) -> Tuple[float, float, float]:
        return (self.face_tx, self.face_ty, self.face_tz)

    @property
    def face_rvec(self) -> Tuple[float, float, float]:
        return (self.face_rx, self.face_ry, self.face_rz)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RandomRanges:
    """무작위 배치의 범위. 성분마다 ±값 안에서 고르게 뽑는다."""

    camera_t: float = 5.0  # mm
    camera_r: float = 2.0  # 도
    face_t: float = 10.0  # mm
    face_r: float = 5.0  # 도
    random_view: bool = True  # 촬영 위치도 세 자리 중에서 뽑는다


def randomized(params: SceneParams, ranges: RandomRanges,
               rng: np.random.Generator) -> SceneParams:
    """카메라 흔들림·얼굴 움직임·배경 무늬를 새로 뽑는다. 해상도·기선·배경 종류는 둔다."""
    values = params.as_dict()
    for prefix, limit_t, limit_r in (("camera", ranges.camera_t, ranges.camera_r),
                                     ("face", ranges.face_t, ranges.face_r)):
        for axis in "xyz":
            values[f"{prefix}_t{axis}"] = float(rng.uniform(-limit_t, limit_t))
            values[f"{prefix}_r{axis}"] = float(rng.uniform(-limit_r, limit_r))
    if ranges.random_view:
        values["view_angle"] = float(rng.choice(list(VIEW_ANGLES.values())))
    values["background_seed"] = int(rng.integers(0, 2**31 - 1))
    return SceneParams(**values)


def noise_texture(seed: int, size: int = 1024) -> np.ndarray:
    """여러 크기의 얼룩을 겹친 무늬. 어느 배율로 찍혀도 대응점을 잡을 질감이 남는다."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), np.float32)
    total = 0.0
    for cells in (4, 8, 16, 32, 64, 128, 256, 512):
        layer = rng.random((cells, cells, 3), dtype=np.float32)
        weight = (4.0 / cells) ** 0.3
        image += weight * cv2.resize(layer, (size, size), interpolation=cv2.INTER_CUBIC)
        total += weight
    image /= total
    lo, hi = np.percentile(image, [1, 99])
    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (image * 255).astype(np.uint8)


def _load_image(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    longest = max(bgr.shape[:2])
    if longest > 4096:
        scale = 4096 / longest
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class Shot:
    """촬영 한 쌍과 그 GT."""

    first: Frame
    second: Frame
    gt: FlowGT
    params: SceneParams


class Scene:
    def __init__(self, renderer: Renderer, base_camera: Optional[Camera] = None) -> None:
        self._renderer = renderer
        self._base_camera = base_camera or cam.device_camera()
        self._mesh: Optional[Mesh] = None
        self._background_key: Optional[tuple] = None
        self.background_error: Optional[str] = None

    @property
    def mesh(self) -> Optional[Mesh]:
        return self._mesh

    def set_mesh(self, mesh: Optional[Mesh]) -> None:
        self._mesh = mesh
        if mesh is None:
            self._renderer.remove_object(FACE_ID)
        else:
            self._renderer.set_mesh(FACE_ID, mesh)

    # ── 구성 요소 ─────────────────────────────────────────────────────────
    def camera(self, params: SceneParams) -> Camera:
        return self._base_camera.scaled(params.scale)

    @property
    def pivot_radius(self) -> float:
        return self._mesh.pivot_radius if self._mesh is not None else cam.device_radius()

    def station(self, angle: float) -> np.ndarray:
        """촬영 위치에 흔들림 없이 선 카메라의 world→camera."""
        return cam.orbit(angle, self.pivot_radius)

    def extrinsics(self, params: SceneParams) -> Tuple[np.ndarray, np.ndarray]:
        first = cam.rigid(params.camera_rvec, params.camera_tvec) @ self.station(
            params.view_angle)
        return first, cam.shifted(first, params.baseline_vector())

    def face_model(self, params: SceneParams) -> np.ndarray:
        if self._mesh is None:
            return np.eye(4)
        return cam.place(self._mesh.center, params.face_rvec, params.face_tvec)

    def models(self, params: SceneParams) -> dict:
        self._sync_background(params)
        models = {}
        if self._mesh is not None:
            models[FACE_ID] = self.face_model(params)
        if self._renderer.has_object(BACKGROUND_ID):
            # 배경판은 촬영 위치의 카메라를 마주 본다 (카메라가 흔들려도 판은 그대로다).
            models[BACKGROUND_ID] = np.linalg.inv(self.station(params.view_angle))
        return models

    def _sync_background(self, params: SceneParams) -> None:
        key = (params.background, round(params.background_z, 3), params.background_seed,
               params.background_image, params.scale, params.baseline_mm,
               params.baseline_axis, params.camera_tvec, params.camera_rvec)
        if key == self._background_key:
            return
        self._background_key = key
        self.background_error = None
        self._renderer.remove_object(BACKGROUND_ID)
        if params.background == BACKGROUND_NONE:
            return

        if params.background == BACKGROUND_IMAGE:
            texture = _load_image(params.background_image) if params.background_image else None
            if texture is None:
                self.background_error = (
                    f"배경 이미지를 읽지 못했습니다: {params.background_image}"
                    if params.background_image else "배경 이미지가 선택되지 않았습니다"
                )
                return
        else:
            texture = noise_texture(params.background_seed)

        # 두 카메라의 시야를 다 덮는 판. 주점이 중앙이 아니어도 모자라지 않게 넉넉히 잡는다.
        camera = self.camera(params)
        z = params.background_z  # 판 좌표계(= 촬영 위치의 카메라 좌표계)에서의 거리
        # 카메라가 옮겨지고 돌아간 만큼 시야가 판 위에서 밀린다.
        turn = np.radians(min(float(np.linalg.norm(params.camera_rvec)), 60.0))
        reach = (abs(params.baseline_mm) + float(np.linalg.norm(params.camera_tvec))
                 + z * np.tan(turn) + 1.0)
        half_w = z * camera.width / camera.fx * 0.75 + reach
        half_h = z * camera.height / camera.fy * 0.75 + reach
        # 사진이 찌그러지지 않게 판의 비율을 텍스처에 맞춘다 (모자란 쪽을 키운다).
        aspect = texture.shape[1] / texture.shape[0]
        if half_w / half_h < aspect:
            half_w = half_h * aspect
        else:
            half_h = half_w / aspect

        vertices = np.array(
            [[-half_w, -half_h, z], [half_w, -half_h, z],
             [-half_w, half_h, z], [half_w, half_h, z]], np.float32)
        uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
        faces = np.array([[0, 2, 1], [1, 2, 3]], np.int32)
        self._renderer.set_object(BACKGROUND_ID, vertices, uvs, faces, texture)

    # ── 촬영 ──────────────────────────────────────────────────────────────
    def shoot(self, params: SceneParams) -> Shot:
        camera = self.camera(params)
        models = self.models(params)
        extrinsic1, extrinsic2 = self.extrinsics(params)
        first = self._renderer.render(camera, extrinsic1, models, background=(0, 0, 0))
        second = self._renderer.render(camera, extrinsic2, models, background=(0, 0, 0))
        return Shot(first=first, second=second, gt=compute_flow(first, second), params=params)

    def overview(self, params: SceneParams, viewer: Camera, extrinsic: np.ndarray) -> np.ndarray:
        """씬을 바깥에서 본 그림. 카메라 두 대의 시야를 선으로 같이 그린다.

        회색은 세 촬영 위치(흔들림 없는 자리)다.
        """
        camera = self.camera(params)
        extrinsic1, extrinsic2 = self.extrinsics(params)
        depth = 60.0
        lines = [
            (cam.frustum_lines(camera, self.station(angle), depth * 0.6), (0.42, 0.42, 0.45))
            for angle in VIEW_ANGLES.values()
        ]
        lines += [
            (cam.frustum_lines(camera, extrinsic1, depth), (1.0, 0.82, 0.25)),
            (cam.frustum_lines(camera, extrinsic2, depth), (0.35, 0.8, 1.0)),
        ]
        frame = self._renderer.render(
            viewer, extrinsic, self.models(params),
            background=BACKGROUND_COLOR, lines=lines, geometry=False,
        )
        return frame.color
