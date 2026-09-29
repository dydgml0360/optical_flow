"""make3D 산출물을 읽는 쪽.

`make3D.py` 는 `<stage>/model/` 아래에 이렇게 남긴다 (suffix 는 `--upsample N` 일 때 `_upNx`)::

    {left,center,right}_vertex{suffix}.npy    병합까지 끝난 3D 격자 (H,W,3 float64)
    {left,center,right}_blend{suffix}.npy     블렌딩된 색 (H,W,3 uint8 BGR)
    {left,center,right}_blend{suffix}.png     위와 같은 것, 눈으로 볼 용도
    {left,center,right}_facemask{suffix}.png
    {left,center,right}_mask{suffix}.png
    render{suffix}.png                        Open3D 오프스크린 정면 렌더
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ofgym.config import PATHS, VIEWS
from ofgym.data import Sample

_VERTEX_RE = re.compile(r"^(left|center|right)_vertex(?P<suffix>.*)\.npy$")


@dataclass
class ReconOutput:
    """한 샘플의 재구성 결과 한 벌."""

    stage_dir: Path
    model_dir: Path
    suffix: str  # "" 또는 "_up2x" 같은 것

    @property
    def render_png(self) -> Optional[Path]:
        p = self.model_dir / f"render{self.suffix}.png"
        return p if p.is_file() else None

    def vertex_path(self, view: str) -> Optional[Path]:
        p = self.model_dir / f"{view}_vertex{self.suffix}.npy"
        return p if p.is_file() else None

    def blend_path(self, view: str) -> Optional[Path]:
        p = self.model_dir / f"{view}_blend{self.suffix}.npy"
        return p if p.is_file() else None

    def layers(self, view: str) -> Dict[str, Path]:
        """프리뷰 콤보에 넣을 레이어들."""
        found: Dict[str, Path] = {}
        for label, name in (
            ("병합 3D Z (vertex)", f"{view}_vertex{self.suffix}.npy"),
            ("블렌딩 색", f"{view}_blend{self.suffix}.png"),
            ("얼굴 마스크", f"{view}_facemask{self.suffix}.png"),
            ("마스크", f"{view}_mask{self.suffix}.png"),
        ):
            p = self.model_dir / name
            if p.is_file():
                found[label] = p
        return found

    @property
    def views(self) -> List[str]:
        return [v for v in VIEWS if self.vertex_path(v) is not None]

    def load_clouds(
        self, max_points_per_view: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """(points Nx3 float32, colors Nx3 uint8 RGB) 를 뷰마다 하나씩.

        vertex 가 (0,0,0) 인 셀은 무효라 버린다 — make3D 의 `getPointCloud` 와 같다.

        `max_points_per_view` 를 주면 균등 간격으로 솎는다. 6x 업샘플이면 뷰당
        300만 점이 넘어(vertex npy 하나가 294MB) 화면에 다 올릴 이유가 없다.
        솎기는 **표시용**이다 — 저장된 격자는 건드리지 않는다.
        """
        clouds: List[Tuple[np.ndarray, np.ndarray]] = []
        for view in self.views:
            vertex = np.load(self.vertex_path(view), mmap_mode="r")
            points = np.asarray(vertex).reshape(-1, 3)
            valid = np.any(points != 0, axis=1)
            index = np.flatnonzero(valid)
            if max_points_per_view and len(index) > max_points_per_view:
                step = len(index) // max_points_per_view + 1
                index = index[::step]

            out_points = points[index].astype(np.float32)
            blend = self.blend_path(view)
            if blend is not None:
                bgr = np.load(blend, mmap_mode="r").reshape(-1, 3)[index]
                colors = np.asarray(bgr)[:, ::-1].astype(np.uint8)  # BGR → RGB
            else:
                colors = np.full((out_points.shape[0], 3), 200, np.uint8)
            clouds.append((out_points, colors))
        return clouds

    def valid_counts(self) -> Dict[str, int]:
        """뷰별 유효 vertex 수 — 솎기 전 실제 격자 기준."""
        counts: Dict[str, int] = {}
        for view in self.views:
            vertex = np.load(self.vertex_path(view), mmap_mode="r")
            counts[view] = int(np.count_nonzero(np.any(np.asarray(vertex) != 0, axis=-1)))
        return counts

    @property
    def grid_shape(self) -> Optional[Tuple[int, int]]:
        for view in self.views:
            return tuple(np.load(self.vertex_path(view), mmap_mode="r").shape[:2])
        return None


def stage_dir_for(sample: Sample) -> Path:
    """샘플 하나가 재구성될 자리. `raw/<세션>/<샘플>` → `recon/<세션>/<샘플>`."""
    return PATHS.recon / sample.path.parent.name / sample.sample_id


def find_output(sample: Sample) -> Optional[ReconOutput]:
    """이미 재구성된 결과가 있으면 돌려준다. suffix 는 붙은 것 중 가장 큰 배율."""
    stage = stage_dir_for(sample)
    model_dir = stage / "model"
    if not model_dir.is_dir():
        return None

    suffixes = set()
    for p in model_dir.glob("*_vertex*.npy"):
        m = _VERTEX_RE.match(p.name)
        if m:
            suffixes.add(m.group("suffix"))
    if not suffixes:
        return None

    # "_up6x" > "_up2x" > "" — 업샘플이 있으면 그쪽을 기본으로 보여 준다.
    def rank(s: str) -> int:
        m = re.match(r"^_up(\d+)x$", s)
        return int(m.group(1)) if m else 1

    suffix = max(suffixes, key=rank)
    return ReconOutput(stage_dir=stage, model_dir=model_dir, suffix=suffix)
