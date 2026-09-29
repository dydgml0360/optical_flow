"""`../dataset/raw` 를 훑어 세션 / 샘플 / 뷰 구조로 만든다.

캡처 세션 한 벌은 다음 모양이다::

    raw/<세션>/<샘플UUID>/
        left.jpg              전체 해상도 원본 (3040x4032)
        left_pair.jpg         같은 순간 짝 카메라 원본
        left_color.jpg        ROI 크롭 컬러
        left.npy              3D 정점맵 (H,W,3) float64
        left_mask.png  left_face_mask.png  left_parts.png ...
        (center_*, right_* 동일)

파일이 빠진 뷰가 있어도 죽지 않고, 있는 것만 담는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ofgym.config import PATHS, VIEWS

# 접미사 -> 사람이 읽는 레이어 이름. 순서가 UI 콤보 순서가 된다.
LAYER_SUFFIXES = (
    ("", ".jpg", "원본 (full)"),
    ("_pair", ".jpg", "짝 카메라 (pair)"),
    ("_color", ".jpg", "ROI 컬러"),
    ("", ".npy", "정점맵 Z (depth)"),
    ("_mask", ".png", "마스크"),
    ("_mask_pair", ".png", "마스크 (pair)"),
    ("_face_mask", ".png", "얼굴 마스크"),
    ("_face_mask_pair", ".png", "얼굴 마스크 (pair)"),
    ("_parts", ".png", "파츠"),
    ("_parts_pair", ".png", "파츠 (pair)"),
    ("_parts_extend", ".png", "파츠 확장"),
    ("_plan_map", ".png", "플랜 맵"),
)


@dataclass
class View:
    """샘플 하나의 카메라 뷰 하나."""

    name: str
    layers: Dict[str, Path] = field(default_factory=dict)

    @property
    def vertex(self) -> Path | None:
        return self.layers.get("정점맵 Z (depth)")

    def __bool__(self) -> bool:
        return bool(self.layers)


@dataclass
class Sample:
    """UUID 디렉터리 하나 = 촬영 한 순간."""

    sample_id: str
    path: Path
    views: Dict[str, View]

    @property
    def label(self) -> str:
        present = [v for v in VIEWS if self.views.get(v)]
        return f"{self.sample_id[:8]}  ({'/'.join(present) or '비어 있음'})"


@dataclass
class Session:
    """날짜_시각 디렉터리 하나."""

    session_id: str
    path: Path
    samples: List[Sample]


def _scan_view(sample_dir: Path, view: str) -> View:
    layers: Dict[str, Path] = {}
    for suffix, ext, label in LAYER_SUFFIXES:
        candidate = sample_dir / f"{view}{suffix}{ext}"
        if candidate.is_file():
            layers[label] = candidate
    return View(name=view, layers=layers)


def _scan_sample(sample_dir: Path) -> Sample | None:
    views = {v: _scan_view(sample_dir, v) for v in VIEWS}
    views = {k: v for k, v in views.items() if v}
    if not views:
        return None
    return Sample(sample_id=sample_dir.name, path=sample_dir, views=views)


def scan_sessions(raw_root: Path | None = None) -> List[Session]:
    """세션 목록을 이름 오름차순으로 반환한다. 없으면 빈 리스트."""

    root = raw_root or PATHS.raw
    if not root.is_dir():
        return []

    sessions: List[Session] = []
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        samples = []
        for sample_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            sample = _scan_sample(sample_dir)
            if sample is not None:
                samples.append(sample)
        if samples:
            sessions.append(
                Session(session_id=session_dir.name, path=session_dir, samples=samples)
            )
    return sessions
