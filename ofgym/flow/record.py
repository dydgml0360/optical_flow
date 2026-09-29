"""디바이스 UI 가 남긴 record 디렉터리를 찾는다.

record 하나는 촬영 한 번이다 (`ToningReacher_QT/data/<process_id>/`)::

    <record>/
        atlas.png             원통 (θ, l) UV 아틀라스, BGR
        atlas_covered.png     아틀라스에서 실제로 채워진 텍셀
        atlas_meta.json       rr / theta0 / l0 / theta_rate / z_rate / width / height
        {left,center,right}.npy   병합된 정점맵 (H,W,3) float64, mm

shared 에는 같은 record 가 여러 공유본에 하드링크로 겹쳐 들어 있다. process_id 가
같으면 같은 촬영이므로 가장 최근 공유본 하나만 남긴다.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from ofgym.config import VIEWS

ATLAS_META = "atlas_meta.json"
ATLAS_IMAGE = "atlas.png"
ATLAS_COVERED = "atlas_covered.png"

# 훑을 때 내려가지 않는 디렉터리. `.stversions` 는 Syncthing 이 남긴 옛 판본이다.
_SKIP_DIRS = {".stversions", ".stfolder", ".git", ".venv", "__pycache__"}


@dataclass
class Record:
    record_id: str
    path: Path
    group: str  # 공유본 이름 — record 가 들어 있던 최상위 폴더
    meta: dict

    @property
    def atlas_path(self) -> Path:
        return self.path / ATLAS_IMAGE

    @property
    def covered_path(self) -> Path:
        return self.path / ATLAS_COVERED

    def vertex_paths(self) -> Dict[str, Path]:
        found = {}
        for view in VIEWS:
            p = self.path / f"{view}.npy"
            if p.is_file():
                found[view] = p
        return found

    @property
    def taken_at(self) -> float:
        """촬영 시각 — 아틀라스 meta 가 쓰인 때. 하드링크라 공유본을 옮겨도 그대로다."""
        try:
            return (self.path / ATLAS_META).stat().st_mtime
        except OSError:
            return 0.0

    @property
    def label(self) -> str:
        views = "/".join(v[0].upper() for v in self.vertex_paths())
        coverage = self.meta.get("coverage")
        tail = f"  {coverage * 100:.0f}%" if isinstance(coverage, float) else ""
        when = time.strftime("%m-%d %H:%M", time.localtime(self.taken_at))
        return f"{when}  {self.record_id[:8]}  ({views}){tail}"


def _load(path: Path, root: Path) -> Record | None:
    try:
        meta = json.loads((path / ATLAS_META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    needed = ("rr", "theta0", "l0", "theta_rate", "z_rate", "width", "height")
    if any(key not in meta for key in needed):
        return None
    if not (path / ATLAS_IMAGE).is_file():
        return None

    try:
        group = path.relative_to(root).parts
    except ValueError:
        group = ()
    # <공유본>/record/<id> 꼴이면 공유본 이름을, 아니면 부모 폴더 이름을 쓴다.
    name = next((p for p in group[:-1] if p != "record"), path.parent.name)
    record = Record(record_id=path.name, path=path, group=name, meta=meta)
    return record if record.vertex_paths() else None


def find_records(root: Path) -> List[Record]:
    """`root` 자신이 record 면 그것 하나, 아니면 아래를 훑어 전부. 최근 것이 앞."""
    root = Path(root)
    if not root.is_dir():
        return []
    if (root / ATLAS_META).is_file():
        record = _load(root, root.parent)
        return [record] if record else []

    by_id: Dict[str, Record] = {}
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        if ATLAS_META not in files:
            continue
        dirs[:] = []  # record 안쪽은 더 볼 것이 없다
        record = _load(Path(current), root)
        if record is None:
            continue
        previous = by_id.get(record.record_id)
        # 공유본 폴더 이름이 날짜_시각이라 경로가 큰 쪽이 나중 것이다.
        if previous is None or str(record.path) > str(previous.path):
            by_id[record.record_id] = record

    return sorted(by_id.values(), key=lambda record: record.taken_at, reverse=True)
