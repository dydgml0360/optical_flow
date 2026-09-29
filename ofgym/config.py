"""경로와 전역 설정.

데이터셋은 저장소 밖(`../dataset`)에서 관리한다. 저장소는 코드만 담는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 뷰 이름. 캡처 세션이 카메라 3대로 찍으므로 항상 이 셋이다.
VIEWS = ("left", "center", "right")

# 스테레오 페어 후보. (기준뷰, 대상뷰)
PAIRS = (("left", "center"), ("center", "right"), ("left", "right"))


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    dataset: Path
    raw: Path      # 캡처 원본
    recon: Path    # make3D 재구성 결과 (스테이징 + model/)
    gt: Path       # 생성된 disparity ground truth
    runs: Path
    depth: Path

    @classmethod
    def resolve(cls) -> "Paths":
        dataset = _env_path("OFGYM_DATASET", (REPO_ROOT / ".." / "dataset").resolve())
        return cls(
            repo_root=REPO_ROOT,
            dataset=dataset,
            raw=dataset / "raw",
            recon=dataset / "recon",
            gt=dataset / "gt",
            runs=REPO_ROOT / "runs",
            depth=REPO_ROOT / "thirdparty" / "depth",
        )


PATHS = Paths.resolve()


def depth_available() -> bool:
    """서브모듈이 체크아웃되어 있는지."""
    return (PATHS.depth / "stereo").is_dir()
