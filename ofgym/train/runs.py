"""파인튜닝의 UI 쪽 절반 — 데이터셋·모델을 찾고, 실행 폴더를 만들고, 결과를 읽는다.

학습 자체는 `worker.py` 가 depth 의 venv 에서 한다. 여기는 torch 를 쓰지 않는다.

실행 하나는 폴더 하나다::

    runs/<모델>_<날짜_시각>/
        config.json            설정 (worker 가 읽는다)
        log.jsonl              worker 가 낸 이벤트 전부
        checkpoint_best.pth    검증 EPE 가 가장 낮았던 가중치
        checkpoint_last.pth    마지막 가중치
        compare/metrics.json   기존 대 튜닝 지표
        compare/<번호>/        검증 쌍별 사진·정답·두 예측
        compare/real_<이름>/   실제 촬영 쌍의 두 예측 (정답 없음)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ofgym.config import PATHS

MODELS = {
    "raft": dict(label="RAFT-Stereo", repo="RAFT-Stereo",
                 checkpoint="RAFT-Stereo/models/raftstereo-eth3d.pth"),
    "crestereo": dict(label="CREStereo", repo="CREStereo-Pytorch",
                      checkpoint="CREStereo-Pytorch/models/crestereo_eth3d.pth"),
}

WORKER = Path(__file__).resolve().parent / "worker.py"


def models_root() -> Path:
    return PATHS.depth / "external" / "stereo_models"


def model_problem(name: str) -> Optional[str]:
    """모델을 쓸 수 없으면 그 까닭과 고치는 법. 쓸 수 있으면 None."""
    spec = MODELS[name]
    root = models_root()
    missing = [p for p in (root / spec["repo"], root / spec["checkpoint"]) if not p.exists()]
    if not missing:
        return None
    return (
        f"{spec['label']} 의 소스나 사전학습 가중치가 없습니다: {missing[0]}\n"
        f"  cd {PATHS.depth} && .venv/bin/python tools/setup_stereo_models.py --models {name}"
    )


# ── 데이터셋 ──────────────────────────────────────────────────────────────
@dataclass
class Dataset:
    path: Path
    pairs: int
    train: int
    val: int
    records: int
    width: int
    height: int
    axis: str
    has_face: bool

    @property
    def label(self) -> str:
        return f"{self.path.name}   {self.pairs:,}쌍 · {self.width}x{self.height}"

    @property
    def summary(self) -> str:
        face = "" if self.has_face else "\n얼굴 마스크 없음 — 얼굴만의 지표는 나오지 않습니다"
        return (f"학습 {self.train:,} / 검증 {self.val:,} 쌍   record {self.records}개\n"
                f"{self.width}x{self.height}   기선 {self.axis}{face}")


def _read_dataset(path: Path) -> Optional[Dataset]:
    index = path / "index.csv"
    if not index.is_file():
        return None
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        split_at, record_at = header.index("split"), header.index("record")
        rows = [line.split(",") for line in lines[1:] if line]
        meta = json.loads((path / "batch.json").read_text(encoding="utf-8"))
        camera = meta["camera"]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    if not rows:
        return None
    train = sum(1 for row in rows if row[split_at] == "train")
    return Dataset(
        path=path, pairs=len(rows), train=train, val=len(rows) - train,
        records=len({row[record_at] for row in rows}),
        width=int(camera["width"]), height=int(camera["height"]),
        axis=str(meta.get("params", {}).get("baseline_axis", "?")),
        has_face="face_mask" in header,
    )


def find_datasets(root: Optional[Path] = None) -> List[Dataset]:
    """일괄 생성 폴더들. 최근 것이 앞."""
    root = root or PATHS.flow
    if not root.is_dir():
        return []
    found = [_read_dataset(p) for p in sorted(root.iterdir(), reverse=True) if p.is_dir()]
    return [d for d in found if d is not None]


# ── 설정 ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """기본값은 메모리 16GB 인 맥(MPS)에서 돌아가는 크기다.

    거기서 잰 값 (RAFT-Stereo, 320x448, 반복 12): 배치 1 에 스텝당 1.5초 · 메모리 5GB,
    배치 2 에 2.8초 · 10GB. 384x512 · 배치 2 는 13.5GB 로 메모리를 넘겨 스텝당 13초가 된다.
    검증 사진 한 장(760x1008)은 약 4초. CREStereo 는 이보다 1.5~2배 느리다.
    기본 설정 전체(학습 1000 스텝 + 검증 5번 + 비교)는 RAFT-Stereo 로 40분쯤 걸린다.
    """

    model: str = "raft"
    dataset: str = ""
    models_root: str = ""
    checkpoint: Optional[str] = None  # None 이면 모델의 사전학습 가중치
    steps: int = 1000
    batch_size: int = 1
    lr: float = 2e-5
    weight_decay: float = 1e-5
    crop_height: int = 320
    crop_width: int = 448
    train_iters: int = 12
    valid_iters: int = 12
    val_every: int = 250
    val_samples: int = 16
    log_every: int = 10
    loader_workers: int = 2
    seed: int = 0
    device: str = "auto"
    compare_samples: int = 60
    compare_shown: int = 16
    compare_real: int = 4


def create_run(config: TrainConfig) -> Path:
    run = PATHS.runs / f"{config.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return run


# ── 결과 ──────────────────────────────────────────────────────────────────
@dataclass
class Run:
    path: Path
    config: dict
    metrics: Optional[dict]  # compare/metrics.json

    @property
    def label(self) -> str:
        name = MODELS.get(self.config.get("model"), {}).get("label", "?")
        if self.metrics is None:
            return f"{self.path.name}   (비교 없음)"
        try:
            base = self.metrics["metrics"]["base"]["all"]["epe"]
            tuned = self.metrics["metrics"]["tuned"]["all"]["epe"]
            return f"{self.path.name}   {name}  EPE {base:.3f} → {tuned:.3f}"
        except KeyError:
            return f"{self.path.name}   {name}"

    def events(self) -> List[dict]:
        out = []
        try:
            for line in (self.path / "log.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        return out

    def samples(self) -> List["Sample"]:
        folder = self.path / "compare"
        if not folder.is_dir():
            return []
        turned = bool((self.metrics or {}).get("vertical", False))
        found = [Sample(p, turned) for p in sorted(folder.iterdir()) if p.is_dir()]
        found = [s for s in found if s.usable]
        # 정답이 있는 합성 쌍이 앞, 실제 촬영이 뒤.
        return sorted(found, key=lambda s: (s.real, s.path.name))


def load_run(path: Path) -> Optional[Run]:
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        metrics = json.loads((path / "compare" / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metrics = None
    return Run(path=path, config=config, metrics=metrics)


def find_runs(root: Optional[Path] = None) -> List[Run]:
    root = root or PATHS.runs
    if not root.is_dir():
        return []
    runs = [load_run(p) for p in sorted(root.iterdir(), reverse=True) if p.is_dir()]
    return [r for r in runs if r is not None]


@dataclass
class Sample:
    """비교용 쌍 하나 — 왼쪽 사진, (정답), 기존 예측, 튜닝 예측.

    세로 기선 데이터는 모델에 넣느라 시계 방향으로 돌려 저장돼 있다. `turned` 면 읽을 때
    되돌려, 얼굴이 바로 선 모습으로 보여 준다.
    """

    path: Path
    turned: bool = False

    @property
    def real(self) -> bool:
        return self.path.name.startswith("real_")

    @property
    def usable(self) -> bool:
        return all((self.path / n).is_file() for n in ("left.png", "base.npy", "tuned.npy"))

    @property
    def label(self) -> str:
        if self.real:
            return f"실제 촬영  {self.path.name[5:]}"
        try:
            source = json.loads((self.path / "source.json").read_text(encoding="utf-8"))
            return f"{self.path.name}  {source['record'][:8]}  {source.get('view_angle')}°"
        except (OSError, ValueError, KeyError):
            return self.path.name

    def load(self) -> Dict[str, np.ndarray]:
        out = dict(
            left=cv2.cvtColor(cv2.imread(str(self.path / "left.png")), cv2.COLOR_BGR2RGB),
            base=np.load(self.path / "base.npy"),
            tuned=np.load(self.path / "tuned.npy"),
        )
        if (self.path / "gt.npy").is_file():
            out["gt"] = np.load(self.path / "gt.npy")
            out["valid"] = cv2.imread(str(self.path / "valid.png"), 0) > 127
            out["face"] = cv2.imread(str(self.path / "face.png"), 0) > 127
        if self.turned:
            out = {key: np.ascontiguousarray(np.rot90(value)) for key, value in out.items()}
        return out
