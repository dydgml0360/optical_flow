"""캡처 한 벌 → 3D 재구성(병합·후처리 포함)을 depth 서브모듈에 위임해서 돌린다.

`thirdparty/depth/example/make3D_ui_log.py` 가 하는 일을 그대로 쓴다:
캡처 디렉터리를 make3D 파일명 규약으로 심링크 스테이징한 뒤 `make3D.py` 를 돌린다.

`Reconstruction.run()` 의 `merge.method != 'roi'` 경로(기본 `incidence`)가 도는데,
그 안에 `stereo.pointmerge.mergeViews` 의 후처리가 전부 들어 있다 — 스파이크 제거 →
정중선 컷 → centerAlign → 합의 필터 → 색 합의 필터 → 입사각 컬링 → center 방위각
게이트 → 색 이득 보정 → 입사각 가중 투영 → 아웃라이어 제거. ROI 하드 컷과 시임
블렌딩은 이 경로에서 **의도적으로** 건너뛴다 (경계맵을 쓰지 않는 병합이라서).

⚠️ 두 가지는 우리가 직접 챙겨야 한다.

1. **업샘플.** `config/stereo.yaml` 의 `upsample`(enabled/method/scale 6)은
   `UpsampleWorker` 만 읽는데, 그 워커는 `setUpsampleCallback()` 으로 콜백이 등록돼야
   돈다. `make3D.py` 는 등록하지 않으므로 그 경로는 항상 `건너뜀 — 콜백 미등록` 이다.
   같은 일을 하는 인라인 경로가 `make3D.py --upsample N` 이고
   (`UpsampleWorker._work` 도 결국 `rec.run(upsample=scale)` 를 부른다),
   기본값이 1 이라 **명시하지 않으면 1/decimation(=1/6) 격자로 끝난다.**
   그래서 여기서는 stereo.yaml 의 `upsample.scale` 을 읽어 기본값으로 쓴다.
   ⚠️ 인라인 경로는 `interpolate`(보간)만 된다. `rematch`(풀해상도 재매칭)는
   `UpsampleWorker` 가 `_upsamplerFactory` 를 주입해야 하므로 make3D 로는 못 쓴다.

2. **로그.** `pytoningreacher` 전체가 `utils.debug_mode.dprint` 로 print 를 가려서
   `PYTR_DEBUG` 가 꺼져 있으면 병합·후처리 로그가 통째로 안 보인다 (동작은 한다).
   무엇이 돌았는지 확인할 수 있도록 기본으로 켠다.

depth 는 python 3.10 에 open3d/torch/onnxruntime 을 쓰고 우리 UI 는 PySide6 만 쓴다.
같은 인터프리터에 밀어 넣지 않고 **depth 의 venv 를 서브프로세스로** 부른다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from PySide6.QtCore import QThread, Signal

from ofgym.config import PATHS
from ofgym.data import Sample
from ofgym.recon.outputs import stage_dir_for

PATCH_ROOT = PATHS.repo_root / "thirdparty" / "patches"

# (패치 파일, 서브모듈 안 목적지) — 자세한 사정은 thirdparty/patches/README.md
PATCHES = (("ml/model/onnx_model.py", "external/ml/model/onnx_model.py"),)


class ReconError(RuntimeError):
    pass


def depth_python() -> Path:
    """depth 서브모듈 venv 의 인터프리터. 없으면 무엇을 하라고 알려 준다."""
    exe = PATHS.depth / ".venv" / "bin" / "python"
    if not exe.is_file():
        raise ReconError(
            f"depth venv 가 없습니다: {exe}\n"
            f"  uv sync --project {PATHS.depth}"
        )
    return exe


def stereo_config() -> dict:
    """depth 의 `config/stereo.yaml`. 못 읽으면 빈 dict."""
    path = PATHS.depth / "config" / "stereo.yaml"
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}


def default_upsample() -> int:
    """`upsample.scale` — 디바이스가 쓰기로 선언한 배율. 읽기 실패하면 1(=업샘플 없음).

    make3D 의 CLI 기본값(1) 이 아니라 이쪽을 기본으로 삼는다. 모듈 docstring 참고.
    """
    cfg = stereo_config().get("upsample") or {}
    if not cfg.get("enabled", False) or cfg.get("method") == "none":
        return 1
    scale = cfg.get("scale", 1)
    return scale if scale in (1, 2, 3, 6) else 1


def default_upsample_method() -> str:
    """`upsample.interpolate.mode`. make3D 인라인 경로는 이것만 지원한다."""
    cfg = stereo_config().get("upsample") or {}
    mode = (cfg.get("interpolate") or {}).get("mode", "bilinear")
    return mode if mode in ("bilinear", "edge_aware") else "bilinear"


def ensure_patches() -> List[str]:
    """upstream 에 커밋 안 된 파일을 서브모듈에 채워 넣는다. 채운 것들을 반환."""
    applied = []
    for rel_src, rel_dst in PATCHES:
        dst = PATHS.depth / rel_dst
        if dst.exists():
            continue
        src = PATCH_ROOT / rel_src
        if not src.is_file():
            raise ReconError(f"패치 원본이 없습니다: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        applied.append(rel_dst)
    return applied


@dataclass
class ReconOptions:
    """model/render 는 make3D_ui_log.py 기본값, upsample 은 stereo.yaml 기본값.

    model="u2net" 인 이유: UI 캡처에는 eye/eyecover 개별 마스크가 없어 (parts 합집합만
    저장) `disk` 로 돌리면 parts ROI 키포인트가 정적 폴백으로 떨어지고 **측면 뷰 vertex 가
    전멸한다**. u2net 은 external/ml 의 onnx 로 개별 마스크를 다시 만든다.

    upsample 이 make3D CLI 기본값(1)이 아닌 이유는 모듈 docstring 참고.
    """

    model: str = "u2net"
    upsample: int = field(default_factory=default_upsample)
    upsample_method: str = field(default_factory=default_upsample_method)
    render: bool = True  # False 면 --no-vis (render.png 안 만듦)
    debug: bool = True  # PYTR_DEBUG — 병합/후처리 로그를 보이게 한다

    def args(self) -> List[str]:
        extra = ["--model", self.model]
        if not self.render:
            extra.append("--no-vis")
        if self.upsample > 1:
            # `--` 뒤는 make3D.py 로 그대로 전달된다.
            extra += ["--", "--upsample", str(self.upsample),
                      "--upsample-method", self.upsample_method]
        return extra


def build_command(sample: Sample, options: ReconOptions) -> tuple[List[str], Path]:
    stage = stage_dir_for(sample)
    script = PATHS.depth / "example" / "make3D_ui_log.py"
    if not script.is_file():
        raise ReconError(f"make3D_ui_log.py 가 없습니다: {script}")
    cmd = [
        str(depth_python()),
        str(script),
        "--capture-dir",
        str(sample.path),
        "--stage-dir",
        str(stage),
        *options.args(),
    ]
    return cmd, stage


def _env(options: "ReconOptions") -> dict:
    env = dict(os.environ)
    # mac: torch libomp ↔ OpenBLAS 충돌로 스테레오가 조용히 멈춘다 (stereo_cpu.py 참고).
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["PYTHONUNBUFFERED"] = "1"
    # pytoningreacher 는 print 를 dprint 로 가려 놨다 — 이게 없으면 병합/후처리
    # 로그가 한 줄도 안 보인다. 켜면 depth/logs/<시각>/ 세션 디렉터리도 생긴다
    # (depth 의 .gitignore 대상).
    env["PYTR_DEBUG"] = "1" if options.debug else "0"
    # 부모(UI) 의 venv 가 자식으로 새지 않도록 정리한다.
    env.pop("VIRTUAL_ENV", None)
    return env


class ReconWorker(QThread):
    """샘플 여러 개를 차례로 재구성한다. 출력은 줄 단위로 흘려보낸다."""

    line = Signal(str)
    sampleDone = Signal(object, bool)  # (Sample, 성공 여부)
    finishedAll = Signal(int, int)  # (성공, 전체)

    def __init__(self, samples: List[Sample], options: ReconOptions, parent=None) -> None:
        super().__init__(parent)
        self._samples = list(samples)
        self._options = options
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def run(self) -> None:  # noqa: D102 (QThread 규약)
        try:
            applied = ensure_patches()
        except ReconError as exc:
            self.line.emit(f"[오류] {exc}")
            self.finishedAll.emit(0, len(self._samples))
            return
        for rel in applied:
            self.line.emit(f"[패치] {rel} 를 서브모듈에 넣었습니다")

        ok_count = 0
        for index, sample in enumerate(self._samples, 1):
            if self._cancelled:
                self.line.emit("[중지] 사용자가 취소했습니다")
                break
            self.line.emit(
                f"\n{'=' * 60}\n[{index}/{len(self._samples)}] {sample.sample_id}\n{'=' * 60}"
            )
            success = self._run_one(sample)
            ok_count += int(success)
            self.sampleDone.emit(sample, success)

        self.finishedAll.emit(ok_count, len(self._samples))

    def _run_one(self, sample: Sample) -> bool:
        try:
            cmd, stage = build_command(sample, self._options)
        except ReconError as exc:
            self.line.emit(f"[오류] {exc}")
            return False

        stage.mkdir(parents=True, exist_ok=True)
        self.line.emit(f"[실행] {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(PATHS.depth),  # make3D.py 가 sys.path 를 레포 루트 기준으로 잡는다
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_env(self._options),
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.line.emit(f"[오류] 실행 실패: {exc}")
            return False

        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            self.line.emit(raw.rstrip("\n"))
        code = self._proc.wait()
        self._proc = None

        if code != 0:
            self.line.emit(f"[실패] 종료 코드 {code}")
            return False
        self.line.emit(f"[완료] {stage / 'model'}")
        return True
