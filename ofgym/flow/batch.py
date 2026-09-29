"""일괄 생성 — record 여러 개에서 촬영 쌍과 flow GT 를 목표 개수만큼 만든다.

한 쌍은 '무작위 배치' 한 번과 같다: 촬영 위치(+50° / 0° / -50°)를 돌아가며 고르고,
카메라와 얼굴을 `tvec` / `rvec` 범위 안에서 흔들고, 배경 무늬를 바꾼다.

**record 단위로 프로세스에 나눠 준다.** 프로세스마다 렌더러(OpenGL 컨텍스트)를 하나씩
갖고, 맡은 record 의 메쉬를 읽어 → 찍고 → 파일로 쓰는 것까지 혼자 한다. 프로세스 사이에
오가는 것은 목록 몇 줄뿐이다. **flow 는 셰이더가 계산한다** (`Renderer.render_pair`).

이렇게 된 까닭 (1/4 해상도, 600쌍, 10코어에서 잰 값):

    스레드로 쓰기만 나눔         12~14 쌍/초  flow 계산이 GIL 에 서로 막힌다
    프로세스로 쓰기만 나눔       12~15 쌍/초  렌더한 프레임(쌍당 24MB)을 넘기느라
                                              그리기 스레드가 쌍당 40ms 를 쓴다
    프로세스마다 렌더러, CPU flow   ~11 쌍/초  numpy 가 큰 임시 배열을 계속 만들고 지워,
                                              프로세스 8개면 단계마다 4배 느려진다
    프로세스마다 렌더러, GPU flow   지금

배경 무늬는 쌍마다 새로 만들지 않는다 (2048² 한 장에 0.1초). 시작할 때 몇 가지를 한 번
만들어 파일로 두고 프로세스들이 읽어 쓴다. 쌍마다 그중 하나를 골라 **판을 제 평면 안에서
돌리고 밀어** 다른 그림이 찍히게 한다.

같은 설정·같은 시드면 어느 프로세스가 어느 record 를 맡든 같은 쌍이 나온다 — 난수는
(시드, record 순번, 쌍 순번) 으로만 정해진다.

결과는 쌍마다 폴더 하나이고 (`gt.save_sample` 과 같은 파일들), 맨 위에 목록이 있다::

    <출력>/
        batch.json      설정
        index.csv       쌍 목록 — split, 파일 경로, 촬영 위치, 통계
        <record>/<번호>/img1.png img2.png flow.flo valid.png occluded.png face.png meta.json
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import queue
import shutil
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ofgym.flow import camera as cam
from ofgym.flow import gt as flow_gt
from ofgym.flow.mesh import build_mesh
from ofgym.flow.record import Record
from ofgym.flow.renderer import Renderer
from ofgym.flow.scene import (
    BACKGROUND_ID,
    BACKGROUND_IMAGE,
    BACKGROUND_NONE,
    FACE_ID,
    VIEW_ANGLES,
    RandomRanges,
    Scene,
    SceneParams,
    load_image,
    noise_texture,
    randomized,
)

# 목표 쌍 수의 기본값. 사전학습된 RAFT / CREStereo 를 한 도메인에 맞출 때 쓰는 규모다 —
# 공개 파인튜닝 세트가 KITTI-2015 200쌍, Sintel 1,041쌍, FlyingChairs(사전학습용) 22,872쌍
# 이고, 얼굴 하나라는 좁은 도메인이라 수천 쌍이면 자세·배경 변이를 덮는다.
DEFAULT_PAIRS = 2000

# 검증용으로 떼어 두는 record 의 비율. 쌍이 아니라 **record 단위**로 나눈다 — 같은 얼굴이
# 학습과 검증 양쪽에 들어가면 검증 점수가 부풀려진다.
VALIDATION_RATIO = 0.1

# 얼굴이 화면에서 이 비율보다 적게 찍히면 다시 뽑는다.
MIN_FACE_RATIO = 0.02
_MAX_RETRIES = 4

# 배경 무늬의 가짓수. 판을 돌리고 밀기 때문에 같은 장이라도 같은 그림은 아니다.
# 프로세스마다 쓰게 된 장을 GPU 에 올려 두므로 (2048² 한 장에 ~16MB) 너무 늘리지 않는다.
_POOL_SIZE = 16
_POOL_SHIFT_MM = 150.0

# 쌍 하나가 디스크에서 차지하는 크기 (픽셀당 바이트). 1/4, 1/2 해상도에서 잰 값이다.
# flow.flo 는 압축이 없어 8, 사진 두 장은 PNG 로 약 3.1, depth.npy 는 4.
_BYTES_PER_PIXEL = 11.2
_BYTES_PER_PIXEL_DEPTH = 4.0

# 배경 무늬를 잠시 담아 두는 폴더 (출력 폴더 안). 끝나면 지운다.
_CACHE_DIR = ".backgrounds"

Progress = Callable[[int, int, str], None]


@dataclass
class BatchSettings:
    total: int = DEFAULT_PAIRS
    seed: int = 0
    params: SceneParams = field(default_factory=SceneParams)  # 해상도·기선·배경·기준 자리
    ranges: RandomRanges = field(default_factory=RandomRanges)
    save_depth: bool = False
    workers: int = 0  # 0 이면 코어 수에 맞춘다

    def worker_count(self, jobs: int) -> int:
        if self.workers > 0:
            return max(1, min(self.workers, jobs))
        # 화면과 다른 프로그램 몫으로 둘을 남긴다.
        return max(1, min((os.cpu_count() or 4) - 2, jobs))


@dataclass
class BatchResult:
    output: Path
    written: int = 0
    requested: int = 0
    failed_records: List[str] = field(default_factory=list)
    cancelled: bool = False
    seconds: float = 0.0
    workers: int = 0


def estimate_bytes(settings: BatchSettings, scene_camera) -> int:
    per_pixel = _BYTES_PER_PIXEL + (_BYTES_PER_PIXEL_DEPTH if settings.save_depth else 0.0)
    return int(settings.total * scene_camera.width * scene_camera.height * per_pixel)


def free_bytes(path: Path) -> int:
    """`path` 가 놓일 디스크의 남은 공간. 아직 없는 경로면 있는 조상까지 올라간다."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def plan(records: Sequence[Record], total: int) -> List[Tuple[Record, int]]:
    """목표 개수를 record 들에 고르게 나눈다. 나머지는 앞에서부터 하나씩 더 준다."""
    if not records or total <= 0:
        return []
    base, extra = divmod(total, len(records))
    counts = [(record, base + (1 if i < extra else 0)) for i, record in enumerate(records)]
    return [(record, count) for record, count in counts if count > 0]


def validation_records(records: Sequence[Record], seed: int) -> set:
    """검증용으로 뗄 record 들. 둘 미만이면 떼지 않는다."""
    if len(records) < 2:
        return set()
    count = max(1, int(round(len(records) * VALIDATION_RATIO)))
    ids = sorted(record.record_id for record in records)
    picked = np.random.default_rng([seed, 0x5EED]).choice(len(ids), count, replace=False)
    return {ids[i] for i in picked}


@dataclass
class _BackgroundSpec:
    """배경판의 크기와 무늬 목록. 시키는 쪽이 정해 일하는 프로세스들에 넘긴다."""

    half: float = 0.0  # 판 한 변의 절반, mm
    z: float = 0.0
    size: int = 0  # 무늬 한 변, 픽셀
    seeds: List[int] = field(default_factory=list)
    image: Optional[str] = None  # 무늬 대신 쓸 이미지 파일
    cache: Optional[str] = None  # 만들어 둔 무늬가 있는 폴더

    @property
    def enabled(self) -> bool:
        return bool(self.seeds)

    def path(self, index: int) -> Path:
        return Path(self.cache) / f"{self.seeds[index]}.npy"


def _background_spec(settings: BatchSettings, camera) -> _BackgroundSpec:
    params = settings.params
    if params.background == BACKGROUND_NONE:
        return _BackgroundSpec()
    z = params.background_z
    # 흔들림이 가장 클 때의 시야를 덮는 원 + 판을 미는 여유. 판을 아무 각도로 돌려도
    # 시야가 판 밖으로 나가지 않도록 정사각형으로 잡는다.
    turn = np.radians(min(settings.ranges.camera_r * np.sqrt(3.0), 60.0))
    reach = (abs(params.baseline_mm) + settings.ranges.camera_t * np.sqrt(3.0)
             + z * np.tan(turn))
    need = np.hypot(z * camera.width / camera.fx, z * camera.height / camera.fy) * 0.6
    half = need + reach + _POOL_SHIFT_MM

    if params.background == BACKGROUND_IMAGE:
        return _BackgroundSpec(half=half, z=z, seeds=[-1], image=params.background_image)
    # 판 위 1mm 가 사진에서 몇 픽셀인지에 맞춰 무늬 해상도를 고른다.
    wanted = 2 * half * camera.fx / z
    size = 4096 if wanted > 2600 else 2048 if wanted > 1300 else 1024
    count = _POOL_SIZE if size < 4096 else _POOL_SIZE // 2
    rng = np.random.default_rng([settings.seed, 0xB6])
    return _BackgroundSpec(half=half, z=z, size=size,
                           seeds=[int(s) for s in rng.integers(0, 2**31 - 1, count)])


def _paint(spec: _BackgroundSpec, index: int) -> None:
    np.save(spec.path(index), noise_texture(spec.seeds[index], spec.size))


def _prepare_backgrounds(spec: _BackgroundSpec, output: Path) -> Optional[str]:
    """무늬를 만들어 파일로 둔다. 문제가 있으면 그 설명을, 없으면 None 을 돌려준다."""
    if not spec.enabled:
        return None
    if spec.image is not None or spec.seeds == [-1]:
        if not spec.image or load_image(spec.image) is None:
            spec.seeds = []
            return f"배경 이미지를 읽지 못했습니다: {spec.image} — 배경 없이 진행합니다"
        return None
    cache = output / _CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    spec.cache = str(cache)
    with ThreadPoolExecutor(max_workers=4) as painters:
        list(painters.map(lambda i: _paint(spec, i), range(len(spec.seeds))))
    return None


class _Backgrounds:
    """일하는 프로세스 쪽 — 넉넉한 판 하나와, 쓰게 된 무늬들의 GPU 텍스처."""

    def __init__(self, renderer: Renderer, spec: _BackgroundSpec) -> None:
        self._renderer = renderer
        self._spec = spec
        self._textures: Dict[int, object] = {}
        self.enabled = spec.enabled
        if not self.enabled:
            return
        self._seeds = spec.seeds
        self._z = spec.z
        half = spec.half
        vertices = np.array([[-half, -half, 0], [half, -half, 0],
                             [-half, half, 0], [half, half, 0]], np.float32)
        uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
        faces = np.array([[0, 2, 1], [1, 2, 3]], np.int32)
        renderer.set_object(BACKGROUND_ID, vertices, uvs, faces,
                            np.zeros((2, 2, 3), np.uint8))

    def _texture(self, index: int):
        if index not in self._textures:
            spec = self._spec
            image = load_image(spec.image) if spec.image else np.load(spec.path(index))
            self._textures[index] = self._renderer.create_texture(image)
        return self._textures[index]

    def place(self, base_pose: np.ndarray, rng: np.random.Generator):
        """(판의 object→world, 기록용 dict). 판은 흔들기 전 카메라를 마주 보고 선다."""
        index = int(rng.integers(0, len(self._seeds)))
        angle = float(rng.uniform(-180.0, 180.0))
        shift = rng.uniform(-_POOL_SHIFT_MM, _POOL_SHIFT_MM, 2)
        self._renderer.swap_texture(BACKGROUND_ID, self._texture(index))

        in_plane = cam.rigid((0.0, 0.0, angle), (shift[0], shift[1], self._z))
        model = np.linalg.inv(base_pose) @ in_plane
        note = dict(texture_seed=self._seeds[index], angle=angle,
                    shift=[float(shift[0]), float(shift[1])])
        return model, note


def _station_for(index: int, offset: int, settings: BatchSettings) -> dict:
    """촬영 위치를 돌아가며 고른다 — 세 자리가 같은 수만큼 나오게."""
    if not settings.ranges.random_view:
        return {}
    angles = list(VIEW_ANGLES.values())
    return dict(view_angle=angles[(index + offset) % len(angles)], free_view=None)


# ── 일하는 프로세스 ───────────────────────────────────────────────────────
class _Worker:
    """프로세스 하나가 뜰 때 만들어져 끝까지 쓰인다."""

    def __init__(self, settings: BatchSettings, output: Path, spec: _BackgroundSpec,
                 ticks, stop) -> None:
        import cv2

        # 프로세스를 이미 코어 수만큼 띄웠다. OpenCV 가 저마다 스레드를 또 띄우면 서로 밀어낸다.
        cv2.setNumThreads(1)
        self.settings = settings
        self.output = output
        self.ticks = ticks
        self.stop = stop
        self.renderer = Renderer()
        self.scene = Scene(self.renderer)
        self.camera = self.scene.camera(settings.params)
        self.backgrounds = _Backgrounds(self.renderer, spec)
        self.fixed = RandomRanges(**{**asdict(settings.ranges), "random_view": False})

    def shoot(self, job_index: int, index: int):
        settings, scene = self.settings, self.scene
        rng = np.random.default_rng([settings.seed, job_index, index])
        station = _station_for(index, job_index, settings)
        for _ in range(_MAX_RETRIES):
            base = SceneParams(**{**settings.params.as_dict(), **station})
            params = randomized(base, self.fixed, rng)
            models = {FACE_ID: scene.face_model(params)}
            note = None
            if self.backgrounds.enabled:
                models[BACKGROUND_ID], note = self.backgrounds.place(
                    scene.base_pose(params), rng)
            extrinsic1, extrinsic2 = scene.extrinsics(params)
            pair = self.renderer.render_pair(
                self.camera, extrinsic1, extrinsic2, models,
                tolerance=flow_gt.OCCLUSION_TOLERANCE_MM,
            )
            if np.count_nonzero(pair.ids == FACE_ID) >= MIN_FACE_RATIO * pair.ids.size:
                break
        return params, note, pair

    def run(self, job_index: int, record: Record, count: int, split: str) -> List[dict]:
        if self.stop.is_set():
            return []
        self.scene.set_mesh(build_mesh(record))
        rows = []
        for index in range(count):
            if self.stop.is_set():
                break
            params, note, pair = self.shoot(job_index, index)
            gt = flow_gt.flow_from_pair(pair)
            relative = Path(record.record_id) / f"{index:05d}"
            flow_gt.save_pair(
                self.output / relative, pair.color1, pair.color2, gt, pair.camera,
                pair.extrinsic1, pair.extrinsic2, pair.models1, pair.models2,
                dict(record=record.record_id, source=str(record.path), split=split,
                     params=params.as_dict(), background_plane=note),
                depth=self.settings.save_depth,
                face=pair.ids == FACE_ID,
            )
            stats = gt.stats()
            rows.append(dict(
                split=split, record=record.record_id, sample=str(relative),
                img1=str(relative / "img1.png"), img2=str(relative / "img2.png"),
                flow=str(relative / "flow.flo"), valid_mask=str(relative / "valid.png"),
                occluded_mask=str(relative / "occluded.png"),
                face_mask=str(relative / "face.png"),
                view_angle=params.view_angle if params.free_view is None else "free",
                valid=round(stats.get("valid", 0.0), 4),
                occluded=round(stats.get("occluded", 0.0), 4),
                flow_mean=round(stats.get("flow_mean", 0.0), 3),
                flow_max=round(stats.get("flow_max", 0.0), 3),
            ))
            self.ticks.put(1)
        return rows


_WORKER: Optional[_Worker] = None
_WORKER_ERROR: Optional[str] = None


def _start_worker(settings: BatchSettings, output: Path, spec: _BackgroundSpec,
                  ticks, stop) -> None:
    global _WORKER, _WORKER_ERROR
    try:
        _WORKER = _Worker(settings, output, spec, ticks, stop)
    except Exception as exc:  # 초기화에서 죽으면 풀 전체가 멈춘다 — 일감마다 알리게 한다
        _WORKER_ERROR = f"렌더러를 만들지 못했습니다: {exc}"


def _run_record(job_index: int, record: Record, count: int, split: str) -> List[dict]:
    if _WORKER is None:
        raise RuntimeError(_WORKER_ERROR or "일하는 프로세스가 준비되지 않았습니다")
    return _WORKER.run(job_index, record, count, split)


# ── 시키는 쪽 ─────────────────────────────────────────────────────────────
def run_batch(
    records: Sequence[Record],
    settings: BatchSettings,
    output: Path,
    progress: Optional[Progress] = None,
    stop: Optional[threading.Event] = None,
) -> BatchResult:
    """끝날 때까지 돌아오지 않는다. UI 에서는 다른 스레드에서 부를 것."""
    started = time.perf_counter()
    stop = stop or threading.Event()
    report = progress or (lambda done, total, message: None)
    jobs = plan(records, settings.total)
    result = BatchResult(output=output, requested=sum(count for _, count in jobs))
    if not jobs:
        return result

    output.mkdir(parents=True, exist_ok=True)
    held_out = validation_records([record for record, _ in jobs], settings.seed)
    result.workers = settings.worker_count(len(jobs))

    camera = Scene.camera_for(settings.params)
    (output / "batch.json").write_text(json.dumps(dict(
        total=result.requested, seed=settings.seed, records=len(jobs),
        params=settings.params.as_dict(), ranges=asdict(settings.ranges),
        camera=dict(width=camera.width, height=camera.height, fx=camera.fx,
                    fy=camera.fy, cx=camera.cx, cy=camera.cy),
        validation_records=sorted(held_out),
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    spec = _background_spec(settings, camera)
    problem = _prepare_backgrounds(spec, output)
    if problem:
        report(0, result.requested, problem)

    # fork 는 OpenGL 컨텍스트와 Qt 를 가진 프로세스를 복제하게 되므로 쓰지 않는다.
    spawn = multiprocessing.get_context("spawn")
    ticks = spawn.Queue()
    halt = spawn.Event()
    pool = ProcessPoolExecutor(
        max_workers=result.workers, mp_context=spawn,
        initializer=_start_worker, initargs=(settings, output, spec, ticks, halt),
    )

    rows: List[dict] = []
    done_pairs = 0
    try:
        # 쌍이 많은 record 부터 — 끝에 긴 일감 하나가 남아 코어가 노는 일을 줄인다.
        order = sorted(range(len(jobs)), key=lambda i: -jobs[i][1])
        pending = {}
        for job_index in order:
            record, count = jobs[job_index]
            split = "val" if record.record_id in held_out else "train"
            pending[pool.submit(_run_record, job_index, record, count, split)] = record

        while pending:
            if stop.is_set() and not halt.is_set():
                halt.set()
                # 아직 시작하지 않은 record 는 버린다 — 두면 메쉬를 읽고 나서야 그만둔다.
                for future in [f for f in pending if f.cancel()]:
                    del pending[future]
                if not pending:
                    break
            finished, _ = wait(list(pending), timeout=0.1, return_when=FIRST_COMPLETED)
            try:
                while True:
                    done_pairs += ticks.get_nowait()
            except queue.Empty:
                pass
            for future in finished:
                record = pending.pop(future)
                try:
                    rows.extend(future.result())
                except Exception as exc:  # record 하나가 실패해도 나머지는 계속한다
                    result.failed_records.append(record.record_id)
                    report(done_pairs, result.requested, f"{record.record_id}: {exc}")
            report(done_pairs, result.requested, "")
    finally:
        halt.set()
        pool.shutdown(wait=True, cancel_futures=True)
        ticks.close()
        if spec.cache:
            shutil.rmtree(spec.cache, ignore_errors=True)

    result.cancelled = stop.is_set()
    result.written = len(rows)
    rows.sort(key=lambda row: row["sample"])
    if rows:
        with (output / "index.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    result.seconds = time.perf_counter() - started
    return result
