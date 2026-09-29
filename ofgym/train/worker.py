"""파인튜닝 · 비교를 실제로 하는 스크립트. **depth 서브모듈의 venv 로 돈다.**

UI(PySide6)와 학습(torch)은 인터프리터가 다르다 — `3D 생성` 과 같은 이유다. 그래서 이
파일은 `ofgym` 의 다른 모듈을 가져다 쓰지 않고 혼자 선다 (numpy, cv2, torch 와 모델
저장소만 쓴다). UI 는 이 스크립트를 서브프로세스로 띄우고 표준 출력을 읽는다.

    python worker.py train   --run <폴더>     # config.json 을 읽어 학습 → 끝나면 비교까지
    python worker.py compare --run <폴더>     # 비교만 다시

표준 출력은 한 줄에 JSON 하나다 (`{"event": ...}`). 그 밖의 줄은 그냥 로그다.

데이터는 Flow Gym 의 일괄 생성 폴더다. 스테레오 모델은 **가로 시차**만 받으므로
쌍을 그 모양으로 바꿔 넣는다.

    기선 세로 (디바이스)   두 장을 시계 방향 90° 돌린다. 시차 = 돌린 flow 의 v 성분
                           — 디바이스가 매칭 전에 하는 것과 같다 (`stereo.py` 의
                           ROTATE_90_CLOCKWISE).
    기선 가로              그대로. 시차 = -flow 의 u 성분

모델마다 부호가 다르다.

    RAFT-Stereo   예측 1채널 = -시차.  RGB 입력
    CREStereo     예측 2채널, 0번 = +시차.  BGR 입력
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# MPS 에 없는 연산은 CPU 로 넘긴다 — 없으면 그 자리에서 죽는다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODELS = {
    "raft": dict(repo="RAFT-Stereo", checkpoint="RAFT-Stereo/models/raftstereo-eth3d.pth",
                 rgb=True, gamma=0.9),
    "crestereo": dict(repo="CREStereo-Pytorch",
                      checkpoint="CREStereo-Pytorch/models/crestereo_eth3d.pth",
                      rgb=False, gamma=0.8),
}

MAX_DISPARITY = 400.0  # 이보다 큰 GT 는 손실에서 뺀다
_FLO_MAGIC = 202021.25


def emit(event: str, **fields) -> None:
    print(json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)


# ── 데이터 ────────────────────────────────────────────────────────────────
def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        magic = np.fromfile(fh, np.float32, 1)
        if magic.size != 1 or magic[0] != np.float32(_FLO_MAGIC):
            raise ValueError(f".flo 가 아닙니다: {path}")
        width, height = np.fromfile(fh, np.int32, 2)
        return np.fromfile(fh, np.float32, 2 * width * height).reshape(height, width, 2)


def is_vertical(dataset: Path) -> bool:
    """기선이 세로인지. batch.json 에 없으면 flow 를 보고 정한다."""
    try:
        axis = json.loads((dataset / "batch.json").read_text("utf-8"))["params"][
            "baseline_axis"]
        return "세로" in axis
    except (OSError, KeyError, ValueError):
        pass
    rows = read_index(dataset)
    flow = read_flo(dataset / rows[0]["flow"])
    return float(np.abs(flow[..., 1]).mean()) > float(np.abs(flow[..., 0]).mean())


def read_index(dataset: Path) -> list:
    with (dataset / "index.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_pair(dataset: Path, row: dict, vertical: bool) -> dict:
    """한 쌍을 가로 시차 모양으로. 사진은 BGR uint8, 시차는 float32 (양수), 마스크는 bool."""
    left = cv2.imread(str(dataset / row["img1"]), cv2.IMREAD_COLOR)
    right = cv2.imread(str(dataset / row["img2"]), cv2.IMREAD_COLOR)
    flow = read_flo(dataset / row["flow"])
    valid = cv2.imread(str(dataset / row["valid_mask"]), cv2.IMREAD_GRAYSCALE) > 127
    face_path = dataset / row["face_mask"] if row.get("face_mask") else None
    if face_path is not None and face_path.is_file():
        face = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE) > 127
    else:
        face = np.zeros(valid.shape, bool)

    if vertical:
        turn = lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)  # noqa: E731
        left, right = turn(left), turn(right)
        disparity = turn(np.ascontiguousarray(flow[..., 1]))
        valid = turn(valid.astype(np.uint8)) > 0
        face = turn(face.astype(np.uint8)) > 0
    else:
        disparity = -flow[..., 0]
    return dict(left=left, right=right, disparity=np.ascontiguousarray(disparity),
                valid=valid, face=face)


def _jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """밝기·대비·감마·채널 이득·잡음. 합성 쌍은 두 장의 색이 똑같아서, 이걸 **장마다 따로**
    걸어 주지 않으면 실제 카메라 두 대의 노출 차이를 본 적 없는 모델이 된다."""
    out = image.astype(np.float32) / 255.0
    out = np.power(np.clip(out, 1e-4, 1.0), rng.uniform(0.8, 1.25))
    out = (out - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.08, 0.08)
    out = out * rng.uniform(0.92, 1.08, 3).astype(np.float32)
    out = out + rng.normal(0.0, rng.uniform(0.0, 0.02), out.shape).astype(np.float32)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.2))
    return np.clip(out * 255.0, 0, 255).astype(np.float32)


class Pairs:
    """torch DataLoader 에 넣는 데이터셋. 워커 프로세스로 넘어가야 해서 모듈 맨 위에 둔다."""

    def __init__(self, dataset: Path, rows: list, vertical: bool, crop, augment: bool,
                 rgb: bool, seed: int) -> None:
        self.dataset = dataset
        self.rows = rows
        self.vertical = vertical
        self.crop = crop
        self.augment = augment
        self.rgb = rgb
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        import torch

        item = load_pair(self.dataset, self.rows[index], self.vertical)
        left, right = item["left"], item["right"]
        disparity, valid, face = item["disparity"], item["valid"], item["face"]
        if self.augment:
            # 워커마다 · epoch 마다 다른 난수. torch 가 워커별 시드를 numpy 에는 안 준다.
            rng = np.random.default_rng(
                [self.seed, index, int(torch.randint(0, 2**31 - 1, (1,)).item())])
            height, width = disparity.shape
            ch, cw = min(self.crop[0], height), min(self.crop[1], width)
            y = int(rng.integers(0, height - ch + 1))
            x = int(rng.integers(0, width - cw + 1))
            window = (slice(y, y + ch), slice(x, x + cw))
            left, right = left[window], right[window]
            disparity, valid, face = disparity[window], valid[window], face[window]
            left, right = _jitter(left, rng), _jitter(right, rng)
            if rng.random() < 0.5:
                # 실제 쌍은 정렬이 완벽하지 않다 — 오른쪽을 세로로 조금 민다.
                shift = np.float32([[1, 0, 0], [0, 1, rng.uniform(-1.0, 1.0)]])
                right = cv2.warpAffine(right, shift, (right.shape[1], right.shape[0]),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        if self.rgb:
            left, right = left[..., ::-1], right[..., ::-1]

        def image(a):
            return torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(2, 0, 1)

        return dict(
            left=image(left), right=image(right),
            disparity=torch.from_numpy(np.ascontiguousarray(disparity))[None],
            valid=torch.from_numpy(np.ascontiguousarray(valid))[None],
            face=torch.from_numpy(np.ascontiguousarray(face))[None],
            index=index,
        )


# ── 모델 ──────────────────────────────────────────────────────────────────
def pick_device(torch, wanted: str):
    if wanted == "auto":
        if torch.cuda.is_available():
            wanted = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            wanted = "mps"
        else:
            wanted = "cpu"
    return torch.device(wanted)


def build_model(torch, name: str, models_root: Path):
    spec = MODELS[name]
    repo = models_root / spec["repo"]
    if not repo.is_dir():
        raise FileNotFoundError(
            f"{name} 소스가 없습니다: {repo}\n"
            f"  {sys.executable} tools/setup_stereo_models.py --models {name}")
    if name == "raft":
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "core"))
        from raft_stereo import RAFTStereo

        return RAFTStereo(SimpleNamespace(
            hidden_dims=[128, 128, 128], corr_implementation="reg",
            shared_backbone=False, corr_levels=4, corr_radius=4,
            n_downsample=2, context_norm="batch", slow_fast_gru=False,
            n_gru_layers=3, mixed_precision=False))

    sys.path.insert(0, str(repo))
    import nets.utils.utils as cre_utils
    from nets import Model

    # 이식본은 ONNX 내보내기용으로 grid_sample 을 손으로 풀어 놨다. 같은 연산이고
    # 미분도 되지만 훨씬 느리다 — 네이티브로 되돌린다 (depth 의 learned.py 와 같다).
    def native(im, grid, align_corners=False):
        return torch.nn.functional.grid_sample(
            im, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)

    cre_utils.bilinear_grid_sample = native
    return Model(max_disp=256, mixed_precision=False, test_mode=False)


def load_weights(torch, model, path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in payload.items()}
    model.load_state_dict(state, strict=True)


def predict(torch, model, name: str, left, right, iters: int, train: bool):
    """시차(양수) 예측들. 학습이면 반복마다 하나씩, 아니면 마지막 하나."""
    if name == "raft":
        if train:
            return [-p for p in model(left, right, iters=iters)]
        return [-model(left, right, iters=iters, test_mode=True)[1]]
    model.test_mode = not train
    out = model(left, right, iters=iters, flow_init=None)
    return [p[:, :1] for p in out] if train else [out[:, :1]]


def pad_to(torch, left, right, multiple: int = 32):
    height, width = left.shape[-2:]
    ph, pw = (-height) % multiple, (-width) % multiple
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    if ph or pw:
        left = torch.nn.functional.pad(left, pad, mode="replicate")
        right = torch.nn.functional.pad(right, pad, mode="replicate")
    return left, right, pad


def unpad(tensor, pad):
    l, r, t, b = pad
    height, width = tensor.shape[-2:]
    return tensor[..., t:height - b if b else height, l:width - r if r else width]


def sequence_loss(torch, predictions, target, valid, gamma: float):
    """RAFT 식 손실 — 뒤 반복일수록 무겁게. gamma 는 반복 수가 달라도 같은 뜻이 되게 맞춘다."""
    count = len(predictions)
    mask = valid & (target.abs() < MAX_DISPARITY)
    if int(mask.sum()) == 0:
        return None
    adjusted = gamma ** (15.0 / max(count - 1, 1))
    loss = 0.0
    for i, prediction in enumerate(predictions):
        weight = adjusted ** (count - i - 1)
        loss = loss + weight * (prediction - target).abs()[mask].mean()
    return loss


# ── 평가 ──────────────────────────────────────────────────────────────────
class Meter:
    """픽셀을 다 모아 평균한다 (사진별 평균의 평균이 아니다)."""

    def __init__(self) -> None:
        self.sums = {}

    def add(self, error: np.ndarray, masks: dict) -> None:
        for key, mask in masks.items():
            values = error[mask]
            s = self.sums.setdefault(key, [0.0, 0.0, 0.0, 0])
            s[0] += float(values.sum())
            s[1] += float((values > 1.0).sum())
            s[2] += float((values > 3.0).sum())
            s[3] += int(values.size)

    def result(self) -> dict:
        out = {}
        for key, (total, bad1, bad3, count) in self.sums.items():
            if count:
                out[key] = dict(epe=total / count, bad1=100.0 * bad1 / count,
                                bad3=100.0 * bad3 / count, pixels=count)
        return out


def evaluate(torch, model, name: str, loader, device, iters: int, keep=None) -> dict:
    """검증. `keep(index, 시차 예측)` 을 주면 사진마다 예측을 넘겨준다."""
    model.eval()
    meter = Meter()
    with torch.inference_mode():
        for batch in loader:
            left, right, pad = pad_to(torch, batch["left"].to(device),
                                      batch["right"].to(device))
            prediction = unpad(predict(torch, model, name, left, right, iters, False)[-1],
                               pad)
            prediction = prediction.float().cpu().numpy()
            target = batch["disparity"].numpy()
            valid = batch["valid"].numpy()
            face = batch["face"].numpy()
            for i in range(prediction.shape[0]):
                error = np.abs(prediction[i, 0] - target[i, 0])
                meter.add(error, dict(all=valid[i, 0], face=valid[i, 0] & face[i, 0],
                                      background=valid[i, 0] & ~face[i, 0]))
                if keep is not None:
                    keep(int(batch["index"][i]), prediction[i, 0])
    return meter.result()


# ── 학습 ──────────────────────────────────────────────────────────────────
def train(run: Path) -> int:
    import torch

    config = json.loads((run / "config.json").read_text("utf-8"))
    name = config["model"]
    spec = MODELS[name]
    dataset = Path(config["dataset"])
    models_root = Path(config["models_root"])
    device = pick_device(torch, config.get("device", "auto"))
    torch.manual_seed(int(config.get("seed", 0)))

    rows = read_index(dataset)
    vertical = is_vertical(dataset)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows:
        raise ValueError("index.csv 에 train 쌍이 없습니다")
    if not val_rows:  # record 가 하나뿐인 데이터 — 끝의 10% 를 검증으로 쓴다
        cut = max(1, len(train_rows) // 10)
        train_rows, val_rows = train_rows[:-cut], train_rows[-cut:]
    val_rows = val_rows[:: max(1, len(val_rows) // int(config["val_samples"]))][
        : int(config["val_samples"])]

    probe = load_pair(dataset, train_rows[0], vertical)
    median = float(np.median(probe["disparity"][probe["valid"]]))
    if median <= 0:
        raise ValueError(
            f"시차가 음수입니다 (중앙값 {median:.1f}px) — img1 이 오른쪽 카메라인 데이터입니다")

    crop = (int(config["crop_height"]), int(config["crop_width"]))
    workers = int(config.get("loader_workers", 4))
    loader = torch.utils.data.DataLoader(
        Pairs(dataset, train_rows, vertical, crop, True, spec["rgb"],
              int(config.get("seed", 0))),
        batch_size=int(config["batch_size"]), shuffle=True, drop_last=True,
        num_workers=workers, persistent_workers=workers > 0)
    val_loader = torch.utils.data.DataLoader(
        Pairs(dataset, val_rows, vertical, crop, False, spec["rgb"], 0),
        batch_size=1, shuffle=False, num_workers=min(2, workers))

    model = build_model(torch, name, models_root)
    base = Path(config.get("checkpoint") or models_root / spec["checkpoint"])
    load_weights(torch, model, base)
    model.to(device)

    steps = int(config["steps"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config.get("weight_decay", 1e-5)),
                                  eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, float(config["lr"]), steps + 100, pct_start=0.05,
        cycle_momentum=False, anneal_strategy="linear")

    emit("start", model=name, device=str(device), train=len(train_rows),
         val=len(val_rows), vertical=vertical, steps=steps, base=str(base),
         height=int(probe["disparity"].shape[0]), width=int(probe["disparity"].shape[1]))

    stopping = dict(flag=False)
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))

    def set_training() -> None:
        model.train()
        if hasattr(model, "freeze_bn"):
            model.freeze_bn()  # 배치가 작아 BN 통계를 새로 잡으면 오히려 망가진다

    def validate(step: int) -> dict:
        metrics = evaluate(torch, model, name, val_loader, device,
                           int(config["valid_iters"]))
        emit("val", step=step, metrics=metrics)
        set_training()
        return metrics

    best = validate(0)["all"]["epe"]  # 0 스텝 = 사전학습 그대로
    baseline = best
    torch.save(model.state_dict(), run / "checkpoint_best.pth")

    set_training()
    step = 0
    window, began = [], time.perf_counter()
    while step < steps and not stopping["flag"]:
        for batch in loader:
            if step >= steps or stopping["flag"]:
                break
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            target = batch["disparity"].to(device)
            valid = batch["valid"].to(device)

            predictions = predict(torch, model, name, left, right,
                                  int(config["train_iters"]), True)
            loss = sequence_loss(torch, predictions, target, valid, spec["gamma"])
            if loss is None or not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            with torch.no_grad():
                mask = valid & (target.abs() < MAX_DISPARITY)
                epe = float((predictions[-1] - target).abs()[mask].mean())
            window.append((float(loss.detach()), epe))
            if step % int(config.get("log_every", 10)) == 0 or step == steps:
                elapsed = time.perf_counter() - began
                emit("step", step=step, steps=steps,
                     loss=float(np.mean([w[0] for w in window])),
                     epe=float(np.mean([w[1] for w in window])),
                     lr=float(scheduler.get_last_lr()[0]),
                     rate=len(window) / max(elapsed, 1e-6))
                window, began = [], time.perf_counter()

            if step % int(config["val_every"]) == 0 or step == steps:
                metrics = validate(step)
                torch.save(model.state_dict(), run / "checkpoint_last.pth")
                if metrics["all"]["epe"] < best:
                    best = metrics["all"]["epe"]
                    torch.save(model.state_dict(), run / "checkpoint_best.pth")
                    emit("best", step=step, epe=best)
                window, began = [], time.perf_counter()

    torch.save(model.state_dict(), run / "checkpoint_last.pth")
    if step > 0 and step % int(config["val_every"]) != 0 and step != steps:
        # 중간에 멈췄다 — 멈춘 자리의 가중치를 한 번 재 둔다. 안 그러면 '가장 좋았던 것' 이
        # 학습 전 가중치로 남아 기존끼리 비교하게 된다.
        metrics = validate(step)
        if metrics["all"]["epe"] < best:
            best = metrics["all"]["epe"]
            torch.save(model.state_dict(), run / "checkpoint_best.pth")
            emit("best", step=step, epe=best)
    emit("trained", steps=step, stopped=stopping["flag"], best_epe=best,
         baseline_epe=baseline)
    del loader, val_loader, optimizer, model
    if step == 0:
        return 0
    return compare(run)


# ── 비교 ──────────────────────────────────────────────────────────────────
def real_pairs(dataset: Path, rows: list, size, limit: int) -> list:
    """검증 record 들의 **실제 촬영** 쌍. 정답은 없고 눈으로 보는 용도다.

    디바이스와 달리 보정(rectify)은 하지 않는다 — 크기만 맞추고 돌린다.
    """
    found, seen = [], set()
    for row in rows:
        if row["record"] in seen:
            continue
        seen.add(row["record"])
        try:
            meta = json.loads((dataset / row["sample"] / "meta.json").read_text("utf-8"))
            source = Path(meta["source"])
        except (OSError, KeyError, ValueError):
            continue
        for view in ("center", "left", "right"):
            first, second = source / f"{view}.jpg", source / f"{view}_pair.jpg"
            if first.is_file() and second.is_file():
                found.append((f"{row['record'][:8]}_{view}", first, second))
        if len(found) >= limit:
            break
    out = []
    for label, first, second in found[:limit]:
        pair = []
        for path in (first, second):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                break
            # 세로로 긴 원본 → 데이터와 같은 크기로 줄이고 시계 방향으로 돌린다.
            image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_AREA)
            pair.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        if len(pair) == 2:
            out.append((label, pair[0], pair[1]))
    return out


def compare(run: Path) -> int:
    """사전학습 그대로(기존)와 튜닝한 것을 같은 검증 쌍으로 잰다."""
    import torch

    config = json.loads((run / "config.json").read_text("utf-8"))
    name = config["model"]
    spec = MODELS[name]
    dataset = Path(config["dataset"])
    models_root = Path(config["models_root"])
    device = pick_device(torch, config.get("device", "auto"))
    tuned_path = run / "checkpoint_best.pth"
    if not tuned_path.is_file():
        raise FileNotFoundError(f"튜닝된 체크포인트가 없습니다: {tuned_path}")

    rows = read_index(dataset)
    vertical = is_vertical(dataset)
    val_rows = [r for r in rows if r["split"] == "val"] or rows[-max(1, len(rows) // 10):]
    limit = int(config.get("compare_samples", 200))
    val_rows = val_rows[:: max(1, math.ceil(len(val_rows) / limit))][:limit]
    shown = int(config.get("compare_shown", 24))
    show = set(range(0, len(val_rows), max(1, len(val_rows) // shown)))

    out = run / "compare"
    if out.is_dir():  # 다시 비교할 때 — 쌍 수가 달라졌으면 옛 폴더가 섞인다
        shutil.rmtree(out)
    out.mkdir()
    loader = torch.utils.data.DataLoader(
        Pairs(dataset, val_rows, vertical, (0, 0), False, spec["rgb"], 0),
        batch_size=1, shuffle=False, num_workers=2)

    model = build_model(torch, name, models_root)
    model.to(device)
    checkpoints = dict(
        base=Path(config.get("checkpoint") or models_root / spec["checkpoint"]),
        tuned=tuned_path)
    iters = int(config["valid_iters"])

    probe = load_pair(dataset, val_rows[0], vertical)
    height, width = probe["disparity"].shape
    # 돌리기 전 크기 (가로, 세로) — 실제 촬영본을 이 크기로 줄인 뒤 돌린다.
    before = (height, width) if vertical else (width, height)
    reals = real_pairs(dataset, val_rows, before, int(config.get("compare_real", 6)))

    metrics = {}
    for which, path in checkpoints.items():
        load_weights(torch, model, path)
        emit("compare", stage=which, samples=len(val_rows))

        def keep(index: int, prediction: np.ndarray, which=which) -> None:
            if index in show:
                folder = out / f"{index:04d}"
                folder.mkdir(exist_ok=True)
                np.save(folder / f"{which}.npy", prediction.astype(np.float32))

        metrics[which] = evaluate(torch, model, name, loader, device, iters, keep)

        model.eval()
        with torch.inference_mode():
            for label, left, right in reals:
                folder = out / f"real_{label}"
                folder.mkdir(exist_ok=True)
                cv2.imwrite(str(folder / "left.png"), left)
                cv2.imwrite(str(folder / "right.png"), right)

                def tensor(a):
                    a = a[..., ::-1] if spec["rgb"] else a
                    return torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(
                        2, 0, 1)[None].to(device)

                l, r, pad = pad_to(torch, tensor(left), tensor(right))
                prediction = unpad(predict(torch, model, name, l, r, iters, False)[-1], pad)
                np.save(folder / f"{which}.npy",
                        prediction[0, 0].float().cpu().numpy().astype(np.float32))

    for index in sorted(show):
        item = load_pair(dataset, val_rows[index], vertical)
        folder = out / f"{index:04d}"
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / "left.png"), item["left"])
        cv2.imwrite(str(folder / "right.png"), item["right"])
        np.save(folder / "gt.npy", item["disparity"].astype(np.float32))
        cv2.imwrite(str(folder / "valid.png"), item["valid"].astype(np.uint8) * 255)
        cv2.imwrite(str(folder / "face.png"), item["face"].astype(np.uint8) * 255)
        (folder / "source.json").write_text(json.dumps(
            dict(sample=val_rows[index]["sample"], record=val_rows[index]["record"],
                 view_angle=val_rows[index].get("view_angle")), ensure_ascii=False), "utf-8")

    summary = dict(model=name, samples=len(val_rows), iters=iters, vertical=vertical,
                   base=str(checkpoints["base"]), tuned=str(checkpoints["tuned"]),
                   metrics=metrics, real=[label for label, _, _ in reals])
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      "utf-8")
    emit("compared", **summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("train", "compare"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        return train(args.run) if args.command == "train" else compare(args.run)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
