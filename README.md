# optical_flow_gym

CREStereo / RAFT-Stereo 계열 모델을 자체 캡처 데이터로 파인튜닝하기 위한 워크벤치.

## 구성

- `ofgym/` — PySide6 데스크톱 UI + 데이터셋/GT 파이프라인
- `ofgym/flow/` — Flow Gym: 아틀라스 3D 를 합성 촬영해 optical flow GT 를 만든다
- `thirdparty/depth` — DepthEstimation 서브모듈. 캘리브레이션·스테레오·3D 재구성 코드를 여기서 가져다 쓴다.
- `../dataset/` — 데이터 저장소(저장소 밖). `raw/<세션>/<샘플UUID>/` 구조.

## 실행

```
uv run main
```

저장소 폴더 안에서는 이것으로 된다. **다른 폴더(홈 등)에서도** 띄우려면 한 번 설치해 둔다:

```
uv tool install --editable .    # 저장소 폴더에서. ~/.local/bin/main 이 생긴다
```

그 뒤로는 어디서든 `uv run main` (또는 그냥 `main`) 이다. editable 이라 코드를 고치면
바로 반영되고, 경로(`thirdparty/depth`, `../dataset`, `../shared`)는 실행한 폴더가 아니라
저장소 위치를 기준으로 잡힌다. 의존성을 바꿨을 때만 `uv tool install --editable . --reinstall`
로 다시 맞춘다. 지울 때는 `uv tool uninstall optical-flow-gym`.

## Flow Gym

FlyingChairs / FlyingThings 처럼 **그래픽스로 찍어서** flow GT 를 얻는다. 실제 촬영으로는
픽셀 단위 정답을 얻을 수 없지만, 3D 를 렌더하면 정답이 계산으로 나온다.

1. `디렉터리 선택…` 으로 record 가 있는 폴더를 고른다 (기본 `../shared`, `OFGYM_SHARED`
   로 바꿀 수 있다). `atlas_meta.json` 이 있는 폴더를 전부 찾아 목록에 올린다.
2. record 를 고르면 세 뷰의 정점맵(`{left,center,right}.npy`)을 원통 (θ, l) 격자에 모아
   한 겹짜리 메쉬를 만들고 `atlas.png` 를 텍스처로 입힌다.
3. 첫 카메라로 한 장, **기선만큼 평행 이동**한 자리에서 한 장 더 찍는다. 씬은 멈춰
   있으므로 flow 는 시차와 같다: `f · B / Z`, 기선 방향으로만.
4. `촬영 결과` 탭에서 확인한다. `되돌린 사진 2` 가 `사진 1` 과 겹치고 `색 오차` 가
   어두우면 GT 가 맞는 것이다.

### 촬영 위치와 무작위 배치

카메라는 디바이스처럼 회전축 둘레의 세 자리 중 하나에 선다 — `left (+50°)`,
`center (0°)`, `right (-50°)`. 흔들림이 0 이면 렌더가 그 뷰의 실제 캡처와 같은 자리에
나온다 (얼굴 마스크 IoU: center 0.97, left/right 0.94).

거기서 카메라와 얼굴을 각각 `tvec` / `rvec` 으로 움직인다. 뜻은 OpenCV 와 같고
(`X' = R(rvec) X + tvec`) rvec 만 도 단위다.

| | 기준 좌표계 | 회전 중심 |
|---|---|---|
| 카메라 흔들림 | 촬영 위치의 카메라 좌표계 | 카메라 |
| 얼굴 움직임 | 월드(캡처) 좌표계 | 메쉬 중심 |

`무작위 배치` 는 성분 여섯 개를 각각 `±범위` 안에서 고르게 뽑는다. 기본 범위는 카메라
±5mm / ±2°, 얼굴 ±10mm / ±5° 이고 `무작위 범위` 에서 바꾼다. `촬영 위치도 뽑기` 를
켜면 세 자리 중 하나도 같이 뽑는다. 배경판은 촬영 위치의 카메라를 마주 보고 서며,
카메라가 흔들려도 따라 움직이지 않는다.

`3D 씬` 탭 맨 위의 `현재 위치에서 촬영` 은 지금 씬을 보고 있는 자리에 카메라를 놓고
찍은 뒤 `촬영 결과` 로 넘어간다. 촬영 위치는 `자유 시점 (3D 씬)` 이 되고 카메라 흔들림은
0 으로 돌아간다. 그 상태에서 흔들림·얼굴 값을 바꾸면 같은 자리에서 다시 찍는다.
세 촬영 위치 중 하나를 고르면 자유 시점에서 빠져나온다.

### 일괄 생성

오른쪽 맨 위 `일괄 생성` 버튼 하나로, 목록에 있는 **record 전부**에서 촬영 쌍과 GT 를
목표 개수만큼 만든다. 한 쌍은 `무작위 배치` 한 번과 같다 — 촬영 위치를 +50° / 0° / -50°
로 돌아가며 고르고, 카메라와 얼굴을 `무작위 범위` 안에서 흔들고, 배경을 바꾼다.
해상도·기선·배경 종류는 그때 화면에 있는 값을 쓴다.

| | |
|---|---|
| 목표 | 기본 **2,000쌍**. record 들에 고르게 나눈다 (71개면 record 당 28~29쌍) |
| 시드 | 같은 설정·같은 시드면 같은 쌍들이 나온다 (프로세스 수와 무관) |
| 출력 | `../dataset/flow/batch_<날짜_시각>/` |
| 용량 | 1/4 해상도에서 쌍당 약 8MB — 2,000쌍에 16GB. 시작 전에 남은 공간을 확인한다 |
| 속도 | 1/4 해상도, M5(10코어)에서 2,000쌍에 75초 (약 27쌍/초) |

기본값 2,000 은 사전학습된 RAFT / CREStereo 를 한 도메인에 맞출 때 쓰는 규모로 잡았다.
공개 파인튜닝 세트가 KITTI-2015 200쌍, Sintel 1,041쌍이고, 22,872쌍인 FlyingChairs 는
처음부터 가르치는 용도다.

```
batch_<날짜_시각>/
    batch.json      설정 (카메라, 범위, 시드, 검증용 record 목록)
    index.csv       쌍 목록 — split, 파일 경로, 촬영 위치, 유효 비율, flow 통계
    <record>/<번호>/img1.png img2.png flow.flo valid.png occluded.png meta.json
```

`index.csv` 의 `split` 은 `train` / `val` 이다. **record 단위로** 10% 를 검증용으로 뗀다 —
같은 얼굴이 양쪽에 들어가면 검증 점수가 부풀려지기 때문이다.

속도를 위해 한 것:

- record 단위로 프로세스에 나눠 주고, 프로세스마다 렌더러를 따로 둔다.
- flow 와 가려짐 판정을 **셰이더에서** 계산한다. CPU 계산(`gt.compute_flow`, 화면에서 한
  쌍씩 볼 때 쓰는 것)과 비교해 flow 차이 최대 0.0004px, 마스크 불일치는 화면당 1픽셀
  수준이다.
- 배경 무늬는 16가지를 한 번 만들어 두고, 쌍마다 판을 돌리고 밀어서 쓴다.

record 수에 비해 목표가 작으면 (예: 71개에 150쌍) 메쉬를 읽는 시간이 대부분이라 쌍당
속도는 느려진다.

### flow 그림

스테레오식 촬영은 flow 방향이 기선 쪽 하나뿐이라, 방향을 색으로 칠하는 색상환으로는
화면 전체가 한 색이 된다. 그래서 기본은 **크기**만 색으로 편다.

| flow 표시 | 색 범위 |
|---|---|
| 크기 — 얼굴 범위 (기본) | 얼굴 픽셀의 1~99 백분위. 얼굴의 굴곡이 가장 잘 보이고 배경은 한쪽 끝 색으로 눌린다 |
| 크기 — 전체 범위 | 화면 전체의 1~99 백분위 |
| 방향·크기 (색상환) | Middlebury 색상환. 물체가 움직이는 flow 를 볼 때 |

`색 오차` 는 0~5 계조로 편다 (유효 픽셀 평균이 0.3~0.5 계조다).

카메라는 디바이스 값(`config/intrinsic.yaml`, `stereo.yaml` 의 baseline 15mm)을 쓴다.
모든 조절값이 0 이면 디바이스가 실제로 찍던 배치다 — 렌더가 `center.jpg` /
`center_pair.jpg` 와 같은 자리에 나온다. 디바이스는 센서를 90° 돌려 달아 짝 카메라가
**세로로** 떨어져 있으므로 기본 기선 방향도 세로다. 가로 시차만 받는 모델에 넣을 때는
`기선 방향` 을 `가로` 로 바꾼다.

`이 쌍 저장` 은 `../dataset/flow/<record>/<번호>/` 에 쓴다:

```
img1.png img2.png   사진
flow.flo            img1 → img2 (Middlebury .flo, RAFT 가 읽는 형식)
valid.png           255 = 손실에 쓸 픽셀
occluded.png        255 = img2 에서 가려졌거나 화면 밖
depth.npy           img1 의 카메라 Z (mm)
meta.json           카메라·자세·기선·통계
```

알아 둘 것:

- flow 는 가려진 픽셀에도 값이 있다 (FlyingThings 와 같다). 손실에는 `valid.png` 를 쓴다.
- 가려짐 경계 1픽셀 폭은 `유효` 로 남는다 (판정을 3x3 이웃 깊이로 하기 때문).
- 사진은 4x MSAA 라 실루엣 픽셀은 앞뒤 색이 섞이지만 GT 는 픽셀 중심의 표면 하나를
  가리킨다. `색 오차` 에서 윤곽선이 밝게 보이는 이유다.
- 조명은 넣지 않는다. 아틀라스에 촬영 당시 조명이 구워져 있어 두 장의 밝기가 같다 —
  실제 스테레오 쌍에 있는 노출·반사 차이는 아직 없다.
- 아틀라스가 덮지 못한 곳(목·귀·머리카락)은 메쉬에 없다.
- 렌더러는 OpenGL 3.3 이 필요하다 (moderngl standalone 컨텍스트).

## 파인튜닝

`파인튜닝` 탭은 일괄 생성한 데이터로 **RAFT-Stereo** 와 **CREStereo** 를 이어 학습시키고,
끝나면 사전학습 그대로(기존)와 나란히 비교한다.

준비 — 모델 소스와 사전학습 가중치 (약 180MB, 서브모듈 안에 받는다):

```
cd thirdparty/depth
.venv/bin/python tools/setup_stereo_models.py --models raft crestereo
```

1. 데이터셋(일괄 생성 폴더)과 모델을 고르고 `파인튜닝 시작`.
2. `학습 곡선` 에 학습 EPE 와 검증 EPE 가 그려진다. 검증의 0 스텝이 기존 모델이다.
3. 학습이 끝나면 자동으로 비교가 돌고 `비교` 탭이 채워진다.

학습은 depth 서브모듈의 venv 에서 `ofgym/train/worker.py` 가 한다 (UI 와 torch 를 한
인터프리터에 섞지 않는다). 결과는 `runs/<모델>_<날짜_시각>/` 에 남고, `지난 실행` 목록에서
다시 열 수 있다.

```
runs/<모델>_<날짜_시각>/
    config.json            설정
    log.jsonl              학습·검증 기록
    checkpoint_best.pth    검증 EPE 가 가장 낮았던 가중치
    checkpoint_last.pth    마지막 가중치
    compare/metrics.json   기존 대 튜닝 지표
    compare/<번호>/        검증 쌍별 사진·정답·두 예측
    compare/real_<이름>/   실제 촬영 쌍의 두 예측
```

체크포인트는 `state_dict` 그대로라 depth 의 `config/stereo.yaml` 에서
`matcher.<backend>.checkpoint` 로 가리키면 디바이스 파이프라인이 읽는다.

### 데이터가 모델에 들어가는 모양

두 모델 모두 **가로 시차**만 받는다. 기선이 세로(디바이스)인 데이터는 두 장을 시계 방향
90° 돌려 넣는다 — 디바이스가 매칭 전에 하는 것과 같다. 시차는 돌린 flow 의 v 성분이다.

학습 때는 크롭한 뒤 **두 장에 서로 다른** 밝기·대비·감마·채널 이득·잡음·흐림을 걸고,
절반은 오른쪽 사진을 세로로 ±1px 민다. 합성 쌍은 두 장의 색이 완전히 같고 정렬도
완벽해서, 그대로 가르치면 실제 카메라 두 대의 차이를 본 적 없는 모델이 된다.

### 비교

| | |
|---|---|
| 지표 | EPE(평균 픽셀 오차), 1px 초과 비율, 3px 초과 비율 — 낮을수록 좋다 |
| 영역 | 전체 / 얼굴 / 배경. 유효(`valid.png`) 픽셀만 센다 |
| 대상 | 검증 split — 학습에 쓰지 않은 record 들 |
| 그림 | 왼쪽 사진, 정답, 두 모델의 예측과 오차(0~3px) |
| 실제 촬영 | 검증 record 의 `center.jpg` / `center_pair.jpg` 등. **정답이 없어** 두 예측과 그 차이만 본다 |

읽을 때 주의:

- **지표는 합성 데이터에서 잰 것이다.** 튜닝한 모델이 합성 검증에서 좋아졌다고 실제
  촬영에서도 좋아진다는 보장은 없다. 실제 촬영 비교는 눈으로 보는 용도다.
- 실제 촬영 쌍은 보정(rectify) 없이 크기만 맞춰 넣는다. 디바이스 파이프라인과 같지 않다.
- 사전학습 모델이 이 합성 데이터에서 이미 EPE 0.1px 아래다. 무늬가 뚜렷하고 조명 차이가
  없는 쉬운 입력이라 그렇다 — 올라갈 여지가 작다는 뜻이기도 하다.

### 기본 설정과 시간

기본값은 메모리 16GB 맥(MPS)에서 돌아가는 크기로 잡았다.

| | 기본 | |
|---|---|---|
| 학습 길이 | 1,000 스텝 | RAFT-Stereo 스텝당 약 1.5초, CREStereo 는 1.5~2배 |
| 배치 / 크롭 | 1 / 320x448 | 메모리 5GB. 배치 2 는 10GB, 384x512 · 배치 2 는 13.5GB 로 넘쳐 스텝당 13초 |
| 검증 | 250 스텝마다 16쌍 | 760x1008 한 장에 약 4초 |
| 비교 | 60쌍 × 모델 둘 | |

RAFT-Stereo 로 전체 40분쯤 걸린다. CUDA 가 있는 장비라면 배치·크롭·스텝을 키우는 편이 낫다.

## 데이터 흐름

1. **캡처** `../dataset/raw/<세션>/<샘플>/` — 뷰(left/center/right)마다 전체 해상도
   `*.jpg`, 스테레오 짝 `*_pair.jpg`, ROI 컬러 `*_color.jpg`, 디바이스가 만든 정점맵
   `*.npy`, 각종 마스크 PNG.
2. **3D 생성** `../dataset/recon/<세션>/<샘플>/` — UI 의 `3D 생성` 탭. 캡처를 make3D
   규약으로 심링크 스테이징한 뒤 `thirdparty/depth/example/make3D_ui_log.py` 를 그대로
   돌린다.

   `Reconstruction.run()` 의 `merge.method != 'roi'` 경로(기본 `incidence`)를 타며,
   `stereo/pointmerge.py` 의 후처리가 그 안에서 전부 돈다:

   ```
   스테레오(sgbm/crestereo) → _alignView(캘리브+지령각+pose T_delta)
     → removeSpikes → midlineCut → centerAlign(2 iters)
     → consensusFilter → colorConsensusFilter → incidenceCull
     → centerAzimuthGate(center 전용) → gainField(뷰간 색 이득)
     → 입사각 가중 투영 → outlier 제거
     → _upsampleViews(6x)
   ```

   ROI 하드 컷(`setROIByParts`)과 시임 블렌딩(`postProcessByResample`)은 이 경로에서
   **의도적으로** 빠진다 — 경계맵을 쓰지 않는 병합이라서다.

   ```
   recon/<세션>/<샘플>/image/     make3D 이름 규약으로 건 심링크
   recon/<세션>/<샘플>/model/     {left,center,right}_vertex.npy  ← 병합된 3D 격자
                                  {…}_blend.npy / .png            ← 블렌딩된 색
                                  {…}_facemask.png / _mask.png
                                  render.png                       ← Open3D 정면 렌더
   ```
3. **Disparity GT** 병합 3D 를 스테레오 페어 카메라로 재투영 → `../dataset/gt/`. (미구현)
4. **파인튜닝** `파인튜닝` 탭 — 위 '파인튜닝' 참고.

## 준비

```
git clone git@github.com:dydgml0360/optical_flow.git optical_flow_gym
cd optical_flow_gym
git submodule update --init                                   # --recursive 쓰지 말 것 (아래)
git -C thirdparty/depth submodule update --init external/ml   # u2net onnx 가중치
uv sync                             # UI (python 3.10 + PySide6)
uv sync --project thirdparty/depth  # 재구성용 depth venv (open3d/torch/onnxruntime)
```

- `--recursive` / `--recurse-submodules` 는 **실패한다.** upstream depth 에
  `external/stereo_models/FoundationStereo` 가 `.gitmodules` 항목 없이 gitlink 로만 남아
  있어 `No url found for submodule path` 로 중단되고, 그 바람에 `external/ml` 도
  체크아웃되지 않는다. 위처럼 두 단계로 나눠 받는다.
- 서브모듈은 비공개(`RejuvenorTeam/DepthEstimation`, `RejuvenorTeam/ml`)라 그쪽 접근
  권한이 있는 SSH 키가 필요하다.
- 데이터셋은 저장소에 없다. `../dataset/raw/` 를 따로 옮겨 놓거나 `OFGYM_DATASET` 으로
  위치를 지정한다. 없어도 UI 는 뜬다 (목록이 빌 뿐).
- uv 는 `exclude-newer = "P3D"` (상대 기간) 를 읽을 수 있는 버전이어야 한다. 0.12.7 에서
  확인.

`3D 생성` 은 `thirdparty/depth/.venv` 를 서브프로세스로 부른다 — UI 와 재구성의
의존성(PySide6 ↔ open3d 0.18/torch)을 한 인터프리터에 섞지 않기 위해서다.

## 업샘플이 기본값이 아닌 이유 (중요)

`config/stereo.yaml` 의 `upsample`(enabled/method=interpolate/scale=6)은 **`UpsampleWorker`
만** 읽고, 그 워커는 `setUpsampleCallback()` 으로 콜백이 등록돼야 돈다. `make3D.py` 도
UI 앱(`ToningReacher_QT`)도 등록하지 않으므로 그 경로는 항상
`[Upsample] 건너뜀 — 콜백 미등록` 이다. 디바이스가 `data/<pid>/*.npy` 로 남기는 vertex 도
1/decimation(=1/6) 격자인 이유가 이것이다.

같은 일을 하는 인라인 경로가 `make3D.py --upsample N` 이고 (`UpsampleWorker._work` 도
결국 `rec.run(upsample=scale)` 를 부른다), **CLI 기본값은 1** 이다. 그래서 `ReconOptions`
는 make3D 기본값 대신 stereo.yaml 의 `upsample.scale` 을 읽어 기본값으로 쓴다.

⚠️ 인라인 경로는 `interpolate`(보간)만 된다. `rematch`(풀해상도 스테레오 재매칭)는
`UpsampleWorker` 가 `Reconstruction._upsamplerFactory` 를 주입해야 하므로 make3D 로는
못 쓴다. **6x 보간은 새 관측이 아니라 1/6 격자의 보간**이라, disparity GT 를 여기서
뽑으면 경계가 뭉개진 GT 가 된다 — GT 생성 단계에서 `rematch` 를 쓰려면 콜백을 등록하는
자체 드라이버가 필요하다.

## 로그가 안 보이면

`pytoningreacher` 는 모듈마다 `from utils.debug_mode import dprint as print` 로 print 를
가려 놨다. `PYTR_DEBUG` 가 꺼져 있으면 **병합·후처리 로그가 한 줄도 안 나온다** (동작은
한다). `ReconOptions.debug=True` (UI 기본)가 `PYTR_DEBUG=1` 을 넣어 준다.
켜면 `thirdparty/depth/logs/<시각>/` 세션 디렉터리도 생긴다 (depth 의 .gitignore 대상).

## 알려진 제약

- `config/stereo.yaml` 의 `matcher.backend` 기본값은 `crestereo` 지만 서브모듈 클론에는
  `external/stereo_models/` 소스가 따라오지 않아(upstream `.gitignore` 대상) **sgbm 으로
  폴백**한다. 학습 백엔드를 쓰려면:
  `thirdparty/depth/.venv/bin/python tools/setup_stereo_models.py --models crestereo raft`
- `external/ml/model/onnx_model.py` 는 upstream 에 커밋된 적이 없다 —
  `thirdparty/patches/` 참고.
