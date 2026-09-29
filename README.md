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
4. **파인튜닝** 생성된 GT 로 CREStereo / RAFT-Stereo 학습. (미구현)

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
