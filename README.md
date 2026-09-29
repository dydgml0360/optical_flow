# optical_flow_gym

CREStereo / RAFT-Stereo 계열 모델을 자체 캡처 데이터로 파인튜닝하기 위한 워크벤치.

## 구성

- `ofgym/` — PySide6 데스크톱 UI + 데이터셋/GT 파이프라인
- `thirdparty/depth` — DepthEstimation 서브모듈. 캘리브레이션·스테레오·3D 재구성 코드를 여기서 가져다 쓴다.
- `../dataset/` — 데이터 저장소(저장소 밖). `raw/<세션>/<샘플UUID>/` 구조.

## 실행

```
uv run main
```

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
uv sync                             # UI (python 3.10 + PySide6)
uv sync --project thirdparty/depth  # 재구성용 depth venv (open3d/torch/onnxruntime)
```

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
