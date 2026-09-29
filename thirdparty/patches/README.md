# thirdparty 패치

서브모듈 upstream 에 **커밋되지 않은** 파일들. 클론만으로는 없어서 실행이 깨지므로
여기 사본을 두고 `ofgym.recon.runner.ensure_patches()` 가 없을 때만 복사해 넣는다.

| 파일 | 들어가는 자리 | 왜 |
|---|---|---|
| `ml/model/onnx_model.py` | `thirdparty/depth/external/ml/model/onnx_model.py` | mac 에는 TensorRT 가 없어 `make3D.py --model u2net` 이 이 ONNX 백엔드를 쓴다. `RejuvenorTeam/ml` 작업본에는 있으나 커밋된 적이 없다 (`git status` 에서 untracked). upstream 에 커밋되면 이 패치는 지워도 된다. |
