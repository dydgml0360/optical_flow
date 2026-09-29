import json
import os
import numpy as np

import onnxruntime as ort

from lib.result import Result


class OnnxModel:
    """ONNX Runtime inference — TrtModel과 동일 인터페이스 (macOS 등 TensorRT 부재 환경용).

    ``{onnx_path basename}.meta.json`` sidecar 를 읽어 Result.class_ids 를 채운다.
    sidecar 가 없으면 class_ids 는 채널 인덱스 기본값을 따른다.
    """

    def __init__(self, onnx_path, meta_path=None, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Engine input (H, W) — meta sidecar 가 없을 때의 fallback.
        _ishape = self.session.get_inputs()[0].shape  # (N, 3, H, W)
        self.input_hw = (int(_ishape[2]), int(_ishape[3]))

        if meta_path is None:
            meta_path = os.path.splitext(onnx_path)[0] + '.meta.json'
        self.class_ids = None
        self.class_names = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.class_ids = meta.get('class_ids')
            self.class_names = meta.get('class_names')
            if meta.get('height') and meta.get('width'):
                self.input_hw = (int(meta['height']), int(meta['width']))
            if self.class_names:
                print(f"OnnxModel loaded class_names: {self.class_names}")
        else:
            print(f"OnnxModel: no metadata sidecar at {meta_path}, "
                  f"class_ids will default to channel index")

    def inference(self, image):
        """image: BGR image (HxWxC) numpy array, already resized to engine input size."""
        img = image[:, :, ::-1].astype(np.float32) / 255.0
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])

        output = self.session.run(None, {self.input_name: x})[0]
        # sigmoid (TrtModel 과 동일) — clip 은 exp overflow 경고 방지용 (결과 동일)
        pred = 1.0 / (1.0 + np.exp(-np.clip(output[0], -88.0, 88.0)))

        return Result(image, pred, class_ids=self.class_ids, class_names=self.class_names)
