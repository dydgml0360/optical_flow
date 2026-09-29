"""레이어 파일 하나를 화면에 띄울 QImage 로 바꾼다.

원본 JPG 가 3040x4032 라 그대로 디코딩하면 프리뷰가 버벅인다.
QImageReader 의 scaled-read 로 긴 변을 MAX_EDGE 로 줄여 읽는다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage, QImageReader

MAX_EDGE = 1600


def _numpy_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    # QImage 는 버퍼를 복사하지 않으므로 copy() 로 수명을 끊어 준다.
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _vertex_to_qimage(path: Path) -> tuple[QImage, str]:
    """정점맵(H,W,3 float) 의 Z 를 컬러맵으로 그린다."""

    vertex = np.load(path)
    if vertex.ndim != 3 or vertex.shape[2] != 3:
        raise ValueError(f"정점맵 모양이 예상과 다릅니다: {vertex.shape}")

    z = vertex[..., 2].astype(np.float32)
    valid = np.isfinite(z) & (z != 0)
    if not valid.any():
        raise ValueError("유효한 정점이 없습니다")

    lo, hi = np.percentile(z[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.zeros(z.shape, np.uint8)
    norm[valid] = np.clip((z[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0  # 무효 픽셀은 검게
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    info = (
        f"{vertex.shape[1]}x{vertex.shape[0]}  float{vertex.dtype.itemsize * 8}  "
        f"유효 {valid.mean() * 100:.1f}%  Z {lo:.1f}~{hi:.1f}mm"
    )
    return _numpy_to_qimage(rgb), info


def load_layer(path: Path) -> tuple[QImage, str]:
    """(이미지, 한 줄 정보) 를 반환한다. 실패하면 예외를 올린다."""

    if path.suffix == ".npy":
        return _vertex_to_qimage(path)

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    info = f"{size.width()}x{size.height()}  {path.suffix.lstrip('.').upper()}"

    longest = max(size.width(), size.height())
    if longest > MAX_EDGE:
        scale = MAX_EDGE / longest
        reader.setScaledSize(
            QSize(max(1, int(size.width() * scale)), max(1, int(size.height() * scale)))
        )
        info += f"  (프리뷰 {scale * 100:.0f}%)"

    image = reader.read()
    if image.isNull():
        raise ValueError(reader.errorString() or "이미지를 읽지 못했습니다")
    return image, info
