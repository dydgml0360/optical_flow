"""오프스크린 래스터라이저 (moderngl).

한 번 찍으면 두 가지가 나온다.

- **색** — 4x MSAA 로 그려 내려받은 사진. 조명 없이 텍스처 색 그대로다 (아틀라스에
  촬영 당시 조명이 이미 구워져 있다).
- **기하** — 픽셀 중심이 본 표면점의 **물체 좌표**와 물체 번호 (RGBA32F). MSAA 없이
  그린다 — 경계에서 앞뒤 물체의 좌표가 섞이면 GT 가 허공을 가리키게 된다.

물체 좌표를 남기는 이유: 두 번째 프레임에서 그 점이 어디로 갔는지는
`카메라2 · 모델2[번호] · 물체좌표` 로 정확히 계산된다. 카메라만 움직이든 물체가
움직이든 같은 식이고, 렌더 두 장을 맞춰 보는 추정이 끼지 않는다.

창을 띄우지 않는 standalone 컨텍스트라 Qt 의 그리기와 얽히지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import moderngl
import numpy as np

from ofgym.flow.camera import Camera
from ofgym.flow.mesh import Mesh

NEAR_MM = 10.0
FAR_MM = 20000.0
BACKGROUND_COLOR = (23, 22, 26)

_VERTEX = """
#version 330
uniform mat4 u_mvp;
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_local;
void main() {
    v_uv = in_uv;
    v_local = in_position;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_COLOR = """
#version 330
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(u_texture, v_uv).rgb, 1.0);
}
"""

_FRAGMENT_GEOMETRY = """
#version 330
uniform float u_id;
in vec3 v_local;
out vec4 f_geometry;
void main() {
    f_geometry = vec4(v_local, u_id);
}
"""

_VERTEX_LINE = """
#version 330
uniform mat4 u_mvp;
in vec3 in_position;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_LINE = """
#version 330
uniform vec3 u_color;
out vec4 f_color;
void main() {
    f_color = vec4(u_color, 1.0);
}
"""


@dataclass
class Frame:
    """한 번 찍은 결과."""

    color: np.ndarray  # (H,W,3) uint8 RGB
    local: np.ndarray  # (H,W,3) float32 — 물체 좌표. 빈 픽셀은 0
    ids: np.ndarray  # (H,W) uint8 — 물체 번호, 0 은 아무것도 없음
    camera: Camera
    extrinsic: np.ndarray  # world → camera
    models: Dict[int, np.ndarray]  # 물체 번호 → object → world

    def camera_points(self) -> np.ndarray:
        """픽셀마다 카메라 좌표 (H,W,3) float64. 빈 픽셀은 0."""
        return self.points_in(self.extrinsic, self.models)

    def points_in(self, extrinsic: np.ndarray, models: Dict[int, np.ndarray]) -> np.ndarray:
        """이 프레임의 표면점들을 다른 시점·다른 물체 자세에서 본 카메라 좌표."""
        out = np.zeros(self.local.shape, np.float64)
        x, y, z = (self.local[..., i].astype(np.float64) for i in range(3))
        for object_id, model in models.items():
            mask = self.ids == object_id
            if not mask.any():
                continue
            matrix = extrinsic @ model
            # 성분별로 곱한다 — (N,3)@(3,3) 은 이 크기에서 BLAS 호출 비용이 더 크다.
            for row in range(3):
                a, b, c, t = matrix[row]
                np.copyto(out[..., row], a * x + b * y + c * z + t, where=mask)
        return out

    def depth(self) -> np.ndarray:
        """카메라 Z (mm). 빈 픽셀은 inf."""
        z = self.camera_points()[..., 2]
        return np.where(self.ids > 0, z, np.inf)


@dataclass
class _Object:
    object_id: int
    color_vao: moderngl.VertexArray
    geometry_vao: moderngl.VertexArray
    texture: moderngl.Texture
    resources: list


class Renderer:
    def __init__(self) -> None:
        self._ctx = moderngl.create_standalone_context(require=330)
        self._color = self._ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT_COLOR)
        self._geometry = self._ctx.program(
            vertex_shader=_VERTEX, fragment_shader=_FRAGMENT_GEOMETRY
        )
        self._line = self._ctx.program(
            vertex_shader=_VERTEX_LINE, fragment_shader=_FRAGMENT_LINE
        )
        self._objects: Dict[int, _Object] = {}
        self._targets: Optional[tuple] = None
        self._target_size: Tuple[int, int] = (0, 0)
        self._samples = min(4, self._ctx.max_samples)

    @property
    def description(self) -> str:
        info = self._ctx.info
        return f"{info.get('GL_RENDERER', '?')} / OpenGL {info.get('GL_VERSION', '?')}"

    # ── 물체 ──────────────────────────────────────────────────────────────
    def set_object(
        self,
        object_id: int,
        vertices: np.ndarray,
        uvs: np.ndarray,
        faces: np.ndarray,
        texture: np.ndarray,
        repeat: bool = False,
    ) -> None:
        """번호 `object_id`(1~255) 자리에 물체를 올린다. 있던 것은 갈아 끼운다."""
        self.remove_object(object_id)
        ctx = self._ctx

        data = np.hstack([vertices.astype("f4"), uvs.astype("f4")])
        vbo = ctx.buffer(np.ascontiguousarray(data).tobytes())
        ibo = ctx.buffer(np.ascontiguousarray(faces.astype("i4")).tobytes())

        height, width = texture.shape[:2]
        # 행 길이가 4의 배수가 아닌 RGB 텍스처도 그대로 올라가게 정렬을 1로 둔다.
        tex = ctx.texture((width, height), 3, np.ascontiguousarray(texture).tobytes(),
                          alignment=1)
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 8.0
        tex.repeat_x = tex.repeat_y = repeat

        layout = [(vbo, "3f 2f", "in_position", "in_uv")]
        color_vao = ctx.vertex_array(self._color, layout, index_buffer=ibo)
        geometry_vao = ctx.vertex_array(
            self._geometry, [(vbo, "3f 8x", "in_position")], index_buffer=ibo
        )
        self._objects[object_id] = _Object(
            object_id, color_vao, geometry_vao, tex, [vbo, ibo, tex, color_vao, geometry_vao]
        )

    def set_mesh(self, object_id: int, mesh: Mesh) -> None:
        self.set_object(object_id, mesh.vertices, mesh.uvs, mesh.faces, mesh.texture)

    def remove_object(self, object_id: int) -> None:
        old = self._objects.pop(object_id, None)
        if old is not None:
            for resource in old.resources:
                resource.release()

    def has_object(self, object_id: int) -> bool:
        return object_id in self._objects

    # ── 렌더 타깃 ─────────────────────────────────────────────────────────
    def _ensure_targets(self, size: Tuple[int, int]) -> tuple:
        if self._targets is not None and self._target_size == size:
            return self._targets
        if self._targets is not None:
            for resource in self._targets[-1]:
                resource.release()
        ctx = self._ctx

        ms_color = ctx.renderbuffer(size, 4, samples=self._samples)
        ms_depth = ctx.depth_renderbuffer(size, samples=self._samples)
        multisampled = ctx.framebuffer([ms_color], ms_depth)

        resolved_color = ctx.renderbuffer(size, 4)
        resolved = ctx.framebuffer([resolved_color])

        geometry_color = ctx.renderbuffer(size, 4, dtype="f4")
        geometry_depth = ctx.depth_renderbuffer(size)
        geometry = ctx.framebuffer([geometry_color], geometry_depth)

        resources = [multisampled, ms_color, ms_depth, resolved, resolved_color,
                     geometry, geometry_color, geometry_depth]
        self._targets = (multisampled, resolved, geometry, resources)
        self._target_size = size
        return self._targets

    # ── 그리기 ────────────────────────────────────────────────────────────
    def render(
        self,
        camera: Camera,
        extrinsic: np.ndarray,
        models: Dict[int, np.ndarray],
        background: Sequence[int] = BACKGROUND_COLOR,
        lines: Optional[List[Tuple[np.ndarray, Sequence[float]]]] = None,
        geometry: bool = True,
    ) -> Frame:
        """`models` 에 있는 물체만 그린다. `lines` 는 (선분들 (K,2,3), RGB 0~1) 목록 — 색에만 들어간다."""
        ctx = self._ctx
        size = (camera.width, camera.height)
        multisampled, resolved, geometry_fbo, _ = self._ensure_targets(size)

        view_projection = camera.gl_projection(NEAR_MM, FAR_MM) @ extrinsic
        drawn = [(self._objects[i], models[i]) for i in sorted(models) if i in self._objects]

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.BLEND)

        multisampled.use()
        multisampled.clear(*(c / 255.0 for c in background), 1.0)
        for obj, model in drawn:
            self._color["u_mvp"].write(self._matrix(view_projection @ model))
            obj.texture.use(0)
            self._color["u_texture"].value = 0
            obj.color_vao.render(moderngl.TRIANGLES)
        if lines:
            self._draw_lines(view_projection, lines)
        ctx.copy_framebuffer(resolved, multisampled)

        height, width = camera.height, camera.width
        color = np.frombuffer(
            resolved.read(components=3, alignment=1), np.uint8
        ).reshape(height, width, 3).copy()

        if geometry:
            geometry_fbo.use()
            geometry_fbo.clear(0.0, 0.0, 0.0, 0.0)
            for obj, model in drawn:
                self._geometry["u_mvp"].write(self._matrix(view_projection @ model))
                self._geometry["u_id"].value = float(obj.object_id)
                obj.geometry_vao.render(moderngl.TRIANGLES)
            raw = np.frombuffer(
                geometry_fbo.read(components=4, dtype="f4", alignment=1), np.float32
            ).reshape(height, width, 4)
            local = raw[..., :3].copy()
            ids = np.rint(raw[..., 3]).astype(np.uint8)
        else:
            local = np.zeros((height, width, 3), np.float32)
            ids = np.zeros((height, width), np.uint8)

        return Frame(
            color=color,
            local=local,
            ids=ids,
            camera=camera,
            extrinsic=np.array(extrinsic, np.float64),
            models={i: np.array(m, np.float64) for i, m in models.items()},
        )

    def _draw_lines(self, view_projection: np.ndarray, lines) -> None:
        self._line["u_mvp"].write(self._matrix(view_projection))
        for segments, color in lines:
            segments = np.asarray(segments, "f4").reshape(-1, 3)
            if not len(segments):
                continue
            vbo = self._ctx.buffer(segments.tobytes())
            vao = self._ctx.vertex_array(self._line, [(vbo, "3f", "in_position")])
            self._line["u_color"].value = tuple(float(c) for c in color)
            vao.render(moderngl.LINES)
            vao.release()
            vbo.release()

    @staticmethod
    def _matrix(matrix: np.ndarray) -> bytes:
        # GLSL 은 열 우선이다.
        return np.ascontiguousarray(matrix.T.astype("f4")).tobytes()

    def release(self) -> None:
        for object_id in list(self._objects):
            self.remove_object(object_id)
        if self._targets is not None:
            for resource in self._targets[-1]:
                resource.release()
            self._targets = None
        self._ctx.release()
