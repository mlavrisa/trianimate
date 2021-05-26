from contextlib import contextmanager
from typing import Callable

import moderngl
import numpy as np


class TriangleShader:
    def __init__(self):
        """TriangleShader renders triangulations using OpenGL.
        
        Please see render_2d and render_3d"""
        self._ctx: moderngl.Context = moderngl.create_standalone_context()

        self._width = -1
        self._height = -1

        self._vbo = None
        self._vao = None
        self._rbo_msaa = None
        self._rbo_ds = None
        self._fbo_msaa = None
        self._fbo_ds = None

        self.vert_shader_2d = ""
        self.frag_shader_2d = ""

        with open("trianimate/shaders/2d_shader.vert", "r") as f:
            self.vert_shader_2d = f.read()

        with open("trianimate/shaders/2d_shader.frag", "r") as f:
            self.frag_shader_2d = f.read()

    def _render_2d(
        self, vertices: np.ndarray, faces: np.ndarray, colours: np.ndarray
    ) -> np.ndarray:
        """Function for 2D rendering: with TriangleShader().render_2d(w, h) as render:

        Do not use outside of the appropriate context, to prevent memory leakage.

        Args:
            vertices (`np.ndarary`, (Nv, 2), `float`[0., 1.])
                2D points specifying the vertices of the triangles, in the range 0 to 1
            faces (`np.ndarary`, (Nf, 3), `int32`[0, Nv - 1])
                each row is a face, with indices referring to the above vertices making
                up that face
            colours (`np.ndarary`, (Nf OR Nv, 3), `uint8`[0, 255])
                each face gets a colour, specified as 3 colour channel values, 0 to 255

        Returns:
            rendered image, `np.ndarray`
        """
        assert self._width > 0 and self._height > 0
        assert self._rbo_msaa is not None
        assert vertices.shape[1] == 2
        assert faces.shape[1] == 3
        assert colours.shape[1] == 3
        assert (
            colours.shape[0] == vertices.shape[0] or colours.shape[0] == faces.shape[0]
        )

        program = self._ctx.program(
            vertex_shader=self.vert_shader_2d, fragment_shader=self.frag_shader_2d,
        )

        xy = vertices[faces.ravel()] * 2.0 - 1.0
        if colours.shape[0] == faces.shape[0]:
            cols = np.repeat(colours / 255.0, 3, axis=0)
        else:
            cols = colours[faces.ravel()] / 255.0
        vertices = np.concatenate([xy, cols], axis=1)

        self._vbo = self._ctx.buffer(vertices.astype("f4").tobytes())
        self._vao = self._ctx.simple_vertex_array(
            program, self._vbo, "in_vert", "in_color"
        )

        self._fbo_msaa.use()
        self._fbo_msaa.clear(0.0, 0.0, 0.0, 1.0)
        self._vao.render()

        self._ctx.copy_framebuffer(self._fbo_ds, self._fbo_msaa)
        frame_bytes = self._fbo_ds.read(components=3, alignment=1)

        # haven't actually tested, but releasing these now
        # vertices might be a different size next frame
        self._vbo.release()
        self._vao.release()
        program.release()

        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )

    @contextmanager
    def render_2d(self, width: int, height: int) -> Callable:
        """Context for image generation for 2D point data.
        
        Used for an image or sequence of images. Yields the 2d render function, which
        is called with arguments:
            vertices (`np.ndarary`, (Nv, 2), `float`[0., 1.])
                2D points specifying the vertices of the triangles, in the range 0 to 1
            faces (`np.ndarary`, (Nf, 3), `int32`[0, Nv - 1])
                each row is a face, with indices referring to the above vertices making
                up that face
            colours (`np.ndarary`, (Nf, 3), `uint8`[0, 255])
                each face gets a colour, specified as 3 colour channel values, 0 to 255

        Args:
            width: `int` width of the resulting image in pixels
            height: `int` height of the resulting image in pixels
        
        Typical usage examples:
            with TriangleShader().render_2d(640, 480) as render:
                img = render(vertices, faces, colours)
            
            frames = np.zeros((n_frames, 640, 480, 3))
            with TriangleShader().render_2d(640, 480) as render:
                for fdx in range(n_frames):
                    frames[fdx] = render(vertices[fdx], faces[fdx], colours[fdx])
        """
        assert width > 0 and height > 0
        self._width = width
        self._height = height
        self._rbo_msaa = self._ctx.renderbuffer((width, height), samples=8)
        self._rbo_ds = self._ctx.renderbuffer((width, height))
        self._fbo_msaa = self._ctx.framebuffer(self._rbo_msaa)
        self._fbo_ds = self._ctx.framebuffer(self._rbo_ds)

        try:
            yield self._render_2d
        finally:
            self._width = -1
            self._height = -1
            self._rbo_msaa.release()
            self._rbo_ds.release()
            self._fbo_msaa.release()
            self._fbo_ds.release()
            self._ctx.release()
            self._vbo = None
            self._vao = None
            self._rbo_msaa = None
            self._rbo_ds = None
            self._fbo_msaa = None
            self._fbo_ds = None
            self._ctx = None
