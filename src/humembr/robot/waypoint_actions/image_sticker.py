import io
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    get_a_tform_b,
    get_vision_tform_body,
)
from OpenGL.GL import (
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_RGB,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    glActiveTexture,
    glBegin,
    glBindFramebuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glClear,
    glClearColor,
    glDeleteFramebuffers,
    glDeleteTextures,
    glEnable,
    glEnd,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    glGetUniformLocation,
    glLoadIdentity,
    glMatrixMode,
    glReadPixels,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glUniform1i,
    glUniformMatrix4fv,
    glUseProgram,
    glVertex3fv,
    glViewport,
    shaders,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from PIL import Image


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-vector kann nicht normalisiert werden.")
    return v / n


def _np_vec(proto_vec) -> np.ndarray:
    return np.array([proto_vec.x, proto_vec.y, proto_vec.z])


def _mat4mul3(mat4: np.ndarray, vec3: np.ndarray, w: float = 1.0) -> np.ndarray:
    """4×4 Matrix ⋅ 3D-Vektor → 3D-Vektor"""
    return (mat4 @ np.append(vec3, w))[:3]


def proto_vec_T_numpy(vec):
    return np.array([vec.x, vec.y, vec.z])


def mat4mul3(mat, vec, vec4=1):
    ret = np.matmul(mat, np.append(vec, vec4))
    return ret[:-1]


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("norm function returned 0.")
    return vec / norm


class ImagePrepped:
    """JPEG-Bytes → RGB-Array + Model-View-Projection-Matrix."""

    def __init__(self, img_resp: image_pb2.ImageResponse):  # type: ignore
        if img_resp.shot.image.format != image_pb2.Image.FORMAT_JPEG:  # type: ignore
            raise ValueError("Nur JPEG unterstützt (ursprünglicher RAW-Pfad entfernt).")

        self.image = np.asarray(
            Image.open(io.BytesIO(img_resp.shot.image.data)).convert("RGB")
        )

        self.vision_T_body = get_vision_tform_body(img_resp.shot.transforms_snapshot)

        self.body_T_sensor = get_a_tform_b(
            img_resp.shot.transforms_snapshot,
            BODY_FRAME_NAME,
            img_resp.shot.frame_name_image_sensor,
        )
        self.vision_T_body = get_vision_tform_body(img_resp.shot.transforms_snapshot)

        if not self.body_T_sensor:
            raise RuntimeError("Transform BODY→SENSOR fehlt.")

        src = img_resp.source
        res = np.array([src.cols, src.rows])
        f = np.array(
            [
                src.pinhole.intrinsics.focal_length.x,
                src.pinhole.intrinsics.focal_length.y,
            ]
        )
        pp = np.array(
            [
                src.pinhole.intrinsics.principal_point.x,
                src.pinhole.intrinsics.principal_point.y,
            ]
        )

        sensor_T_vo = (self.vision_T_body * self.body_T_sensor).inverse()  # type: ignore

        proj = np.eye(4)
        proj[0, 0], proj[0, 2] = f[0] / res[0], pp[0] / res[0]
        proj[1, 1], proj[1, 2] = f[1] / res[1], pp[1] / res[1]

        self.mvp = proj @ sensor_T_vo.to_matrix()


class GLTexture:
    """Erzeugt/aktualisiert GL_TEXTURE_2D für RGB-Daten."""

    def __init__(self, img: np.ndarray):
        glEnable(GL_TEXTURE_2D)
        self.handle = glGenTextures(1)
        self._upload(img)

    @contextmanager
    def bound(self):
        glBindTexture(GL_TEXTURE_2D, self.handle)
        try:
            yield
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)

    def update(self, img: np.ndarray):
        with self.bound():
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                img.shape[1],
                img.shape[0],
                GL_RGB,
                GL_UNSIGNED_BYTE,
                img,
            )

    def _upload(self, img: np.ndarray):
        with self.bound():
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                img.shape[1],
                img.shape[0],
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                img,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


class StitchCamera:
    """Positioniert virtuelle Kamera mittig zwischen den Sensoren (SDK-Stil!)."""

    def __init__(self, img1: "ImagePrepped", img2: "ImagePrepped"):
        rect_stitching_distance_meters = 2.0

        vo_T_body = img1.vision_T_body.to_matrix()  # type: ignore

        eye_wrt_body = proto_vec_T_numpy(
            img1.body_T_sensor.position  # type: ignore
        ) + proto_vec_T_numpy(img2.body_T_sensor.position)  # type: ignore
        eye_norm_wrt_body = np.array(
            img1.body_T_sensor.rot.transform_point(0, 0, 1)  # type: ignore
        ) + np.array(img2.body_T_sensor.rot.transform_point(0, 0, 1))  # type: ignore

        eye_wrt_body[1] = 0
        eye_norm_wrt_body[1] = 0

        eye_norm_wrt_body = normalize(eye_norm_wrt_body)

        plane_wrt_body = (
            eye_wrt_body + eye_norm_wrt_body * rect_stitching_distance_meters
        )

        self.plane_wrt_vo = mat4mul3(vo_T_body, plane_wrt_body)
        self.plane_norm_wrt_vo = mat4mul3(vo_T_body, eye_norm_wrt_body, 0)

        self.eye_wrt_vo = mat4mul3(vo_T_body, eye_wrt_body)
        self.up_wrt_vo = mat4mul3(vo_T_body, np.array([0, 0, 1]), 0)


def _draw_rect(center: np.ndarray, normal: np.ndarray, size_m: float):
    """Veraltetes Immediate-Mode-Quad → für Demo ausreichend."""
    left = _normalize(np.cross([0, 0, 1], normal)) * size_m
    up = _normalize(np.cross(normal, left)) * size_m

    v = (
        center + left - up,
        center + left + up,
        center - left + up,
        center - left - up,
    )
    idx = (0, 1, 2, 0, 2, 3)

    glBegin(GL_TRIANGLES)
    for i in idx:
        glVertex3fv(v[i])
    glEnd()


class StitchShader:
    """GLSL-Programm mit zwei Texture-Sampler & zwei MVP-Uniforms."""

    def __init__(self, vert: Path, frag: Path):
        self.program = shaders.compileProgram(
            shaders.compileShader(vert.read_text(), shaders.GL_VERTEX_SHADER),  # type: ignore
            shaders.compileShader(frag.read_text(), shaders.GL_FRAGMENT_SHADER),  # type: ignore
        )

        self.u_cam1 = glGetUniformLocation(self.program, "camera1_MVP")
        self.u_cam2 = glGetUniformLocation(self.program, "camera2_MVP")
        self.u_tex1 = glGetUniformLocation(self.program, "image1")
        self.u_tex2 = glGetUniformLocation(self.program, "image2")

        self.tex1 = self.tex2 = None
        self.mvp1 = self.mvp2 = None

    def feed(self, left: ImagePrepped, right: ImagePrepped):
        if self.tex1 is None:
            self.tex1 = GLTexture(left.image)
            self.tex2 = GLTexture(right.image)
        else:
            self.tex1.update(left.image)
            self.tex2.update(right.image)  # type: ignore
        self.mvp1, self.mvp2 = left.mvp, right.mvp


class OffscreenStitcher:
    def __init__(
        self, width=640, height=480, shader_dir: Path = Path(__file__).with_suffix("")
    ):
        import pygame

        pygame.init()
        pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)

        self.width, self.height = width, height

        self.fbo = glGenFramebuffers(1)
        self.color_tex = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.color_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_tex, 0
        )
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer nicht komplett.")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.shader = StitchShader(
            shader_dir / "shader_vert.glsl", shader_dir / "shader_frag.glsl"
        )

        glClearColor(0, 0, 0, 0)
        self.frame_count = 0

    def render(
        self,
        left_resp: image_pb2.ImageResponse,  # type: ignore
        right_resp: image_pb2.ImageResponse,  # type: ignore
    ) -> np.ndarray:
        """Stitcht genau EIN Frame und gibt BGR-ndarray zurück."""
        left = ImagePrepped(left_resp)
        right = ImagePrepped(right_resp)

        cam = StitchCamera(left, right)

        self.shader.feed(left, right)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(110, (self.width / self.height), 0.1, 50.0)

        glUseProgram(self.shader.program)

        glActiveTexture(GL_TEXTURE0)
        with self.shader.tex1.bound():  # type: ignore
            glUniform1i(self.shader.u_tex1, 0)
            glActiveTexture(GL_TEXTURE0 + 1)  # type: ignore
            with self.shader.tex2.bound():  # type: ignore
                glUniform1i(self.shader.u_tex2, 1)

                glUniformMatrix4fv(self.shader.u_cam1, 1, True, self.shader.mvp1)
                glUniformMatrix4fv(self.shader.u_cam2, 1, True, self.shader.mvp2)

                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                gluLookAt(
                    cam.eye_wrt_vo[0],
                    cam.eye_wrt_vo[1],
                    cam.eye_wrt_vo[2],
                    cam.plane_wrt_vo[0],
                    cam.plane_wrt_vo[1],
                    cam.plane_wrt_vo[2],
                    cam.up_wrt_vo[0],
                    cam.up_wrt_vo[1],
                    cam.up_wrt_vo[2],
                )

                _draw_rect(cam.plane_wrt_vo, cam.plane_norm_wrt_vo, 7)

        buf = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        frame = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)  # type: ignore
        frame = np.flipud(frame)
        frame = frame[..., ::-1]

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.frame_count += 1
        return frame

    def __del__(self):
        try:
            glDeleteFramebuffers(1, [self.fbo])
            glDeleteTextures(1, [self.color_tex])
        except Exception:
            pass
