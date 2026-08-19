"""Render posed meshes into DA3 cameras with pyrender (EGL offscreen).

DA3 extrinsics are world->cam in the computer-vision convention
(X_cam = R @ X_world + t, +z into the scene, y down). pyrender wants a
cam->world node pose in the OpenGL convention (-z into the scene, y up), so we
invert E and flip the y/z axes.

Meshes are given together with their mesh->world similarity transforms; the
transform is baked into a mesh copy so the pyrender node pose stays identity
(avoids any non-rigid-pose assumptions in the shader).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh

# CV camera (x right, y down, z forward) -> GL camera (x right, y up, z back).
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def cv_extrinsic_to_gl_pose(E: np.ndarray) -> np.ndarray:
    """(3,4) world->cam CV extrinsic -> (4,4) cam->world OpenGL node pose."""
    E = np.asarray(E, dtype=np.float64)
    R = E[:3, :3]
    t = E[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w @ _CV_TO_GL


def _baked_mesh(mesh: trimesh.Trimesh, T_world: np.ndarray) -> trimesh.Trimesh:
    m = mesh.copy()
    m.apply_transform(np.asarray(T_world, dtype=np.float64))
    return m


class SceneRenderer:
    """Reusable EGL offscreen renderer for a fixed image size."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        cache_mesh_resources: bool = False,
    ):
        self.width = int(width)
        self.height = int(height)
        self._r = pyrender.OffscreenRenderer(self.width, self.height)
        self._cache_mesh_resources = bool(cache_mesh_resources)
        self._mesh_cache: dict[int, pyrender.Mesh] = {}

    def _scene_mesh(
        self,
        scene: pyrender.Scene,
        mesh: trimesh.Trimesh,
        transform: np.ndarray,
    ) -> None:
        if not self._cache_mesh_resources:
            scene.add(
                pyrender.Mesh.from_trimesh(
                    _baked_mesh(mesh, transform), smooth=False
                )
            )
            return
        # Candidate scoring repeatedly renders the same visual asset.  Upload
        # its vertex/texture buffers once and apply the uniform similarity in
        # the scene graph; recreating and uploading a textured mesh for every
        # camera/candidate dominates runtime by orders of magnitude.
        key = id(mesh)
        resource = self._mesh_cache.get(key)
        if resource is None:
            resource = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            self._mesh_cache[key] = resource
        scene.add(resource, pose=np.asarray(transform, dtype=np.float64))

    def close(self):
        self._r.delete()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _camera_node(self, scene: pyrender.Scene, K: np.ndarray, E: np.ndarray):
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                        znear=0.01, zfar=100.0)
        pose = cv_extrinsic_to_gl_pose(E)
        scene.add(cam, pose=pose)
        return pose

    def render(
        self,
        parts: list[tuple[trimesh.Trimesh, np.ndarray]],
        K: np.ndarray,
        E: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Textured color (H,W,3 uint8) + depth (H,W float, 0 = background)."""
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=[0.45, 0.45, 0.45])
        for mesh, T in parts:
            self._scene_mesh(scene, mesh, T)
        cam_pose = self._camera_node(scene, K, E)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
                  pose=cam_pose)
        color, depth = self._r.render(scene)
        return color[..., :3].copy(), depth

    def render_seg(
        self,
        parts: list[tuple[str, trimesh.Trimesh, np.ndarray]],
        K: np.ndarray,
        E: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Per-part occlusion-aware boolean masks via one SEG pass.

        parts: list of (name, mesh, T_world). Returns {name: (H,W) bool}.
        """
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1, 1, 1])
        node_color = {}
        for i, (name, mesh, T) in enumerate(parts):
            if self._cache_mesh_resources:
                key = id(mesh)
                resource = self._mesh_cache.get(key)
                if resource is None:
                    resource = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                    self._mesh_cache[key] = resource
                node = scene.add(
                    resource, pose=np.asarray(T, dtype=np.float64)
                )
            else:
                node = scene.add(
                    pyrender.Mesh.from_trimesh(
                        _baked_mesh(mesh, T), smooth=False
                    )
                )
            # distinct, well-separated colors
            node_color[node] = np.array([(i * 67 + 40) % 256,
                                         (i * 113 + 90) % 256,
                                         (i * 191 + 150) % 256], dtype=np.uint8)
        self._camera_node(scene, K, E)
        seg, _ = self._r.render(scene, flags=pyrender.RenderFlags.SEG,
                                seg_node_map=node_color)
        out = {}
        for (name, _, _), node in zip(parts, node_color):
            c = node_color[node]
            out[name] = np.all(seg == c[None, None, :], axis=2)
        return out


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-pixel surface normal map (H,W,3 in [0,1]) from a depth image."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ys, xs = np.mgrid[0:H, 0:W]
    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    P = np.stack([x, y, z], axis=2)
    dzx = np.zeros_like(P)
    dzy = np.zeros_like(P)
    dzx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dzy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dzx, dzy)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-9)
    rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    rgb[depth <= 0] = 0
    return rgb
