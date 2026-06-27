import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
import pyrender
from pyrender import RenderFlags


class FaceRenderer:
    def __init__(self, width=1200, height=1200, bg_color=(0.0, 0.0, 0.0, 1.0), yfov=None, target=None):
        self.width = width
        self.height = height
        self.bg_color = bg_color

        if yfov is None:
            yfov = np.pi / 5.0
        self.yfov = yfov

        self.scene = pyrender.Scene(bg_color=bg_color)

        lookat_target = [0, 0, 0] if target is None else target
        cam = pyrender.PerspectiveCamera(
            yfov=yfov,
            aspectRatio=width / height,
        )
        self.camera_node = self.scene.add(cam, pose=self._look_at(
            eye=[0, 0, 3.0],
            target=lookat_target,
            up=[0, 1, 0],
        ))

        ambient = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.scene.ambient_light = ambient[:3]

        self._renderer = None
        try:
            self._renderer = pyrender.OffscreenRenderer(width, height)
        except Exception:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            self._renderer = pyrender.OffscreenRenderer(width, height)

        self._mesh_node = None

        self._eye_sphere_nodes = []      # list of pyrender Node, one per eye
        self._eye_sphere_data = None     # list of sphere data dicts

        self.faces = None
        self.uv_coords = None
        self.texture_img = None

        self._static_meshes_built = False

    def set_mesh_data(self, faces, uv_coords, texture_img):
        self.faces = faces
        self.uv_coords = uv_coords.astype(np.float32)
        self.texture_img = texture_img

    def set_eye_spheres(self, eye_spheres):
        self._eye_sphere_data = eye_spheres

    def set_eye_pose(self, index, center, scale=1.0, yaw=0.0, pitch=0.0):
        if self._eye_sphere_data is None or index >= len(self._eye_sphere_nodes):
            return
        R_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                           [0, 1, 0],
                           [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float64)
        R_pitch = np.array([[1, 0, 0],
                             [0, np.cos(pitch), -np.sin(pitch)],
                             [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float64)
        S = np.diag([scale, scale, scale]).astype(np.float64)
        R = R_yaw @ R_pitch
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R @ S
        pose[:3, 3] = np.array(center, dtype=np.float64)
        self.scene.set_pose(self._eye_sphere_nodes[index], pose)

    def _build_static_meshes(self):
        if self._static_meshes_built:
            return
        self._static_meshes_built = True

        # --- eye spheres (textured hemispheres, positioned per frame) ---
        if self._eye_sphere_data:
            for sphere in self._eye_sphere_data:
                tri = trimesh.Trimesh(vertices=sphere['verts_local'].astype(np.float64),
                                       faces=sphere['faces'], process=False)
                tri.visual = trimesh.visual.TextureVisuals(
                    uv=sphere['uv'].astype(np.float64),
                    image=sphere['texture'])
                em = pyrender.Mesh.from_trimesh(tri, smooth=True)
                em.primitives[0].material.doubleSided = True
                em.primitives[0].material.metallicFactor = 0.0
                em.primitives[0].material.roughnessFactor = 1.0
                node = self.scene.add(em)
                self._eye_sphere_nodes.append(node)
                # Initial pose: translate to photo eye centre (verts are local-space)
                init_pose = np.eye(4, dtype=np.float64)
                init_pose[:3, 3] = sphere['center'].astype(np.float64)
                self.scene.set_pose(node, init_pose)

    def render(self, deformed_vertices, faces=None):
        if faces is None:
            faces = self.faces
        if self.texture_img is None or faces is None:
            return np.full(
                (self.height, self.width, 3), 128, dtype=np.uint8
            )

        tri_mesh = trimesh.Trimesh(
            vertices=deformed_vertices.astype(np.float64),
            faces=faces,
            process=False,
        )

        tri_mesh.visual = trimesh.visual.TextureVisuals(
            uv=self.uv_coords.copy().astype(np.float64),
            image=self.texture_img,
        )

        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)
        render_mesh.primitives[0].material.doubleSided = True
        render_mesh.primitives[0].material.metallicFactor = 0.0
        render_mesh.primitives[0].material.roughnessFactor = 1.0

        if self._mesh_node is not None:
            self.scene.remove_node(self._mesh_node)
        self._mesh_node = self.scene.add(render_mesh)

        self._build_static_meshes()

        color, _ = self._renderer.render(self.scene, flags=RenderFlags.SKIP_CULL_FACES)
        return color

    def _look_at(self, eye, target, up):
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        z = eye - target
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = x
        pose[:3, 1] = y
        pose[:3, 2] = z
        pose[:3, 3] = eye
        return pose

    def delete(self):
        if self._renderer is not None:
            try:
                self._renderer.delete()
            except Exception:
                pass
