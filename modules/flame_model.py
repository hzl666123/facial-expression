import os
import pickle
import numpy as np
from scipy.spatial import cKDTree


class FLAMEModel:
    """Minimal FLAME model loader for topology and vertex transfer.

    Loads the FLAME 2020 generic_model.pkl and provides:
    - faces: (F, 3) triangle indices for high-quality face topology
    - v_template: (5023, 3) canonical template vertices
    - build_transfer_map(): maps FLAME vertices → subdivided mesh via barycentric lookup
    - transfer(): applies displacements from subdivided mesh to FLAME vertices
    """

    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.v_template = np.array(data['v_template'], dtype=np.float32)  # (5023, 3)
        self.faces = np.array(data['f'], dtype=np.int64).astype(np.int32)  # (9976, 3)
        self.shapedirs = np.array(data['shapedirs'], dtype=np.float32)  # (5023,3,400)

        self.n_verts = 5023
        self.n_faces = 9976

        # Normalize template to [-0.5, 0.5] approx range for easier correspondence
        self._template_scale = 1.0 / max(
            self.v_template[:, 0].max() - self.v_template[:, 0].min(),
            self.v_template[:, 1].max() - self.v_template[:, 1].min(),
        )
        self._template_center = self.v_template.mean(axis=0)

    def build_transfer_map(self, source_verts_2d, source_faces):
        """Build a barycentric mapping from FLAME vertices → source mesh.

        Maps each FLAME vertex to a triangle on the source subdivided mesh
        via nearest-point lookup, then computes barycentric coordinates.

        Args:
            source_verts_2d: (V_src, 2) 2D vertex positions of subdivided mesh
            source_faces: (F_src, 3) triangle indices of subdivided mesh

        Returns:
            mapping dict with keys: tri_idx, bary_coords
        """
        # Build KD-tree for source vertices
        tree = cKDTree(source_verts_2d)

        # For each FLAME vertex, find its position in the source 2D space
        # We project FLAME template onto 2D by simply taking x,y and centering/scaling
        flame_2d = np.zeros((self.n_verts, 2), dtype=np.float32)
        flame_2d[:, 0] = (self.v_template[:, 0] - self._template_center[0]) * self._template_scale
        flame_2d[:, 1] = (self.v_template[:, 1] - self._template_center[1]) * self._template_scale

        # Center flame 2D to match source mesh center
        src_center = source_verts_2d.mean(axis=0)
        src_scale = max(
            source_verts_2d[:, 0].max() - source_verts_2d[:, 0].min(),
            source_verts_2d[:, 1].max() - source_verts_2d[:, 1].min(),
        )
        flame_2d = flame_2d / np.ptp(flame_2d[:, 0]) * src_scale * 0.95 + src_center

        # For each FLAME vertex, find closest source vertex
        _, nn_indices = tree.query(flame_2d)

        # For each FLAME vertex, find which triangle the nearest source vertex belongs to
        # and compute barycentric coordinates
        tri_idx = np.zeros(self.n_verts, dtype=np.int32)
        bary_coords = np.zeros((self.n_verts, 3), dtype=np.float32)

        vertex_to_tri = {}
        for fi, (a, b, c) in enumerate(source_faces):
            for v_idx in (a, b, c):
                if v_idx not in vertex_to_tri:
                    vertex_to_tri[v_idx] = []
                vertex_to_tri[v_idx].append(fi)

        for flame_i in range(self.n_verts):
            src_v = nn_indices[flame_i]
            if src_v in vertex_to_tri and len(vertex_to_tri[src_v]) > 0:
                fi = vertex_to_tri[src_v][0]
                a, b, c = source_faces[fi]
                pa = source_verts_2d[a]
                pb = source_verts_2d[b]
                pc = source_verts_2d[c]
                bary = self._barycentric(flame_2d[flame_i], pa, pb, pc)
            else:
                fi = 0
                bary = np.array([1.0, 0.0, 0.0])

            tri_idx[flame_i] = fi
            bary_coords[flame_i] = bary

        return {
            "tri_idx": tri_idx,
            "bary_coords": bary_coords,
            "source_faces": source_faces,
        }

    def transfer_vertices(self, source_verts_3d, mapping):
        """Transfer 3D vertex positions from source mesh to FLAME vertices.

        Args:
            source_verts_3d: (V_src, 3) current 3D positions of source mesh
            mapping: result from build_transfer_map()

        Returns:
            flame_verts: (5023, 3) transferred FLAME vertex positions
        """
        tri_idx = mapping["tri_idx"]
        bary = mapping["bary_coords"]
        source_faces = mapping["source_faces"]

        flame_verts = np.zeros((self.n_verts, 3), dtype=np.float32)
        for i in range(self.n_verts):
            fi = tri_idx[i]
            a, b, c = source_faces[fi]
            u, v, w = bary[i]
            flame_verts[i] = (u * source_verts_3d[a]
                              + v * source_verts_3d[b]
                              + w * source_verts_3d[c])
        return flame_verts

    def transfer_displacement(self, source_disp_3d, mapping):
        """Transfer per-vertex displacement from source mesh to FLAME vertices.

        Uses the barycentric mapping to interpolate displacements from
        the source mesh triangles to each FLAME vertex.

        Args:
            source_disp_3d: (V_src, 3) displacement vectors for source mesh
            mapping: result from build_transfer_map()

        Returns:
            flame_disp: (5023, 3) interpolated displacement for FLAME vertices
        """
        return self.transfer_vertices(source_disp_3d, mapping)

    @staticmethod
    def _barycentric(pt, a, b, c):
        """Compute barycentric coordinates of pt in triangle (a,b,c)."""
        v0 = b - a
        v1 = c - a
        v2 = pt - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return np.array([1.0, 0.0, 0.0])

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        s = u + v + w
        if s > 0:
            u /= s; v /= s; w /= s

        return np.array([u, v, w])
