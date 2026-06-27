import ctypes
import os
import sys

if sys.platform == "linux":
    _LIBS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libs"
    )
    _lib_path = os.path.join(_LIBS_DIR, "libGLESv2.so.2.1.0")
    if os.path.exists(_lib_path):
        ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

from .flame_model import FLAMEModel

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task",
)


def _edges_to_triangles(num_vertices, edges):
    adj = {i: set() for i in range(num_vertices)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    triangles = set()
    for v in range(num_vertices):
        neighbors = sorted(adj[v])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj[neighbors[i]]:
                    tri = tuple(sorted([v, neighbors[i], neighbors[j]]))
                    triangles.add(tri)

    return np.array(sorted(triangles), dtype=np.int32)


def _fix_winding_2d(faces, vertices_2d):
    """Ensure all faces have consistent CW winding in image space
    (becomes CCW in canonical/OpenGL space after y-axis flip)."""
    fixed = faces.copy()
    for i, (a, b, c) in enumerate(faces):
        va = vertices_2d[a]
        vb = vertices_2d[b]
        vc = vertices_2d[c]
        cross = (vb[0] - va[0]) * (vc[1] - va[1]) - (vb[1] - va[1]) * (vc[0] - va[0])
        if cross < 0:
            fixed[i] = [a, c, b]
    return fixed


def _loop_subdivide(vertices, faces):
    """One iteration of Loop subdivision for triangular meshes.

    Args:
        vertices: (V, D) vertex positions (D=2 or 3)
        faces: (F, 3) triangle indices

    Returns:
        new_vertices: (V', D)
        new_faces: (F*4, 3)
    """
    V = len(vertices)
    F = len(faces)

    # --- build edge-to-face adjacency ---
    edge_to_faces = {}
    edge_verts = []
    for fi, (i0, i1, i2) in enumerate(faces):
        for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append(fi)
            edge_verts.append((key, a, b))

    # Deduplicate edges
    unique_edges = list(edge_to_faces.keys())
    edge_index = {e: idx for idx, e in enumerate(unique_edges)}
    E = len(unique_edges)

    # --- compute vertex valence and neighbour list ---
    valence = np.zeros(V, dtype=np.int32)
    adjacency = {i: {} for i in range(V)}
    for (a, b), face_list in edge_to_faces.items():
        valence[a] += 1
        valence[b] += 1
        adjacency[a][b] = True
        adjacency[b][a] = True

    # --- even vertex positions (original vertices) ---
    safe_valence = np.maximum(valence, 1)
    beta = np.where(
        valence == 3,
        3.0 / 16.0,
        3.0 / (8.0 * safe_valence),
    )

    even_vertices = vertices.copy()
    for v in range(V):
        if valence[v] == 0:
            continue
        neighbors = list(adjacency[v].keys())
        neighbor_sum = np.sum(vertices[neighbors], axis=0)
        even_vertices[v] = (1.0 - valence[v] * beta[v]) * vertices[v] + beta[v] * neighbor_sum

    # --- odd vertex positions (edge midpoints) ---
    odd_vertices = np.zeros((E, vertices.shape[1]), dtype=vertices.dtype)
    for idx, (a, b) in enumerate(unique_edges):
        adj_faces = edge_to_faces[(a, b)]
        if len(adj_faces) == 2:
            f0, f1 = adj_faces
            # Find the two opposite vertices
            v0_set = set(faces[f0])
            v1_set = set(faces[f1])
            common = {a, b}
            opp0 = (v0_set - common).pop()
            opp1 = (v1_set - common).pop()
            odd_vertices[idx] = (
                3.0 / 8.0 * (vertices[a] + vertices[b])
                + 1.0 / 8.0 * (vertices[opp0] + vertices[opp1])
            )
        else:
            # Boundary edge: simple midpoint
            odd_vertices[idx] = 0.5 * (vertices[a] + vertices[b])

    # --- new faces (each original triangle → 4 new triangles) ---
    new_vertices = np.vstack([even_vertices, odd_vertices])
    new_faces = np.zeros((F * 4, 3), dtype=np.int32)

    for fi, (i0, i1, i2) in enumerate(faces):
        e0 = V + edge_index[(min(i0, i1), max(i0, i1))]
        e1 = V + edge_index[(min(i1, i2), max(i1, i2))]
        e2 = V + edge_index[(min(i2, i0), max(i2, i0))]

        new_faces[fi * 4 + 0] = [i0, e0, e2]
        new_faces[fi * 4 + 1] = [i1, e1, e0]
        new_faces[fi * 4 + 2] = [i2, e2, e1]
        new_faces[fi * 4 + 3] = [e0, e1, e2]

    return new_vertices, new_faces


def _cleanup_mesh(vertices, faces):
    """Remove degenerate faces and merge near-duplicate vertices."""
    # 1. Remove faces with near-zero area
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt((cross ** 2).sum(axis=1)) if vertices.shape[1] > 2 else \
            0.5 * np.abs(cross) if vertices.shape[1] == 2 else \
            0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
    keep = areas > 1e-8
    faces = faces[keep]

    if faces.size == 0:
        return vertices, faces

    # 2. Merge near-duplicate vertices
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(0.5, output_type='ndarray')
    if len(pairs) > 0:
        remap = np.arange(len(vertices))
        # Use union-find style: merge each duplicate to the lower index
        for a, b in pairs:
            remap[b] = remap[a]
        # Collapse chains
        for _ in range(3):
            remap = remap[remap]
        # Remap faces
        faces = remap[faces]
        # Remove orphan vertices
        used = np.unique(faces)
        new_idx = np.full(len(vertices), -1, dtype=np.int32)
        new_idx[used] = np.arange(len(used))
        faces = new_idx[faces]
        vertices = vertices[used]

    # 3. Remove any faces that became degenerate after vertex merge
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    dup = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    faces = faces[~dup]

    return vertices, faces


def _smooth_z_laplacian(z, faces, iterations=2, lambda_=0.4):
    """Smooth z values using mesh Laplacian (neighbor averaging via face topology)."""
    n = len(z)
    neighbors = {i: [] for i in range(n)}
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    for v in range(n):
        neighbors[v] = list(set(neighbors[v]))

    z_smooth = z.copy().astype(np.float64)
    for _ in range(iterations):
        z_new = z_smooth.copy()
        for v in range(n):
            nbrs = neighbors.get(v, [])
            if nbrs:
                z_new[v] = (1 - lambda_) * z_smooth[v] + lambda_ * np.mean(z_smooth[nbrs])
        z_smooth = z_new
    return z_smooth.astype(z.dtype)


def _filter_points_in_hull(ref_points, query_points, query_values, margin=0.05):
    """Keep only query points within convex hull of reference points."""
    from scipy.spatial import Delaunay
    centroid = ref_points.mean(axis=0)
    expanded = centroid + (ref_points - centroid) * (1.0 + margin)
    tri = Delaunay(expanded)
    inside = tri.find_simplex(query_points) >= 0
    return query_points[inside], query_values[inside]


def _point_in_polygon(pts, poly, eps=1e-4):
    """Ray casting: test which points lie inside a closed polygon.

    Args:
        pts: (N, 2) query points
        poly: (M, 2) polygon vertices in order (closed automatically)
        eps: tiny vertical offset to avoid vertex-collision edge cases

    Returns:
        (N,) bool array
    """
    n = len(poly)
    inside = np.zeros(len(pts), dtype=bool)
    px, py = poly[:, 0], poly[:, 1]
    x, y = pts[:, 0], pts[:, 1] + eps
    for i in range(n):
        j = (i + 1) % n
        cond = (py[i] > y) != (py[j] > y)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_intersect = px[i] + (y - py[i]) * (px[j] - px[i]) / (py[j] - py[i] + 1e-12)
        inside ^= cond & (x < x_intersect)
    return inside


def _build_eye_spheres(canonical_verts, sv2d, img_rgb, landmark_indices, landmarks_2d,
                        image_w, image_h, scale):
    """Create textured hemisphere meshes for both eyes from photo crop.

    Replaces the old per-vertex corneal bulge with simple textured
    half-spheres positioned behind the eye openings. Gaze tracking
    is handled by rotating the spheres per-frame.
    """
    right_eye_contour = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    right_eye_iris = [468, 469, 470, 471, 472]
    left_eye_contour = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
    left_eye_iris = [473, 474, 475, 476, 477]

    eye_spheres = []

    for contour_lm, iris_lm in [(right_eye_contour, right_eye_iris),
                                  (left_eye_contour, left_eye_iris)]:
        contour_v = landmark_indices[contour_lm]
        contour_3d = canonical_verts[contour_v]

        # Uniform hemisphere radius — sphere stays perfectly round.
        # The face mesh's eyelids naturally occlude the overflow.
        ex = float(contour_3d[:, 0].max() - contour_3d[:, 0].min())
        ey = float(contour_3d[:, 1].max() - contour_3d[:, 1].min())
        r_base = max(ex, ey) * 0.61
        rx = r_base
        ry = r_base
        rz = r_base * 0.85

        # 3D eye centre from contour bounding-box midpoint — immune to uneven
        # landmark distribution around the eyelid (7 lower vs 6 upper landmarks)
        center = np.array([
            (contour_3d[:, 0].min() + contour_3d[:, 0].max()) / 2.0,
            (contour_3d[:, 1].min() + contour_3d[:, 1].max()) / 2.0,
            0.0,
        ], dtype=np.float32)
        center[2] = -rz - 0.002  # sphere front just behind face surface

        # --- Eye texture: external photo or procedural ---
        import config
        eye_photo_path = getattr(config, 'EYE_PHOTO_PATH', None)
        use_external = False

        if eye_photo_path and os.path.exists(eye_photo_path):
            eye_img = cv2.imread(eye_photo_path)
            if eye_img is not None:
                eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
                h, w = eye_img.shape[:2]

                # Auto-detect: circular eyeball on dark background vs full eye photo
                corners = [eye_img[0, 0], eye_img[0, -1], eye_img[-1, 0], eye_img[-1, -1]]
                is_eyeball = all(c.sum() < 30 for c in corners)

                tex_size = 64

                if is_eyeball:
                    # Circular eyeball image: use directly
                    eye_tex = cv2.resize(eye_img, (tex_size, tex_size),
                                          interpolation=cv2.INTER_AREA)
                    fade_start = tex_size * 0.35
                    fade_end = tex_size * 0.65
                else:
                    # Full eye photo: detect iris and crop
                    gray = cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)
                    x0, x1 = int(w * 0.2), int(w * 0.8)
                    y0, y1 = int(h * 0.2), int(h * 0.8)
                    roi = gray[y0:y1, x0:x1]
                    min_y, min_x = np.unravel_index(np.argmin(roi), roi.shape)
                    iris_x = x0 + min_x
                    iris_y = y0 + min_y

                    half = int(min(iris_x, iris_y, w - iris_x, h - iris_y) * 0.7)
                    cx0 = max(0, iris_x - half)
                    cx1 = min(w, iris_x + half)
                    cy0 = max(0, iris_y - half)
                    cy1 = min(h, iris_y + half)
                    crop = eye_img[cy0:cy1, cx0:cx1]
                    eye_tex = cv2.resize(crop, (tex_size, tex_size),
                                          interpolation=cv2.INTER_AREA)
                    fade_start = tex_size * 0.30
                    fade_end = tex_size * 0.50

                # Radial mask: keep original at centre, fade to white at edges
                cy_c, cx_c = tex_size / 2.0, tex_size / 2.0
                yy, xx = np.ogrid[:tex_size, :tex_size]
                dist = np.sqrt((xx - cx_c) ** 2 + (yy - cy_c) ** 2)
                mask = np.clip(1.0 - (dist - fade_start) / (fade_end - fade_start + 1e-6), 0.0, 1.0)
                mask = mask[:, :, np.newaxis]
                sclera = np.full((tex_size, tex_size, 3), 220, dtype=np.uint8)
                eye_tex = (eye_tex.astype(np.float32) * mask + sclera.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

                iris_u = 0.5
                iris_v = 0.5
                use_external = True

        if not use_external:
            # Sample iris colour from the photo's iris landmarks
            iris_2d = landmarks_2d[iris_lm]
            iris_colors = []
            for px, py in iris_2d:
                x, y = int(round(px)), int(round(py))
                if 0 <= x < image_w and 0 <= y < image_h:
                    iris_colors.append(img_rgb[y, x])
            if iris_colors:
                iris_color = np.mean(iris_colors, axis=0).astype(np.uint8)
            else:
                iris_color = np.array([80, 60, 40], dtype=np.uint8)

            tex_size = 64
            eye_tex = np.full((tex_size, tex_size, 3), 220, dtype=np.uint8)
            cy, cx = tex_size / 2.0, tex_size / 2.0
            yy, xx = np.ogrid[:tex_size, :tex_size]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

            iris_r = tex_size * 0.35
            eye_tex[dist <= iris_r] = iris_color

            pupil_r = iris_r * 0.4
            eye_tex[dist <= pupil_r] = np.array([15, 15, 15], dtype=np.uint8)

            iris_u = 0.5
            iris_v = 0.5

        # Mirror the external photo horizontally for the left eye
        if use_external and contour_lm[0] > 300:
            eye_tex = eye_tex[:, ::-1, :].copy()

        # Build UV hemisphere (front half of UV sphere, front axis = +Z)
        # UVs are centred on the iris so the pole (theta=0) shows the iris.
        n_lat = 12
        n_lon = 24
        verts = []
        uvs = []

        # Max UV radius: distance from iris to nearest crop edge
        r_max = min(iris_u, 1.0 - iris_u, iris_v, 1.0 - iris_v) * 0.92

        for j in range(n_lat + 1):
            theta = (j / n_lat) * (np.pi / 2.0)  # 0 (north pole / +Z) to pi/2 (equator)
            sin_t = np.sin(theta)
            r_uv = (theta / (np.pi / 2.0)) * r_max  # 0 at pole → r_max at equator
            for i in range(n_lon + 1):
                phi = (i / n_lon) * 2.0 * np.pi
                x = rx * sin_t * np.cos(phi)
                y = ry * sin_t * np.sin(phi)
                z = rz * np.cos(theta)
                verts.append([x, y, z])
                # UV: pole → iris centre, equator → near crop edge
                u = iris_u + r_uv * np.cos(phi)
                v = iris_v + r_uv * np.sin(phi)
                u = np.clip(u, 0.001, 0.999)
                v = np.clip(v, 0.001, 0.999)
                uvs.append([u, v])

        verts_local = np.array(verts, dtype=np.float32)
        uvs = np.array(uvs, dtype=np.float32)

        # Triangle fan faces
        faces = []
        for j in range(n_lat):
            for i in range(n_lon):
                a = j * (n_lon + 1) + i
                b = a + 1
                c = a + (n_lon + 1)
                d = c + 1
                faces.append([a, b, d])
                faces.append([a, d, c])
        faces = np.array(faces, dtype=np.int32)


        eye_spheres.append({
            'verts_local': verts_local,
            'faces': faces,
            'uv': uvs,
            'texture': eye_tex,
            'center': center.astype(np.float32),
            'rx': rx,
            'ry': ry,
            'rz': rz,
            'iris_lm': np.array(iris_lm, dtype=np.int32),
            'contour_v_idx': contour_v.astype(np.int32),
            'static_span': max(ex, ey),
        })

    return eye_spheres


def _build_mouth_meshes(canonical, landmark_indices):
    upper_inner_lm = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lower_inner_lm = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    mouth_inner_all = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
                       78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

    upper_v = landmark_indices[upper_inner_lm]
    lower_v = landmark_indices[lower_inner_lm]
    all_v = landmark_indices[mouth_inner_all]
    n_teeth = len(upper_v)

    upper_pos = canonical[upper_v].copy()
    lower_pos = canonical[lower_v].copy()

    # --- Mouth cavity (dark fill inside mouth) ---
    all_pos = canonical[all_v].copy()
    all_pos[:, 2] -= 0.003
    center = all_pos.mean(axis=0)
    center[2] -= 0.003
    cavity_verts = np.vstack([all_pos, center], dtype=np.float32)
    center_idx = len(all_pos)
    cavity_faces = np.array([
        [center_idx, i, (i + 1) % len(all_pos)]
        for i in range(len(all_pos))
    ], dtype=np.int32)

    # --- Teeth: solid block sized to fill the mouth interior ---
    # Three rows: upper, middle, lower — creates a smooth curved block.
    # The block extends from 0.02 above the upper lip to 0.02 below the
    # lower lip to make teeth visible even in a closed-mouth photo.
    teeth_h = 0.05  # total height in canonical units (~13 px)

    teeth_z_offset = 0.015  # clear depth separation from face mesh lips
    teeth_verts = []
    for i in range(n_teeth):
        mid = (upper_pos[i] + lower_pos[i]) / 2.0
        teeth_verts.append([mid[0], mid[1] + teeth_h * 0.5, mid[2] + teeth_z_offset])
    for i in range(n_teeth):
        mid = (upper_pos[i] + lower_pos[i]) / 2.0
        teeth_verts.append([mid[0], mid[1], mid[2] + teeth_z_offset])
    for i in range(n_teeth):
        mid = (upper_pos[i] + lower_pos[i]) / 2.0
        teeth_verts.append([mid[0], mid[1] - teeth_h * 0.5, mid[2] + teeth_z_offset])
    teeth_verts = np.array(teeth_verts, dtype=np.float32)

    teeth_faces = []
    # Upper band: row 0 → row 1 (CCW winding, +Z normal)
    # Mouth contour goes right-to-left so x decreases. Winding uses
    # middle-row vertex first to keep cross-product Z positive.
    for i in range(n_teeth - 1):
        u0, u1 = i, i + 1
        m0, m1 = n_teeth + i, n_teeth + i + 1
        teeth_faces.append([m1, u0, u1])
        teeth_faces.append([m0, u0, m1])
    # Lower band: row 1 → row 2 (CCW winding, +Z normal)
    for i in range(n_teeth - 1):
        m0, m1 = n_teeth + i, n_teeth + i + 1
        l0, l1 = 2 * n_teeth + i, 2 * n_teeth + i + 1
        teeth_faces.append([l1, m0, m1])
        teeth_faces.append([l0, m0, l1])
    teeth_faces = np.array(teeth_faces, dtype=np.int32)

    return {
        "mouth_cavity_verts": cavity_verts,
        "mouth_cavity_faces": cavity_faces,
        "teeth_verts": teeth_verts,
        "teeth_faces": teeth_faces,
    }


def _build_hair_quad(canonical, img_rgb, face_bbox):
    x_min, y_min, x_max, y_max = face_bbox
    h, w = img_rgb.shape[:2]

    # Forehead-top canonical Y (highest valid vertex)
    forehead_y = float(canonical[:, 1].max())

    # Hair region in photo: everything above the face bounding box top
    hair_img_top = 0
    hair_img_bot = max(0, int(y_min) - 5)  # include a few px of forehead for blend
    hair_img = img_rgb[hair_img_top:hair_img_bot, :].copy()
    if hair_img.size == 0:
        hair_img = np.full((1, w, 3), 128, dtype=np.uint8)
        hair_img_bot = 1

    # Hair quad in canonical space: spans full photo width at the
    # forehead Y level, extending upward well above the camera frustum.
    face_x_span = float(canonical[:, 0].max() - canonical[:, 0].min())
    hair_margin = face_x_span * 0.4
    hair_x_left = float(canonical[:, 0].min()) - hair_margin
    hair_x_right = float(canonical[:, 0].max()) + hair_margin
    hair_y_bot = forehead_y - 0.02   # tiny overlap with face mesh
    hair_y_top = forehead_y + 0.4    # extend upward within camera frustum

    hair_verts = np.array([
        [hair_x_left,  hair_y_bot, -0.3],
        [hair_x_right, hair_y_bot, -0.3],
        [hair_x_right, hair_y_top, -0.3],
        [hair_x_left,  hair_y_top, -0.3],
    ], dtype=np.float32)
    hair_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    # UV: map the hair crop region of the photo.
    # OpenGL convention: v=0.0 = bottom of texture (last image row),
    # v=1.0 = top of texture (first image row).
    # Bottom of quad (forehead) → v=0.0 → near-forehead crop row
    # Top of quad (hair) → v=1.0 → top-of-photo crop row
    hair_uv = np.array([
        [0.0, 0.0],  # bottom-left  → near forehead
        [1.0, 0.0],  # bottom-right → near forehead
        [1.0, 1.0],  # top-right    → sky / hair
        [0.0, 1.0],  # top-left     → sky / hair
    ], dtype=np.float32)

    return dict(verts=hair_verts, faces=hair_faces, uv=hair_uv, img=hair_img)


class FaceReconstructor:
    """High-precision 3D face reconstruction from a single photo.

    Backends:
    - "mediapipe_subdiv": MediaPipe 478 landmarks + Loop subdivision
    - "flame": FLAME model via DECA (when FLAME model files are available)
    """

    def __init__(self, backend="mediapipe_subdiv", subdiv_iterations=2,
                 flame_path=None, deca_path=None):
        self.backend = backend
        self.subdiv_iterations = subdiv_iterations
        self.flame_path = flame_path
        self.deca_path = deca_path

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Build MediaPipe face mesh triangulation
        mp_edges = [(c.start, c.end)
                    for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
        self._face_triangulation = _edges_to_triangles(478, mp_edges)

    def _subdivision_backend(self, img_rgb, landmarks_2d):
        h, w = img_rgb.shape[:2]

        # Build 2D mesh using MediaPipe face tesselation topology
        verts_2d = landmarks_2d.astype(np.float64)
        faces = self._face_triangulation.copy()

        # Fix winding order for consistent front-facing rendering
        faces = _fix_winding_2d(faces, verts_2d)

        # Loop subdivision
        for _ in range(self.subdiv_iterations):
            verts_2d, faces = _loop_subdivide(verts_2d, faces)
            faces = _fix_winding_2d(faces, verts_2d)

        # Clean up degenerate faces and near-duplicate vertices
        verts_2d, faces = _cleanup_mesh(verts_2d, faces)
        faces = _fix_winding_2d(faces, verts_2d)

        # Rebuild landmark→vertex index mapping (cleanup may have reordered vertices)
        landmark_to_vertex = np.zeros(478, dtype=np.int32)
        for i in range(478):
            landmark_to_vertex[i] = int(np.argmin(
                np.sum((verts_2d - landmarks_2d[i]) ** 2, axis=1)))

        # --- Delete faces spanning the eye interior (prevent Z-fighting with fan triangles) ---
        reye_contour_lm = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        leye_contour_lm = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        all_eye_contour_lm = reye_contour_lm + leye_contour_lm
        eye_contour_v = set(int(landmark_to_vertex[lm]) for lm in all_eye_contour_lm)

        keep_mask = np.ones(len(faces), dtype=bool)
        for fi in range(len(faces)):
            f = faces[fi]
            count = sum(1 for v in f if v in eye_contour_v)
            if count >= 2:
                keep_mask[fi] = False
        faces = faces[keep_mask]
        faces = _fix_winding_2d(faces, verts_2d)

        # --- Crop face texture ---
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(landmarks_2d.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        x_min, y_min = landmarks_2d[:, 0].min(), landmarks_2d[:, 1].min()
        x_max, y_max = landmarks_2d[:, 0].max(), landmarks_2d[:, 1].max()
        face_size = max(x_max - x_min, y_max - y_min)
        margin = face_size * 0.25

        x_min = max(0, int(x_min - margin))
        y_min = max(0, int(y_min - margin))
        x_max = min(w, int(x_max + margin))
        y_max = min(h, int(y_max + margin))

        texture_img = img_rgb[y_min:y_max, x_min:x_max].copy()
        tex_h, tex_w = texture_img.shape[:2]

        # --- UV coordinates (barycentric interpolation within mesh topology) ---
        uv_coords = np.zeros((len(verts_2d), 2), dtype=np.float32)
        for i in range(478):
            v = landmark_to_vertex[i]
            uv_coords[v, 0] = (landmarks_2d[i, 0] - x_min) / tex_w
            uv_coords[v, 1] = 1.0 - (landmarks_2d[i, 1] - y_min) / tex_h

        # Build UV lookup table keyed by landmark ID (0-477)
        lm_uv = uv_coords[landmark_to_vertex]

        # Barycentric UV interpolation for subdivided vertices
        # Uses MediaPipe tesselation topology (same as mesh faces), NOT Delaunay
        # Also precomputes barycentric basis for per-frame displacement interpolation
        n_verts_total = len(verts_2d)
        subdiv_bary_v = np.zeros((n_verts_total, 3), dtype=np.int32)
        subdiv_bary_w = np.zeros((n_verts_total, 3), dtype=np.float32)
        is_landmark_vertex = np.zeros(n_verts_total, dtype=bool)
        is_landmark_vertex[landmark_to_vertex] = True
        # Landmarks: identity mapping (each landmark is its own 100% source)
        for i in range(478):
            v = landmark_to_vertex[i]
            subdiv_bary_v[v] = [i, i, i]
            subdiv_bary_w[v, 0] = 1.0

        if self.subdiv_iterations > 0:
            tri_v0 = landmarks_2d[self._face_triangulation[:, 0]]
            tri_v1 = landmarks_2d[self._face_triangulation[:, 1]]
            tri_v2 = landmarks_2d[self._face_triangulation[:, 2]]
            tri_uv0 = lm_uv[self._face_triangulation[:, 0]]
            tri_uv1 = lm_uv[self._face_triangulation[:, 1]]
            tri_uv2 = lm_uv[self._face_triangulation[:, 2]]

            for i in range(n_verts_total):
                if is_landmark_vertex[i]:
                    continue
                px, py = verts_2d[i, 0], verts_2d[i, 1]

                eu = tri_v0 - tri_v2
                ev = tri_v1 - tri_v2
                ed = np.array([px, py]) - tri_v2
                det = eu[:, 0] * ev[:, 1] - ev[:, 0] * eu[:, 1]
                valid = np.abs(det) > 1e-12

                w0 = np.full(len(tri_v0), np.nan)
                w1 = np.full(len(tri_v0), np.nan)
                w0[valid] = (ed[valid, 0] * ev[valid, 1] - ev[valid, 0] * ed[valid, 1]) / det[valid]
                w1[valid] = (eu[valid, 0] * ed[valid, 1] - ed[valid, 0] * eu[valid, 1]) / det[valid]
                w2 = 1.0 - w0 - w1

                tol = 0.01
                error = np.full(len(tri_v0), np.inf)
                error[valid] = (
                    np.maximum(0.0, -w0[valid] - tol) + np.maximum(0.0, -w1[valid] - tol) +
                    np.maximum(0.0, -w2[valid] - tol) +
                    np.maximum(0.0, w0[valid] - 1.0 - tol) + np.maximum(0.0, w1[valid] - 1.0 - tol) +
                    np.maximum(0.0, w2[valid] - 1.0 - tol)
                )

                best = np.argmin(error)
                w0_b, w1_b, w2_b = w0[best], w1[best], w2[best]
                uv_coords[i, 0] = w0_b * tri_uv0[best, 0] + w1_b * tri_uv1[best, 0] + w2_b * tri_uv2[best, 0]
                uv_coords[i, 1] = w0_b * tri_uv0[best, 1] + w1_b * tri_uv1[best, 1] + w2_b * tri_uv2[best, 1]
                subdiv_bary_v[i] = self._face_triangulation[best]
                subdiv_bary_w[i] = [w0_b, w1_b, w2_b]

            uv_coords = np.nan_to_num(uv_coords, nan=0.5)
            uv_coords = np.clip(uv_coords, 0.0, 1.0)

        # --- Canonical 3D vertices (neutral pose) ---
        canonical = np.zeros((len(verts_2d), 3), dtype=np.float32)
        canonical[:, 0] = -(verts_2d[:, 0] / w - 0.5)
        canonical[:, 1] = -(verts_2d[:, 1] / h - 0.5)

        scale = 1.0 / max(x_max - x_min, y_max - y_min) * max(w, h)
        canonical[:, :2] *= scale
        canonical[:, 1] *= (float(h) / float(w))

        center = canonical[:, :2].mean(axis=0)
        canonical[:, 0] -= center[0]
        canonical[:, 1] -= center[1]

        landmark_indices = landmark_to_vertex.astype(np.int32)

        # Pre-compute mouth interior vertex mask for Z recession when jaw opens.
        # Vertices inside the inner mouth polygon (2D) are nudged backward to
        # hide stretched upper↔lower lip connecting triangles.
        from matplotlib.path import Path
        mouth_inner_lm = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
                           78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        mouth_inner_v = landmark_to_vertex[mouth_inner_lm]
        mouth_poly = verts_2d[mouth_inner_v]
        mouth_path = Path(mouth_poly)
        mouth_interior_mask = mouth_path.contains_points(verts_2d)

        # Also identify which FACES are inside the mouth interior.
        # These are the triangles that connect upper↔lower lip and create
        # visible "webbing" when the mouth opens — culled dynamically per frame.
        # Mark faces where >= 2 vertices fall inside the inner-lip polygon + any
        # faces that directly span upper↔lower inner lip landmark vertices.
        face_v_all = faces.ravel()
        in_poly = mouth_path.contains_points(verts_2d[face_v_all])
        in_poly = in_poly.reshape(-1, 3)
        mouth_interior_face_mask = in_poly.sum(axis=1) >= 2

        upper_inner_v = set(landmark_to_vertex[
            [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]])
        lower_inner_v = set(landmark_to_vertex[
            [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]])
        for fi in range(len(faces)):
            if mouth_interior_face_mask[fi]:
                continue
            vs = faces[fi]
            if (any(int(v) in upper_inner_v for v in vs) and
                    any(int(v) in lower_inner_v for v in vs)):
                mouth_interior_face_mask[fi] = True

        eye_spheres = _build_eye_spheres(canonical, verts_2d, img_rgb,
                                          landmark_indices, landmarks_2d,
                                          w, h, scale)

        # Store coordinate system parameters for webcam→canonical mapping
        coord_params = {
            "image_size": (w, h),
            "face_size_px": float(max(x_max - x_min, y_max - y_min)),
            "aspect": float(h) / float(w),
            "center_xy": center,
            "scale_xy": scale,
        }

        return {
            "canonical_vertices": canonical,
            "subdiv_verts_2d": verts_2d,
            "faces": faces,
            "uv_coords": uv_coords,
            "texture_img": texture_img,
            "landmark_indices": landmark_indices,
            "landmarks_2d": landmarks_2d,
            "image_size": (w, h),
            "face_bbox": (x_min, y_min, x_max, y_max),
            "texture_size": (tex_w, tex_h),
            "model_type": "mediapipe_subdiv",
            "coord_params": coord_params,
            "scale": scale,
            "center": center,
            "aspect": float(h) / float(w),
            "subdiv_bary_v": subdiv_bary_v,
            "subdiv_bary_w": subdiv_bary_w,
            "eye_spheres": eye_spheres,
            "mouth_interior_mask": mouth_interior_mask,
            "mouth_interior_face_mask": mouth_interior_face_mask,
        }

    def process(self, photo_path):
        if isinstance(photo_path, (list, tuple)):
            if self.backend == "emoca":
                return self.process_multi_view(list(photo_path))
            else:
                photo_path = photo_path[0]
        img_bgr = cv2.imread(photo_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image from {photo_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb,
        )
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            raise ValueError(f"No face detected in photo: {photo_path}")

        landmarks_raw = result.face_landmarks[0]
        landmarks_2d = np.array(
            [[lm.x * w, lm.y * h] for lm in landmarks_raw],
            dtype=np.float32,
        )

        if self.backend == "mediapipe_subdiv":
            return self._subdivision_backend(img_rgb, landmarks_2d)
        elif self.backend == "flame":
            return self._flame_backend(img_rgb, landmarks_2d)
        elif self.backend == "emoca":
            return self._emoca_backend(img_rgb, landmarks_2d)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def process_multi_view(self, photo_paths, front_index=0):
        """Multi-view 3D face reconstruction: average EMOCA shapecodes
        from multiple photos for better 3D shape (nose/chin/philtrum).

        Args:
            photo_paths: list of photo file paths (front + 1-2 side views)
            front_index: index of front-facing photo in the list
        """
        import torch
        n = len(photo_paths)

        # 1. Load all images
        all_imgs = []
        for path in photo_paths:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                raise ValueError(f"Cannot read image from {path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            all_imgs.append(img_rgb)

        # 2. Encode all photos via EMOCA → collect shapecodes
        shapecodes = []
        for i, img_rgb in enumerate(all_imgs):
            codedict = self._emoca_encode_one(img_rgb)
            shapecodes.append(codedict['shapecode'].squeeze(0))
            print(f"  EMOCA encoded view {i}: {photo_paths[i]}")

        # 3. Average shapecode for better 3D shape
        shapecode_avg = torch.stack(shapecodes).mean(dim=0, keepdim=True)

        # 4. Decode with averaged shape + neutral expression/pose
        emoca = self._emoca_model
        expcode_zero = torch.zeros(1, 50)
        posecode_zero = torch.zeros(1, 6)
        flame = emoca.deca.flame
        with torch.no_grad():
            result = flame(shape_params=shapecode_avg,
                           expression_params=expcode_zero,
                           pose_params=posecode_zero)
            if len(result) >= 4:
                verts_world, _, _, _ = result
            else:
                verts_world, _, _ = result
        verts_world = verts_world.squeeze(0).numpy()

        # 5. Detect landmarks on front photo → build subdivided mesh
        front_img = all_imgs[front_index]
        h, w = front_img.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=front_img)
        res = self.landmarker.detect(mp_image)
        if not res.face_landmarks or len(res.face_landmarks) == 0:
            raise ValueError(f"No face detected in front photo: {photo_paths[front_index]}")
        lm_raw = res.face_landmarks[0]
        front_lm = np.array([[lm.x * w, lm.y * h] for lm in lm_raw], dtype=np.float32)
        subdiv_data = self._subdivision_backend(front_img, front_lm)

        # 6. Transfer subdiv canonical to FLAME → flame_lm_idx
        from modules.flame_model import FLAMEModel
        flame_model = FLAMEModel(self.flame_path or '/mnt/f/FLAME2020/FLAME2020/generic_model.pkl')
        mapping = flame_model.build_transfer_map(
            subdiv_data["subdiv_verts_2d"],
            subdiv_data["faces"],
        )
        flame_canonical = flame_model.transfer_vertices(
            subdiv_data["canonical_vertices"], mapping
        )

        sub_scale = subdiv_data["scale"]
        sub_center = subdiv_data["center"]
        sub_aspect = subdiv_data["aspect"]
        flame_verts_2d = np.zeros((flame_model.n_verts, 2), dtype=np.float32)
        flame_verts_2d[:, 0] = w * (-(flame_canonical[:, 0] + sub_center[0]) / sub_scale + 0.5)
        flame_verts_2d[:, 1] = h * (-(flame_canonical[:, 1] + sub_center[1]) / (sub_scale * sub_aspect) + 0.5)

        flame_lm_idx = np.zeros(478, dtype=np.int32)
        for i in range(478):
            flame_lm_idx[i] = int(np.argmin(np.sum((flame_verts_2d - front_lm[i]) ** 2, axis=1)))

        # 7. Compute z from the multi-view FLAME mesh at landmark positions
        lm_canon = flame_canonical[flame_lm_idx]
        can_face_w = float(lm_canon[:, 0].max() - lm_canon[:, 0].min())
        mod_face_w = float(verts_world[:, 0].max() - verts_world[:, 0].min())
        z_scale = can_face_w / max(mod_face_w, 1e-6) * 0.08
        flame_z_canon = verts_world[:, 2] * z_scale

        # Filter FLAME vertices to face hull (exclude back-of-head/neck)
        flame_v2d_filt, flame_z_filt = _filter_points_in_hull(
            front_lm, flame_verts_2d, flame_z_canon)

        # 8. Interpolate z from face-region FLAME vertices

        subdiv_z = LinearNDInterpolator(flame_v2d_filt, flame_z_filt)(
            subdiv_data["subdiv_verts_2d"][:, 0],
            subdiv_data["subdiv_verts_2d"][:, 1],
        )
        subdiv_z = np.nan_to_num(subdiv_z, nan=0.0)

        # Smooth z to reduce normal deviations around nose/mouth
        sv2d = subdiv_data["subdiv_verts_2d"]
        subdiv_z = _smooth_z_laplacian(subdiv_z, subdiv_data["faces"], iterations=2, lambda_=0.4)

        # 8c. Philtrum groove depression (flat topology stretches texture → visual streak)
        philtrum_cx = (float(front_lm[0, 0]) + float(front_lm[164, 0]) + float(front_lm[267, 0])) / 3.0
        philtrum_cy = (float(front_lm[0, 1]) + float(front_lm[164, 1]) + float(front_lm[267, 1])) / 3.0
        philtrum_r = float(np.linalg.norm(front_lm[164] - front_lm[267])) * 0.45
        dist_p = np.sqrt((sv2d[:, 0] - philtrum_cx) ** 2 + (sv2d[:, 1] - philtrum_cy) ** 2)
        in_phil = dist_p < philtrum_r
        if np.any(in_phil):
            z_span = max(np.ptp(subdiv_z), 0.01)
            groove_depth = max(z_span * 0.3, 0.012)
            falloff = np.clip(1.0 - dist_p[in_phil] / philtrum_r, 0.0, 1.0)
            falloff = falloff * falloff
            subdiv_z[in_phil] -= groove_depth * falloff

        # 9. Apply z to canonical vertices
        canonical_z = subdiv_data["canonical_vertices"].copy()
        canonical_z[:, 2] = subdiv_z

        eye_spheres = _build_eye_spheres(canonical_z, sv2d, front_img,
                                          subdiv_data["landmark_indices"], front_lm,
                                          w, h, subdiv_data["scale"])

        print(f"Multi-view: averaged {n} shapecodes, z_range=[{subdiv_z.min():.4f},{subdiv_z.max():.4f}]")

        # 10. Build result
        result = dict(subdiv_data)
        result.update({
            "canonical_vertices": canonical_z,
            "landmark_indices": subdiv_data["landmark_indices"],
            "model_type": "emoca_subdiv",
            "subdiv_data": subdiv_data,
            "eye_spheres": eye_spheres,
            "emoca_codes": {
                "shapecode": shapecode_avg.squeeze(0).numpy(),
            },
        })
        return result

    def _emoca_load(self):
        import sys
        from pathlib import Path
        _emoca_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "emoca")
        if _emoca_root not in sys.path:
            sys.path.insert(0, _emoca_root)
        from gdl_apps.EMOCA.utils.load import load_model
        import gdl as _gdl
        if not hasattr(self, '_emoca_model'):
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
            models_path = str(Path(_gdl.__file__).parents[1] / 'assets' / 'EMOCA' / 'models')
            self._emoca_model, self._emoca_conf = load_model(
                models_path, 'EMOCA_v2_lr_mse_20', 'detail')
            self._emoca_model.eval()
        return self._emoca_model

    def _emoca_encode_one(self, img_rgb):
        """Run EMOCA encode on a single image, return codedict."""
        import torch
        emoca = self._emoca_load()
        h, w = img_rgb.shape[:2]
        sz = min(h, w)
        crop = img_rgb[(h - sz) // 2:(h + sz) // 2, (w - sz) // 2:(w + sz) // 2]
        crop = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5
        batch = {'image': img_tensor.unsqueeze(1)}
        with torch.no_grad():
            codedict = emoca.encode(batch, training=False)
        return codedict

    def _flame_backend(self, img_rgb, landmarks_2d):
        """FLAME-topology face reconstruction via subdivided MediaPipe mesh transfer."""
        # 1. Build subdivided mesh (same as mediapipe_subdiv)
        subdiv_data = self._subdivision_backend(img_rgb, landmarks_2d)

        # 2. Load FLAME model and build transfer map
        flame = FLAMEModel(self.flame_path)

        mapping = flame.build_transfer_map(
            subdiv_data["subdiv_verts_2d"],
            subdiv_data["faces"],
        )

        # 3. Transfer canonical vertices to FLAME
        flame_canonical = flame.transfer_vertices(
            subdiv_data["canonical_vertices"], mapping
        )

        # 4. Transfer UVs to FLAME
        flame_uvs = flame.transfer_vertices(
            np.column_stack([
                subdiv_data["uv_coords"],
                np.zeros(len(subdiv_data["uv_coords"])),
            ]),
            mapping,
        )[:, :2]
        flame_uvs = np.clip(flame_uvs, 0.0, 1.0)

        # 5. Fix face winding directly on transferred canonical vertices
        flame_faces = _fix_winding_2d(flame.faces, flame_canonical[:, :2])

        # 6. Build FLAME vertex 2D positions for landmark index lookup
        sub_scale = subdiv_data["scale"]
        sub_center = subdiv_data["center"]
        sub_aspect = subdiv_data["aspect"]
        h, w = img_rgb.shape[:2]

        flame_verts_2d = np.zeros((flame.n_verts, 2), dtype=np.float32)
        flame_verts_2d[:, 0] = flame_canonical[:, 0]
        flame_verts_2d[:, 1] = flame_canonical[:, 1]
        # Invert canonical normalization to get image-space 2D coords
        flame_verts_2d[:, 0] = w * (-(flame_canonical[:, 0] + sub_center[0]) / sub_scale + 0.5)
        flame_verts_2d[:, 1] = h * (-(flame_canonical[:, 1] + sub_center[1]) / (sub_scale * sub_aspect) + 0.5)

        flame_lm_idx = np.zeros(478, dtype=np.int32)
        for i in range(478):
            flame_lm_idx[i] = int(np.argmin(np.sum((flame_verts_2d - landmarks_2d[i]) ** 2, axis=1)))

        result = dict(subdiv_data)
        result.update({
            "canonical_vertices": flame_canonical,
            "flame_vertices": flame_canonical,
            "faces": flame_faces,
            "uv_coords": flame_uvs,
            "flame_mapping": mapping,
            "landmark_indices": flame_lm_idx,
            "model_type": "flame",
            "subdiv_data": subdiv_data,
            "eye_spheres": subdiv_data.get("eye_spheres", None),
        })
        return result

    def _emoca_backend(self, img_rgb, landmarks_2d):
        """EMOCA-based FLAME reconstruction: encode photo via EMOCA, align mesh to landmarks."""
        import sys
        import torch
        from pathlib import Path

        # Lazy import EMOCA (add emoca/ to path if needed)
        _emoca_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "emoca")
        if _emoca_root not in sys.path:
            sys.path.insert(0, _emoca_root)
        
        from gdl_apps.EMOCA.utils.load import load_model
        import gdl as _gdl

        # 1. Build subdivided mesh (for texture + canonical space params)
        subdiv_data = self._subdivision_backend(img_rgb, landmarks_2d)
        h, w = img_rgb.shape[:2]

        # 2. Load EMOCA model (one-time, cached)
        if not hasattr(self, '_emoca_model'):
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
            models_path = str(Path(_gdl.__file__).parents[1] / 'assets' / 'EMOCA' / 'models')
            self._emoca_model, self._emoca_conf = load_model(
                models_path, 'EMOCA_v2_lr_mse_20', 'detail')
            self._emoca_model.eval()
        emoca = self._emoca_model

        # 3. Preprocess photo for EMOCA (center crop + resize to 224x224)
        sz = min(h, w)
        crop = img_rgb[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2]
        crop = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(crop).permute(2,0,1).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5
        batch = {'image': img_tensor.unsqueeze(1)}

        # 4. Encode photo
        with torch.no_grad():
            codedict = emoca.encode(batch, training=False)

        # 5. Get FLAME mesh with neutral expression/pose
        shapecode = codedict['shapecode']
        texcode = codedict['texcode']
        cam = codedict['cam']
        expcode_zero = torch.zeros_like(codedict['expcode'])
        posecode_zero = torch.zeros_like(codedict['posecode'])

        flame = emoca.deca.flame
        with torch.no_grad():
            result = flame(
                shape_params=shapecode,
                expression_params=expcode_zero,
                pose_params=posecode_zero)
            if len(result) >= 4:
                verts_world, _, _, _ = result
            else:
                verts_world, _, _ = result
        verts_world = verts_world.squeeze(0).numpy()

        # 6. Transfer subdivided canonical to FLAME (needed for flame_lm_idx matching)
        from modules.flame_model import FLAMEModel
        flame_model = FLAMEModel(self.flame_path or '/mnt/f/FLAME2020/FLAME2020/generic_model.pkl')
        mapping = flame_model.build_transfer_map(
            subdiv_data["subdiv_verts_2d"],
            subdiv_data["faces"],
        )
        flame_canonical = flame_model.transfer_vertices(
            subdiv_data["canonical_vertices"], mapping
        )

        # 7. Find nearest FLAME vertex for each landmark in pixel space
        sub_scale = subdiv_data["scale"]
        sub_center = subdiv_data["center"]
        sub_aspect = subdiv_data["aspect"]
        flame_verts_2d = np.zeros((flame_model.n_verts, 2), dtype=np.float32)
        flame_verts_2d[:, 0] = w * (-(flame_canonical[:, 0] + sub_center[0]) / sub_scale + 0.5)
        flame_verts_2d[:, 1] = h * (-(flame_canonical[:, 1] + sub_center[1]) / (sub_scale * sub_aspect) + 0.5)

        flame_lm_idx = np.zeros(478, dtype=np.int32)
        for i in range(478):
            flame_lm_idx[i] = int(np.argmin(np.sum((flame_verts_2d - landmarks_2d[i]) ** 2, axis=1)))

        # 8. Compute EMOCA z at landmark positions (canonical space)
        lm_canon = flame_canonical[flame_lm_idx]
        can_face_w = float(lm_canon[:, 0].max() - lm_canon[:, 0].min())
        mod_face_w = float(verts_world[:, 0].max() - verts_world[:, 0].min())
        z_scale = can_face_w / max(mod_face_w, 1e-6) * 0.08
        flame_z_canon = verts_world[:, 2] * z_scale

        # Filter FLAME vertices to face hull (exclude back-of-head/neck)
        flame_v2d_filt, flame_z_filt = _filter_points_in_hull(
            landmarks_2d, flame_verts_2d, flame_z_canon)

        # 9. Interpolate EMOCA z from face-region FLAME vertices
        subdiv_z = LinearNDInterpolator(flame_v2d_filt, flame_z_filt)(
            subdiv_data["subdiv_verts_2d"][:, 0],
            subdiv_data["subdiv_verts_2d"][:, 1],
        )
        subdiv_z = np.nan_to_num(subdiv_z, nan=0.0)

        sv2d = subdiv_data["subdiv_verts_2d"]

        # Smooth z to reduce normal deviations around nose/mouth
        subdiv_z = _smooth_z_laplacian(subdiv_z, subdiv_data["faces"], iterations=2, lambda_=0.4)

        # 9c. Philtrum groove depression (flat topology stretches texture → visual streak)
        philtrum_cx = (float(landmarks_2d[0, 0]) + float(landmarks_2d[164, 0]) + float(landmarks_2d[267, 0])) / 3.0
        philtrum_cy = (float(landmarks_2d[0, 1]) + float(landmarks_2d[164, 1]) + float(landmarks_2d[267, 1])) / 3.0
        philtrum_r = float(np.linalg.norm(landmarks_2d[164] - landmarks_2d[267])) * 0.45
        dist_p = np.sqrt((sv2d[:, 0] - philtrum_cx) ** 2 + (sv2d[:, 1] - philtrum_cy) ** 2)
        in_phil = dist_p < philtrum_r
        if np.any(in_phil):
            z_span = max(np.ptp(subdiv_z), 0.01)
            groove_depth = max(z_span * 0.3, 0.012)
            falloff = np.clip(1.0 - dist_p[in_phil] / philtrum_r, 0.0, 1.0)
            falloff = falloff * falloff
            subdiv_z[in_phil] -= groove_depth * falloff

        # 10. Override subdivided mesh z with EMOCA depth
        canonical_z = subdiv_data["canonical_vertices"].copy()
        canonical_z[:, 2] = subdiv_z

        eye_spheres = _build_eye_spheres(canonical_z, sv2d, img_rgb,
                                          subdiv_data["landmark_indices"], landmarks_2d,
                                          w, h, subdiv_data["scale"])

        # 11. Build result (subdiv mesh, subdiv-style displacement, no FLAME render)
        result = dict(subdiv_data)
        result.update({
            "canonical_vertices": canonical_z,
            "landmark_indices": subdiv_data["landmark_indices"],
            "model_type": "emoca_subdiv",
            "subdiv_data": subdiv_data,
            "eye_spheres": eye_spheres,
            "emoca_codes": {
                "shapecode": shapecode.squeeze(0).numpy(),
                "expcode": codedict['expcode'].squeeze(0).numpy(),
            },
        })
        return result

    def close(self):
        self.landmarker.close()
