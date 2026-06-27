# AGENTS.md — Facial Expression Transfer System

## Project Overview

Real-time 3D facial expression transfer: take a single photo (or multiple) → reconstruct a textured 3D face mesh → drive it with webcam facial landmarks, producing a live avatar that mimics the user's expressions.

## Environment

| Item | Detail |
|------|--------|
| OS | WSL2 (Ubuntu) on Windows |
| GPU | RTX 5060 Ti (sm_120) |
| Python | 3.10 (venv at `venv/`) |
| CUDA | **NOT supported** (sm_120 too new for stable PyTorch) — all PyTorch runs on CPU |
| PyTorch3D | **NOT installed** — not needed for runtime rendering (pyrender used instead) |
| Windows mounts | `F:\` → `/mnt/f/` |

## Architecture

```
Photo ──→ [Backend Choice] ──→ Canonical 3D Face Mesh
                                      │
Webcam ──→ FaceTracker (MediaPipe) ──→ Deformation per frame ──→ pyrender (EGL/OpenGL GPU) ──→ Live Avatar
```

### Three Backends

| Backend | mesh size | reconstruction method | speed |
|---------|-----------|----------------------|-------|
| `mediapipe_subdiv` (default) | ~6669v / 13063f | MediaPipe 478 landmarks + Loop subdiv | fast |
| `flame` | 5023v / 9976f | Subdiv mesh → barycentric transfer to FLAME topology | fast |
| `emoca` | ~6669v / 13063f | EMOCA encoder (ResNet/Swin) → FLAME decode → Z-depth transfer to subdiv mesh | **CPU-only, ~30s** |

All backends render via the same `FaceRenderer` in `modules/renderer.py` (pyrender / PBR metallic-roughness).

## Key Files

### Core Pipeline
- **`main.py`** — Entry point. `--backend {mediapipe_subdiv|flame|emoca}` `--photo PATH`
- **`modules/pipeline.py`** — Main loop: webcam capture → FaceTracker → displacement compute → deformation → render. Contains all gaze/blink/blendshape logic.
- **`modules/face_tracker.py`** — MediaPipe FaceLandmarker wrapper. `output_face_blendshapes=True` → returns blendshapes dict.

### Reconstruction
- **`modules/face_reconstructor.py`** — `FaceReconstructor` class. All three backends (`_subdivision_backend`, `_flame_backend`, `_emoca_backend`). Also generates hair quad, teeth mesh, mouth cavity mesh, eye data, corneal bulge, iris Z flattening.
- **`modules/flame_model.py`** — `FLAMEModel`: loads FLAME 2020 `.pkl`, barycentric vertex transfer from subdiv → FLAME topology.

### Rendering
- **`modules/renderer.py`** — `FaceRenderer`: pyrender-based. Multi-layer scene: `_hair_node` (z=-0.3) → `_mesh_node` (face, z≈0) → `_mouth_cavity_node` (dark fan) → `_teeth_node` (white strip). PBR: metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True, SKIP_CULL_FACES.

### EMOCA (3rd party, in `emoca/`)
- **`emoca/gdl_apps/EMOCA/utils/load.py`** — `load_model()` -> `DecaModule`
- **`emoca/gdl_apps/EMOCA/deca.py`** — `DecaModule.__init__` wraps `SRenderY` import in try/except (PyTorch3D optional)
- EMOCA model files: `emoca/assets/EMOCA_v2_lr_mse_20/` (DECA), `emoca/assets/FLAME/geometry/`
- EMOCA is used for **encode only** (expression/pose/shape params); rendering always done by pyrender.

### Data
- `assets/test_face.png` — default test photo
- `assets/canonical_face_data.pkl` — cached reconstructed data

## Feature Status

### Eye Geometry (complete)
| Feature | Details | Location |
|---------|---------|----------|
| Corneal bulge | Parabolic falloff (1-t²), R_px=max(ex,ey)*0.75, bulge_h max(span)*0.20*px_to_canon, distance-based vertex selection | `face_reconstructor.py:_apply_corneal_bulge` |
| Multi-vertex iris fan | 4 existing subdivided-mesh iris landmark vertices per eye as fan poles; contour→iris triangulation; eliminates single-pole normal singularity | `face_reconstructor.py:_subdivision_backend` (lines 424-473) |
| Iris Z flattening | Iris vertex Z averaged to mean both offline and per-frame → eliminates diamond artefact | `face_reconstructor.py:_subdivision_backend` + `pipeline.py:_apply_gaze_bulge` |
| Gaze tracking | EMA-smoothed iris displacement (α=0.4), amplified ×1.3 XY. Gaze-centered bulge with 30% static baseline. Scale-by-span (bulge_h and R_canon scaled by live_eye_span / static_eye_span, floor 0.3) | `pipeline.py:_compute_displacement` + `_apply_gaze_bulge` |
| Blink Z scaling | Only z_bulge (z_deformed - z_base) scaled by (1.0 - closure); z_base preserved; full blink returns Z to z_base=0.0 | `pipeline.py:_apply_gaze_bulge` |
| Eyelid blendshapes | Unified 5-shape: eyeBlink, eyeSquint, eyeWide, eyeLookUp, eyeLookDown (mediapipe blendshapes). Additive with deadzone | `pipeline.py:_apply_eyelid` |

### Visual Polish (complete)
| Feature | Details | Location |
|---------|---------|----------|
| Hair | Quad mesh from forehead Y upward to Y+1.5, z=-0.3. Textured with photo crop above face bbox (img[:y_min, :]). 4v/2f, doubleSided | `face_reconstructor.py:_build_hair_quad` + `renderer.py:_build_static_meshes` `_hair_node` |
| Teeth | 3-row white block (upper/middle/lower) from inner lip landmarks, 33v/40f, height 0.05 canonical units. White 4×4 texture (TextureVisuals), doubleSided=True. Z=+0.001 (in front of lip surface) | `face_reconstructor.py:_build_mouth_meshes` + `renderer.py:_build_static_meshes` `_teeth_node` |
| Mouth cavity | Dark fan mesh from inner lip polygon cycle (20 landmarks cycled + center), 21v/20f. Z=-0.008 (behind lip surface). | `face_reconstructor.py:_build_mouth_meshes` + `renderer.py:_build_static_meshes` `_mouth_cavity_node` |
| Philtrum groove | Z depression at triangle (0,164,267) centroid with quadratic falloff | `pipeline.py:_compute_displacement` |

### Other Features
| Feature | Details |
|---------|---------|
| Multi-view reconstruction | Average shapecodes from multiple photos |
| Dynamic FOV | yfov computed from canonical y_span targeting 50% viewport fill |
| Displacement clamp | face_min ± margin*span, margin=0.3 |
| Translation removal | Nose bridge displacement subtracted, face locked at origin |
| FLAME-dense Z interpolation | Filtered to face hull, Delaunay + Laplacian smooth (λ=0.4, 2 iter) |
| UV + displacement barycentric | MediaPipe tessellation triangles, vectorized triangle search |
| subdiv_bary_v/w init | Uses landmark_to_vertex[i] instead of assuming index=i |

## Known Constraints

1. **EMOCA CPU-only** — ~30s encode, no PyTorch3D → no SRenderY decode visualization
2. **Teeth/mouth cavity static** — positions from photo only, no jaw tracking
3. **Hair static** — from photo only, no dynamic changes
4. **Lighting fixed** — pure ambient (1.0), no dynamic lighting from webcam
5. **CUDA unavailable** — sm_120 not in pre-built PyTorch; builds from source unreliable
6. **Mesh topology fixed** — teeth/mouth/eye vertices at fixed indices; may break if MediaPipe updates landmarks

## MediaPipe Landmark Reference

### Inner mouth (teeth + cavity)
- Upper inner lip: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
- Lower inner lip: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
- Full cycle: [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]

### Eyes
- Left iris: [468, 469, 470, 471, 472]
- Right iris: [473, 474, 475, 476, 477]
- MediaPipe tessellation: eye interiors are topological **holes** (not in faces); mouth is continuous surface

## Common Commands

```bash
# Run with default backend
python main.py --photo assets/test_face.png

# Run with FLAME backend
python main.py --backend flame --photo assets/test_face.png

# Run with EMOCA (CPU-only, slow)
python main.py --backend emoca --photo assets/test_face.png

# Multi-view (multiple photos)
python main.py --photo photo1.png photo2.png photo3.png

# Full integration test
python -c "
from modules.face_reconstructor import FaceReconstructor
r = FaceReconstructor(backend='mediapipe_subdiv', subdiv_iterations=2)
data = r.process('assets/test_face.png')
print(f'Face: {len(data[\"canonical_vertices\"])}v/{len(data[\"faces\"])}f')
print(f'Teeth: {data[\"teeth_verts\"].shape} verts Z=[{data[\"teeth_verts\"][:,2].min():.3f},{data[\"teeth_verts\"][:,2].max():.3f}]')
print(f'Cavity: {data[\"mouth_cavity_verts\"].shape} verts Z=[{data[\"mouth_cavity_verts\"][:,2].min():.3f},{data[\"mouth_cavity_verts\"][:,2].max():.3f}]')
print(f'Hair Y: [{data[\"hair_verts\"][:,1].min():.2f},{data[\"hair_verts\"][:,1].max():.2f}]')
r.close()
"

# Activate venv
source venv/bin/activate
```

## Code Conventions

- Pure NumPy/scipy for mesh operations; no torch in core modules (except EMOCA backend)
- pyrender for all rendering with PBR materials
- `SKIP_CULL_FACES` on all meshes
- `doubleSided=True` on all pyrender Materials
- Static meshes built once in `_build_static_meshes()`, updated on `set_*()` calls
- Hair/teeth/mouth cavity forwarded through pipeline constructor from `FaceReconstructor` output dict
