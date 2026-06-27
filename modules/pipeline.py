import time
import math
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .camera import Camera
from .face_tracker import FaceTracker
from .renderer import FaceRenderer
from .face_reconstructor import FaceReconstructor
from .flame_model import FLAMEModel

DEFAULT_MODEL_PATH = "models/face_landmarker.task"
DEFAULT_FLAME_PATH = None


def _landmarks_to_canonical(landmarks_2d, img_w, img_h, scale, center, aspect):
    pts = np.zeros((len(landmarks_2d), 3), dtype=np.float32)
    pts[:, 0] = -(landmarks_2d[:, 0] / img_w - 0.5)
    pts[:, 1] = -(landmarks_2d[:, 1] / img_h - 0.5)
    pts[:, :2] *= scale
    pts[:, 1] *= aspect
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts


def _webcam_to_canonical(landmarks_3d, center, aspect, photo_span=1.0):
    pts = np.zeros((len(landmarks_3d), 3), dtype=np.float32)
    pts[:, 0] = -(landmarks_3d[:, 0] - 0.5)
    pts[:, 1] = -(landmarks_3d[:, 1] - 0.5)
    z_min = landmarks_3d[:, 2].min()
    pts[:, 2] = -(landmarks_3d[:, 2] - z_min)

    # Normalise by webcam face width so canonical span = photo_span (face size locked)
    webcam_x = max(landmarks_3d[:, 0].max() - landmarks_3d[:, 0].min(), 0.05)
    norm = photo_span / webcam_x
    pts[:, 0] *= norm
    pts[:, 1] *= norm
    pts[:, 2] *= 0.15 * norm  # Z uses same scale

    pts[:, 1] *= aspect
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts


class Pipeline:
    def __init__(self, photo_path, camera_device=0, camera_width=640,
                 camera_height=480, render_width=1200, render_height=1200,
                 model_path=None, subdiv_iterations=2, backend="mediapipe_subdiv",
                 flame_path=None):
        self.backend = backend
        self._flame = None

        self.camera = Camera(
            device_id=camera_device,
            width=camera_width,
            height=camera_height,
        )

        photo_label = photo_path if isinstance(photo_path, str) else photo_path[0]
        n_views = 1 if isinstance(photo_path, str) else len(photo_path)
        print(f"Reconstructing face from photo: {photo_label} (backend={backend}, views={n_views})")
        reconstructor = FaceReconstructor(
            backend=backend,
            subdiv_iterations=subdiv_iterations,
            flame_path=flame_path,
        )
        photo_data = reconstructor.process(photo_path)
        reconstructor.close()

        self.canonical_vertices = photo_data["canonical_vertices"].astype(np.float32)
        self.faces = photo_data["faces"]
        self.uv_coords = photo_data["uv_coords"]
        self.texture_img = photo_data["texture_img"]
        self.mouth_interior_mask = photo_data.get("mouth_interior_mask", None)

        # Pre-build face list with mouth interior triangles culled
        mouth_face_mask = photo_data.get("mouth_interior_face_mask")
        if mouth_face_mask is not None and len(mouth_face_mask) == len(self.faces):
            self.faces_no_mouth = self.faces[~mouth_face_mask]
            self._mouth_face_mask = mouth_face_mask
        else:
            self.faces_no_mouth = self.faces
            self._mouth_face_mask = None
        self.photo_landmarks_2d = photo_data["landmarks_2d"]
        self.image_size = photo_data["image_size"]
        self.scale = photo_data["scale"]
        self.center = photo_data["center"]
        self.aspect = photo_data["aspect"]
        self.photo_face_span = float(self.canonical_vertices[:, 0].max() - self.canonical_vertices[:, 0].min())
        self.model_type = photo_data.get("model_type", backend)

        n_verts = len(self.canonical_vertices)
        n_faces = len(self.faces)
        print(f"Reconstructed mesh: {n_verts} vertices, {n_faces} faces "
              f"(backend={self.model_type})")

        self.photo_lm_canonical = _landmarks_to_canonical(
            self.photo_landmarks_2d,
            self.image_size[0], self.image_size[1],
            self.scale, self.center, self.aspect,
        )

        # Precompute eye closure target centers (iris mean from photo)
        right_center_2d = self.photo_lm_canonical[[468, 469, 470, 471, 472]].mean(axis=0)[:2]
        left_center_2d = self.photo_lm_canonical[[473, 474, 475, 476, 477]].mean(axis=0)[:2]
        self._eyelid_closure = {
            'right': {
                'upper': [246, 161, 160, 159, 158, 157, 173],
                'lower': [7, 163, 144, 145, 153, 154, 155],
                'center': right_center_2d,
            },
            'left': {
                'upper': [398, 384, 385, 386, 387, 388, 466],
                'lower': [382, 381, 380, 374, 373, 390, 249],
                'center': left_center_2d,
            },
        }

        # Handle FLAME-specific data
        if self.model_type in ("flame", "emoca"):
            subdiv_data = photo_data["subdiv_data"]
            self._subdiv_verts_2d = subdiv_data["subdiv_verts_2d"]
            self._flame_mapping = photo_data["flame_mapping"]
            self._flame_lm_idx = photo_data["landmark_indices"]
            self._subdiv_bary_v = subdiv_data.get("subdiv_bary_v", None)
            self._subdiv_bary_w = subdiv_data.get("subdiv_bary_w", None)
        else:
            self._subdiv_verts_2d = photo_data.get("subdiv_verts_2d", None)
            self._flame_mapping = None
            self._flame_lm_idx = None
            self._subdiv_bary_v = photo_data.get("subdiv_bary_v", None)
            self._subdiv_bary_w = photo_data.get("subdiv_bary_w", None)

        mp_path = model_path or DEFAULT_MODEL_PATH
        self.tracker = FaceTracker(model_path=mp_path)

        y_center = float(self.canonical_vertices[:, 1].mean())
        self.renderer = FaceRenderer(
            width=render_width,
            height=render_height,
            yfov=self._compute_yfov(photo_data, target_fill=0.50),
            target=[0.0, y_center, 0.0],
        )
        self.renderer.set_mesh_data(self.faces, self.uv_coords, self.texture_img)

        # Eye spheres (textured hemispheres behind eye openings)
        es = photo_data.get("eye_spheres")
        if es is not None:
            self.renderer.set_eye_spheres(es)

        self.camera_width = camera_width
        self.camera_height = camera_height
        self.render_width = render_width
        self.render_height = render_height

        self._running = False
        self._frame_count = 0
        self._start_time = 0.0

        self._setup_displacement_clamp()

        self._eye_spheres = photo_data.get("eye_spheres", None)
        self._iris_smooth_disp = None  # EMA state for iris landmark displacement
        self._iris_raw_disp = None     # unsmoothed, for gaze sync
        self._eye_yaw_smooth = None    # EMA-smoothed yaw per eye
        self._eye_pitch_smooth = None  # EMA-smoothed pitch per eye
        self._right_pitch_delta = 0.0
        self._blink_right = 0.0
        self._blink_left = 0.0

    def _diag_philtrum(self, canonical, disp, deformed):
        """Diagnostic: print z stats for philtrum area landmarks."""
        idx = [2, 164, 267, 0, 19, 94, 1, 5]
        names = {
            2: "nose_bottom", 164: "philtrum_R", 267: "philtrum_L",
            0: "upper_lip", 19: "mid_columella", 94: "columella_base",
            1: "upper_columella", 5: "nose_lower",
        }
        n = len(canonical)
        print(f"\n[D] Frame {self._frame_count} ({n} verts)")
        print(f"  {'landmark':<16} {'can_z':>8} {'disp_z':>8} {'def_z':>8} {'can_x':>8} {'disp_x':>8}")
        for i in idx:
            if i < n:
                print(f"  {names.get(i,str(i)):<16} "
                      f"{canonical[i,2]:8.4f} {disp[i,2]:8.4f} {deformed[i,2]:8.4f} "
                      f"{canonical[i,0]:8.4f} {disp[i,0]:8.4f}")
        # Check: does the groove (2) get pushed behind the lip (0)?
        if 2 < n and 0 < n:
            dz_20 = deformed[2, 2] - deformed[0, 2]
            dz_canon_20 = canonical[2, 2] - canonical[0, 2]
            print(f"  deformed[2].z - deformed[0].z = {dz_20:.4f}  (canon: {dz_canon_20:.4f}) "
                  f"{'SELF-INTERSECT?' if dz_20 > 0.01 else 'OK'}")
        print(f"  def_z range in region: [{deformed[idx,2].min():.4f}, {deformed[idx,2].max():.4f}]")

    @staticmethod
    def _compute_yfov(photo_data, target_fill=0.50, camera_dist=3.0):
        """Compute yfov so the static face spans target_fill of viewport height.

        The clamped displacement margin (0.3 in _setup_displacement_clamp)
        is verified to still fit within this FOV for typical face meshes.
        """
        cv = photo_data["canonical_vertices"]
        x_span = float(cv[:, 0].max() - cv[:, 0].min())
        y_span = float(cv[:, 1].max() - cv[:, 1].min())
        max_span = max(x_span, y_span)
        visible_h = max_span / target_fill
        return 2.0 * math.atan2(visible_h / 2.0, camera_dist)

    def _setup_displacement_clamp(self):
        cv = self.canonical_vertices
        span_w = float(cv[:, 0].max() - cv[:, 0].min())
        span_h = float(cv[:, 1].max() - cv[:, 1].min())
        span_z = float(cv[:, 2].max() - cv[:, 2].min())
        margin = 0.3
        self._clamp_x = (float(cv[:, 0].min()) - margin * span_w,
                         float(cv[:, 0].max()) + margin * span_w)
        self._clamp_y = (float(cv[:, 1].min()) - margin * span_h,
                         float(cv[:, 1].max()) + margin * span_h)
        self._clamp_z = (float(cv[:, 2].min()) - margin * max(span_z, 0.05),
                         float(cv[:, 2].max()) + margin * max(span_z, 0.05))

    def _update_eye_spheres(self, deformed):
        """Position and rotate eye spheres to track face + gaze.

        Eye centre follows the deformed contour so the spheres move with
        head rotation.  Sphere size is fixed (scale=1.0) — face size is
        already locked by _webcam_to_canonical.
        """
        if self._iris_smooth_disp is None or self._eye_spheres is None:
            return

        n_eyes = len(self._eye_spheres)
        if self._eye_yaw_smooth is None:
            self._eye_yaw_smooth = [0.0] * n_eyes
            self._eye_pitch_smooth = [0.0] * n_eyes

        for i, sphere in enumerate(self._eye_spheres):
            # Eye centre from deformed contour (tracks head rotation)
            contour_cur = deformed[sphere['contour_v_idx']]
            cur_center = contour_cur.mean(axis=0).copy()
            cur_center[2] = float(contour_cur[:, 2].mean()) - sphere['rz'] - 0.002

            # Push sphere back during blink so gap between eyelids won't reveal it
            blink = self._blink_right if i == 0 else self._blink_left
            if blink > 0.01:
                cur_center[2] -= blink * sphere['rz'] * 0.8
            scale = 1.0

            # Default gaze baseline from STATIC sphere centre — immune to
            # asymmetric expression displacement between left and right eyes
            cam_vec = np.array([0.0, 0.0, 3.0]) - sphere['center']
            default_yaw = float(np.arctan2(cam_vec[0], cam_vec[2]))
            default_pitch = float(np.arctan2(-cam_vec[1], cam_vec[2])) * 0.6

            # Gaze from iris displacement → raw yaw/pitch
            start = i * 5
            disp = self._iris_smooth_disp[start:start + 5].mean(axis=0) * 1.2
            rx_scaled = sphere['rx'] * scale + 1e-6
            ry_scaled = sphere['ry'] * scale + 1e-6
            raw_mag_yaw = abs(disp[0])

            # Pitch tracking: left eye mirrors right eye's vertical movement.
            # Both eyes move together vertically, so copying the right eye's
            # pitch delta eliminates the persistent MediaPipe left-eye bias.
            pitch_delta = float(-disp[1] / ry_scaled)
            if i == 1:
                pitch_delta = self._right_pitch_delta

            dz_thresh = 0.0005
            if raw_mag_yaw < dz_thresh:
                target_yaw = default_yaw
            else:
                target_yaw = default_yaw + float(disp[0] / rx_scaled)

            if i == 0:
                self._right_pitch_delta = pitch_delta
                raw_mag_pitch = abs(disp[1])
                if raw_mag_pitch < dz_thresh:
                    target_pitch = default_pitch
                else:
                    target_pitch = default_pitch + pitch_delta
            else:
                target_pitch = default_pitch + pitch_delta

            # Clamp to visible hemisphere range (front face only)
            max_angle = np.deg2rad(35)
            target_yaw = np.clip(target_yaw, -max_angle, max_angle)
            target_pitch = np.clip(target_pitch, -max_angle, max_angle)

            # EMA smooth: fast attack when gaze moves, slower release to zero
            cur_yaw = self._eye_yaw_smooth[i]
            cur_pitch = self._eye_pitch_smooth[i]
            if abs(target_yaw) > abs(cur_yaw) or abs(target_pitch) > abs(cur_pitch):
                alpha = 0.35   # fast: track new gaze direction
            else:
                alpha = 0.08   # slow: drift back to centre smoothly
            self._eye_yaw_smooth[i] = cur_yaw + alpha * (target_yaw - cur_yaw)
            self._eye_pitch_smooth[i] = cur_pitch + alpha * (target_pitch - cur_pitch)

            self.renderer.set_eye_pose(i, cur_center, scale,
                                        self._eye_yaw_smooth[i],
                                        self._eye_pitch_smooth[i])



    def _compute_displacement(self, webcam_landmarks_3d, blendshapes=None):
        webcam_lm_canonical = _webcam_to_canonical(
            webcam_landmarks_3d,
            self.center, self.aspect, self.photo_face_span,
        )

        disp_478 = webcam_lm_canonical - self.photo_lm_canonical

        # Remove global translation: lock face at origin using stable nose bridge
        stable_lm = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1]
        disp_478[:, :2] -= disp_478[stable_lm, :2].mean(axis=0)

        # Amplify iris landmark displacement for visible gaze tracking
        iris_lm = np.array(list(range(468, 478)), dtype=np.int32)
        self._iris_raw_disp = disp_478[iris_lm].copy()  # unsmoothed, for Z bulge sync
        if self._iris_smooth_disp is None:
            self._iris_smooth_disp = self._iris_raw_disp.copy()
        self._iris_smooth_disp = 0.6 * self._iris_smooth_disp + 0.4 * self._iris_raw_disp
        disp_478[iris_lm] = self._iris_smooth_disp * 1.3

        # Eyelid closure from MediaPipe blendshapes
        if blendshapes is not None and hasattr(self, '_eyelid_closure'):
            blink_left   = blendshapes.get('eyeBlinkLeft',   0.0)
            blink_right  = blendshapes.get('eyeBlinkRight',  0.0)
            self._blink_right = blink_right
            self._blink_left = blink_left
            squint_left  = blendshapes.get('eyeSquintLeft',  0.0)
            squint_right = blendshapes.get('eyeSquintRight', 0.0)
            wide_left    = blendshapes.get('eyeWideLeft',    0.0)
            wide_right   = blendshapes.get('eyeWideRight',   0.0)
            look_up_l    = blendshapes.get('eyeLookUpLeft',  0.0)
            look_up_r    = blendshapes.get('eyeLookUpRight', 0.0)
            look_dn_l    = blendshapes.get('eyeLookDownLeft', 0.0)
            look_dn_r    = blendshapes.get('eyeLookDownRight',0.0)

            def _apply_eyelid(lm_ids, direction_sign, scale, upper_factor, lower_factor):
                """Apply scaled eyelid movement. direction_sign +1=toward center, -1=away."""
                sc = max(0.0, scale * (1.0 - 0.05))  # deadzone
                if sc < 1e-4:
                    return
                for lm in lm_ids:
                    vec = (self._eyelid_closure[side_key]['center'] -
                           self.photo_lm_canonical[lm, :2])
                    disp_478[lm, :2] += vec * sc * direction_sign * upper_factor
                for lm in self._eyelid_closure[side_key]['lower']:
                    vec = (self._eyelid_closure[side_key]['center'] -
                           self.photo_lm_canonical[lm, :2])
                    disp_478[lm, :2] += vec * sc * direction_sign * lower_factor

            for side_key in ('right', 'left'):
                is_left = (side_key == 'left')
                bk  = blink_left  if is_left else blink_right
                sq  = squint_left if is_left else squint_right
                wd  = wide_left   if is_left else wide_right
                lup = look_up_l   if is_left else look_up_r
                ldn = look_dn_l   if is_left else look_dn_r
                upper = self._eyelid_closure[side_key]['upper']

                # Blink: close asymmetric (upper faster than lower)
                _apply_eyelid(upper, +1, bk * 0.35, 0.9, 0.4)
                # Squint: close symmetric (upper≈lower)
                _apply_eyelid(upper, +1, sq * 0.20, 0.6, 0.5)
                # Wide: open (push away from center)
                _apply_eyelid(upper, -1, wd * 0.10, 0.5, 0.3)
                # Look up: upper eyelid retracts
                _apply_eyelid(upper, -1, lup * 0.08, 0.7, 0.0)
                # Look down: upper eyelid drops, lower eyelid rises
                _apply_eyelid(upper, +1, ldn * 0.08, 0.7, 0.3)

        if self.model_type in ("flame", "emoca") and self._subdiv_verts_2d is not None:
            subdiv_disp = self._interpolate_disp_barycentric(disp_478)
            disp = self._transfer_flame_disp(subdiv_disp)
        elif len(self.canonical_vertices) == 478:
            disp = disp_478
        elif self._subdiv_verts_2d is not None:
            disp = self._interpolate_disp_barycentric(disp_478)
        else:
            disp = disp_478

        # Mouth interior Z recession: push inner-mouth surface backward
        # when jaw opens so stretched upper↔lower lip triangles are
        # occluded behind the lip surfaces.
        if (blendshapes is not None and self.mouth_interior_mask is not None
                and len(self.mouth_interior_mask) == len(disp)):
            jaw_open = blendshapes.get('jawOpen', 0.0)
            if jaw_open > 0.05:
                disp[self.mouth_interior_mask, 2] += -jaw_open * 0.05

        return disp

    def _interpolate_disp_barycentric(self, disp_478):
        if self._subdiv_bary_v is None:
            # Fallback: Delaunay-based interpolation
            interp_x = LinearNDInterpolator(self.photo_landmarks_2d, disp_478[:, 0])
            interp_y = LinearNDInterpolator(self.photo_landmarks_2d, disp_478[:, 1])
            interp_z = LinearNDInterpolator(self.photo_landmarks_2d, disp_478[:, 2])
            s2d = self._subdiv_verts_2d
            n_verts = len(self.canonical_vertices)
            disp = np.zeros((n_verts, 3), dtype=np.float32)
            disp[:, 0] = interp_x(s2d[:, 0], s2d[:, 1])
            disp[:, 1] = interp_y(s2d[:, 0], s2d[:, 1])
            disp[:, 2] = interp_z(s2d[:, 0], s2d[:, 1])
            disp[:478] = disp_478
            disp = np.nan_to_num(disp, nan=0.0)
            return disp

        # Vectorized barycentric interpolation (mesh-topology, not Delaunay)
        v0 = self._subdiv_bary_v[:, 0]
        v1 = self._subdiv_bary_v[:, 1]
        v2 = self._subdiv_bary_v[:, 2]
        w0 = self._subdiv_bary_w[:, 0:1]
        w1 = self._subdiv_bary_w[:, 1:2]
        w2 = self._subdiv_bary_w[:, 2:3]
        return (w0 * disp_478[v0] + w1 * disp_478[v1] + w2 * disp_478[v2]).astype(np.float32)

    def _transfer_flame_disp(self, subdiv_disp):
        mapping = self._flame_mapping
        tri_idx = mapping["tri_idx"]
        bary = mapping["bary_coords"]
        source_faces = mapping["source_faces"]

        n_flame = len(self.canonical_vertices)
        flame_disp = np.zeros((n_flame, 3), dtype=np.float32)
        for i in range(n_flame):
            fi = tri_idx[i]
            a, b, c = source_faces[fi]
            u, v, w = bary[i]
            flame_disp[i] = (u * subdiv_disp[a]
                             + v * subdiv_disp[b]
                             + w * subdiv_disp[c])
        return flame_disp

    def run(self):
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()

        print("Pipeline started. Press 'q' to quit.")
        print(f"Camera: {self.camera_width}x{self.camera_height}")
        print(f"Renderer: {self.render_width}x{self.render_height}")

        while self._running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.tracker.process(frame_rgb)

            if result is not None:
                disp = self._compute_displacement(result["landmarks_3d"], result.get("blendshapes"))
                deformed = self.canonical_vertices + disp
                deformed[:, 0] = np.clip(deformed[:, 0], *self._clamp_x)
                deformed[:, 1] = np.clip(deformed[:, 1], *self._clamp_y)
                deformed[:, 2] = np.clip(deformed[:, 2], *self._clamp_z)

                self._update_eye_spheres(deformed)

                if self._frame_count % 30 == 0:
                    self._diag_philtrum(self.canonical_vertices, disp, deformed)

                # Dynamic mouth interior face culling: when jaw opens, remove
                # the triangles connecting upper↔lower lip to hide webbing.
                use_faces = self.faces
                if self._mouth_face_mask is not None:
                    jaw_open = result.get("blendshapes", {}).get("jawOpen", 0.0)
                    if jaw_open > 0.05:
                        use_faces = self.faces_no_mouth

                rendered = self.renderer.render(deformed, faces=use_faces)
                rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

                cam_display = frame.copy()
                lm_pix = result["landmarks_pixel"]
                if len(lm_pix) > 0:
                    for pt in lm_pix[::3]:
                        cv2.circle(cam_display,
                                   (int(pt[0]), int(pt[1])), 1,
                                   (0, 255, 0), -1)

                self._frame_count += 1
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    fps = self._frame_count / elapsed
                    cv2.putText(cam_display, f"FPS: {fps:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    cv2.putText(rendered_bgr, f"FPS: {fps:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2)

                cv2.imshow("Camera + Landmarks", cv2.resize(cam_display, (320, 240)))
                cv2.imshow("3D Face (Photo Texture)", rendered_bgr)
            else:
                no_face = np.full((self.render_height, self.render_width, 3), 64,
                                  dtype=np.uint8)
                cv2.putText(no_face, "No face detected",
                            (50, self.render_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2)
                cv2.imshow("Camera + Landmarks",
                           cv2.resize(frame, (320, 240)))
                cv2.imshow("3D Face (Photo Texture)", no_face)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                self._running = False
                break

        self._cleanup()

    def _cleanup(self):
        self.camera.release()
        self.tracker.close()
        self.renderer.delete()
        cv2.destroyAllWindows()
        print("Pipeline stopped.")
