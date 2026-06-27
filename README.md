# Facial Expression Tracing Demo / 面部表情追踪demo

 [中文](#中文) | [English](#english)

---

<a name="中文"></a>

## 中文

为了检验当前模型能力边界，通过opencode实现了一套demo，能力如下：

实时 3D 面部表情迁移：拍摄一张照片 → 重建带纹理的 3D 面部网格 → 通过摄像头面部关键点驱动，生成一个实时模仿你表情的 **虚拟头像**。

```
照片 ──→ [重建后端] ──→ 标准 3D 面部网格
                              │
摄像头 ──→ 面部追踪器 ──────→ 逐帧变形计算 ──→ pyrender 渲染 ──→ 实时头像
```

### 功能特性

- **3 种重建后端**（见下表）
- **摄像头实时驱动**，基于 MediaPipe FaceLandmarker（52 个 blendshape）
- **PBR 物理渲染**（pyrender，metallic-roughness，双面渲染，GPU EGL）
- **眼部几何**：视线追踪、眨眼、眼睑 blendshape、角膜凸起、虹膜 Z 轴压平
- **视觉润色**：头发平面、牙齿块、口腔腔体、人中凹槽
- **多视角重建**：多张照片平均 shapecode（EMOCA 后端）
- **动态 FOV**：根据标准面部跨度自动计算

### 运行环境

| 项目 | 详情 |
|------|------|
| 操作系统 | Linux（WSL2 Ubuntu 已测试），macOS/Windows 应可用 |
| Python | 3.10 |
| GPU | 不需要（CPU 渲染通过 pyrender/EGL） |
| 摄像头 | 标准 USB 摄像头（OpenCV） |

### 重建后端

| 后端 | 顶点/面数 | 重建方法 | 速度 |
|------|----------|----------|------|
| `mediapipe_subdiv`（默认） | ~6669 / ~13063 | MediaPipe 478 关键点 + Loop 细分 | 快 |
| `flame` | 5023 / 9976 | 细分网格 → 重心坐标迁移至 FLAME 拓扑 | 快 |
| `emoca` | ~6669 / ~13063 | EMOCA 编码器（ResNet/Swin）→ FLAME 解码 → Z 深度迁移 | 仅 CPU，约 30s |

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/hzl666123/facial-expression.git
cd facial-expression

# 2. 创建并激活虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

**FLAME 后端**需获取 FLAME 2020 模型（`generic_model.pkl`），通过 `--flame-path` 指定路径。

**EMOCA 后端**需将模型文件下载到 `emoca/assets/`（参考 `emoca/gdl_apps/EMOCA/demos/download_assets.sh`）。

**Linux EGL**：项目自动加载 `libs/libGLESv2.so.2.1.0` 以支持无头渲染，建议使用 `run.py` 启动。

### 使用方法

```bash
# 默认后端 (mediapipe_subdiv)
python main.py --photo assets/test_face.png

# 或使用 run.py（Linux 下自动处理 EGL）
python run.py --photo assets/test_face.png

# FLAME 后端
python main.py --backend flame --photo assets/test_face.png \
               --flame-path /path/to/FLAME2020/generic_model.pkl

# EMOCA 后端（启动较慢）
python main.py --backend emoca --photo assets/test_face.png

# 多视角重建（更优的 3D 形状）
python main.py --photo front.jpg left.jpg right.jpg

# 自定义分辨率
python main.py --photo assets/test_face.png --render-width 800 --render-height 800
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|----------|---------|------|
| `--photo` / `-p` | （必填） | 人脸照片路径（可多张） |
| `--camera` / `-c` | `0` | 摄像头设备 ID |
| `--cam-width` | `640` | 摄像头捕获宽度 |
| `--cam-height` | `480` | 摄像头捕获高度 |
| `--render-width` | `1200` | 渲染窗口宽度 |
| `--render-height` | `1200` | 渲染窗口高度 |
| `--model` / `-m` | `models/face_landmarker.task` | MediaPipe 模型路径 |
| `--subdiv` | `2` | Loop 细分迭代次数 |
| `--backend` | `mediapipe_subdiv` | `mediapipe_subdiv` / `flame` / `emoca` |
| `--flame-path` | `/mnt/f/FLAME2020/...` | FLAME `generic_model.pkl` 路径 |
| `--eye-photo` | `None` | 外部眼部照片用于逼真眼球纹理 |

### 测试

```bash
# 模块验证
python verify.py

# 集成测试（无需摄像头）
python test_integration.py

# 静态多角度渲染
python demo_render.py --photo assets/test_face.png --output output/
```

### 项目结构

```
facial-expression/
├── main.py                    # 入口，命令行参数解析
├── run.py                     # 跨平台启动器（EGL 预加载）
├── config.py                  # 默认配置
├── verify.py                  # 模块验证
├── test_integration.py        # 端到端集成测试
├── demo_render.py             # 静态多角度渲染器
├── requirements.txt           # Python 依赖
├── models/
│   └── face_landmarker.task   # MediaPipe 面部关键点模型
├── assets/
│   └── test_face.png          # 默认测试照片
├── libs/
│   └── libGLESv2.so.2.1.0     # 无头渲染 EGL 库
├── emoca/                     # EMOCA 第三方库
├── modules/
│   ├── pipeline.py            # 主循环：捕获 → 追踪 → 变形 → 渲染
│   ├── face_reconstructor.py  # 3D 面部重建（3 种后端）
│   ├── flame_model.py         # FLAME 2020 模型 + 重心坐标迁移
│   ├── renderer.py            # pyrender PBR 渲染器
│   ├── face_tracker.py        # MediaPipe FaceLandmarker 封装
│   ├── camera.py              # OpenCV VideoCapture 封装
│   └── texture_provider.py    # 照片 → 纹理/UV 提取
└── uploads/                   # 用户上传照片（已 gitignore）
```

### 已知限制

1. **EMOCA 仅 CPU** — 编码约 30s，无 PyTorch3D 无法做解码可视化
2. **牙齿/口腔静态** — 仅从照片提取，无下巴追踪
3. **头发静态** — 仅从照片提取，无动态变化
4. **光照固定** — 纯环境光，不跟随摄像头动态光照
5. **网格拓扑固定** — 硬编码的顶点索引，MediaPipe 更新关键点布局后可能失效

---
<a name="english"></a>

## English

Real-time 3D facial expression transfer: take a single photo → reconstruct a textured 3D face mesh → drive it with webcam facial landmarks, producing a **live avatar** that mimics your expressions.

```
Photo ──→ [Backend] ──→ Canonical 3D Face Mesh
                              │
Webcam ──→ FaceTracker ──────→ Deformation per frame ──→ pyrender ──→ Live Avatar
```

### Features

- **3 backends** for face reconstruction (see below)
- **Real-time webcam driving** via MediaPipe FaceLandmarker (52 blendshapes)
- **PBR rendering** with pyrender (metallic-roughness, double-sided, GPU EGL)
- **Eye geometry**: gaze tracking, blink, eyelid blendshapes, corneal bulge, iris Z flattening
- **Visual polish**: hair quad, teeth block, mouth cavity, philtrum groove
- **Multi-view reconstruction**: average shapecodes from multiple photos (EMOCA backend)
- **Dynamic FOV**: auto-computed from canonical face span

### Environment

| Item | Detail |
|------|--------|
| OS | Linux (WSL2 Ubuntu tested), macOS/Windows likely |
| Python | 3.10 |
| GPU | Not required (CPU-only rendering via pyrender/EGL) |
| Camera | Standard USB webcam (OpenCV) |

### Backends

| Backend | Vertices / Faces | Method | Speed |
|---------|-------------------|--------|-------|
| `mediapipe_subdiv` (default) | ~6669 / ~13063 | MediaPipe 478 landmarks + Loop subdivision | Fast |
| `flame` | 5023 / 9976 | Subdiv mesh → barycentric transfer to FLAME topology | Fast |
| `emoca` | ~6669 / ~13063 | EMOCA encoder (ResNet/Swin) → FLAME decode → Z-depth transfer | CPU-only, ~30s |

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/hzl666123/facial-expression.git
cd facial-expression

# 2. Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**For the FLAME backend**, obtain the FLAME 2020 model (`generic_model.pkl`) and specify its path with `--flame-path`.

**For the EMOCA backend**, download EMOCA model files into `emoca/assets/` (see `emoca/gdl_apps/EMOCA/demos/download_assets.sh`).

**Linux EGL**: The project auto-loads `libs/libGLESv2.so.2.1.0` for headless rendering. Use `run.py` which handles EGL preloading automatically.

### Usage

```bash
# Default backend (mediapipe_subdiv)
python main.py --photo assets/test_face.png

# Or use run.py (handles EGL on Linux)
python run.py --photo assets/test_face.png

# FLAME backend
python main.py --backend flame --photo assets/test_face.png \
               --flame-path /path/to/FLAME2020/generic_model.pkl

# EMOCA backend (slow startup)
python main.py --backend emoca --photo assets/test_face.png

# Multi-view (better 3D shape)
python main.py --photo front.jpg left.jpg right.jpg

# Custom resolution
python main.py --photo assets/test_face.png --render-width 800 --render-height 800
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--photo` / `-p` | (required) | Path(s) to face photo(s) |
| `--camera` / `-c` | `0` | Camera device ID |
| `--cam-width` | `640` | Camera capture width |
| `--cam-height` | `480` | Camera capture height |
| `--render-width` | `1200` | Render window width |
| `--render-height` | `1200` | Render window height |
| `--model` / `-m` | `models/face_landmarker.task` | MediaPipe model path |
| `--subdiv` | `2` | Loop subdivision iterations |
| `--backend` | `mediapipe_subdiv` | `mediapipe_subdiv` / `flame` / `emoca` |
| `--flame-path` | `/mnt/f/FLAME2020/...` | Path to FLAME `generic_model.pkl` |
| `--eye-photo` | `None` | External eye photo for realistic eyeball |

### Testing

```bash
# Module verification
python verify.py

# Integration test (no camera needed)
python test_integration.py

# Static multi-angle render
python demo_render.py --photo assets/test_face.png --output output/
```

### Project Structure

```
facial-expression/
├── main.py                    # Entry point, CLI parsing
├── run.py                     # Cross-platform launcher (EGL preloading)
├── config.py                  # Default configuration
├── verify.py                  # Module verification
├── test_integration.py        # End-to-end integration test
├── demo_render.py             # Static multi-angle/perspective renderer
├── requirements.txt           # Python dependencies
├── models/
│   └── face_landmarker.task   # MediaPipe face landmarker model
├── assets/
│   └── test_face.png          # Default test photo
├── libs/
│   └── libGLESv2.so.2.1.0     # EGL library for headless rendering
├── emoca/                     # EMOCA third-party library
├── modules/
│   ├── pipeline.py            # Main loop: capture → track → deform → render
│   ├── face_reconstructor.py  # 3D face reconstruction (3 backends)
│   ├── flame_model.py         # FLAME 2020 model + barycentric transfer
│   ├── renderer.py            # pyrender-based PBR renderer
│   ├── face_tracker.py        # MediaPipe FaceLandmarker wrapper
│   ├── camera.py              # OpenCV VideoCapture wrapper
│   └── texture_provider.py    # Photo → texture/UV extraction
└── uploads/                   # User-uploaded photos (gitignored)
```

### Known Limitations

1. **EMOCA CPU-only** — ~30s encode, no PyTorch3D for decode visualization
2. **Teeth/mouth cavity static** — from photo only, no jaw tracking
3. **Hair static** — from photo only, no dynamic changes
4. **Lighting fixed** — pure ambient, no dynamic lighting from webcam
5. **Mesh topology fixed** — hard-coded vertex indices may break if MediaPipe updates landmarks

---


## License

This project includes third-party code from [EMOCA](https://github.com/radekd91/emoca). See `emoca/LICENSE` for details.
