# ComfyUI Camera Motion (Three.js Powered)

[English](#english) | [中文](#chinese)

一个基于 Three.js 的 ComfyUI 扩展，用于可视化地设计、预览和生成复杂的 3D 相机运镜与人物运动轨迹。生成的运动数据/图像序列可直接用于 AnimateDiff、ControlNet 或 VideoHelperSuite，特别是用于uni3c与wan2.1结合，可以实现超强的运镜控制。





---

<a name="english"></a>
## ✨ Features

*   **Visual 3D Preview**: Real-time preview of camera movement and character animation inside a ComfyUI node using Three.js.
*   **Modular Design**: Separate nodes for Scene Settings, Motion Definitions, and Rendering to keep workflows clean.
*   **Dual Camera Modes**:
    *   **Follow Target**: Camera automatically tracks the moving character (great for orbiting shots).
    *   **Fixed Target**: Camera stays focused on a fixed point while the character moves away.
*   **Complex Motion Chaining**: Chain multiple motion segments (Pan, Tilt, Zoom, Roll, XYZ movement) with customizable easing functions.
*   **Web Integration**: Seamlessly embeds a lightweight 3D engine within the ComfyUI interface.

## 📦 Installation

1.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/tdrminglin/ComfyUI_camera_videogen.git
    ```
3.  Restart ComfyUI.

## Nodes Usage

### 1. CM_SceneSettings (🎥 Scene Settings)
Defines the global environment.
*   **figure_scale**: Size of the reference character.
*   **camera_follow_mode**: `follow` (track character) or `fixed` (static look-at point).
*   **initial_distance/elevation/azimuth**: Starting camera position.

### 2. CM_CameraAction (🎬 Camera/Figure Action)
Defines a single segment of motion. You can chain multiple actions together.
*   **action_type**: Choose between Camera moves (Distance, Elevation, Azimuth, Roll, FOV, Pan) or Figure moves (X/Y/Z Pos).
*   **start/end frame**: Timeline for this specific action.
*   **easing**: Smoothness of the transition (Linear, EaseIn, EaseOut, etc.).

### 3. CM_Renderer (📺 Motion Preview & Render)
The output node that visualizes the data.
*   **Inputs**: Accepts scene config and action lists.
*   **Preview**: Shows a black window initially, updates with 3D preview after running the queue.
*   **Output**: Generates an `IMAGE` batch (Tensor) for downstream nodes (e.g., AnimateDiff).

---

<a name="chinese"></a>
## ✨ 功能特点

*   **可视化 3D 预览**: 在 ComfyUI 节点内直接嵌入 Three.js 窗口，实时预览相机和人物的运动轨迹。
*   **模块化设计**: 将场景设置、动作定义和渲染分离为不同节点，符合 ComfyUI 的连线逻辑。
*   **双相机模式**:
    *   **跟随目标 (Follow)**: 无论人物如何移动，相机始终聚焦人物（适合环绕跟拍）。
    *   **固定目标 (Fixed)**: 相机聚焦在初始位置，人物可以走出画面（适合固定机位）。
*   **复杂运动组合**: 支持无限串联多个动作片段（推拉、摇移、旋转、横滚、人物位移），并支持多种缓动曲线。
*   **无缝集成**: 它可以生成单纯的导引图像，完美配合 AnimateDiff 或 ControlNet 使用。

## 📦 安装说明

1.  进入你的 ComfyUI 插件目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone https://github.com/tdrminglin/ComfyUI_camera_videogen.git
    ```
3.  重启 ComfyUI。

## 🛠️ 节点使用指南

### 1. 场景设置 (CM_SceneSettings)
用于定义全局的初始状态。
*   **figure_scale**: 参考小人的大小。
*   **camera_follow_mode**: 选择相机是“跟随人物移动”还是“盯着固定点”。
*   **initial_...**: 定义相机的初始距离、角度和视野。

### 2. 动作定义 (CM_CameraAction)
定义一段具体的运动。支持链式连接（将上一个动作连入 `prev_action`）。
*   **action_type**: 选择是相机运动（距离、高度角、方位角、横滚、FOV、平移）还是人物运动（X/Y/Z轴位移）。
*   **start/end frame**: 该动作持续的帧数范围。
*   **easing**: 运动的缓动效果（线性、渐入、渐出等）。

### 3. 预览与渲染 (CM_Renderer)
核心节点，用于接收数据并生成图像。
*   **输入**: 连接上面的场景配置和动作列表。
*   **预览**: 点击 Queue Prompt 运行后，节点中间的窗口会加载 3D 场景并根据参数播放动画。
*   **输出**: 输出 `IMAGE` 格式的图片序列，可以直接连入 VideoHelperSuite 的 `Video Combi
