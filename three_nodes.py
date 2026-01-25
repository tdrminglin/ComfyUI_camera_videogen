import time
import uuid
import json
import base64
import numpy as np
import torch
import os
from io import BytesIO
from PIL import Image
from server import PromptServer
from aiohttp import web
import folder_paths

RENDER_CACHE = {}

try:
    @PromptServer.instance.routes.post("/threejs/render_result")
    async def receive_render_result(request):
        data = await request.json()
        req_id = data.get("request_id")
        images_base64 = data.get("images")
        if req_id and images_base64:
            RENDER_CACHE[req_id] = images_base64
            return web.json_response({"status": "ok"})
        return web.json_response({"status": "error"}, status=400)
except Exception as e:
    pass

def get_input_files():
    try:
        files = folder_paths.get_filename_list("input")
        model_files = [f for f in files if f.lower().endswith(('.glb', '.gltf'))]
        return ["none"] + model_files
    except Exception:
        return ["none"]

# --- 新增节点: 人物专用动作 (Figure Action) ---
class ThreeJSFigureAction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action_type": ([
                    "x_pos (左右移动)", 
                    "y_pos (上下移动)", 
                    "z_pos (前后移动)", 
                    "rotation_y (转身/旋转)"
                ],),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "end_frame": ("INT", {"default": 60, "min": 0, "max": 9999}),
                "start_value": ("FLOAT", {"default": 0.0, "step": 0.1,"min": -9999, "max": 9999}),
                "end_value": ("FLOAT", {"default": 10.0, "step": 0.1,"min": -9999, "max": 9999}),
                "easing": ([
                    "linear", "easeInQuad", "easeOutQuad", "easeInOutQuad",
                    "easeInCubic", "easeOutCubic", "easeInOutCubic"
                ],),
            }
        }
    RETURN_TYPES = ("ACTION_SEGMENT",)
    RETURN_NAMES = ("figure_action",)
    FUNCTION = "create_segment"
    CATEGORY = "ThreeJS/Animation"

    def create_segment(self, action_type, start_frame, end_frame, start_value, end_value, easing):
        # 清理中文注释，只保留 code
        raw_type = action_type.split(" ")[0]
        return ({
            "target": "figure",
            "type": raw_type,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "startValue": start_value,
            "endValue": end_value,
            "easing": easing,
            "enabled": True
        },)

# --- 新增节点: 相机专用动作 (Camera Action) ---
class ThreeJSCameraAction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action_type": ([
                    "distance (距离)", 
                    "elevation (高度角/俯仰)", 
                    "azimuth (方位角/环绕)", 
                    "panX (横向平移)", 
                    "panY (纵向平移)", 
                    "fov (视野)", 
                    "roll (镜头旋转)"
                ],),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "end_frame": ("INT", {"default": 60, "min": 0, "max": 9999}),
                "start_value": ("FLOAT", {"default": 0.0, "step": 0.1, "min": -9999, "max": 9999}),
                "end_value": ("FLOAT", {"default": 10.0, "step": 0.1,"min": -9999, "max": 9999}),
                "easing": ([
                    "linear", "easeInQuad", "easeOutQuad", "easeInOutQuad",
                    "easeInCubic", "easeOutCubic", "easeInOutCubic"
                ],),
            }
        }
    RETURN_TYPES = ("ACTION_SEGMENT",)
    RETURN_NAMES = ("camera_action",)
    FUNCTION = "create_segment"
    CATEGORY = "ThreeJS/Animation"

    def create_segment(self, action_type, start_frame, end_frame, start_value, end_value, easing):
        raw_type = action_type.split(" ")[0]
        return ({
            "target": "camera",
            "type": raw_type,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "startValue": start_value,
            "endValue": end_value,
            "easing": easing,
            "enabled": True
        },)

# --- 人物配置 ---
class ThreeJSFigureConfig:
    @classmethod
    def INPUT_TYPES(cls):
        file_list = get_input_files()
        return {
            "required": {
                "figure_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
                "limb_length": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0}),
                "show_face": ("BOOLEAN", {"default": True}),
                "custom_model": (file_list, ),
            }
        }
    RETURN_TYPES = ("FIGURE_CONFIG",)
    RETURN_NAMES = ("fig_config",)
    FUNCTION = "config_figure"
    CATEGORY = "ThreeJS/Config"

    def config_figure(self, figure_scale, limb_length, show_face, custom_model):
        model_url = ""
        if custom_model and custom_model != "none":
            model_url = f"/view?filename={custom_model}&type=input"
        return ({
            "scale": figure_scale,
            "limbLength": limb_length,
            "showFace": show_face,
            "modelUrl": model_url
        },)

# --- 环境配置 ---
class ThreeJSEnvironment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skybox_type": (["None (Gray)", "Grid Only", "Simple Sky (Blue)", "Texture Skybox (Folder)"],),
            }
        }
    RETURN_TYPES = ("ENV_CONFIG",)
    RETURN_NAMES = ("env_config",)
    FUNCTION = "config_env"
    CATEGORY = "ThreeJS/Config"

    def config_env(self, skybox_type):
        return ({"skyboxType": skybox_type},)

# --- 相机初始配置 ---
class ThreeJSCameraConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_mode": (["follow", "fixed"],),
                "look_at_height": ("FLOAT", {"default": 0.0, "step": 0.1}),
                "initial_distance": ("FLOAT", {"default": 7.0, "min": 0.1}),
                "initial_elev_deg": ("FLOAT", {"default": 10.0, "min": -90.0, "max": 90.0}),
                "initial_azim_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
                "initial_fov": ("FLOAT", {"default": 50.0, "min": 5.0, "max": 150.0}),
            }
        }
    RETURN_TYPES = ("CAMERA_CONFIG",)
    RETURN_NAMES = ("cam_config",)
    FUNCTION = "create_config"
    CATEGORY = "ThreeJS/Config"

    def create_config(self, camera_mode, look_at_height, initial_distance, initial_elev_deg, initial_azim_deg, initial_fov):
        return ({
            "cameraFollowMode": camera_mode,
            "lookAtHeightOffset": look_at_height,
            "initialDistance": initial_distance,
            "initialElevationDeg": initial_elev_deg,
            "initialAzimuthDeg": initial_azim_deg,
            "fov": initial_fov
        },)

# --- 动作组合器 ---
class ThreeJSActionCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "action_1": ("ACTION_SEGMENT",), },
            "optional": {
                "action_2": ("ACTION_SEGMENT",),
                "action_3": ("ACTION_SEGMENT",),
                "action_4": ("ACTION_SEGMENT",),
                "previous_list": ("ACTION_LIST",),
            }
        }
    RETURN_TYPES = ("ACTION_LIST",)
    RETURN_NAMES = ("action_list",)
    FUNCTION = "combine"
    CATEGORY = "ThreeJS/Animation"

    def combine(self, action_1, action_2=None, action_3=None, action_4=None, previous_list=None):
        actions = []
        if previous_list: actions.extend(previous_list)
        actions.append(action_1)
        if action_2: actions.append(action_2)
        if action_3: actions.append(action_3)
        if action_4: actions.append(action_4)
        return (actions,)

# --- 渲染器 (更新：增加了 seed) ---
class ThreeJSRenderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_config": ("CAMERA_CONFIG",),
                "figure_config": ("FIGURE_CONFIG",),
                "total_frames": ("INT", {"default": 60, "min": 1}),
                "width": ("INT", {"default": 512, "min": 64}),
                "height": ("INT", {"default": 512, "min": 64}),
                "fps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # 新增 Seed
            },
            "optional": {
                "env_config": ("ENV_CONFIG",),
                "action_list": ("ACTION_LIST",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "render"
    CATEGORY = "ThreeJS/Renderer"
    OUTPUT_NODE = True

    def render(self, camera_config, figure_config, total_frames, width, height, fps, seed, env_config=None, action_list=None):
        request_id = str(uuid.uuid4())
        node_id = getattr(self, "id", None)

        camera_segments = []
        figure_segments = []
        if action_list:
            for seg in action_list:
                if seg['target'] == 'camera': camera_segments.append(seg)
                elif seg['target'] == 'figure': figure_segments.append(seg)

        payload = {
            **camera_config,
            **figure_config,
            **(env_config if env_config else {"skyboxType": "Grid Only"}),
            "numFrames": total_frames,
            "width": width,
            "height": height,
            "fps": fps,
            "seed": seed, # 传递 seed
            "cameraSegments": camera_segments,
            "figureSegments": figure_segments,
            "request_id": request_id,
            "node_id": node_id
        }

        print(f"[ThreeJS] Request ID: {request_id}, Seed: {seed}")
        PromptServer.instance.send_sync("threejs_render_request", payload)
        timeout = 200
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in RENDER_CACHE:
                images_b64 = RENDER_CACHE.pop(request_id)
                return (self._convert_to_tensor(images_b64, width, height),)
            time.sleep(0.1)
        print(f"[ThreeJS] Error: Render timed out. Request ID {request_id} not received from frontend.")
        print("[ThreeJS] Possible causes: 1. Browser tab closed. 2. JS Extension failed to load (Check F12 Console).")

        return (torch.zeros((total_frames, height, width, 3)),)

    def _convert_to_tensor(self, b64_list, width, height):
        image_list = []
        for b64 in b64_list:
            if "," in b64: b64 = b64.split(",")[1]
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            if img.width != width or img.height != height:
                img = img.resize((width, height))
            image_list.append(np.array(img).astype(np.float32) / 255.0)
        return torch.from_numpy(np.array(image_list))

NODE_CLASS_MAPPINGS = {
    "ThreeJSCameraAction": ThreeJSCameraAction,
    "ThreeJSFigureAction": ThreeJSFigureAction,
    "ThreeJSCameraConfig": ThreeJSCameraConfig,
    "ThreeJSFigureConfig": ThreeJSFigureConfig,
    "ThreeJSEnvironment": ThreeJSEnvironment,
    "ThreeJSActionCombiner": ThreeJSActionCombiner,
    "ThreeJSRenderNode": ThreeJSRenderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThreeJSCameraAction": "3D Camera Action",
    "ThreeJSFigureAction": "3D Figure Action",
    "ThreeJSCameraConfig": "3D Camera Config",
    "ThreeJSFigureConfig": "3D Figure Config",
    "ThreeJSEnvironment": "3D Environment Config",
    "ThreeJSActionCombiner": "3D Action Combiner",
    "ThreeJSRenderNode": "3D Preview & Render"
}
