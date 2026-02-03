#!/usr/bin/env python3
"""
Setup script to copy HY-Motion model files from ComfyUI-HY-Motion1 to ComfyUI_camera_videogen

Usage:
    python setup_hymotion_models.py [--source PATH] [--target PATH]

This script copies the necessary binary model files from HY-Motion1 to the
camera_videogen plugin's web/geom_models/hymotion_models directory.

Required files:
    - v_template.bin    : Vertex positions
    - j_template.bin    : Joint positions
    - skinWeights.bin   : Skin weights
    - skinIndice.bin    : Skin indices
    - kintree.bin       : Bone hierarchy
    - faces.bin         : Face indices
    - joint_names.json  : Joint names (will be created automatically)
"""

import os
import shutil
import argparse
import json
from pathlib import Path

def find_hymotion_source(start_path=None):
    """Find ComfyUI-HY-Motion1 plugin directory"""
    if start_path is None:
        start_path = os.path.dirname(os.path.abspath(__file__))

    # Try common locations
    possible_paths = [
        # Relative to this script
        os.path.join(start_path, "..", "..", "..", "ComfyUI-HY-Motion1"),
        os.path.join(start_path, "..", "..", "ComfyUI-HY-Motion1"),
        os.path.join(start_path, "..", "ComfyUI-HY-Motion1"),
        r"E:\ai_models\ComfyUI-aki-v1.2\custom_nodes\ComfyUI_camera_videogen\ComfyUI-HY-Motion1",
        # Common ComfyUI installation paths
        os.path.join(os.path.expanduser("~"), "ComfyUI", "custom_nodes", "ComfyUI-HY-Motion1"),
        os.path.join(os.path.expanduser("~"), "Desktop", "ComfyUI", "custom_nodes", "ComfyUI-HY-Motion1"),
        # Try to find in parent directories
        os.path.dirname(start_path),
        os.path.dirname(os.path.dirname(start_path)),
        os.path.dirname(os.path.dirname(os.path.dirname(start_path))),
    ]

    for path in possible_paths:
        dump_path = os.path.join(path, "web", "dump_wooden")
        if os.path.exists(dump_path):
            print(f"✓ Found HY-Motion1 at: {path}")
            return dump_path

    return None

def create_joint_names_json(output_dir):
    """Create joint_names.json for the wooden model"""
    # HY-Motion 22 joint names (SMPL topology)
    joint_names = [
        "Root",           # 0
        "LHip",           # 1
        "RHip",           # 2
        "Spine1",         # 3
        "LKnee",          # 4
        "RKnee",          # 5
        "Spine2",         # 6
        "LAnkle",         # 7
        "RAnkle",         # 8
        "Spine3",         # 9
        "LFoot",          # 10
        "RFoot",          # 11
        "Neck",           # 12
        "LShoulder",      # 13
        "RShoulder",      # 14
        "Head",           # 15
        "LElbow",         # 16
        "RElbow",         # 17
        "LWrist",         # 18
        "RWrist",         # 19
        "LHand",          # 20
        "RHand"           # 21
    ]

    json_path = os.path.join(output_dir, "joint_names.json")
    with open(json_path, 'w') as f:
        json.dump(joint_names, f)
    print(f"  Created: joint_names.json")

def copy_model_files(source_dir, target_dir, model_name="wooden"):
    """Copy binary model files from source to target"""
    # Required files
    required_files = [
        'v_template.bin',
        'j_template.bin',
        'skinWeights.bin',
        'skinIndice.bin',
        'kintree.bin',
        'faces.bin'
    ]

    # Create target directory
    target_model_dir = os.path.join(target_dir, model_name)
    os.makedirs(target_model_dir, exist_ok=True)
    print(f"  Target directory: {target_model_dir}")

    # Check source exists
    if not os.path.exists(source_dir):
        print(f"✗ Error: Source directory not found: {source_dir}")
        return False

    # Check and copy required files
    copied = []
    missing = []
    for filename in required_files:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(filename)
            print(f"  ✓ {filename}")
        else:
            missing.append(filename)
            print(f"  ✗ {filename} (missing)")

    if missing:
        print(f"\n⚠ Warning: {len(missing)} required files are missing from source!")
        print(f"  Missing files: {', '.join(missing)}")

    # Create joint_names.json
    create_joint_names_json(target_model_dir)

    if copied:
        print(f"\n✓ Success! Model '{model_name}' is ready at:")
        print(f"  {target_model_dir}")
        print(f"\nFiles copied: {len(copied)}/{len(required_files)}")
        return True
    else:
        print(f"\n✗ No files were copied. Please check your HY-Motion installation.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Setup HY-Motion model files for ComfyUI_camera_videogen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect HY-Motion1 location
  python setup_hymotion_models.py

  # Specify source manually
  python setup_hymotion_models.py --source /path/to/ComfyUI-HY-Motion1/web/dump_wooden

  # Specify custom target directory
  python setup_hymotion_models.py --target /path/to/ComfyUI_camera_videogen/web/geom_models/hymotion_models
        """
    )
    parser.add_argument("--source", "-s", help="Path to ComfyUI-HY-Motion1/web/dump_wooden")
    parser.add_argument("--target", "-t", help="Target directory (default: web/geom_models/hymotion_models)")
    parser.add_argument("--model-name", "-m", default="wooden", help="Model folder name")

    args = parser.parse_args()

    print("=" * 60)
    print("HY-Motion Model Setup for ComfyUI_camera_videogen")
    print("=" * 60)

    # Determine source
    source = args.source
    if not source:
        print("\n[1/3] Searching for ComfyUI-HY-Motion1...")
        source = r"E:\ai_models\ComfyUI-aki-v1.2\custom_nodes\ComfyUI_camera_videogen\ComfyUI-HY-Motion1\assets\dump_wooden"#find_hymotion_source()
        if not source:
            print("\n✗ Could not auto-detect ComfyUI-HY-Motion1 location.")
            print("\nPlease provide the path with --source flag.")
            print("\nExample paths to check:")
            print("  - ~/ComfyUI/custom_nodes/ComfyUI-HY-Motion1/web/dump_wooden")
            print("  - /path/to/ComfyUI/custom_nodes/ComfyUI-HY-Motion1/web/dump_wooden")
            return 1

    # Determine target
    if not args.target:
        target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "geom_models", "hymotion_models")
    else:
        target_dir = r"E:\ai_models\ComfyUI-aki-v1.2\custom_nodes\ComfyUI_camera_videogen\web\geom_models\hymotion_models\wooden"#args.target

    print(f"\n[2/3] Copying model files...")
    print(f"  Source: {source}")
    print(f"  Target: {target_dir}")
    print(f"  Model:  {args.model_name}")

    if not copy_model_files(source, target_dir, args.model_name):
        return 1

    print("\n[3/3] Setup complete!")
    print("-" * 60)
    print("Next steps:")
    print("  1. Restart ComfyUI to reload the plugin")
    print("  2. In your workflow, use '3D HY-Motion Figure' node")
    print("  3. Select 'wooden' as the model")
    print("  4. Connect to '3D Action Combiner' via '3D HY-Motion Import'")
    print("-" * 60)
    print("\nTroubleshooting:")
    print("  - If model doesn't appear, check browser console (F12)")
    print("  - Look for 'HY-Motion: Loaded' messages")
    print("  - Ensure all 6 .bin files + joint_names.json exist")
    return 0

if __name__ == "__main__":
    exit(main())
