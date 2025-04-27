import importlib
import rtde_control
import torch
import torch.version
print(torch.version.__version__)

modules = [
    "pyrealsense2",
    "cv2",
    "cupy",
    "numpy",
    "ultralytics",
    "PyQt5",
    "open3d",
    "rtde",
    "matplotlib",
    "scipy",
    "pandas"
]

print("Library Versions:\n")

for module_name in modules:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'Unknown Version Attribute')

        if module_name == "cv2":
            version = mod.__version__
        elif module_name == "pyrealsense2":
            version = mod.__version__ if hasattr(mod, '__version__') else "Unknown (RealSense sometimes has no __version__)"
        elif module_name == "rtde":
            version = "Custom or system installed"  # rtde sometimes doesn't have __version__

        print(f"{module_name:15}: {version}")

    except ImportError:
        print(f"{module_name:15}: Not Installed")
