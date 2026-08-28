#!/usr/bin/env python
"""List the bounded EGL device enumeration used by pyrender."""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import write_json


MAX_DEVICES = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from pyrender.platforms import egl as pyrender_egl
    from OpenGL import EGL as egl

    query_attribute = pyrender_egl._get_egl_func(
        "eglQueryDeviceAttribEXT",
        egl.EGLBoolean,
        pyrender_egl._EGLDeviceEXT,
        egl.EGLint,
        ctypes.POINTER(ctypes.c_ssize_t),
    )
    cuda_device_attribute = 0x323A  # EGL_CUDA_DEVICE_NV

    devices = pyrender_egl.query_devices()
    if not devices or len(devices) > MAX_DEVICES:
        raise RuntimeError(
            f"expected 1..{MAX_DEVICES} EGL devices, found {len(devices)}"
        )
    values = []
    for index, device in enumerate(devices):
        name = device.name
        entry = {"egl_index": index, "drm_device": name}
        if query_attribute is not None:
            cuda_index = ctypes.c_ssize_t(-1)
            if query_attribute(
                device._display,
                cuda_device_attribute,
                ctypes.pointer(cuda_index),
            ):
                entry["cuda_device"] = int(cuda_index.value)
        if name:
            sysfs = Path("/sys/class/drm") / Path(name).name / "device/uevent"
            if sysfs.is_file():
                fields = {}
                for line in sysfs.read_text(encoding="utf-8").splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        fields[key] = value
                entry["pci_slot_name"] = fields.get("PCI_SLOT_NAME")
                entry["driver"] = fields.get("DRIVER")
        values.append(entry)
    report = {"device_count": len(values), "devices": values}
    write_json(Path(args.output), report)
    for value in values:
        print(value, flush=True)


if __name__ == "__main__":
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/list_egl_devices.py")
    main()
