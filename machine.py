#!/usr/bin/env python3
"""Machine identity + hardware-spec detection for multi-machine result tracking.

Every result row in results/ is tagged with a ``machine_id`` so runs from several
boxes can share one CSV and one repo. This module:

  1. Auto-detects hardware **specs** (CPU / RAM / GPUs / PCIe / OS) — cross-platform,
     stdlib only, matching the harness's dependency-free ethos.
  2. Manages a stable per-machine id, stored locally in ``.machine_id`` (gitignored,
     so each machine that pulls the repo mints its OWN id on first run — ids are
     never inherited across a clone).
  3. Writes a committed registry file ``machines/<id>.json`` describing the box, so
     anyone reading the CSV can look up what a machine_id means.

PRIVACY: this records hardware **specs only**. It deliberately never reads the
hostname, username, MAC, or serial numbers — the ids are non-identifying (a
spec-derived slug + a random suffix), which keeps the repo safe to open-source.

CLI:
  python3 machine.py --show      # print this machine's id + detected specs (JSON)
  python3 machine.py --id        # print just the id (mints + registers if needed)
  python3 machine.py --ensure    # mint id if needed and write machines/<id>.json
"""
from __future__ import annotations  # keep annotations lazy for older pythons

import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ID_FILE = os.path.join(HERE, ".machine_id")          # gitignored, per-machine
REGISTRY_DIR = os.path.join(HERE, "machines")         # committed


# --------------------------------------------------------------------------- #
# Spec detection (stdlib only, cross-platform, specs-only — no identifiers)
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return val.strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            return out.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown-cpu"


def _cpu_physical_cores() -> int | None:
    try:
        if platform.system() == "Linux":
            phys, cores = set(), {}
            cur_phys = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        cur_phys = line.split(":", 1)[1].strip()
                        phys.add(cur_phys)
                    elif line.startswith("cpu cores"):
                        cores[cur_phys] = int(line.split(":", 1)[1].strip())
            if cores:
                return sum(cores.values())
        elif platform.system() == "Windows":
            # PowerShell is always present; sum cores across sockets.
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | "
                 "Measure-Object -Property NumberOfCores -Sum).Sum"],
                text=True, stderr=subprocess.DEVNULL)
            return int(out.strip())
    except Exception:
        pass
    return None


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    return None


def _gpus() -> list[dict]:
    """Per-GPU specs via nvidia-smi (works on Windows + Linux). Empty if none."""
    q = ("index,name,memory.total,pcie.link.gen.max,pcie.link.width.max,"
         "pcie.link.gen.current,pcie.link.width.current")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [c.strip() for c in line.split(",")]
        if len(p) < 7:
            continue
        gpus.append({
            "index": int(p[0]),
            "name": p[1],
            "vram_mib": int(p[2]),
            "pcie_gen_max": int(p[3]),
            "pcie_width_max": int(p[4]),
            "pcie_gen_cur": int(p[5]),
            "pcie_width_cur": int(p[6]),
        })
    return gpus


def _gpu_summary(gpus: list[dict]) -> str:
    if not gpus:
        return "no-nvidia-gpu"
    # Group identical models; flag any card on a narrow PCIe link (e.g. x1/x4).
    by_name: dict[str, int] = {}
    for g in gpus:
        vram_gb = round(g["vram_mib"] / 1024)
        key = f"{g['name']} {vram_gb}GB"
        by_name[key] = by_name.get(key, 0) + 1
    parts = [f"{n}x {k}" if n > 1 else k for k, n in by_name.items()]
    # Flag cards running on a narrow link (current negotiated width, e.g. an x1
    # riser/slot) — that's the bottleneck that actually affects multi-GPU splits.
    narrow = [g for g in gpus if g["pcie_width_cur"] and g["pcie_width_cur"] <= 4]
    summary = " + ".join(parts)
    if narrow:
        widths = ", ".join(f"x{g['pcie_width_cur']}" for g in narrow)
        summary += f" ({len(narrow)} on PCIe {widths})"
    return summary


def detect_specs() -> dict:
    gpus = _gpus()
    return {
        "cpu": _cpu_model(),
        "cpu_cores_physical": _cpu_physical_cores(),
        "cpu_cores_logical": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpus": gpus,
        "gpu_summary": _gpu_summary(gpus),
        # OS family/release only — never platform.node() (hostname).
        "os": f"{platform.system()} {platform.release()}".strip(),
    }


# --------------------------------------------------------------------------- #
# Identity: spec-slug + random suffix, non-identifying, stable per machine
# --------------------------------------------------------------------------- #
def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(amd|intel|nvidia|geforce|rtx|gtx|processor|cpu|core|tm|r)\b",
                  " ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)


def _spec_slug(specs: dict) -> str:
    toks = [t for t in _slugify(specs["cpu"]).split("-") if t]
    # Prefer tokens that look like a model number (e.g. "9800x3d"); they identify
    # the chip far better than the leading brand word.
    model = [t for t in toks if any(c.isdigit() for c in t)]
    cpu_tag = "-".join((model or toks)[:2]) or "cpu"
    gpus = specs["gpus"]
    if gpus:
        vram_gb = round(gpus[0]["vram_mib"] / 1024)
        gpu_tag = f"{len(gpus)}x{_slugify(gpus[0]['name']).replace('-', '')}{vram_gb}g"
    else:
        gpu_tag = "cpuonly"
    ram = f"{round(specs['ram_gb'])}g" if specs.get("ram_gb") else ""
    return "-".join(t for t in (cpu_tag, gpu_tag, ram) if t)[:48]


def get_machine_id(specs: dict | None = None) -> str:
    """Return this machine's stable id, minting + persisting it on first call."""
    if os.path.exists(ID_FILE):
        with open(ID_FILE) as f:
            mid = f.read().strip()
            if mid:
                return mid
    specs = specs or detect_specs()
    mid = f"{_spec_slug(specs)}-{uuid.uuid4().hex[:4]}"
    with open(ID_FILE, "w", encoding="utf-8") as f:
        f.write(mid + "\n")
    return mid


def ensure_registered(machine_id: str | None = None, specs: dict | None = None) -> str:
    """Write machines/<id>.json (committed) if absent. Returns the id."""
    specs = specs or detect_specs()
    machine_id = machine_id or get_machine_id(specs)
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    path = os.path.join(REGISTRY_DIR, machine_id + ".json")
    if not os.path.exists(path):
        record = {"machine_id": machine_id,
                  "first_seen": int(time.time()),
                  "specs": specs}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"[machine] registered {path}", file=sys.stderr)
    return machine_id


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true", help="print id + detected specs")
    p.add_argument("--id", action="store_true", help="print just the machine id")
    p.add_argument("--ensure", action="store_true",
                   help="mint id if needed and write machines/<id>.json")
    a = p.parse_args()

    specs = detect_specs()
    mid = get_machine_id(specs)
    if a.ensure or a.show:
        ensure_registered(mid, specs)
    if a.id:
        print(mid)
    else:
        print(json.dumps({"machine_id": mid, "specs": specs}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
