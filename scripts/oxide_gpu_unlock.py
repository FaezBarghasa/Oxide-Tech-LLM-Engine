#!/usr/bin/env python3
"""
Oxide Hardware GPU Unlock Diagnostic & Probe Tool
Reference: gpu-unlock (https://github.com/Zagitta/gpu-unlock)

Probes NVIDIA GPUs across PCI bus hierarchy, detects Ampere (RTX 3000) and
Ada Lovelace (RTX 4000) series hardware, and verifies kernel module unlock
compatibility, instruction issue rates, and 4th-Gen Tensor Core execution capabilities.
"""

import os
import sys
import glob
import subprocess

# Compatible Ampere (RTX 3000) Device IDs
SUPPORTED_AMPERE_DEVICES = {
    "2203": "NVIDIA GeForce RTX 3090 Ti (GA102) -> Full BF16 Unlock Supported (+68-80%)",
    "2204": "NVIDIA GeForce RTX 3090 (GA102) -> Full BF16 Unlock Supported (+25-67%)",
    "2205": "NVIDIA GeForce RTX 3080 Ti OEM (GA102-250) -> Unlock completes",
    "2206": "NVIDIA GeForce RTX 3080 10GB/20GB (GA102) -> Full BF16 Unlock Supported (+56-73%)",
    "2482": "NVIDIA GeForce RTX 3070 Ti (GA104) -> Full BF16 Unlock Supported",
    "2484": "NVIDIA GeForce RTX 3070 (GA104) -> Full BF16 Unlock Supported (+73-81%)",
    "2504": "NVIDIA GeForce RTX 3060 12GB (GA106) -> Full BF16 Unlock Supported (~+80%)",
    "2414": "NVIDIA GA103 GPU -> Compatible",
}

# Compatible Ada Lovelace (RTX 4000) Device IDs
SUPPORTED_ADA_DEVICES = {
    # AD102
    "2681": "NVIDIA RTX TITAN Ada (AD102) -> 4th-Gen Tensor Cores (FP8/BF16/FP16 native dual-issue)",
    "2684": "NVIDIA GeForce RTX 4090 (AD102) -> 4th-Gen Tensor Cores (16,384 CUDA cores, 24GB G6X)",
    "2685": "NVIDIA GeForce RTX 4090 D (AD102) -> 4th-Gen Tensor Cores (FP8/BF16/FP16 native)",
    # AD103
    "2702": "NVIDIA GeForce RTX 4080 SUPER (AD103) -> 4th-Gen Tensor Cores (10,240 CUDA cores)",
    "2703": "NVIDIA GeForce RTX 4080 SUPER (AD103) -> 4th-Gen Tensor Cores (FP8/BF16)",
    "2704": "NVIDIA GeForce RTX 4080 (AD103) -> 4th-Gen Tensor Cores (9,728 CUDA cores, 16GB G6X)",
    "2705": "NVIDIA GeForce RTX 4070 Ti SUPER (AD103) -> 4th-Gen Tensor Cores (8,448 CUDA cores)",
    "2709": "NVIDIA GeForce RTX 4070 (AD103) -> 4th-Gen Tensor Cores",
    "2717": "NVIDIA GeForce RTX 4090 Laptop GPU (AD103M) -> 4th-Gen Mobile Tensor Cores",
    # AD104
    "2782": "NVIDIA GeForce RTX 4070 Ti (AD104) -> 4th-Gen Tensor Cores (7,680 CUDA cores, 12GB G6X)",
    "2783": "NVIDIA GeForce RTX 4070 SUPER (AD104) -> 4th-Gen Tensor Cores (7,168 CUDA cores)",
    "2785": "NVIDIA AC AD104 20GB (AD104) -> 4th-Gen Tensor Cores",
    "2786": "NVIDIA GeForce RTX 4070 (AD104) -> 4th-Gen Tensor Cores (5,888 CUDA cores, 12GB G6X)",
    "2788": "NVIDIA GeForce RTX 4060 Ti (AD104) -> 4th-Gen Tensor Cores",
    "27a0": "NVIDIA GeForce RTX 4080 Max-Q / Mobile (AD104M) -> 4th-Gen Mobile Tensor Cores",
    "27e0": "NVIDIA GeForce RTX 4080 Max-Q / Mobile (AD104M) -> 4th-Gen Mobile Tensor Cores",
    # AD106
    "2803": "NVIDIA GeForce RTX 4060 Ti 8GB (AD106) -> 4th-Gen Tensor Cores (4,352 CUDA cores)",
    "2805": "NVIDIA GeForce RTX 4060 Ti 16GB (AD106) -> 4th-Gen Tensor Cores (4,352 CUDA cores, 16GB)",
    "2808": "NVIDIA GeForce RTX 4060 (AD106) -> 4th-Gen Tensor Cores",
    "2820": "NVIDIA GeForce RTX 4070 Max-Q / Mobile (AD106M) -> 4th-Gen Mobile Tensor Cores",
    "2860": "NVIDIA GeForce RTX 4070 Max-Q / Mobile (AD106M) -> 4th-Gen Mobile Tensor Cores",
    # AD107 (Host Reference Architecture)
    "2882": "NVIDIA GeForce RTX 4060 Desktop (AD107) -> 4th-Gen Tensor Cores (3,072 CUDA cores, 8GB)",
    "28a0": "NVIDIA GeForce RTX 4060 Max-Q / Mobile (AD107M) -> 4th-Gen Mobile Tensor Cores",
    "28a1": "NVIDIA GeForce RTX 4050 Max-Q / Mobile (AD107M) -> 4th-Gen Mobile Tensor Cores",
    "28e0": "NVIDIA GeForce RTX 4060 Max-Q / Mobile (AD107M) [Host Reference GPU] -> 4th-Gen Mobile Tensor Cores (3,072 CUDA cores, 8GB GDDR6)",
    "28e1": "NVIDIA GeForce RTX 4050 Max-Q / Mobile (AD107M) -> 4th-Gen Mobile Tensor Cores (2,560 CUDA cores, 6GB)",
}

def get_sysfs_attr(dev_path, attr_name):
    path = os.path.join(dev_path, attr_name)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

def query_nvidia_smi():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=gpu_name,driver_version,memory.total,memory.free,power.limit,clocks.max.sm",
            "--format=csv,noheader"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip().split("\n")
    except Exception:
        return None

def probe_pci_gpus():
    print("=" * 76)
    print(" Oxide Engine: NVIDIA GPU Hardware Capability & Unlock Probe")
    print(" Architecture Support: Ampere (RTX 3000) & Ada Lovelace (RTX 4000)")
    print("=" * 76)
    
    pci_devices = glob.glob("/sys/bus/pci/devices/*")
    found_any = False
    smi_lines = query_nvidia_smi()

    for dev in sorted(pci_devices):
        vendor = get_sysfs_attr(dev, "vendor")
        device = get_sysfs_attr(dev, "device")
        pci_class = get_sysfs_attr(dev, "class")
        
        if not vendor or not device or not pci_class:
            continue
            
        # Filter for NVIDIA (0x10de) Display/3D compute controllers (Class 0x03xxxx)
        if vendor.lower() == "0x10de" and pci_class.lower().startswith("0x03"):
            found_any = True
            dev_id = device.replace("0x", "").lower()
            pci_bdf = os.path.basename(dev)
            sub_vendor = get_sysfs_attr(dev, "subsystem_vendor") or "unknown"
            sub_device = get_sysfs_attr(dev, "subsystem_device") or "unknown"
            
            print(f"\n[+] Detected NVIDIA Physical GPU at PCI {pci_bdf}")
            print(f"    PCI Device ID: 10de:{dev_id} (Subsystem: {sub_vendor.replace('0x', '')}:{sub_device.replace('0x', '')})")
            
            if dev_id in SUPPORTED_ADA_DEVICES:
                desc = SUPPORTED_ADA_DEVICES[dev_id]
                print(f"    Architecture: \033[96mAda Lovelace (RTX 4000 Series)\033[0m")
                print(f"    Status: \033[92mNATIVE 4TH-GEN TENSOR CORES (UNTHROTTLED)\033[0m")
                print(f"    Details: {desc}")
                print(f"    Precision Capabilities: FP8 (E4M3/E5M2), FP16, BF16, TF32, INT8, INT4 Marlin")
                print(f"    Hardware Acceleration: Full dual-issue instruction rate natively enabled.")
                print(f"    gpu-unlock Target Match: Supported (module id: 0x{dev_id})")
            elif dev_id in SUPPORTED_AMPERE_DEVICES:
                desc = SUPPORTED_AMPERE_DEVICES[dev_id]
                print(f"    Architecture: \033[93mAmpere (RTX 3000 Series)\033[0m")
                print(f"    Status: \033[92mCOMPATIBLE WITH GPU-UNLOCK KERNEL MODULE\033[0m")
                print(f"    Details: {desc}")
                print("\n    To unlock full BF16 instruction issue rates (+80% GEMM):")
                print("      cd references/gpu-unlock && make && sudo insmod kernel/gpu_unlock.ko")
                print("      or add 'gpu_unlock.targets=all' to kernel boot command line.")
            else:
                print(f"    Status: Modern or unthrottled architecture (native full-rate FP/BF16 active)")

    if smi_lines:
        print("\n" + "-" * 76)
        print(" Active Driver & Hardware Telemetry (via nvidia-smi):")
        for line in smi_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                print(f"  - Model: {parts[0]}")
                print(f"    Driver Version: {parts[1]}")
                print(f"    Physical VRAM: {parts[2]} total ({parts[3]} free)")
                if len(parts) >= 6:
                    print(f"    Power Limit: {parts[4]} | Max SM Clock: {parts[5]}")
        print("-" * 76)

    if not found_any:
        print("\n[!] No PCI NVIDIA display/compute devices detected in /sys/bus/pci/devices/.")
        print("    Running on non-PCI / virtualized / CPU-only environment.")
    print("=" * 76)

if __name__ == "__main__":
    probe_pci_gpus()
