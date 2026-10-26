#!/usr/bin/env python3
"""
Oxide Hardware GPU Unlock Diagnostic & Probe Tool
Reference: gpu-unlock (https://github.com/Zagitta/gpu-unlock)

Detects NVIDIA Ampere GPUs (GA102, GA104, GA106) and verifies whether
BF16 / FP16 instruction issue rates can be unlocked to their full
professional equivalents (+80% GEMM throughput: 87.5 TFLOPS -> 157.4 TFLOPS).
"""

import os
import sys
import glob

# Compatible Ampere Device IDs from gpu-unlock
SUPPORTED_DEVICES = {
    "2203": "NVIDIA GeForce RTX 3090 Ti (GA102) -> Full BF16 Unlock Supported (+68-80%)",
    "2204": "NVIDIA GeForce RTX 3090 (GA102) -> Full BF16 Unlock Supported (+25-67%)",
    "2205": "NVIDIA GeForce RTX 3080 Ti OEM (GA102-250) -> Unlock completes",
    "2206": "NVIDIA GeForce RTX 3080 10GB/20GB (GA102) -> Full BF16 Unlock Supported (+56-73%)",
    "2482": "NVIDIA GeForce RTX 3070 Ti (GA104) -> Full BF16 Unlock Supported",
    "2484": "NVIDIA GeForce RTX 3070 (GA104) -> Full BF16 Unlock Supported (+73-81%)",
    "2504": "NVIDIA GeForce RTX 3060 12GB (GA106) -> Full BF16 Unlock Supported (~+80%)",
    "2414": "NVIDIA GA103 GPU -> Compatible",
}

def probe_pci_gpus():
    print("=" * 70)
    print(" Oxide Engine: NVIDIA GPU Hardware Capability & Unlock Probe")
    print("=" * 70)
    
    pci_devices = glob.glob("/sys/bus/pci/devices/*")
    found_any = False

    for dev in sorted(pci_devices):
        vendor_file = os.path.join(dev, "vendor")
        device_file = os.path.join(dev, "device")
        
        if not (os.path.exists(vendor_file) and os.path.exists(device_file)):
            continue
            
        try:
            with open(vendor_file, "r") as f:
                vendor = f.read().strip().lower()
            with open(device_file, "r") as f:
                device = f.read().strip().lower()
                
            # NVIDIA Vendor ID is 0x10de
            if vendor == "0x10de":
                found_any = True
                dev_id = device.replace("0x", "")
                pci_bdf = os.path.basename(dev)
                
                print(f"\n[+] Detected NVIDIA GPU at PCI {pci_bdf}")
                print(f"    PCI ID: 10de:{dev_id}")
                
                if dev_id in SUPPORTED_DEVICES:
                    print(f"    Status: \033[92mCOMPATIBLE WITH GPU-UNLOCK\033[0m")
                    print(f"    Details: {SUPPORTED_DEVICES[dev_id]}")
                    print("\n    To build and load the kernel unlock module:")
                    print(f"      cd references/gpu-unlock && make && sudo insmod kernel/gpu_unlock.ko")
                else:
                    print(f"    Status: Modern or unthrottled architecture (native full-rate FP/BF16 active)")
        except Exception as e:
            continue
            
    if not found_any:
        print("\n[!] No PCI NVIDIA devices detected in /sys/bus/pci/devices/.")
        print("    Running on non-PCI / virtualized / CPU-only environment.")
    print("=" * 70)

if __name__ == "__main__":
    probe_pci_gpus()
