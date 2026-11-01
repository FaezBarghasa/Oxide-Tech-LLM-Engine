#!/usr/bin/env bash
# Hardware Firmware Clock & Voltage Stabilization for NVIDIA Ada / Ampere dGPUs
set -euo pipefail

echo "===================================================================="
echo " Oxide Engine: NVIDIA GPU Hardware Clock & Power Stabilization"
echo "===================================================================="

TARGET_MHZ="${1:-2550}"
TARGET_POWER="${2:-120}"

if [ "$EUID" -ne 0 ]; then
  echo "[-] Not running as root. Invoking sudo to lock hardware clock frequencies..."
  SUDO="sudo"
else
  SUDO=""
fi

echo "[*] Step 1: Enabling NVIDIA Persistence Mode (prevents idle power gating)..."
$SUDO nvidia-smi -pm 1

echo "[*] Step 2: Locking SM Graphics Clock to ${TARGET_MHZ} MHz..."
$SUDO nvidia-smi -lgc "${TARGET_MHZ},${TARGET_MHZ}"

echo "[*] Step 3: Setting Target Power Limit to ${TARGET_POWER} W..."
$SUDO nvidia-smi -pl "${TARGET_POWER}" || echo "[!] Power limit change not supported on this VBIOS (using stock max)."

echo "===================================================================="
echo "[+] Hardware Clocks Successfully Stabilized"
echo "===================================================================="
nvidia-smi --query-gpu=name,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,power.draw,power.limit,temperature.gpu --format=csv

