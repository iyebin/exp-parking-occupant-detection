#!/usr/bin/env bash
# One-time setup on the Raspberry Pi 4 (Raspberry Pi OS 64-bit, Bookworm or newer).
#   bash setup_pi.sh
set -e
cd "$(dirname "$0")"

echo "== 0. checks"
[ "$(uname -m)" = "aarch64" ] || { echo "Needs 64-bit Raspberry Pi OS (aarch64). Found $(uname -m)."; exit 1; }
cat /proc/device-tree/model; echo
free -h | head -2

echo "== 1. system packages"
sudo apt update
sudo apt install -y v4l-utils python3-venv python3-dev libgl1 git
sudo usermod -aG video "$USER"

echo "== 2. python environment (keeps system gpiozero for the LED test)"
python3 -m venv --system-site-packages ~/exitguard-venv
source ~/exitguard-venv/bin/activate
df -h / | tail -1
# big wheels: unpack on the SD card, not in /tmp (RAM-backed on newer Pi OS), and don't cache
mkdir -p ~/pip-tmp
export TMPDIR=~/pip-tmp PIP_NO_CACHE_DIR=1
pip install --upgrade pip
# CPU-only PyTorch FIRST - otherwise pip pulls the CUDA build (~3 GB of nvidia_* packages) the Pi can't use
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
rm -rf ~/pip-tmp

echo "== 3. versions"
python - <<'PY'
import importlib
for m in ["ultralytics", "torch", "cv2", "onnxruntime", "ncnn", "psutil", "pandas"]:
    try:
        print(f"{m:12s}", importlib.import_module(m).__version__)
    except Exception as e:
        print(f"{m:12s} MISSING ({e})")
PY

echo "== 4. camera"
v4l2-ctl --list-devices || true
lsusb | grep -i 1532 || echo "Razer camera (1532:0e05) not found on USB - check cable / blue USB 3.0 port"
echo "usbcore quirks currently set: $(cat /sys/module/usbcore/parameters/quirks 2>/dev/null)"

echo "== 5. power + temperature"
vcgencmd get_throttled
vcgencmd measure_temp

cat <<'TXT'

Done. Before each test session:
  source ~/exitguard-venv/bin/activate
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
(the governor resets on reboot; log out/in once so the 'video' group applies)
TXT
