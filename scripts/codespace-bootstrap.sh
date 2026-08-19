#!/usr/bin/env bash
# One-shot setup for a running default-image Codespace (no rebuild required).
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cmake ninja-build pkg-config

python3 -m pip install --user --upgrade pip
python3 -m pip install --user kaggle pillow
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "${HOME}/.kaggle"
chmod 700 "${HOME}/.kaggle"
if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    umask 077
    printf '%s\n' "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" \
      > "${HOME}/.kaggle/kaggle.json"
    chmod 600 "${HOME}/.kaggle/kaggle.json"
  else
    echo "Missing ~/.kaggle/kaggle.json and KAGGLE_USERNAME/KAGGLE_KEY."
    echo "In the Codespace: mkdir -p ~/.kaggle && chmod 700 ~/.kaggle"
    echo "then paste your kaggle.json there (chmod 600)."
    exit 1
  fi
fi
chmod 600 "${HOME}/.kaggle/kaggle.json"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

echo "EdgeDet Codespace ready. Tools in build/: edtrain edeval edviz edpack-yolo edpack-voc"
python3 -m kaggle datasets list --max-size 1 >/dev/null
echo "Kaggle CLI authenticated."
