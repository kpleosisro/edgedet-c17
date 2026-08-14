#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cmake ninja-build pkg-config

python3 -m pip install --user --upgrade pip
python3 -m pip install --user kaggle pillow

mkdir -p "${HOME}/.kaggle"
chmod 700 "${HOME}/.kaggle"
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "{\"username\":\"${KAGGLE_USERNAME}\",\"key\":\"${KAGGLE_KEY}\"}" \
    > "${HOME}/.kaggle/kaggle.json"
  chmod 600 "${HOME}/.kaggle/kaggle.json"
  echo "Wrote ~/.kaggle/kaggle.json from Codespace secrets."
else
  echo "Kaggle credentials not in the environment. Set KAGGLE_USERNAME and KAGGLE_KEY as Codespace secrets, or copy kaggle.json to ~/.kaggle/."
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
