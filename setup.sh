#!/bin/bash
set -e

git config --global user.email "paul.plays.a.pun@gmail.com"
git config --global user.name "better-call-paul"

mkdir -p ~/devlibs
cd ~/devlibs

apt-get update
apt-get install -y cmake git curl build-essential

apt-get install -y \
    cuda-toolkit-12-6 \
    libcublas-dev-12-6 \
    libcudnn8-dev \
    libthrust-dev

if [ ! -d "cutlass" ]; then
  git clone https://github.com/NVIDIA/cutlass.git
  cd cutlass
  mkdir -p build && cd build
  cmake .. && make -j
  cd ../..
fi

if ! command -v ncu &> /dev/null; then
  echo "Installing Nsight Compute CLI..."
  apt-get update && apt-get install -y nvidia-nsight-compute
else
  echo "Nsight Compute already installed."
fi

echo "Setup complete."
