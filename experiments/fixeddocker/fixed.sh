#!/bin/bash
set -e

echo "Installing Python packages..."
pip3 install --no-build-isolation robosuite gym gymnasium mujoco mujoco-py robomimic --no-cache-dir
rm -rf /root/.cache/pip

echo "Downloading and installing MuJoCo 210..."
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
cd /tmp
tar -xf /tmp/mujoco210.tar.gz
mkdir -p /root/.mujoco
mv /tmp/mujoco210 /root/.mujoco/mujoco210
rm /tmp/mujoco210.tar.gz

echo "Configuring LD_LIBRARY_PATH..."
if ! grep -q "mujoco210/bin" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc
fi

echo "Downgrading Cython..."
pip install cython==0.29.36

echo "Fix script completed successfully!"
