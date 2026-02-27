# ============================================================
# DemoGen Dockerfile - Multi-stage build with proper layering
# Supports training and inference of experiments on GPU servers
# ============================================================

# ------ Stage 1: Base image with system dependencies ------
FROM luobaiguu/demogen

RUN pip3 install --no-build-isolation robosuite gym gymnasium mujoco mujoco-py robomimic --no-cache-dir && \
    rm -rf /root/.cache/pip
ENV DEBIAN_FRONTEND=https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz && \
    cd /tmp && \
    tar -xf /tmp/mujoco210.tar.gz && \
    mkdir -p /root/.mujoco && \
    mv /tmp/mujoco210 /root/.mujoco/mujoco210 && \
    rm /tmp/mujoco210.tar.gz

# writr export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin to ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc

# 降级cpython
RUN pip install cython==0.29.36
# download mujuco to /root/.mujoco/mujoco210
CMD ["bash"]
