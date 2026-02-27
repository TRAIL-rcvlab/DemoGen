# ============================================================
# DemoGen Dockerfile - Multi-stage build with proper layering
# Supports training and inference of experiments on GPU servers
# ============================================================

# ------ Stage 1: Base image with system dependencies ------
FROM luobaigu/demogen:latest

RUN pip3 install --no-build-isolation robosuite gym gymnasium mujoco mujoco-py robomimic cython==0.29.36 --no-cache-dir && \
    rm -rf /root/.cache/pip
ENV DEBIAN_FRONTEND=noninteractive
# 假设你已经在宿主机的当前目录下载了 mujoco210-linux-x86_64.tar.gz
COPY mujoco210-linux-x86_64.tar.gz /tmp/mujoco210.tar.gz
RUN cd /tmp && \
    tar -xf /tmp/mujoco210.tar.gz && \
    mkdir -p /root/.mujoco && \
    mv /tmp/mujoco210 /root/.mujoco/mujoco210 && \
    rm /tmp/mujoco210.tar.gz

# writr export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin to ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc
# download mujuco to /root/.mujoco/mujoco210
CMD ["bash"]
