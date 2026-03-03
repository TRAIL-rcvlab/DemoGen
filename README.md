

# <a href="https://demo-generation.github.io/">𝑫𝒆𝒎𝒐𝑮𝒆𝒏: 面向数据高效视觉运动策略学习的合成示教生成</a>

<a href="https://demo-generation.github.io/"><strong>项目主页</strong></a> | <a href="https://arxiv.org/abs/2502.16932"><strong>arXiv</strong></a> | <a href="https://x.com/ZhengrongX/status/1899134914416800123"><strong>Twitter</strong></a> 

**Robotics: Science and Systems (RSS) 2025**


# 概述

<div align="center">
  <img src="pics/teaser.png" alt="teaser" width="100%">
</div>

𝑫𝒆𝒎𝒐𝑮𝒆𝒏 是一种面向机器人操作的合成数据生成方法。仅需一条在真实世界中采集的人类示教，𝑫𝒆𝒎𝒐𝑮𝒆𝒏 即可在数秒内生成数百条经空间增强的合成示教。实验表明，这些示教可有效训练具有强泛化能力的视觉运动策略（如 [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy)）。

<br>
<div align="center">
  <img src="pics/method.png" alt="method" width="100%">
</div>

在动作生成方面，𝑫𝒆𝒎𝒐𝑮𝒆𝒏 采用任务与运动规划（TAMP）的思想，根据新的物体配置调整源动作。在视觉观测生成方面，𝑫𝒆𝒎𝒐𝑮𝒆𝒏 以3D点云为模态，通过3D编辑重新排列场景中的目标物体。

# 更新日志
* **2025/04/02**，正式发布 𝑫𝒆𝒎𝒐𝑮𝒆𝒏。


---

# 环境分离说明

由于 gymnasium 版本冲突（`metaworld` 要求 `gymnasium>=1.1`，`mani_skill` 要求 `gymnasium==0.29.1`），我们使用两个独立的 micromamba 环境：

| 环境名 | gymnasium 版本 | 用途 |
|--------|---------------|------|
| `metaworld_teleop` | 1.2.3 | Metaworld 遥操作 |
| `maniskill_teleop` | 0.29.1 | ManiSkill 遥操作 + 轨迹回放 |

**激活环境的完整命令：**

```bash
# 通用前缀（设置 micromamba）
export MAMBA_EXE='/home/axgu/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/axgu/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"

# 激活 Metaworld 环境
micromamba activate /data2/axgu/micromamba/envs/metaworld_teleop

# 或激活 ManiSkill 环境
micromamba activate /data2/axgu/micromamba/envs/maniskill_teleop
```

> 以下各节中所有命令均假设已进入项目根目录 `cd /data2/axgu/code/DemoGen`，并已激活对应环境。


---

# Metaworld 仿真环境

我们集成了 [Meta-World](https://github.com/Farama-Foundation/Metaworld) 作为多任务机器人操作的仿真基准，提供基于 MuJoCo 的 50 个多样化操作任务。

## 克隆仓库（含子模块）
```bash
git clone --recurse-submodules https://github.com/YOUR_ORG/DemoGen.git
# 如果已克隆：
git submodule update --init --recursive
```

## 安装 Metaworld
```bash
pip install metaworld gymnasium
# 或从子模块安装：
cd third_party/Metaworld && pip install -e . && cd ../..
```

## 键盘/鼠标遥操作
提供遥操作脚本，可通过键盘和鼠标控制 Metaworld 机械臂并采集示教数据。

```bash
# 基本遥操作（带 GUI 窗口）
python metaworld_teleop/teleop_keyboard_mouse.py --task pick-place-v3

# 将采集的示教保存为 zarr 格式
python metaworld_teleop/teleop_keyboard_mouse.py --task pick-place-v3 --save-path data/datasets/teleop
```

**按键说明：**
| 按键 | 功能 |
|------|------|
| W / S | 末端执行器 Y 轴前进/后退 |
| A / D | 末端执行器 X 轴左/右移动 |
| Q / E | 末端执行器 Z 轴上升/下降 |
| 鼠标左键 | 闭合夹爪（抓取） |
| 鼠标右键 | 松开夹爪（释放） |
| R | 重置环境 |
| ESC | 退出并保存 |

## Web 遥操作（无头服务器）
针对无头服务器，我们提供基于浏览器的遥操作界面。支持离屏 GPU 渲染（EGL），通过 WebSocket 将画面流式传输到浏览器。

### 方式 A：本地安装（Conda / Micromamba）

**1. 创建环境（Metaworld v3 要求 Python >= 3.10）**
```bash
micromamba create -n metaworld_teleop python=3.10 -y
micromamba activate metaworld_teleop
```

**2. 安装依赖**
```bash
cd /data2/axgu/code/DemoGen

# PyTorch（必须使用 PyTorch 官方索引获取 CUDA 版本）
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Metaworld + gymnasium
pip install metaworld gymnasium -i https://mirrors.sustech.edu.cn/pypi/web/simple

# Web 服务器 + 数据采集 + 其他
pip install fastapi "uvicorn[standard]" websockets Pillow \
    pynput zarr==2.12.0 numpy h5py imageio imageio-ffmpeg \
    matplotlib hydra-core==1.2.0 termcolor opencv-python==4.8.1.78 \
    -i https://mirrors.sustech.edu.cn/pypi/web/simple

# EGL 离屏渲染库（无需 sudo）
micromamba install -c conda-forge mesalib glew mesa-libegl-cos7-x86_64 -y
```

**3. 验证**
```bash
# 测试 Metaworld 导入
python -c "import metaworld; print('metaworld OK')"

# 测试离屏渲染
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "
import os; os.environ.pop('DISPLAY', None)
import metaworld, gymnasium as gym
os.environ['MUJOCO_GL'] = 'egl'
env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=42, render_mode='rgb_array')
obs, _ = env.reset(); frame = env.render()
print(f'EGL 渲染 OK  frame_shape={frame.shape}')
env.close()
"
```

**4. 启动服务器**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -m metaworld_teleop.teleop_server --port 9527 --task pick-place-v3
```

### 方式 B：Docker（推荐）
```bash
# 拉取预构建镜像
docker pull luobaigu/demogen_telop

# 启动服务器
docker compose -f docker-compose.teleop.yml up

# 或本地构建
# docker compose -f docker-compose.teleop.yml build && docker compose -f docker-compose.teleop.yml up
```

然后在浏览器中打开 `http://服务器IP:9527`。

### 局域网访问：启用 WebHID 和并行下载（Chrome / Edge 设置）

通过局域网 IP 访问服务器时（如 `http://192.168.x.x:9527`），浏览器会将其视为**不安全上下文**并阻止 WebHID（Joy-Con 所需）。需要在浏览器标志中将该地址加入白名单。

**步骤 1 — 将服务器 URL 标记为安全来源：**

| 浏览器 | 地址栏输入 |
|--------|-----------|
| Chrome | `chrome://flags/#unsafely-treat-insecure-origin-as-secure` |
| Edge   | `edge://flags/#unsafely-treat-insecure-origin-as-secure` |

添加你的服务器 URL（如 `http://192.168.1.100:9527`），将标志设为 **Enabled**，然后重启浏览器。

**步骤 2 —（可选）启用并行下载以加快数据传输：**

| 浏览器 | 地址栏输入 |
|--------|-----------|
| Chrome | `chrome://flags/#enable-parallel-downloading` |
| Edge   | `edge://flags/#enable-parallel-downloading` |

设为 **Enabled** 并重启。

> 完成以上步骤后，WebHID（Joy-Con）即可在局域网 HTTP 下正常工作。

### JoyCon 按键调试/映射工具

在 `http://服务器IP:9527/debug` 提供了独立的调试页面，可以：
- 通过 WebHID 连接 Joy-Con，实时查看所有按键/摇杆/IMU 状态
- 为每个按键分配自定义功能名称
- 导出/导入映射为 JSON

### JoyCon 按键映射（右 JoyCon）

| 按键 | 功能 |
|------|------|
| 摇杆（上下） | 前进/后退（Y 轴） |
| 摇杆（左右） | 左/右移动（X 轴） |
| R | Z+（上升） |
| R-Stick 按下 | Z-（下降） |
| ZR | 夹爪开合（切换） |
| A | 开始录制 |
| B | 停止录制 |
| X | 下一个场景（任务） |
| Y | 上一个场景（任务） |
| Home | 重置环境 |

### 关于末端执行器姿态的说明（4-DOF 动作空间）

Metaworld 使用 **4-DOF 动作空间** `[dx, dy, dz, gripper]`。末端执行器姿态（roll/pitch/yaw）**不属于**标准动作空间。

尽管 JoyCon IMU 数据已被解码，且存在实验性的 `apply_orientation()` 函数可直接操作 MuJoCo 的 `mocap_quat` 刚体，**但姿态信息不会被包含在录制的遥操作数据中**。具体而言：

1. **保存的动作仅有 4-DOF** — `[dx, dy, dz, gripper]`，不含姿态分量。
2. **姿态绕过了动作空间** — 它通过直接写入 MuJoCo 的 `mocap_quat` 实现，不被 `env.step()` 捕获。
3. **Metaworld 观测（39维）不包含末端执行器姿态** — 仅包含末端位置、夹爪状态和物体位姿，不含机械臂姿态。
4. **训练的策略无法利用姿态** — 由于动作和观测中均不包含姿态，基于此数据训练的扩散策略（如 DP3）无法学习姿态控制。

如果你的任务需要姿态控制，请考虑使用真实世界管线（`real_world/collect_demo.py`，支持6-DOF 臂位姿：位置+欧拉角），或扩展 Metaworld 动作空间以包含姿态维度。

> **故障排除：** 如果 EGL 失败，尝试 `export MUJOCO_GL=osmesa`。确保防火墙允许 9527 端口。

## 列出可用任务
```bash
python -c "from metaworld_teleop.utils import list_available_tasks; print('\n'.join(list_available_tasks()))"
```


---

# ManiSkill 3 仿真环境

我们还支持 [ManiSkill 3](https://github.com/haosulab/ManiSkill) 作为仿真后端。ManiSkill 提供 75+ 个基于 SAPIEN 的操作任务，使用 Vulkan 渲染（原生支持无头模式，无需 DISPLAY）。

## 安装 ManiSkill

ManiSkill 需要安装在独立的 `maniskill_teleop` 环境中（由于 gymnasium 版本冲突，见上文"环境分离说明"）：

```bash
micromamba activate /data2/axgu/micromamba/envs/maniskill_teleop

# ManiSkill 3
pip install mani_skill

# 修复 numpy/setuptools 兼容性（如需要）
pip install "numpy<2" "setuptools<81"
```

### 验证

```bash
python -c "
import mani_skill, gymnasium as gym
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
env = gym.make('PickCube-v1', num_envs=1, obs_mode='state', control_mode='pd_ee_delta_pos',
               render_mode='rgb_array', human_render_camera_configs=dict(width=640, height=480))
env = CPUGymWrapper(env)
obs, _ = env.reset(); frame = env.render()
print(f'ManiSkill OK  obs={obs.shape}, frame={frame.shape}')
env.close()
"
```

## Web 遥操作（ManiSkill）

同一个 Web 遥操作服务器通过 `--simulator maniskill` 参数支持 ManiSkill：

```bash
# 启动 ManiSkill 服务器
python -m metaworld_teleop.teleop_server --simulator maniskill --port 9527 --task PickCube-v1

# 使用 7-DOF 控制（位置 + 姿态）
python -m metaworld_teleop.teleop_server --simulator maniskill --task PickCube-v1 \
    --control-mode pd_ee_delta_pose

# 使用 RGBD 观测模式进行数据采集
python -m metaworld_teleop.teleop_server --simulator maniskill --task PickCube-v1 \
    --obs-mode rgbd
```

在浏览器中打开 `http://服务器IP:9527`，键盘/JoyCon 操控方式与 Metaworld 相同。

### ManiSkill 控制模式

| 模式 | 自由度 | 动作空间 | 说明 |
|------|--------|---------|------|
| `pd_ee_delta_pos` | 4 | `[dx, dy, dz, gripper]` | 仅位置（与 Metaworld 相同） |
| `pd_ee_delta_pose` | 7 | `[dx, dy, dz, droll, dpitch, dyaw, gripper]` | 位置 + 姿态 |

### 列出可用 ManiSkill 任务

```bash
python -c "from maniskill_teleop.utils import list_available_tasks; print('\n'.join(list_available_tasks()))"
```

## ManiSkill 轨迹回放

ManiSkill 提供官方示教数据集（运动规划、RL、遥操作），可以下载、回放，并转换为不同的控制/观测模式。我们提供 CLI 工具和 Web 可视化回放器两种方式。

### CLI 工具

```bash
# 下载某个环境的官方示教
python -m maniskill_teleop.replay download --env-id PickCube-v1

# 列出本地所有已有的示教数据
python -m maniskill_teleop.replay list

# 查看轨迹信息（集数、成功率、动作维度）
python -m maniskill_teleop.replay info \
    --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5

# 回放：转换控制模式（如 joint → 末端增量）
# 从记录的环境状态重新执行每个 episode，保存为新的 .h5 文件
python -m maniskill_teleop.replay replay \
    --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
    --control-mode pd_ee_delta_pos --obs-mode state --count 10
```

回放工具封装了 ManiSkill 官方的 `mani_skill.trajectory.replay_trajectory` 模块。输出文件保存在原文件旁边，带有描述性后缀（如 `trajectory.state.pd_ee_delta_pos.physx_cpu.h5`）。

### Web 可视化回放

遥操作服务器内置了轨迹回放查看器，地址为 `http://服务器IP:9527/replay`（仅在 `--simulator maniskill` 模式下可用）。

```bash
# 启动 ManiSkill 服务器
python -m metaworld_teleop.teleop_server --simulator maniskill --port 9527

# 在浏览器中打开 http://服务器IP:9527/replay
```

回放查看器功能：
- 浏览和选择本地已有的示教数据集
- 直接在浏览器中下载官方示教
- 逐帧可视化回放，支持播放/暂停/停止控制
- 可调节回放速度（0.25x 到 4x）
- 进度条与步数计数器
- 显示轨迹元数据（集数、成功率、动作维度）

### 轨迹数据格式

ManiSkill 轨迹以 HDF5 格式存储：

```
trajectory.h5
├── traj_0/
│   ├── actions         (steps, action_dim) float32
│   ├── rewards         (steps,) float32
│   ├── success         (steps,) bool
│   ├── terminated      (steps,) bool
│   ├── truncated       (steps,) bool
│   └── env_states/
│       ├── actors/
│       │   └── <name>  (steps+1, 13) float32
│       └── articulations/
│           └── <name>  (steps+1, 31) float32
├── traj_1/
│   └── ...
```

同名的 JSON 伴随文件（`.json` 扩展名）包含 episode 元数据：`env_id`、`episodes` 列表（含 `episode_id`、`episode_seed`、`control_mode`、`elapsed_steps`、`reset_kwargs`、`success`）。


---

# 快速测试启动指南

以下是在本服务器上测试各个模块的完整流程。

## 前置步骤：激活 micromamba

所有测试命令都需要先设置 micromamba 环境。以下前缀在每个新终端中只需执行一次：

```bash
export MAMBA_EXE='/home/axgu/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/axgu/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
cd /data2/axgu/code/DemoGen
```

## 测试 1：Metaworld 遥操作服务器

```bash
# 激活 Metaworld 环境
micromamba activate /data2/axgu/micromamba/envs/metaworld_teleop

# 启动服务器（EGL 离屏渲染）
MUJOCO_GL=egl python -m metaworld_teleop.teleop_server --port 9527 --task pick-place-v3
```

浏览器打开 `http://服务器IP:9527`，应能看到 Sawyer 机械臂场景，可用 WASD/QE 控制。

## 测试 2：ManiSkill 遥操作服务器

```bash
# 激活 ManiSkill 环境
micromamba activate /data2/axgu/micromamba/envs/maniskill_teleop

# 启动服务器（Vulkan 渲染，无需设置 DISPLAY）
python -m metaworld_teleop.teleop_server --simulator maniskill --port 9527 --task PickCube-v1
```

浏览器打开：
- 遥操作页面：`http://服务器IP:9527`
- 轨迹回放页面：`http://服务器IP:9527/replay`

## 测试 3：ManiSkill 轨迹回放 CLI

```bash
# 激活 ManiSkill 环境
micromamba activate /data2/axgu/micromamba/envs/maniskill_teleop

# 列出本地已有的示教数据
python -m maniskill_teleop.replay list

# 查看 PickCube-v1 运动规划示教的信息
python -m maniskill_teleop.replay info \
    --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5

# 回放 3 条轨迹，转换为 pd_ee_delta_pos 控制模式
python -m maniskill_teleop.replay replay \
    --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
    --control-mode pd_ee_delta_pos --obs-mode state --count 3
```

## 测试 4：下载新环境的示教数据

```bash
# 激活 ManiSkill 环境
c

# 下载 StackCube-v1 的官方示教
python -m maniskill_teleop.replay download --env-id StackCube-v1

# 确认下载成功
python -m maniskill_teleop.replay list
```

## 测试 5：两个仿真器同时运行（不同端口）

在两个终端中分别运行：

```bash
# 终端 1：Metaworld（端口 9527）
c

# 终端 2：ManiSkill（端口 9528）
micromamba activate /data2/axgu/micromamba/envs/maniskill_teleop
python -m metaworld_teleop.teleop_server --simulator maniskill --port 9528 --task PickCube-v1
```

浏览器分别打开 `http://服务器IP:9527` 和 `http://服务器IP:9528`。

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| Metaworld 渲染报错 `GLEW initialization error` | 设置 `MUJOCO_GL=egl`（无头服务器必须） |
| ManiSkill 报错 `gymnasium version` | 确认使用 `maniskill_teleop` 环境（gymnasium 0.29.1） |
| Metaworld 报错 `gymnasium version` | 确认使用 `metaworld_teleop` 环境（gymnasium 1.2.3） |
| 浏览器无法连接 WebSocket | 检查防火墙是否放行对应端口 |
| JoyCon 无法连接 WebHID | 需要将服务器地址加入浏览器安全来源白名单（见上文） |
| `MUJOCO_GL=egl` 失败 | 尝试 `export MUJOCO_GL=osmesa` 作为软件渲染后备 |
| numpy 报错 ABI 不兼容 | 降级至 `pip install "numpy<2"`（需要 1.26.x） |
| SAPIEN/setuptools 报错 `pkg_resources` | 降级至 `pip install "setuptools<81"` |


---

# 5 分钟快速体验（DemoGen 合成示教生成）

## 1. 最小安装

#### 1.0. 创建 conda 环境
```bash
conda remove -n demogen --all
conda create -n demogen python=3.8
conda activate demogen
```

#### 1.1. 安装 pip 包
```bash
pip3 install imageio imageio-ffmpeg termcolor hydra-core==1.2.0 zarr==2.12.0 matplotlib setuptools==59.5.0 pynput h5py scikit-video tqdm
```

#### 1.2. 安装 diffusion_policies
我们只需要 diffusion_policies 包中的数据集加载器。
```bash
cd diffusion_policies
pip install -e .
cd ..
```

#### 1.3. 安装 𝑫𝒆𝒎𝒐𝑮𝒆𝒏
```bash
cd demo_generation
pip install -e .
cd ..
```

## 2. 使用 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 生成合成示教

#### 2.1. 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 实现
𝑫𝒆𝒎𝒐𝑮𝒆𝒏 的核心流程实现在 `demo_generation/demo_generation/demogen.py` 中。运行代码需要指定 `demo_generation/demo_generation/config` 文件夹下的 `.yaml` 配置文件，其中提供了一些示例供参考。结合主代码和配置的外部入口是 `demo_generation/gen_demo.py`。

#### 2.2. 输入与输出
我们在 `data/datasets/source` 文件夹下准备了一些包含 1~3 条源示教的 `.zarr` 数据集。运行 `gen_demo.py` 脚本并使用适当的配置文件后，可以生成合成示教数据集，输出到 `data/datasets/generated` 文件夹。当配置文件中的 `generation:render_video` 标志设为 `True` 时，可在 `data/videos` 文件夹中查看渲染的视频，以直观了解生成结果。

**注意：** 示教生成过程本身非常快，但渲染单条轨迹视频大约需要 ~10 秒。因此建议仅在调试时渲染视频。

#### 2.3. 开始生成！
我们在 `demo_generation/run_gen_demo.sh` 脚本中提供了示例命令，涵盖四个任务：**花-花瓶**、**杯-挂架**、**铲-蛋**和**涂酱**。可以尝试运行并在 `data/datasets/generated` 和 `data/videos` 文件夹中对比合成示教与源示教的结果，视频文件名指示了物体的变换方式。
```bash
cd demo_generation
bash run_gen_demo.sh
```


---

# 在自己的任务上使用

只要你的任务需要采集少量示教来克服空间泛化问题，𝑫𝒆𝒎𝒐𝑮𝒆𝒏 就能帮助你节省重复的人力劳动。正如我们在论文中的实验所证明的，𝑫𝒆𝒎𝒐𝑮𝒆𝒏 对各种类型的任务普遍有效，甚至包括接触密集型运动技能。为了帮助你将 𝑫𝒆𝒎𝒐𝑮𝒆𝒏 应用到自己的任务，我们在 `docs` 文件夹下准备了详细指南，有兴趣请查阅！

# 可视化
```shell
micromamba run -n metaworld_teleop python metaworld_teleop/visualize_zarr.py data/datasets/teleop/teleop_assembly-v3_20260303_000546.zarr
```
---

# 许可证
本仓库基于 MIT 许可证发布。详见 [LICENSE](LICENSE)。

# 致谢
我们的代码主要基于以下项目构建：[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[UMI](https://github.com/real-stanford/universal_manipulation_interface)、[MimicGen](https://github.com/NVlabs/mimicgen)。感谢所有作者开源代码并对社区做出的贡献。

如有任何问题或建议，请联系 [Zhengrong Xue](https://steven-xzr.github.io/)。

# 引用

如果我们的工作对你有帮助，请考虑引用：
```
@article{xue2025demogen,
  title={DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning},
  author={Xue, Zhengrong and Deng, Shuying and Chen, Zhenyang and Wang, Yixuan and Yuan, Zhecheng and Xu, Huazhe},
  journal={arXiv preprint arXiv:2502.16932},
  year={2025}
}
```
