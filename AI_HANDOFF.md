# DemoGen 项目交接说明（给其他 AI）

本文档用于让新的 AI 助手快速接手当前项目状态，避免重复踩坑。

## 1) 项目目标与当前重点

- 项目：`DemoGen`
- 主要场景：网页遥操作 + 轨迹回放 + 导出训练数据
- 当前主要工作流：
  - `metaworld_teleop`: Web 服务入口（同时承载 Metaworld / ManiSkill）
  - `maniskill_teleop`: ManiSkill 环境管理、轨迹回放、下载、导出

本阶段重点：
- 修复 ManiSkill 回放稳定性（已完成：env_states 回放 + 强制 `physx_cpu`）
- 任务列表与下载能力对齐官方数据源（已完成）
- Replay 页面支持“当前环境重录并导出 DP3 zarr”（已实现）

---

## 2) 环境与运行配置（最关键）

### 2.1 两套 micromamba 环境

由于 `gymnasium` 版本冲突，必须分环境：

- `metaworld_teleop` 环境：`gymnasium 1.2.3`
- `maniskill_teleop` 环境：`gymnasium 0.29.1`

当前常用激活方式（ManiSkill）：

```bash
export MAMBA_EXE='/home/axgu/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/axgu/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate /data2/axgu/micromamba/envs/maniskill_teleop
```

运行项目命令时建议统一带：

```bash
PYTHONPATH=.
```

### 2.2 LSP 报错说明

若 IDE 报 `numpy/h5py/fastapi/gymnasium` unresolved，多半是 IDE 绑定错环境，不是代码真错。

可用如下命令做真校验：

```bash
PYTHONPATH=. python -m py_compile metaworld_teleop/teleop_server.py maniskill_teleop/replay.py maniskill_teleop/utils.py
```

---

## 3) 服务入口与页面

- 主服务：`metaworld_teleop/teleop_server.py`
- Replay 页面：`metaworld_teleop/static/replay.html`
- 默认端口：`9527`

ManiSkill 模式启动示例：

```bash
PYTHONPATH=. python -m metaworld_teleop.teleop_server --simulator maniskill --port 9527 --task PickCube-v1
```

---

## 4) 现有 API（ManiSkill 相关）

### 4.1 任务与数据

- `GET /api/available_tasks`
  - 返回所有任务，包含：
    - `has_demos`: 本地是否已有数据
    - `downloadable`: 是否在 ManiSkill 官方可下载 UID 列表内

- `GET /api/demos`
  - 返回本地 demo 文件列表

- `POST /api/demos/download/{env_id}`
  - 仅当 `env_id` 在官方可下载列表中才允许下载
  - 非官方 UID 直接 400（避免之前 KeyError）

### 4.2 轨迹回放与导出

- `GET /api/demos/trajectory_info?traj_path=...`
  - 返回轨迹 metadata 信息

- `WS /ws/replay`
  - 可视化流式回放

- `POST /api/demos/export_dp3`
  - 新增：在当前环境重录指定 episode，并导出 DP3 训练格式 zarr
  - body 示例：

```json
{
  "traj_path": "/home/axgu/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5",
  "episode_id": 0,
  "control_mode": null,
  "n_points": 512
}
```

---

## 5) Replay 页面当前行为

文件：`metaworld_teleop/static/replay.html`

- 环境分组：
  - `已下载`
  - `可下载(官方)`
  - `暂无官方下载`
- 下载按钮对“暂无官方下载”任务会禁用，并提示不可下载
- 新增按钮：`当前环境重录并导出DP3 Zarr`
  - 触发 `POST /api/demos/export_dp3`
  - 成功后在信息栏显示导出路径

---

## 6) DP3 zarr 导出格式与点云流程

导出函数：`maniskill_teleop/replay.py` 中 `export_episode_to_dp3_zarr(...)`

输出结构（与 DemoGen/DP3 训练读法对齐）：

- `data/agent_pos`  -> `(T, D)` float32
- `data/point_cloud` -> `(T, 512, 3)` float32
- `data/action` -> `(T, A)` float32
- `meta/episode_ends` -> `(1,)` int64

点云处理流程（按 Metaworld/DP3 流程一致）：

1. 深度图 -> 相机坐标 3D 点
2. 有效深度过滤（`0.1m ~ 5.0m`）
3. 工作空间统计裁剪（median +/- 3*std）
4. DBSCAN 去离群（可选，依赖 sklearn）
5. FPS 下采样到固定点数（默认 512）

注意：ManiSkill `depth` 常见为 int16 毫米，导出里已自动转换到米（`/1000`）。

---

## 7) 关键代码位置

- 官方可下载 UID 映射
  - `maniskill_teleop/utils.py`
  - `list_official_downloadable_tasks()`

- 轨迹回放/导出
  - `maniskill_teleop/replay.py`
  - `visual_replay_generator(...)`
  - `export_episode_to_dp3_zarr(...)`

- Web API
  - `metaworld_teleop/teleop_server.py`
  - `api_available_tasks`
  - `api_download_demos`
  - `api_export_dp3`

- 前端 Replay
  - `metaworld_teleop/static/replay.html`

---

## 8) 已知限制与风险

- 官方 demo 下载受 `mani_skill.utils.download_demo.DATASET_SOURCES` 限制。
  - 不在官方列表的任务（如 `InsertFlower-v1`）无法官方下载。
- `physx_cuda` 与现有 server 进程共存限制，视觉回放统一用 `physx_cpu`。
- 导出 DP3 zarr 是同步阻塞接口，长轨迹会占用请求时间。

---

## 9) 建议后续改进

1. 把 `/api/demos/export_dp3` 做成后台任务（异步队列 + 进度轮询）
2. 导出时增加可选字段：输出目录、点数（512/1024）、是否保存调试视频
3. 在 Replay 页面增加“导出列表刷新”与“打开目录”提示
4. 增加 zarr 完整性检查接口（shape/dtype/episode_ends 一致性）

---

## 10) 快速验收清单（给接手 AI）

1. 启动 ManiSkill 服务并打开 Replay 页面
2. 选一个本地可回放轨迹（如 PickCube）
3. 点“当前环境重录并导出DP3 Zarr”
4. 确认返回路径存在，且 zarr 包含：
   - `data/agent_pos`
   - `data/point_cloud`
   - `data/action`
   - `meta/episode_ends`
5. 用训练侧读取逻辑验证 shape 与 dtype
