# 片段加权实验

## 实验动机

在 DemoGen 框架中，轨迹被分解为 **motion（运动）** 和 **skill（技能/规划）** 片段：
- **Motion 片段**（motion-1, motion-2）：接近/后退运动，动作通过插值重新规划
- **Skill 片段**（skill-1, skill-2）：接触/操作阶段，动作直接从源演示复制

本实验旨在验证**数据的不同片段对模型的贡献是否不同**。我们通过在训练过程中对 motion 和 skill 片段施加**不同的采样权重**，然后观察任务成功率的变化来进行验证。

## 实验方法

1. **生成合成数据**：使用 DemoGen 的仿真流程（标准流程）
2. **训练时使用片段感知的加权采样**：
   - 训练数据中的每条序列根据其中心帧所在的片段进行分类
   - 对 motion 和 skill 片段施加不同的采样倍数
   - 所有权重 >= 1.0（至少采样一倍原始数据）
3. **使用不同权重配置训练策略**
4. **评估**任务成功率并进行比较

## 实验配置

| 配置名称          | Motion 权重  | Skill 权重   | 说明                     |
|------------------|-------------|-------------|--------------------------|
| `balanced`       | 1.0         | 1.0         | 基准（均匀采样）           |
| `motion_2x`     | 2.0         | 1.0         | 2倍过采样 motion          |
| `motion_3x`     | 3.0         | 1.0         | 3倍过采样 motion          |
| `skill_2x`      | 1.0         | 2.0         | 2倍过采样 skill           |
| `skill_3x`      | 1.0         | 3.0         | 3倍过采样 skill           |

## 使用方法

### 1. 生成数据（标准 DemoGen 流程）
```bash
# cd demo_generation
python demo_generation/gen_demo.py --config-name=flower generation.render_video=False
```

### 2. 运行所有实验
```bash
cd experiments/segment_weighting
bash run_experiments.sh
```

### 3. 运行单个实验
```bash
cd diffusion_policies
python train.py --config-path=../experiments/segment_weighting/config \
    --config-name=segment_weight_exp \
    task.segment_weights.motion=2.0 \
    task.segment_weights.skill=1.0 \
    exp_name=motion_2x
```

### 4. 评估训练好的策略（推理 / Inference）

训练完成后，使用 `eval_experiments.sh` 在仿真环境中评估策略的平均成功率：

```bash
cd experiments/segment_weighting

# 批量评估：遍历基础目录下所有含 checkpoint 的子目录（默认 robosuite，20 episodes）
bash eval_experiments.sh ../diffusion_policies/data/outputs/2026.02.26


# （Docker 内等价写法）
bash eval_experiments.sh /workspace/DemoGen/diffusion_policies/data/outputs/2026.02.26

# 批量评估：使用 metaworld 环境，50 episodes
bash eval_experiments.sh ../diffusion_policies/data/outputs/2026.02.26 metaworld 50

# 单个实验评估：指定具体的 checkpoint 目录
bash eval_experiments.sh ../diffusion_policies/data/outputs/2026.02.26/17.09.36_train_diffusion_unet_hybrid_segment_weight_exp robosuite 20 balanced
```

其中第一个参数是训练输出目录。批量模式会自动搜索该目录下所有包含 checkpoint 的子目录（优先使用 `checkpoints/latest.ckpt`，若不存在则自动选择最新编号的 `*.ckpt`）；
单个实验模式（提供第 4 个参数 `exp_name`）则直接使用给定目录的 checkpoint。

> 注意：不要在终端里输入 Markdown 链接形式（例如 `[eval_experiments.sh](...)`），应直接输入命令本身。

也可以直接调用 `diffusion_policies/eval.py` 评估单个 checkpoint：

```bash
cd diffusion_policies
python eval.py \
    --config-path=../experiments/segment_weighting/config \
    --config-name=segment_weight_exp \
    hydra.run.dir=<checkpoint_dir> \
    +task.env_runner._target_=diffusion_policies.env_runner.robosuite_runner.RobosuiteRunner \
    +task.env_runner.eval_episodes=20 \
    +task.env_runner.n_obs_steps='${n_obs_steps}' \
    +task.env_runner.n_action_steps='${n_action_steps}' \
    +task.env_runner.shape_meta='${shape_meta}'
```

## 文件结构

```
experiments/segment_weighting/
├── README.md                          # 本文件
├── segment_weighted_sampler.py        # 加权采样器实现
├── weighted_segment_dataset.py        # 支持片段加权采样的数据集
├── config/
│   ├── task/
│   │   └── segment_weight_exp.yaml    # 任务配置（片段边界）
│   └── segment_weight_exp.yaml        # 主训练配置
├── run_experiments.sh                 # 批量运行所有权重配置的脚本（训练）
└── eval_experiments.sh                # 批量评估训练好策略的脚本（推理）
```

## 片段边界参考

片段边界由任务配置中的 `parsing_frames` 定义（与 DemoGen 约定一致）：

```
帧:     0          skill_1     motion_2     skill_2      end
        |--motion1--|---skill1---|--motion2---|---skill2---|
```

- **motion-1**: 帧 `[0, skill_1)` — 接近运动
- **skill-1**: 帧 `[skill_1, motion_2)` — 第一阶段操作技能
- **motion-2**: 帧 `[motion_2, skill_2)` — 过渡运动
- **skill-2**: 帧 `[skill_2, end)` — 第二阶段操作技能


# 缺失
docker环境中缺少最终测试需要环境，补充安装
pip install gym robommic