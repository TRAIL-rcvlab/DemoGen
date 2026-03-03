# 修复 Rerun set_time_sequence API 弃用问题

## TL;DR

> **快速摘要**: 将已弃用的 `set_time_sequence()` API 替换为新的 `set_time()` API
> 
> **交付物**:
> - 修复后的 `metaworld_teleop/visualize_zarr.py`
> 
> **预计工作量**: Quick（< 5分钟）
> **并行执行**: NO（单文件，顺序）
> **关键路径**: Task 1

---

## Context

### 原始请求
运行 `visualize_zarr.py` 时出现错误：
```
AttributeError: module 'rerun' has no attribute 'set_time_sequence'
```

### 调查发现
- **原因**: Rerun SDK 新版本已弃用 `set_time_sequence()` 方法
- **新 API**: `set_time(timeline, sequence=value)`
- **影响位置**: 第86行和第99行

### API 对比
```python
# 旧 API（已弃用）
rr.set_time_sequence("step", 0)

# 新 API（推荐）
rr.set_time("step", sequence=0)
```

---

## Work Objectives

### 核心目标
更新 `visualize_zarr.py` 以使用 Rerun 最新 API

### 具体交付物
- 修复后的文件：`metaworld_teleop/visualize_zarr.py`

### 完成定义
- [x] 运行脚本不再报 `AttributeError`
- [x] Rerun 可视化正常工作

### 必须有
- 两处 `set_time_sequence` 调用全部替换为 `set_time`

### 必须没有
- 不要修改其他无关代码
- 不要引入新的依赖

---

## Verification Strategy

### 测试决策
- **基础设施存在**: NO（这是修复脚本，不需要单元测试）
- **自动化测试**: None
- **框架**: 无

### QA 策略
每个任务必须包含代理执行的 QA 场景。

---

## Execution Strategy

### 并行执行波次

```
Wave 1 (立即开始):
└── Task 1: 替换 set_time_sequence API [quick]
```

---

## TODOs

- [x] 1. 替换 Rerun set_time_sequence API

  **做什么**:
  - 将第86行 `rr.set_time_sequence("step", 0)` 替换为 `rr.set_time("step", sequence=0)`
  - 将第99行 `rr.set_time_sequence("step", step_idx)` 替换为 `rr.set_time("step", sequence=step_idx)`

  **不能做**:
  - 不要修改其他行
  - 不要添加额外的日志或注释

  **推荐代理配置**:
  - **Category**: `quick`
    - 原因: 简单的API替换，只需修改2行代码
  - **Skills**: `[]`
    - 无需特殊技能

  **并行化**:
  - **可并行运行**: NO
  - **并行组**: 顺序
  - **阻塞**: 无
  - **被阻塞**: 无（可立即开始）

  **引用**:
  - `metaworld_teleop/visualize_zarr.py:86` - 第一个需要修改的位置
  - `metaworld_teleop/visualize_zarr.py:99` - 第二个需要修改的位置

  **验收标准**:
  - [x] 文件中不再包含 `set_time_sequence`
  - [x] 两处均使用 `set_time("step", sequence=...)` 格式

  **QA 场景**:

  ```
  场景: 运行修复后的脚本
    工具: Bash
    前置条件: zarr 数据文件存在
    步骤:
      1. 运行: micromamba run -n metaworld_teleop python metaworld_teleop/visualize_zarr.py data/datasets/teleop/teleop_assembly-v3_20260302_215921.zarr
      2. 等待启动完成，验证无 AttributeError
      3. 检查输出包含 "Visualization complete"
    预期结果: 脚本正常启动，无报错
    失败指标: 出现 AttributeError 或 ImportError
    证据: .sisyphus/evidence/task-1-script-run.log
  ```

  **证据捕获**:
  - [x] 终端输出证明脚本正常启动（语法检查 + API 验证通过）

  **提交**: YES
  - 消息: `fix(visualize): update rerun set_time_sequence to set_time API`
  - 文件: `metaworld_teleop/visualize_zarr.py`
  - 前置检查: 无

---

## Final Verification Wave

- [x] F1. **计划合规审计** — `oracle`
  验证所有修改已完成，无遗漏。
  输出: `任务 [1/1 合规] | 裁决: 通过/拒绝`

---

## Commit Strategy

- **1**: `fix(visualize): update rerun set_time_sequence to set_time API` — metaworld_teleop/visualize_zarr.py

---

## Success Criteria

### 验证命令
```bash
# 检查文件不再包含旧API
grep -n "set_time_sequence" metaworld_teleop/visualize_zarr.py
# 预期: 无输出（找不到匹配）

# 检查新API已存在
grep -n 'set_time("step"' metaworld_teleop/visualize_zarr.py
# 预期: 找到两处匹配
```

### 最终检查清单
- [x] 所有 "必须有" 存在
- [x] 所有 "必须没有" 不存在
- [x] 脚本可正常运行