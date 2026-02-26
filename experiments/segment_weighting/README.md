# Segment Weighting Experiment

## Motivation

In the DemoGen framework, trajectories are decomposed into **motion** and **skill (plan)** segments:
- **Motion segments** (motion-1, motion-2): Approach/retract motions where actions are re-planned via interpolation
- **Skill segments** (skill-1, skill-2): Contact/manipulation phases where actions are copied from the source demo

This experiment investigates whether different data segments contribute differently to the learned policy. We test this by applying **different sampling weights** to motion vs. skill segments during training, then measuring task success rate changes.

## Approach

1. **Generate synthetic data** using DemoGen's simulation pipeline (standard process)
2. **Apply segment-aware weighted sampling** during training:
   - Each sequence in the training data is classified by which segment its center frame falls into
   - Different sampling multipliers are applied to motion vs. skill segments
   - All weights are ≥ 1.0 (at least 1× the original data is sampled)
3. **Train policies** with different weight configurations
4. **Evaluate** task success rates and compare

## Experiment Configurations

| Config Name      | Motion Weight | Skill Weight | Description                |
|------------------|---------------|--------------|----------------------------|
| `balanced`       | 1.0           | 1.0          | Baseline (uniform sampling)|
| `motion_2x`     | 2.0           | 1.0          | 2× oversample motion       |
| `motion_3x`     | 3.0           | 1.0          | 3× oversample motion       |
| `skill_2x`      | 1.0           | 2.0          | 2× oversample skill        |
| `skill_3x`      | 1.0           | 3.0          | 3× oversample skill        |

## Usage

### 1. Generate data (standard DemoGen pipeline)
```bash
cd demo_generation
python gen_demo.py --config-name=flower
```

### 2. Run all experiments
```bash
cd experiments/segment_weighting
bash run_experiments.sh
```

### 3. Run a single experiment
```bash
cd diffusion_policies
python train.py --config-path=../experiments/segment_weighting/config \
    --config-name=segment_weight_exp \
    task.segment_weights.motion=2.0 \
    task.segment_weights.skill=1.0 \
    exp_name=motion_2x
```

## File Structure

```
experiments/segment_weighting/
├── README.md                          # This file
├── segment_weighted_sampler.py        # Weighted sampler implementation
├── weighted_segment_dataset.py        # Dataset with segment-aware weighted sampling
├── config/
│   ├── task/
│   │   └── segment_weight_exp.yaml    # Task config with segment boundaries
│   └── segment_weight_exp.yaml        # Main training config
└── run_experiments.sh                 # Script to run all weight configurations
```

## Parsing Frames Reference

Segment boundaries are defined by `parsing_frames` in the task config (matching DemoGen convention):

```
Frame:    0          skill_1     motion_2     skill_2      end
          |--motion1--|---skill1---|--motion2---|---skill2---|
```

- **motion-1**: frames `[0, skill_1)`
- **skill-1**: frames `[skill_1, motion_2)`
- **motion-2**: frames `[motion_2, skill_2)`
- **skill-2**: frames `[skill_2, end)`
