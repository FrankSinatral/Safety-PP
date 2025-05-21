# Safety-Polarized and Prioritized Reinforcement Learning
This repository contains the implementation of Safety-Polarized and Prioritized Reinforcement Learning.

## ✅Requirements
REQUIRE:
pip install gymnasium==0.29.1

## 🧪Environments:
The framework supports the following four autonomous driving tasks from the [Highway-env](https://github.com/Farama-Foundation/highway-env) benchmark:
1. Intersection
2. Roundabout
3. Merge
4. TwoWay
and two classic control tasks:
1. Circle
2. Acc

## 🚀Running codes
To launch **SPOM_PER** on the Merge environment for a total of 1e5 steps, run:
```bash
python main_highway.py \
  --env_config configs/MergeEnv/env.json \
  --agent_config configs/MergeEnv/agents/sp_perdqn.json \
  --seed 0
```
To launch **SPOM_PER** on the Circle environment, run:
```bash
python main_classic_control.py
```
