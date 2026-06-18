<p align="center">
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=240&color=0f172a&text=WanwanLab&fontColor=60d0ff&fontSize=72&desc=Robot RL Simulation Framework&descColor=94a3b8" alt="Lightweight RL Simulation Platform for Humanoid Robots, Optimized for AgiBot X2">
</p>

项目简介
WanwanLab 基于 UniLab 机器人强化学习框架，用于人形机器人 / 机械臂仿真、任务规划与强化学习策略训练，适配 CUDA / CPU 多设备。
环境前置要求

```bash
# 0. If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/xiaotzhao/WanwanLab.git
cd WanwanLab

# 2. Install dependencies
# Pick the setup command for your platform.

# Linux CUDA or macOS
uv sync
# Without shell completion setup: uv sync --extra motrix
# If `make` is not installed: uv sync --extra motrix && uv run --no-sync unilab-complete install
