<p align="center">
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=240&color=0f172a&text=WanwanLab&fontColor=60d0ff&fontSize=72&desc=Robot RL Simulation Framework&descColor=94a3b8" alt="Lightweight RL Simulation Platform for Humanoid Robots, Optimized for AgiBot X2">
</p>

项目简介
WanwanLab 基于 UniLab 机器人强化学习框架，用于人形机器人 / 机械臂仿真、任务规划与强化学习策略训练，适配 CUDA / CPU 多设备。
环境前置要求

    安装 Anaconda / Miniconda（推荐 Miniconda）
    Python 版本：3.11（项目统一版本，不建议更换）
    NVIDIA 用户：提前安装 CUDA Toolkit（匹配 PyTorch 版本）

    git clone https://github.com/xiaotzhao/WanwanLab.git
    cd WanwanLab

创建并激活虚拟环境 wanwanlab

    # 创建名为 wanwanlab 的conda环境，指定python3.11
    conda create -n wanwanlab python=3.11 -y
    
激活环境

    conda activate wanwanlab

安装全部依赖
    # 优先升级pip
    pip install --upgrade pip
    
    # 读取 requirements.txt 批量安装依赖
    pip install -r requirements.txt
