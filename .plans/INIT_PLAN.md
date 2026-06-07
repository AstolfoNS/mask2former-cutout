# Mask2Former-Cutout 项目初始化文档

> 状态说明：本文是项目初始化阶段的历史计划，不代表当前实现的完整事实。
> 当前准确用法以根目录 `README.md`、`ai-core/docs/dataset_build.md`
> 和实际源码为准。当前项目使用 `uv` workspace；训练入口为
> `ai-core/scripts/train.sh`，推荐数据集为 COCO+VOC 构建结果。

> **项目代号:** `Mask2Former-Cutout`
> **核心目标:** 打造一个极致性能、高精度的双类别（人、车）图像抠图全栈微服务系统。
> **硬件环境:** 独立 Linux(Ubuntu) 裸机 (单卡 RTX 4090 24GB VRAM)

---

## 1. 技术栈全景选型 (Tech Stack)

### 1.1 算法与模型侧 (AI/ML)

- **核心架构**: `Mask2Former` (Swin-Small Backbone)
- **深度学习框架**: PyTorch 2.5.1
- **核心加速库**:
- `transformers` (Hugging Face)
- `accelerate` (分布式与混合精度控制)

- **包与环境管理**: `uv` (追求极致的解析与安装速度)
- **实验监控**: `wandb` (Weights & Biases)
- **配置管理**: `hydra-core` (用于层级化 YAML 参数管理)

### 1.2 后端服务侧 (Backend API)

- **Web 框架**: `FastAPI` (原生异步、极高吞吐量、自动生成 Swagger UI)
- **应用服务器**: `Uvicorn` + `Gunicorn`
- **图像处理**: `OpenCV-Python` (无头版) + `Pillow` (PIL)
- **序列化与验证**: `Pydantic`

### 1.3 前端交互侧 (Frontend UI)

- **核心框架**: `Vue 3` (Composition API) + `TypeScript` + `Vite` (极速构建工具) + Axios
- **样式框架**: `Tailwind CSS` + `Naive UI` (组件库)
- **交互体验**: 实现图像上传、拖拽区域、以及实时高亮抠图结果的画布 (Canvas) 渲染。

---

## 2. 工程目录架构 (Project Structure)

项目将采用严格的**三端解耦**策略（算法研究、后端服务、前端界面分离），当前优先聚焦于 AI 核心（`ai-core`）的搭建。

```text
mask2former-cutout/
├── ai-core/                  # 【阶段一】算法微调与推理核心
│   ├── configs/              # Hydra YAML 配置中心
│   │   ├── data/             # 数据路径与增强配置
│   │   ├── model/            # 模型超参数配置
│   │   ├── train/            # 硬件与训练循环配置 (batch_size, lr等)
│   │   └── default.yaml      # 全局聚合入口
│   ├── data/                 # 统一数据挂载点 (仅限 hvm_coco_512 规范格式)
│   ├── docker/               # (可选) 容器化部署脚本
│   ├── scripts/              # 自动化 Shell 脚本入口
│   │   ├── setup_env.sh      # uv 环境一键初始化
│   │   └── train.sh          # 训练启动命令 (支持 Hydra 参数重写)
│   ├── src/                  # 核心 Python 源码
│   │   ├── data/             # DataLoader 与 COCO 解析
│   │   ├── engine/           # 训练/验证循环 (Trainer)
│   │   ├── models/           # 模型构建与损失函数微调
│   │   └── tools/            # 离线数据处理 (如 COCO 转换脚本)
│   ├── weights/              # 模型权重存放 (HF 预训练与微调 checkpoint)
│   ├── .env.example          # 环境变量模板 (Wandb Key 等)
│   └── pyproject.toml        # 核心依赖清单 (由 uv 解析)
│
├── backend/                  # 【阶段二】FastAPI 推理微服务
│   ├── app/
│   │   ├── api/              # 路由定义 (Routers)
│   │   ├── core/             # 引擎集成 (调用 ai-core/src/models)
│   │   └── schemas/          # Pydantic 数据模型
│   └── main.py               # 服务启动入口
│
└── frontend/                 # 【阶段三】用户交互界面 (Vue3)

```

---

## 3. 极速环境初始化 (Environment Setup)

我们采用 `uv` 在裸机上进行极速、隔离的环境构建，全面释放 4090 算力。

### 3.1 编写 `pyproject.toml`

在 `ai-core/` 目录下创建 `pyproject.toml`：

```toml
[project]
name = "mask2former-cutout-ai"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch==2.5.1",
    "torchvision==0.20.1",
    "transformers==4.57.6",
    "accelerate==1.13.0",
    "datasets",
    "pycocotools",
    "numpy",
    "pillow",
    "opencv-python-headless",
    "wandb",
    "python-dotenv",
    "rich",
    "hydra-core"
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

```

### 3.2 一键部署脚本 (`scripts/setup_env.sh`)

```bash
#!/usr/bin/env bash
set -e

echo "=> 安装 uv 包管理器..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

echo "=> 初始化 Python 3.11 虚拟环境..."
uv venv --python 3.11 .venv

echo "=> 极速同步依赖 (根据 pyproject.toml)..."
uv sync

echo "=> 验证 PyTorch CUDA 环境..."
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"

echo "=> 环境搭建完成！请执行: source .venv/bin/activate"

```

---

## 4. RTX 4090 算力榨取策略 (Hardware Tuning)

在编写训练核心逻辑时，必须确保配置以下参数以彻底打满 24GB 显存，拒绝 GPU 饥饿：

- **关闭检查点**：`gradient_checkpointing=False`（释放计算资源，专注吞吐）。
- **精度跃迁**：`bf16=True`，彻底弃用 `fp16`（根治梯度溢出，无需动态缩放）。
- **并行极致**：`dataloader_num_workers=8` 起步，配合 `dataloader_persistent_workers=True` 和 `pin_memory=True`。
- **TF32 满血**：代码头部强制注入 `torch.set_float32_matmul_precision("high")`。
- **批次探顶**：基于 512x512 输入，`batch_size` 建议从 `4` 起步，安全冒烟测试后可尝试推高至 `6` 或 `8`。

---

## 5. 项目启动路线图 (Roadmap)

- [ ] **Step 1 (基石搭建)**: 在新 4090 服务器上执行 `setup_env.sh`，完成底层隔离。
- [ ] **Step 2 (算力点火)**: 编写并执行 `scripts/train.sh`，以 BF16 满血姿态启动 Mask2Former 微调。
- [ ] **Step 3 (服务封装)**: 训练达标后，切入 `backend/` 目录，封装 FastAPI 推理接口。
- [ ] **Step 4 (全栈闭环)**: 开发前端可视化画布，打通端到端交互。
