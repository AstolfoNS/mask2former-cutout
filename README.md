# Mask2Former-Cutout

基于 Mask2Former 的双类别图像分割与抠图项目，面向 `person` 和 `car` 两类目标，支持模型训练、模型推理服务和前端交互界面建设。

当前推荐推理模型保存在：

```text
ai-core/weights/mask2former-cutout-coco-voc-v1/best_model/
```

该目录是后续推理和部署应优先使用的最小模型目录。旧 HVM 训练权重仍保留在
`ai-core/weights/mask2former-cutout/best_model/`，主要用于回滚和对比，不建议作为默认泛化模型。

## 项目目标

项目最终目标是搭建一个完整的图像抠图系统：

```text
用户上传图片
    -> 前端页面提交请求
    -> FastAPI 后端接收图片
    -> Mask2Former 模型推理
    -> 生成 mask / overlay / transparent cutout
    -> 前端展示并下载结果
```

核心能力：

- 对人和车进行高精度分割。
- 输出透明 PNG 抠图。
- 输出 mask 图。
- 输出 overlay 预览图。
- 支持本地开发和 GPU 服务器部署。
- 支持 RTX 4090 训练与 RTX 4060 8GB 单图推理。

## 项目结构

```text
mask2former-cutout/
    ai-core/
        src/
            train.py
        scripts/
            data/
                build_cutout_dataset.py
                download_and_build_coco_cutout.py
                prepare_pascal_voc2012.py
            train.sh
            setup_env.sh
        data/
        weights/
            mask2former-cutout-coco-voc-v1/
                best_model/
                training_metrics.csv
            mask2former-cutout/
                best_model/
                training_metrics.csv
        docs/

    backend/
        main.py
        app/
            api/
            core/
            schemas/
        pyproject.toml

    frontend/
        src/
            components/
            composables/
            types/
            App.vue
            main.ts
        package.json

    .docs/
        project_build_guide.md

    pyproject.toml
    README.md
```

## 当前模型状态

训练任务：

- 模型：`facebook/mask2former-swin-small-coco-instance`
- 任务：person / car 双类别分割
- 输入尺寸：`512x512`
- 数据格式：COCO JSON
- 当前推荐训练集：COCO 2017 小规模 person/car 子集 + Pascal VOC 2012 trainval 语义分割数据
- 当前推荐权重：`ai-core/weights/mask2former-cutout-coco-voc-v1/best_model/`
- 当前推荐权重最佳验证指标：`mIoU ~= 0.7221`
- 旧 HVM 权重最佳验证指标曾达到 `mIoU ~= 0.9795`，但数据分布更窄，泛化效果不作为当前主判断依据。

当前数据集构建与训练命令见：

```text
ai-core/docs/dataset_build.md
```

当前权重目录中包含：

```text
ai-core/weights/mask2former-cutout-coco-voc-v1/
    best_model/
    checkpoint-5504/
    checkpoint-6880/
    checkpoint-7224/
    training_metrics.csv

ai-core/weights/mask2former-cutout/
    best_model/
    checkpoint-3178/
    checkpoint-4994/
    checkpoint-5448/
    training_metrics.csv
    validation_preview.png
```

说明：

- `mask2former-cutout-coco-voc-v1/best_model/`：当前推荐用于推理和部署。
- `mask2former-cutout/best_model/`：旧 HVM 训练权重，仅建议作为回滚或对比模型。
- `checkpoint-*`：用于恢复训练，推理部署通常不需要。
- `training_metrics.csv`：训练与验证指标历史，可用于绘制 loss 和 mIoU 曲线。
- `validation_preview.png`：验证集可视化预览图。

## 环境要求

推荐环境：

```text
Python >= 3.11, < 3.12
uv
PyTorch 2.5.1
CUDA 12.4
Node.js 18+
npm
```

训练环境推荐：

```text
RTX 4090 24GB
```

推理环境推荐：

```text
RTX 4060 8GB 或更高
```

RTX 4060 8GB 只做单图推理时可以正常运行，但建议：

- 只加载 `best_model/`。
- `batch_size=1`。
- 使用 `torch.inference_mode()`。
- 使用 `float16` 或硬件支持时使用 `bfloat16`。
- 后端限制 GPU 并发推理请求。

## 安装依赖

项目使用 uv workspace 管理 Python 子项目：

```bash
uv sync
```

AI Core 依赖位于：

```text
ai-core/pyproject.toml
```

后端依赖位于：

```text
backend/pyproject.toml
```

前端依赖安装：

```bash
cd frontend
npm install
```

## 模型训练

训练入口：

```bash
cd ai-core
./scripts/train.sh
```

也可以传入参数：

```bash
cd ai-core
./scripts/train.sh \
    --annotation_file data/cutout_mix_512/annotations.json \
    --image_dir data/cutout_mix_512/images \
    --output_dir ./weights/mask2former-cutout-coco-voc-v1 \
    --epochs 40 \
    --batch_size 8 \
    --gradient_accumulation_steps 2
```

训练主脚本：

```text
ai-core/src/train.py
```

训练指标会持久化到：

```text
ai-core/weights/<run-name>/training_metrics.csv
```

CSV 字段包括：

```text
event, step, epoch, loss, learning_rate,
eval_loss, mIoU, Dice, PixelAcc,
iou_person, iou_car, dice_person, dice_car
```

## 模型推理建议

推理部署时只需要拷贝：

```text
ai-core/weights/mask2former-cutout-coco-voc-v1/best_model/
```

不要把 `checkpoint-*` 作为生产推理模型目录使用。checkpoint 中包含 optimizer、scheduler 和随机状态文件，主要用于恢复训练。

推荐推理流程：

```text
1. 加载 best_model。
2. 读取上传图片并转为 RGB。
3. 使用与训练一致的 letterbox 预处理：等比例缩放，RGB 114 填充到 512x512。
4. 模型推理得到 person/car mask。
5. 先裁掉 letterbox padding 区域，再将 mask resize 回原图尺寸。
6. 生成 mask、overlay、透明 PNG。
7. 返回结果 URL。
```

## 后端服务

后端目录：

```text
backend/
```

当前入口：

```text
backend/main.py
```

开发启动：

```bash
cd /home/timeleafing/projects/mask2former-cutout
uv run --project backend uvicorn main:app \
    --app-dir backend \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

不要直接运行系统全局的 `uvicorn`。项目依赖由 `uv` 管理，直接运行
`uvicorn main:app ...` 可能会使用系统 Python，导致 `python-dotenv`、
`torch` 等依赖缺失。

指定新训练模型作为默认推理模型：

```bash
cd /home/timeleafing/projects/mask2former-cutout
MODEL_DIR=/home/timeleafing/projects/mask2former-cutout/ai-core/weights/mask2former-cutout-coco-voc-v1/best_model \
uv run --project backend uvicorn main:app \
    --app-dir backend \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

生产启动示例：

```bash
cd /home/timeleafing/projects/mask2former-cutout
uv run --project backend gunicorn main:app \
    --chdir backend \
    -k uvicorn.workers.UvicornWorker \
    -w 1 \
    --bind 0.0.0.0:8000
```

建议后续核心接口：

```text
GET  /api/health
GET  /api/models
POST /api/segment
GET  /api/results/{job_id}
```

`POST /api/segment` 推荐返回：

```json
{
    "job_id": "string",
    "status": "success",
    "classes": ["person", "car"],
    "model_id": "default",
    "model_label": "default",
    "files": {
        "original_url": "/static/outputs/{job_id}/original.png",
        "cutout_url": "/static/outputs/{job_id}/cutout.png",
        "mask_url": "/static/outputs/{job_id}/mask_combined.png",
        "overlay_url": "/static/outputs/{job_id}/overlay.png"
    },
    "timing": {
        "total_ms": 120,
        "inference_ms": 80
    }
}
```

## 前端应用

前端目录：

```text
frontend/
```

技术栈：

- Vue 3
- TypeScript
- Vite
- Naive UI
- Axios
- Tailwind CSS

开发启动：

```bash
cd /home/timeleafing/projects/mask2former-cutout/frontend
npm run dev
```

构建：

```bash
cd frontend
npm run build
```

预览构建产物：

```bash
cd frontend
npm run preview
```

当前前端已有模块：

```text
frontend/src/components/
    HistoryList.vue
    ImageUploader.vue
    MaskCanvas.vue
    ResultPanel.vue
    ResultViewer.vue

frontend/src/composables/
    useCutoutApi.ts
```

建议完善方向：

- 上传前图片预览。
- 推理中 loading 状态。
- 原图 / mask / overlay / cutout 多视图切换。
- person / car / both 类别选择。
- 下载透明 PNG。
- 最近推理历史。

## 数据与权重管理

`.gitignore` 当前忽略：

```text
ai-core/data/
ai-core/weights/
*.pt
*.pth
*.bin
*.safetensors
```

原因：

- 数据目录约 GB 级。
- checkpoint 与模型权重为大二进制文件。
- 普通 Git 不适合长期管理这些文件。

推荐方案：

- 使用 Git LFS 管理模型权重。
- 使用对象存储管理训练数据。
- 生产部署只同步 `best_model/`。

## 文档

更详细的项目完善计划见：

```text
.docs/project_build_guide.md
```

该文档覆盖：

- 模型推理层设计。
- 后端 API 设计。
- 前端页面设计。
- 后处理策略。
- 性能建议。
- 部署方案。
- 测试清单。
- 实施里程碑。

## 推荐开发顺序

1. 完成后端模型加载单例。
2. 完成单张图片推理函数。
3. 完成 mask、overlay、cutout 输出。
4. 完成 `/api/segment`。
5. 前端接入上传和结果展示。
6. 增加类别选择和下载能力。
7. 增加请求队列和结果清理。
8. 完成生产部署。

## 验收标准

最小可用版本应满足：

- 后端启动后模型只加载一次。
- 上传一张图片后能成功返回透明 PNG。
- 前端能展示原图、mask、overlay 和 cutout。
- RTX 4060 8GB 上单图推理不 OOM。
- 连续推理多张图片时显存稳定。

## 许可证

见根目录 `LICENSE`。
