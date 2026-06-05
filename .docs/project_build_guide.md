# Mask2Former-Cutout 项目完善搭建文档

## 1. 目标

本文档用于指导 `Mask2Former-Cutout` 从已完成训练的模型，完善为一个可本地运行、可部署、可维护的完整图像分割抠图系统。

系统最终应包含三层：

```text
frontend/
    Vue 3 + TypeScript 用户界面

backend/
    FastAPI 推理服务、文件管理、结果返回

ai-core/
    Mask2Former 模型权重、推理逻辑、图像后处理
```

核心能力：

- 上传图片。
- 调用已训练好的 `best_model` 做 person/car 分割。
- 返回透明 PNG 抠图、mask 图、overlay 预览图。
- 支持 person/car/both 类别选择。
- 支持结果下载。
- 支持后续部署到 RTX 4090 或 RTX 4060 8GB 推理机器。

## 2. 当前项目状态

当前已有：

```text
ai-core/
    src/train.py
    scripts/train.sh
    weights/mask2former-cutout/best_model/
    weights/mask2former-cutout/training_metrics.csv

backend/
    main.py
    app/

frontend/
    src/
    package.json
```

模型侧已经具备：

- 已完成训练。
- 最佳权重位于 `ai-core/weights/mask2former-cutout/best_model/`。
- 模型类别为 `person` 和 `car`。
- 训练输入尺寸为 `512x512`。
- 当前验证最佳 mIoU 约为 `0.9795`。

## 3. 推荐最终目录结构

建议保持前端、后端、模型代码隔离：

```text
mask2former-cutout/
    ai-core/
        src/
            train.py
            inference.py
            postprocess.py
        weights/
            mask2former-cutout/
                best_model/
        docs/

    backend/
        app/
            api/
                __init__.py
                health.py
                segment.py
                files.py
            core/
                __init__.py
                config.py
                model_runtime.py
                image_ops.py
                storage.py
            schemas/
                __init__.py
                segment.py
            static/
                uploads/
                outputs/
        main.py
        pyproject.toml

    frontend/
        src/
            components/
                ImageUploader.vue
                ResultViewer.vue
                InferencePanel.vue
                ClassToggle.vue
                HistoryList.vue
            composables/
                useSegmentation.ts
            types/
                segmentation.ts
            App.vue
            main.ts
        package.json

    docs/
        project_build_guide.md
```

## 4. 模型推理层设计

### 4.1 只加载 `best_model`

部署和推理只应使用：

```text
ai-core/weights/mask2former-cutout/best_model/
```

不要加载：

```text
checkpoint-3178/
checkpoint-4994/
checkpoint-5448/
```

原因：

- `best_model/` 是推理所需的最小模型目录。
- `checkpoint-*` 包含 `optimizer.pt`、`scheduler.pt`、`rng_state.pth`，只用于恢复训练。
- 每个 checkpoint 约 788MB，部署时冗余。

### 4.2 推理输入规范

模型训练时使用 512x512 数据，因此推理侧应统一为：

- 输入图片读取为 RGB。
- 保留原始尺寸用于结果映射。
- 推理前 resize 到 512x512，或按训练预处理逻辑处理。
- 输出 mask 后再 resize 回原图尺寸。

建议接口参数：

```text
score_threshold: float = 0.5
target_classes: person | car | both = both
return_mask: bool = true
return_overlay: bool = true
return_cutout: bool = true
```

### 4.3 推理显存策略

RTX 4090 和 RTX 4060 8GB 都可推理，但部署默认应保守：

```python
model.eval()
model.to("cuda")

with torch.inference_mode():
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(**inputs)
```

策略：

- `batch_size=1`。
- 后端请求队列化，避免多请求同时抢占 GPU。
- 不启用梯度。
- 不加载 optimizer。
- 启动时模型常驻 GPU。

### 4.4 输出结果

每次推理建议输出：

```text
outputs/{job_id}/
    original.png
    mask_person.png
    mask_car.png
    mask_combined.png
    overlay.png
    cutout.png
    result.json
```

`result.json` 示例：

```json
{
    "job_id": "20260605_120000_abcd",
    "classes": ["person", "car"],
    "image": {
        "width": 1280,
        "height": 720
    },
    "timing": {
        "preprocess_ms": 12,
        "inference_ms": 85,
        "postprocess_ms": 18,
        "total_ms": 115
    },
    "files": {
        "cutout": "/static/outputs/.../cutout.png",
        "mask": "/static/outputs/.../mask_combined.png",
        "overlay": "/static/outputs/.../overlay.png"
    }
}
```

## 5. 后端设计

### 5.1 技术栈

推荐：

- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Pillow
- OpenCV
- Pydantic

### 5.2 后端模块职责

```text
backend/app/core/config.py
    读取环境变量、路径配置、推理参数。

backend/app/core/model_runtime.py
    模型加载、单例管理、GPU 推理。

backend/app/core/image_ops.py
    图片读取、resize、mask 后处理、透明 PNG 生成。

backend/app/core/storage.py
    上传文件和输出文件管理。

backend/app/api/segment.py
    POST /api/segment 推理接口。

backend/app/api/health.py
    GET /api/health 健康检查。

backend/app/api/files.py
    静态文件或结果文件访问辅助接口。
```

### 5.3 后端配置

建议使用环境变量：

```text
MODEL_DIR=/root/autodl-tmp/projects/mask2former-cutout/ai-core/weights/mask2former-cutout/best_model
DEVICE=cuda
INFERENCE_DTYPE=float16
INPUT_SIZE=512
SCORE_THRESHOLD=0.5
MAX_UPLOAD_MB=20
UPLOAD_DIR=backend/app/static/uploads
OUTPUT_DIR=backend/app/static/outputs
```

### 5.4 API 设计

#### 健康检查

```text
GET /api/health
```

响应：

```json
{
    "status": "ok",
    "device": "cuda",
    "model_loaded": true
}
```

#### 图像分割

```text
POST /api/segment
```

请求类型：

```text
multipart/form-data
```

字段：

```text
file: image
target_classes: both
score_threshold: 0.5
return_overlay: true
return_mask: true
return_cutout: true
```

响应：

```json
{
    "job_id": "string",
    "status": "success",
    "classes": ["person", "car"],
    "files": {
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

#### 查询结果

```text
GET /api/results/{job_id}
```

用于前端刷新或历史记录。

### 5.5 后端异常处理

需要明确处理：

- 非图片文件。
- 图片过大。
- 图片损坏。
- GPU 不可用。
- 模型加载失败。
- 推理超时。
- 输出文件写入失败。

错误响应统一：

```json
{
    "status": "error",
    "code": "INVALID_IMAGE",
    "message": "Uploaded file is not a valid image."
}
```

## 6. 前端设计

### 6.1 技术栈

当前项目适合继续使用：

- Vue 3
- TypeScript
- Vite
- Naive UI
- Axios

### 6.2 页面布局

首屏就是工具界面，不做营销页。

推荐布局：

```text
┌───────────────────────────────────────────────┐
│ 顶部栏：项目名 / 后端状态 / GPU 状态           │
├───────────────┬───────────────────────────────┤
│ 上传与参数面板 │ 结果预览区                    │
│               │ 原图 / mask / overlay / cutout │
├───────────────┴───────────────────────────────┤
│ 历史记录 / 下载按钮                            │
└───────────────────────────────────────────────┘
```

### 6.3 核心组件

```text
ImageUploader.vue
    拖拽上传、文件校验、图片预览。

InferencePanel.vue
    score threshold、类别选择、开始推理按钮。

ClassToggle.vue
    person/car/both 类别切换。

ResultViewer.vue
    原图、mask、overlay、cutout 多视图切换。

HistoryList.vue
    最近推理结果列表。
```

### 6.4 前端状态

建议定义：

```ts
export interface SegmentResult {
    job_id: string
    status: 'success' | 'error'
    classes: string[]
    files: {
        cutout_url?: string
        mask_url?: string
        overlay_url?: string
    }
    timing?: {
        total_ms: number
        inference_ms: number
    }
}
```

`useSegmentation.ts` 负责：

- 上传文件。
- 调用 `/api/segment`。
- 管理 loading/error/result。
- 暴露下载 URL。

### 6.5 前端交互细节

必须支持：

- 上传前预览。
- 推理中禁用按钮。
- 失败时显示错误原因。
- 成功后展示下载按钮。
- 支持重新上传。
- 支持结果视图切换。

建议视图：

```text
Original
Mask
Overlay
Cutout
```

## 7. 图像后处理策略

### 7.1 mask 合成

模型输出 person/car 两类 mask。

当 `target_classes=both`：

```text
combined_mask = person_mask OR car_mask
```

当 `target_classes=person`：

```text
combined_mask = person_mask
```

当 `target_classes=car`：

```text
combined_mask = car_mask
```

### 7.2 透明 PNG

透明 PNG 生成逻辑：

```text
RGBA = original RGB + alpha
alpha = combined_mask * 255
```

建议可选优化：

- 小区域过滤。
- mask 边缘轻微 blur。
- 形态学 close。
- alpha feather。

默认参数：

```text
min_area_ratio = 0.001
morph_kernel = 3
blur_kernel = 3
```

## 8. 性能建议

### 8.1 RTX 4090

推荐：

- `float16` 或 `bfloat16`
- 单请求 batch size 1
- 可接受少量并发，但应限制 GPU 同时推理数量

### 8.2 RTX 4060 8GB

推荐：

- 只加载 `best_model/`
- `batch_size=1`
- `float16`
- 后端请求串行化
- 禁止同时跑训练任务

预期显存：

```text
约 2GB - 4GB
```

## 9. 部署方案

### 9.1 本地开发

后端：

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

前端：

```bash
cd frontend
npm run dev
```

### 9.2 生产部署

推荐：

```text
Nginx
    /              -> frontend/dist
    /api/*         -> FastAPI backend
    /static/*      -> backend static outputs

FastAPI
    Uvicorn/Gunicorn
    单 GPU 推理服务
```

### 9.3 Docker 化

建议后续增加：

```text
backend/Dockerfile
frontend/Dockerfile
docker-compose.yml
```

GPU 容器需要：

```text
nvidia-container-toolkit
CUDA runtime
PyTorch CUDA wheel
```

## 10. 测试清单

### 10.1 模型推理测试

- 本地加载 `best_model/` 成功。
- 单张 512x512 图片推理成功。
- 大图 resize 后推理成功。
- person 图片输出正确。
- car 图片输出正确。
- 空背景或非目标图像不崩溃。

### 10.2 后端测试

- `GET /api/health` 正常。
- `POST /api/segment` 支持 jpg/png/webp。
- 错误文件返回明确错误。
- 输出文件 URL 可访问。
- 多次请求不会重复加载模型。
- GPU 显存不会持续增长。

### 10.3 前端测试

- 上传前预览正常。
- 推理 loading 状态正常。
- mask/overlay/cutout 展示正常。
- 下载按钮正常。
- 后端错误能展示。
- 移动端布局不重叠。

## 11. 实施里程碑

### Milestone 1: 后端最小推理闭环

- 完成模型加载单例。
- 完成单张图片推理函数。
- 完成 mask 和 cutout 输出。
- 完成 `/api/segment`。

验收标准：

```text
curl 上传一张图片后，可以得到 cutout.png。
```

### Milestone 2: 前端上传与展示

- 完成上传组件。
- 完成参数面板。
- 完成结果展示。
- 完成下载按钮。

验收标准：

```text
浏览器上传图片后，可以看到 overlay 和透明抠图结果。
```

### Milestone 3: 后处理与稳定性

- 增加小区域过滤。
- 增加 alpha 平滑。
- 增加错误处理。
- 增加结果清理任务。

验收标准：

```text
连续推理 50 张图片，服务不崩溃，显存稳定。
```

### Milestone 4: 部署

- 前端 build。
- 后端生产启动。
- Nginx 反向代理。
- GPU 机器部署文档。

验收标准：

```text
局域网或公网访问页面，可完成上传、推理、下载。
```

## 12. 优先级建议

建议按以下顺序推进：

1. 后端模型推理闭环。
2. 后端 API 与文件输出。
3. 前端上传和结果展示。
4. 前端下载和历史记录。
5. 后处理质量优化。
6. 性能与部署。

最关键的第一步是后端推理 API。只要 `/api/segment` 稳定，前端和部署都可以围绕它快速完善。
