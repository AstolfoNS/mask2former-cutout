  # 《模型核查与训练压榨报告》

  依据 Hugging Face 官方文档，Mask2Former 是统一的 panoptic / instance / semantic segmentation 架构，适合你的双类别实例/语义分割任务；
  Mask2FormerImageProcessor/AutoImageProcessor 是官方推荐的数据预处理入口，Mask2FormerForUniversalSegmentation 可用于 instance/semantic 后处理。参考：
  Hugging Face Mask2Former 文档、Trainer 文档、单卡性能优化文档。
  来源：

  - https://huggingface.co/docs/transformers/v4.47.0/model_doc/mask2former
  - https://huggingface.co/docs/transformers/main_classes/trainer
  - https://huggingface.co/docs/transformers/v4.37.2/en/perf_train_gpu_one

  说明：本文是训练方案核查记录，部分时间估算来自历史 7755 张训练集。
  当前 `ai-core/data/cutout_mix_512/annotations.json` 为 5200 张图片、
  22710 条标注；实际训练步数应按当前数据集重新估算。启动命令以
  `uv run python` 为准。

  ## 1. 架构与任务适配性诊断

  结论：Mask2Former Swin-Small 用于 512x512、person/car 双类别 COCO 实例分割是合理且偏强的基线。

  当前处理后数据集约 5200 张图片；历史训练集曾为 7755 张。类别只有 2 类，且图像已统一到 512x512，这对 Swin-Small + Mask2Former 来说属于“中等数据量、低类别复杂度、高边界质量要求”的场
  景。Mask2Former 的 query-based mask decoder 对实例边界、重叠目标、多人多车场景会比普通 semantic-only 模型更合适。

  加载阶段必须注意：

  from transformers import Mask2FormerForUniversalSegmentation

  id2label = {
      0: "person",
      1: "car",
  }
  label2id = {v: k for k, v in id2label.items()}

  model = Mask2FormerForUniversalSegmentation.from_pretrained(
      "facebook/mask2former-swin-small-coco-instance",
      num_labels=2,
      id2label=id2label,
      label2id=label2id,
      ignore_mismatched_sizes=True,
  )

  关键点：

  - num_labels=2：你的训练标签必须是连续类别索引 0, 1，不要直接把 COCO category_id 的 1/person、3/car 原样喂给模型。
  - id2label / label2id：显式绑定类别，避免评估和推理后处理时标签错位。
  - ignore_mismatched_sizes=True：COCO 预训练头是 80 类，你改成 2 类后分类头尺寸不一致，必须允许重建分类头。官方 from_pretrained 文档中该参数就是用于
    处理 checkpoint 与目标类别数不一致的场景。

  - 分类头会重新初始化：这不是 bug。骨干网络、像素解码器、mask decoder 的大部分权重会保留，只有类别预测相关层因维度变化被重建。

  关于“类别灾难性遗忘”：
  如果你直接 num_labels=2 + ignore_mismatched_sizes=True，person/car 的 COCO 分类头语义不会被完整继承，但视觉特征和 mask 生成能力会继承。对于当前 5200 张
  高质量数据，这通常足够。如果你追求极致冷启动，可以手动从原 80 类分类头迁移 person 和 car 两行权重到新 2 类头；但这需要确认当前 Transformers 版本中分
  类层命名，复杂度略高，不是首轮训练的必要条件。

  ## 2. 显存压榨推演

  硬件：RTX 4090 24GB，512x512，bf16，关闭 gradient checkpointing。

  我的建议：

   方案          per_device_train_batch_size    gradient_accumulation_steps    有效 batch    判断                                                     
  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   稳健首发                                8                              2            16    最推荐，速度/显存/稳定性均衡
  ────────────  ─────────────────────────────  ─────────────────────────────  ────────────  ──────────────────────────────
   压榨方案                               10                              2            20    可能可跑，需监控峰值显存
  ────────────  ─────────────────────────────  ─────────────────────────────  ────────────  ──────────────────────────────
   极限尝试                               12                              2            24    有机会 OOM，不建议作为首发
  ────────────  ─────────────────────────────  ─────────────────────────────  ────────────  ──────────────────────────────
   保守高稳定                              6                         3 或 4         18/24    数据增强或评估占显存时使用

  结论：首发建议 batch size = 8，gradient_accumulation_steps = 2。

  不建议一开始直接冲 12。Mask2Former 的显存不只由输入分辨率决定，还受以下因素影响：

  - 每张图的实例数量；
  - num_queries=100；
  - Hungarian matching；
  - auxiliary losses；
  - point sampling loss；
  - 数据 collator 是否保留了过多 CPU/GPU tensor；
  - eval 阶段是否累计大 mask 输出。

  如果 batch 8 的显存峰值低于 19GB，可以再试 batch 10。batch 12 更适合在确认数据 pipeline 很轻、实例数量不多、eval 不同时跑大 batch 的情况下测试。

  ## 3. 训练时间沙盘推演

  以 batch size 8 为例：

  - 当前数据量：5200 张；
  - micro-batch steps / epoch：约 ceil(5200 / 8) = 650；
  - accumulation=2 后 optimizer steps / epoch：约 325。

  4090 + bf16 + 512x512 的合理吞吐预估：

  - 若数据加载和 mask 构建高效：约 1.8-3.0 micro-it/s；
  - 若 COCO mask 解码、augmentation、collator 较重：约 1.0-1.8 micro-it/s；
  - 折算 images/s：大约 8-24 images/s。

  单 epoch 训练时间粗估：

  - 理想：6-10 分钟；
  - 正常：10-18 分钟；
  - 数据管线偏慢：18-25 分钟。

  最大 40 epoch：

  - 纯训练理论值：约 4-10 小时；
  - 加上 eval、checkpoint、metric、early stopping：约 5-12 小时。

  如果 Early Stopping 设置合理，例如 patience=5，且数据质量较高，我预计真实收敛大概率在 18-30 epoch，今晚实际耗时更可能是 4-8 小时，而不是完整跑满 40
  epoch。

  ## 4. 性能优化终极蓝图

  推荐配置如下：

  import torch
  from transformers import TrainingArguments

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.set_float32_matmul_precision("high")

  training_args = TrainingArguments(
      output_dir="./outputs/mask2former-cutout-swin-small",

      num_train_epochs=40,
      per_device_train_batch_size=8,
      per_device_eval_batch_size=4,
      gradient_accumulation_steps=2,

      learning_rate=5e-5,
      weight_decay=0.05,
      max_grad_norm=1.0,
      warmup_ratio=0.05,
      lr_scheduler_type="cosine",

      bf16=True,
      fp16=False,
      tf32=True,
      gradient_checkpointing=False,

      dataloader_num_workers=8,
      dataloader_pin_memory=True,
      dataloader_persistent_workers=True,
      dataloader_prefetch_factor=4,

      remove_unused_columns=False,

      eval_strategy="epoch",
      save_strategy="epoch",
      logging_strategy="steps",
      logging_steps=50,

      load_best_model_at_end=True,
      metric_for_best_model="eval_mean_iou",
      greater_is_better=True,
      save_total_limit=3,

      optim="adamw_torch_fused",

      report_to="none",
      seed=42,
      data_seed=42,
  )

  补充建议：

  - remove_unused_columns=False 对 segmentation 任务非常重要，否则 Trainer 可能错误丢弃 mask_labels、class_labels 等字段。
  - per_device_eval_batch_size 建议先设为 4，不要和 train batch 一样激进；后处理和指标计算可能比训练更吃内存。
  - bf16=True 优于 fp16=True，4090 支持 bf16，数值稳定性更好。
  - tf32=True 对残留 FP32 matmul/convolution 有帮助，官方 Trainer 支持该参数。
  - 首轮不要开 torch_compile=True。Mask2Former + 动态 target/mask 结构可能带来 compile 开销或不稳定，等 baseline 跑通后再单独测试。

  最终发车参数：batch_size=8，grad_acc=2，bf16=True，tf32=True，workers=8，eval batch=4。这组配置最适合作为第一晚的高性能稳定基线。
