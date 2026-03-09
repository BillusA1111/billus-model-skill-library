# Billus 大模型技能库

面向真实模型工程工作的 Codex 技能仓库，重点服务这些场景：

- LLM 训练、调参、扩容、蒸馏、剪枝、量化
- VL 多模态视觉模型训练与结构修改
- 图像生成、图像编辑、inpainting、ControlNet、IP-Adapter
- Hugging Face Trainer / Accelerate / DeepSpeed / FSDP / PEFT / diffusers

当前仓库提供的核心技能：

- `skills/billus-model-ops`

## 技能定位

`billus-model-ops` 不是泛泛的“模型知识摘要”，而是面向仓库落地的模型工程技能。它的目标是让 Codex 在进入训练仓库后，先识别真实框架路径，再做最小变更、最小验证、最清晰的风险交付。

它特别适合下面这些请求：

- 帮我修改训练参数，但不要破坏现有 launch 和 resume 流程
- 帮我定位这个仓库到底是 Trainer、Accelerate、DeepSpeed 还是 FSDP
- 帮我做 QLoRA / LoRA / PEFT 参数调整
- 帮我处理 LLaVA / Qwen-VL / InternVL 这类视觉塔和 projector 改动
- 帮我检查 diffusers 图像生成或图像编辑链路里的 scheduler / VAE / ControlNet 风险
- 帮我写实验计划，并总结训练日志下一步怎么调

## 覆盖框架

### 文本与通用训练栈

- PyTorch
- Hugging Face Trainer
- Accelerate
- DeepSpeed
- FSDP

### 参数高效与低比特

- PEFT
- LoRA
- QLoRA
- bitsandbytes

### 视觉语言模型

- LLaVA 类
- Qwen-VL 类
- InternVL 类

### 图像生成与图像编辑

- diffusers
- text-to-image
- image-to-image
- inpainting
- instruction-based image editing
- ControlNet
- IP-Adapter

## 仓库结构

```text
skills/
  billus-model-ops/
    SKILL.md
    agents/
      openai.yaml
    references/
      frameworks-hf-accelerate-deepspeed.md
      frameworks-peft-and-lowbit.md
      frameworks-vl-stacks.md
      frameworks-diffusers-image.md
      validation-and-release.md
    scripts/
      detect_training_stack.py
      summarize_training_log.py
      new_experiment_note.py
docs/
  billus-model-ops-paper.md
```

## 使用方式

### 1. 在 Codex 中显式调用

```text
用 $billus-model-ops 帮我识别这个训练仓库的实际框架入口，并安全修改 batch size 和 learning rate
```

```text
用 $billus-model-ops 帮我检查这个 VL 项目的 projector 改动是否会破坏 image token 对齐
```

```text
用 $billus-model-ops 帮我评估这个 diffusers 图像编辑链路里 scheduler 和 VAE 的修改风险
```

### 2. 用辅助脚本做快速分析

检测训练栈：

```bash
python skills/billus-model-ops/scripts/detect_training_stack.py <repo-root>
```

生成实验计划：

```bash
python skills/billus-model-ops/scripts/new_experiment_note.py \
  --title "QLoRA on VL projector" \
  --goal "Reduce memory while preserving multimodal accuracy" \
  --baseline "vl_full_ft_baseline" \
  --change "4-bit base model" \
  --change "LoRA on projector and attention qkv" \
  --metric "multimodal eval drop <= 1%"
```

总结训练日志：

```bash
python skills/billus-model-ops/scripts/summarize_training_log.py <train.log>
```

## 设计原则

- 先识别真实框架，再改代码
- 先收敛变更范围，再谈优化
- 先做最小可验证路径，再给结论
- 对 LLM、VL、图像生成分别保留不同的回归检查重点
- 输出必须包含风险、验证范围和剩余未验证项

## 说明文档

更完整的设计说明与应用说明见：

- [docs/billus-model-ops-paper.md](D:/CODEX/billus-model-skill-library/docs/billus-model-ops-paper.md)
