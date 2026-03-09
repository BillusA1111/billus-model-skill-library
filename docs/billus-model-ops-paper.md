# Billus Model Ops 技能说明

## 1. 背景

模型工程类仓库通常不是单纯的“改一个参数”。一次看似简单的改动，往往同时影响：

- launch 命令
- config 合并逻辑
- 分布式包装
- checkpoint 加载
- tokenizer 或 image processor
- eval 脚本
- 导出或推理链路

传统的通用型提示经常只能给出局部建议，难以保证“改动真的落在仓库真实路径上”。`billus-model-ops` 的设计目标，就是把 Codex 的工作方式从“泛化回答”收束成“仓库实证驱动”的模型工程工作流。

## 2. 目标

这个技能的目标不是替代训练框架文档，而是让 Codex 在处理模型工程任务时遵循一套稳定流程：

1. 识别训练或推理仓库的真实框架栈
2. 判断这次请求属于哪类模型工程变更
3. 锁定最小变更面
4. 执行最小但有效的验证
5. 给出可复现、可回滚、可交接的结果说明

## 3. 适用范围

### 3.1 文本模型

- continued pretraining
- SFT
- DPO 类偏好训练
- 扩容、蒸馏、剪枝、量化

### 3.2 多模态视觉语言模型

- 视觉塔替换
- projector / resampler / connector 修改
- 图像 token 对齐
- 多模态对话模板和 masking

### 3.3 图像生成与图像编辑模型

- text-to-image
- image-to-image
- inpainting
- instruction editing
- ControlNet
- IP-Adapter
- scheduler / VAE / denoiser 修改

## 4. 支持框架

- PyTorch
- Hugging Face Trainer
- Accelerate
- DeepSpeed
- FSDP
- PEFT
- diffusers

## 5. 技能内容组成

### 5.1 SKILL.md

定义技能触发范围、主工作流、参考文档入口、以及脚本用途。

### 5.2 references

按常见工程栈分拆，避免一次性加载无关内容：

- `frameworks-hf-accelerate-deepspeed.md`
- `frameworks-peft-and-lowbit.md`
- `frameworks-vl-stacks.md`
- `frameworks-diffusers-image.md`
- `validation-and-release.md`

### 5.3 scripts

- `detect_training_stack.py`
  用于快速识别仓库的框架栈和候选入口。
- `summarize_training_log.py`
  用于把训练日志整理成结构化摘要。
- `new_experiment_note.py`
  用于生成标准化实验计划。

## 6. 工作流设计

### 第一步：识别真实栈

技能优先判断仓库到底使用：

- Trainer
- Accelerate 自定义 loop
- DeepSpeed
- FSDP
- PEFT / QLoRA
- VL 视觉塔路径
- diffusers 图像链路

这一步的目的，是避免只改到“看起来像入口”的文件。

### 第二步：归类任务

请求会被归到以下类型之一：

- 训练调参
- 结构扩容或 checkpoint 迁移
- PEFT / 低比特改动
- VL 结构或数据格式改动
- 图像生成 / 图像编辑链路改动
- 回归验证与实验分析

### 第三步：限制变更面

技能强调只动当前任务必须动的表面，不把多个互相耦合的调整混在一次提交里。

### 第四步：最小验证

验证并不追求昂贵，而追求有效。比如：

- config 是否真的生效
- 模型是否能 instantiate
- checkpoint 是否还能 load
- 一步 forward 是否能通
- 固定 seed 的图像输出是否可比较
- VL 的 image token 对齐是否还成立

### 第五步：可交付结果

最后输出不只是“改了什么”，还包括：

- 为什么这样改
- 本地验证了什么
- 哪些部分还没被完整训练验证
- 主要风险点
- 快速回滚方式

## 7. 应用方式

### 7.1 直接让 Codex 调用技能

示例：

```text
用 $billus-model-ops 帮我分析这个 Hugging Face + DeepSpeed 仓库的训练入口，并安全调低显存占用
```

```text
用 $billus-model-ops 帮我检查这个 LLaVA 项目里 vision tower 和 projector 的维度兼容性
```

```text
用 $billus-model-ops 帮我验证这个 diffusers 图像编辑模型更换 scheduler 之后的风险
```

### 7.2 单独使用脚本

适合做训练仓库快速体检、实验记录和日志摘要。

## 8. 适合的人群

- 大模型训练工程师
- 多模态模型工程师
- 图像生成模型工程师
- 负责训练基础设施、模型迭代和回归验证的算法工程师

## 9. 价值

这个技能的核心价值不是“知道很多名词”，而是帮助 Codex 在复杂模型仓库里：

- 更快找到真实改动入口
- 更少做错误假设
- 更稳地完成小步快跑式改动
- 更清楚地交付验证范围和风险边界
