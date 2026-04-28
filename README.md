# agent_router_sft

**智能体路由模型 SFT 训练数据自动化生成工程**

本工程为 100 个垂直领域智能体构建高质量路由模型微调（SFT）数据集，目标规模约 **10 万条**，支持三阶段分层生成策略、多维度数据清洗、困难负样本定向构造，以及多格式导出。

---

## 目录

- [工程概述](#工程概述)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [三阶段数据规划](#三阶段数据规划)
- [模块说明](#模块说明)
- [配置参数详解](#配置参数详解)
- [命令行用法](#命令行用法)
- [数据格式说明](#数据格式说明)
- [算力与规模估算](#算力与规模估算)
- [常见问题](#常见问题)

---

## 工程概述

### 背景

当系统中部署了 100 个专业垂直领域智能体时，需要一个**路由模型**将用户的自然语言查询精准分发到对应的智能体。训练这样的路由模型，需要大规模、高质量、覆盖多种表达方式的训练数据。

### 目标数据规模

| 阶段 | 类型 | 规模 | 说明 |
|------|------|------|------|
| 阶段一 | 冷启动样本 | ~15,000 条 | 每智能体 150 条简单核心查询 |
| 阶段二 | 生产泛化样本 | ~70,000 条 | 每智能体 700 条混合复杂度样本 |
| 阶段三 | 困难负样本 | ~15,000 条 | 针对混淆智能体对定向生成 |
| **合计** | **全量数据集** | **~100,000 条** | 经清洗、去重、均衡后 |

### 数据样本格式

每条训练样本为一个 JSON 对象，包含两个字段：

```json
{
  "question": "我们公司的货轮在南海遭遇了12级台风，需要紧急调整航线，请协助规划备用路径",
  "answer": "民航绕偏航处置"
}
```

---

## 目录结构

```
agent_router_sft/
├── run.py                          # 命令行入口（主程序）
├── pipeline.py                     # 全流程 Pipeline 主控
├── requirements.txt                # Python 依赖
├── README.md                       # 本文档
│
├── config/                         # 配置模块
│   ├── config.yaml                 # 主配置文件（所有参数均在此调整）
│   └── config_loader.py            # 配置加载与验证
│
├── agents/                         # 智能体元数据模块
│   └── agent_registry.py           # 100 个智能体的完整定义
│                                   # （名称、领域、描述、关键词、混淆列表）
│
├── generators/                     # 数据生成模块
│   ├── prompt_builder.py           # Prompt 模板构造器
│   ├── llm_client.py               # LLM API 调用封装（支持 OpenAI 兼容接口）
│   ├── stage_generator.py          # 三阶段批量生成器（断点续传）
│   └── hard_negative_builder.py    # 困难负样本专项构造器
│
├── cleaners/                       # 数据清洗模块
│   └── data_cleaner.py             # 三层过滤流水线（硬过滤/质量评分/去重）
│
├── utils/                          # 工具模块
│   └── dataset_builder.py          # 数据集合并、统计分析、多格式导出、划分
│
└── output/                         # 输出目录（运行后自动创建）
    ├── raw/                        # LLM 原始生成数据
    │   ├── stage1/                 # 阶段一原始数据（每智能体一个 .jsonl 文件）
    │   ├── stage2/                 # 阶段二原始数据
    │   └── stage3/                 # 阶段三困难负样本
    ├── clean/                      # 清洗后数据
    │   ├── stage1/
    │   ├── stage2/
    │   └── stage3/
    ├── merged/
    │   └── full_dataset.jsonl      # 合并后完整数据集
    ├── split/                      # 划分后数据集
    │   ├── train/                  # 训练集（多格式）
    │   ├── val/                    # 验证集
    │   └── test/                   # 测试集
    ├── export/                     # 多格式导出
    │   ├── agent_router_sft.jsonl
    │   ├── agent_router_sft_alpaca.json
    │   ├── agent_router_sft_sharegpt.json
    │   └── agent_router_sft_openai.jsonl
    ├── reports/                    # 统计报告
    │   ├── dataset_report.json
    │   └── confusable_pairs.json
    └── logs/
        └── pipeline.log
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
export OPENAI_API_KEY="your-api-key"
# 如使用本地部署的模型（vLLM、Ollama 等）：
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

### 3. 修改配置（可选）

编辑 `config/config.yaml`，调整模型名称、样本数量、输出目录等参数。

### 4. 运行完整流程

```bash
python run.py
```

### 5. 测试流程（不调用 LLM）

```bash
python run.py --dry-run
```

---

## 三阶段数据规划

### 阶段一：冷启动（Cold Start）

**目标：** 每个智能体生成 100~200 条简单、直接的核心查询样本，快速建立基础覆盖。

**样本特征：**
- 问题表达简洁直接，意图明确
- 包含智能体的核心关键词
- 长度 15~50 字

**示例：**
```
"飞机偏航了怎么处理？" → 民航绕偏航处置
"帮我审核这份飞行计划" → 飞行计划审核
"查一下这个患者的历史病历" → 医疗助手
```

### 阶段二：生产泛化（Production Generalization）

**目标：** 每个智能体生成 500~1000 条混合复杂度样本，覆盖真实用户的多样化表达。

**样本特征（40% 简单 + 60% 复杂）：**

| 类型 | 占比 | 特征 |
|------|------|------|
| 口语化表达 | 15% | 非正式语言、缩写、方言词汇 |
| 长文本描述 | 20% | 包含背景信息的详细描述（50~200字） |
| 多轮对话截断 | 15% | 模拟从对话中间截取的上下文片段 |
| 隐含意图 | 10% | 不直接说明需求，需推理才能判断 |
| 简单直接 | 40% | 清晰表达，意图明确 |

### 阶段三：困难负样本（Hard Negatives）

**目标：** 针对功能相近的智能体对，定向生成区分性样本，强制模型学习细微语义差异。

**混淆对发现策略：**
1. **Registry 标注**：在 `agent_registry.py` 中为每个智能体手动标注 `confusable` 列表
2. **关键词重叠分析**：自动计算智能体间的关键词 Jaccard 相似度
3. **同领域相似**：同一业务领域内的智能体天然存在混淆风险

**困难负样本生成策略（5 种）：**

| 策略 | 说明 | 示例 |
|------|------|------|
| 关键词替换型 | 在目标场景中混入混淆智能体的关键词 | 问题含"飞行计划"词汇，但核心需求是偏航处置 |
| 场景嫁接型 | 以混淆智能体的场景为背景，核心需求指向目标 | "在审核飞行计划时发现飞机已偏航，需要..." |
| 反向确认型 | 先描述混淆场景，再转折到目标需求 | "虽然计划已提交，但实际飞行中出现了偏航..." |
| 多需求混合型 | 同时涉及两个智能体，主需求指向目标 | 问题同时提到两个领域，但主要需求明确 |
| 隐含意图型 | 描述现象而非需求，需推理才能判断 | "航班突然改变了预定航路，机组在联系地面..." |

---

## 模块说明

### `agents/agent_registry.py`

定义 100 个智能体的完整元数据，每个智能体包含：

```python
{
    "name": "民航绕偏航处置",          # 唯一标识符（也是训练标签）
    "domain": "交通运输",              # 所属领域
    "description": "...",              # 功能描述（用于 Prompt 构造）
    "keywords": ["飞机偏航", "备降"],   # 核心关键词（用于相关性评分）
    "confusable": ["飞行计划审核"],     # 易混淆的智能体列表
    "typical_intents": ["..."],        # 典型用户意图（用于 Prompt 多样化）
}
```

### `generators/prompt_builder.py`

负责将智能体元数据转换为 LLM 可理解的生成指令。提供以下 Prompt 类型：

- `build_stage1_prompt`：单条简单样本生成
- `build_batch_prompt`：批量样本生成（可指定数量和风格）
- `build_stage2_complex_prompt`：复杂样本生成（含多样性指令）
- `build_hard_negative_prompt`：困难负样本生成
- `get_diversity_instructions`：多样性指令轮换列表

### `generators/llm_client.py`

封装 OpenAI 兼容接口调用，提供：

- 自动重试（指数退避）
- JSON 响应解析（鲁棒处理 Markdown 代码块等格式）
- Token 消耗统计
- `dry_run` 模式（返回 Mock 数据，用于流程测试）
- 并发请求控制

### `cleaners/data_cleaner.py`

三层过滤流水线：

```
原始数据
  │
  ▼ Layer 1: 硬过滤（HardFilter）
  │  ├── 字段完整性检查
  │  ├── answer 合法性验证
  │  ├── 问题长度限制（8~500字符）
  │  ├── 标签泄露检测（question 中不能含 answer）
  │  ├── 中文比例检测（防乱码）
  │  └── 无意义内容检测
  │
  ▼ Layer 2: 质量评分（QualityScorer）
  │  ├── 长度适当性（0.20）
  │  ├── 关键词相关性（0.30）
  │  ├── 表达多样性（0.20）
  │  ├── 信息密度（0.15）
  │  └── 自然度（0.15）
  │
  ▼ Layer 3: 去重过滤（DuplicateFilter）
  │  ├── 精确哈希去重
  │  └── N-gram 相似度去重（同智能体内）
  │
  ▼ 清洗后数据
```

### `utils/dataset_builder.py`

提供以下功能：

- **DatasetMerger**：按配置权重合并三阶段数据，支持全局去重
- **DatasetAnalyzer**：统计样本分布、长度分布、领域分布、均衡度评分
- **DatasetExporter**：导出为 JSONL、JSON、CSV、Alpaca、ShareGPT、OpenAI Chat 格式
- **TrainTestSplitter**：按类别分层采样，划分训练/验证/测试集

---

## 配置参数详解

所有参数均在 `config/config.yaml` 中配置。以下列出关键参数：

### LLM 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm.model` | `gpt-4o-mini` | 使用的模型名称 |
| `llm.temperature` | `0.9` | 生成温度（越高越多样） |
| `llm.max_workers` | `5` | 并发请求数 |
| `llm.dry_run` | `false` | 是否启用 Dry Run 模式 |

### 生成配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `generation.stage1.samples_per_agent` | `150` | 阶段一每智能体样本数 |
| `generation.stage2.samples_per_agent` | `700` | 阶段二每智能体样本数 |
| `generation.stage2.simple_ratio` | `0.40` | 阶段二简单样本比例 |
| `generation.stage3.samples_per_pair` | `50` | 每个混淆对的样本数 |

### 清洗配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cleaning.min_quality_score` | `0.55` | 最低质量分数阈值 |
| `cleaning.dedup_method` | `both` | 去重方法（exact/ngram/both） |
| `cleaning.ngram_similarity_threshold` | `0.85` | N-gram 相似度去重阈值 |

### 合并配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `merging.target_per_agent` | `1000` | 最终每智能体目标样本数 |
| `merging.stage1_weight` | `0.15` | 阶段一样本占比 |
| `merging.stage2_weight` | `0.70` | 阶段二样本占比 |
| `merging.stage3_weight` | `0.15` | 阶段三样本占比 |

---

## 命令行用法

```bash
# 完整流程（默认配置）
python run.py

# 指定配置文件
python run.py --config config/config.yaml

# Dry Run 模式（测试流程，不调用 LLM）
python run.py --dry-run

# 仅运行阶段一（冷启动生成 + 清洗）
python run.py --stage 1

# 仅运行阶段二（生产泛化生成 + 清洗）
python run.py --stage 2

# 仅运行阶段三（困难负样本生成 + 清洗）
python run.py --stage 3

# 仅合并与导出（假设各阶段数据已存在）
python run.py --merge-only

# 指定特定智能体（逗号分隔）
python run.py --agents "民航绕偏航处置,飞行计划审核"

# 跳过指定阶段
python run.py --skip-stages "2,3"

# 查看当前生成进度
python run.py --progress

# 生成混淆对分析报告
python run.py --analyze-pairs

# 统计分析已有数据集
python run.py --analyze-dataset ./output/merged/full_dataset.jsonl

# 调整日志级别
python run.py --log-level DEBUG
```

---

## 数据格式说明

### 基础格式（JSONL）

每行一个 JSON 对象，包含 `question` 和 `answer` 两个字段：

```jsonl
{"question": "飞机偏航了怎么处理？", "answer": "民航绕偏航处置"}
{"question": "帮我审核这份飞行计划的合规性", "answer": "飞行计划审核"}
```

### Alpaca 格式

适用于 LLaMA-Factory、FastChat 等框架：

```json
[
  {
    "instruction": "你是一个智能体路由助手。根据用户的问题，从100个智能体中选择最合适的一个，只输出智能体名称。",
    "input": "飞机偏航了怎么处理？",
    "output": "民航绕偏航处置"
  }
]
```

### ShareGPT 格式

适用于需要多轮对话格式的 SFT 框架：

```json
[
  {
    "system": "你是一个智能体路由助手...",
    "conversations": [
      {"from": "human", "value": "飞机偏航了怎么处理？"},
      {"from": "gpt", "value": "民航绕偏航处置"}
    ]
  }
]
```

### OpenAI Chat Fine-tuning 格式

适用于 OpenAI Fine-tuning API：

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "飞机偏航了怎么处理？"}, {"role": "assistant", "content": "民航绕偏航处置"}]}
```

---

## 算力与规模估算

### 数据生成成本

以 `gpt-4o-mini` 为例（约 $0.15/1M input tokens，$0.60/1M output tokens）：

| 阶段 | 样本数 | 估算 Token | 估算成本 |
|------|--------|-----------|---------|
| 阶段一 | 15,000 | ~3M | ~$2 |
| 阶段二 | 70,000 | ~15M | ~$10 |
| 阶段三 | 15,000 | ~5M | ~$4 |
| **合计** | **100,000** | **~23M** | **~$16** |

> 实际成本因模型选择和 Prompt 长度而异。使用本地部署模型（Qwen、LLaMA 等）可将成本降至接近零。

### 训练成本

处理约 10 万条高质量微调数据，在主流硬件上的训练时间估算：

| 硬件 | 方法 | 模型规模 | 预计时间 |
|------|------|---------|---------|
| 8× H100 | Full SFT（2 Epoch） | 70B | 4~8 小时 |
| 8× A800 | Full SFT（2 Epoch） | 14B | 2~4 小时 |
| 4× 910B | LoRA（r=64） | 72B | 1~3 小时 |
| 单卡 A100 | LoRA（r=16） | 7B | 30~60 分钟 |

---

## 常见问题

**Q: 如何添加新的智能体？**

编辑 `agents/agent_registry.py`，在 `AGENT_REGISTRY` 列表中添加新的智能体字典，填写 `name`、`domain`、`description`、`keywords`、`confusable` 等字段。

**Q: 如何使用本地部署的模型？**

在 `config/config.yaml` 中设置 `llm.base_url` 为本地服务地址（如 `http://localhost:8000/v1`），并将 `llm.model` 设置为对应的模型名称。

**Q: 生成中途中断了怎么办？**

所有阶段均支持断点续传（`resume: true`）。直接重新运行相同命令，程序会自动检测已有数据并跳过已完成的部分。

**Q: 如何控制每个智能体的最终样本数量？**

调整 `merging.target_per_agent` 参数。同时可以通过 `merging.stage1_weight`、`stage2_weight`、`stage3_weight` 控制各阶段样本的比例。

**Q: 清洗后数据量不足怎么办？**

可以降低 `cleaning.min_quality_score` 阈值（如从 0.55 降到 0.45），或增加 `generation.stage2.samples_per_agent` 的目标数量，让生成更多原始数据以弥补清洗损耗。

**Q: 如何验证生成数据的质量？**

运行 `python run.py --analyze-dataset ./output/merged/full_dataset.jsonl` 查看详细的统计报告，包括每个智能体的样本数量、问题长度分布、均衡度评分等指标。
