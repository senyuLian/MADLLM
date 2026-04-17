# MADLLM: Multi-Agent Decision Large Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**MADLLM** 是一个基于大语言模型（LLM）监督微调（SFT）的多智能体协同决策框架。项目将多智能体交互决策建模为序列生成任务，利用 LLM 的上下文理解与模式识别能力，替代传统多智能体强化学习（MARL）中的策略网络，在 AR 移动边缘计算（MEC）资源分配场景中实现了显著的性能提升与跨场景泛化。

---

## 目录

- [核心思想](#核心思想)
- [系统架构](#系统架构)
  - [多智能体模型](#多智能体模型)
  - [LLM 策略网络](#llm-策略网络)
  - [序列化决策建模](#序列化决策建模)
  - [增量贡献度机制](#增量贡献度机制)
- [训练流程](#训练流程)
- [支持的大语言模型](#支持的大语言模型)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [技术特点](#技术特点)

---

## 核心思想

传统多智能体强化学习（如 COMA、MADDPG）依赖独立的策略网络，面临 **信用分配困难**、**训练不稳定**、**跨场景泛化弱** 等挑战。

MADLLM 的核心创新在于：

> **将多智能体决策转化为 LLM 的序列生成任务。**

借鉴 Decision Transformer 的思想，将多智能体在环境中的交互轨迹结构化为序列，通过 LLM 的自注意力机制同时建模智能体间的协作关系与时间维度的上下文依赖，实现端到端的策略学习。

| 维度 | 传统 MARL | MADLLM (本框架) |
| :--- | :--- | :--- |
| 策略表示 | 独立神经网络 | 预训练 LLM + LoRA 微调 |
| 智能体交互建模 | 显式消息传递 / 隐式参数共享 | LLM 自注意力统一建模 |
| 信用分配 | COMA / VDN / QMIX | 增量贡献度令牌（Incremental Contribution） |
| 泛化能力 | 有限（需重新训练） | 强（跨智能体数量迁移） |
| 参数效率 | 全参数训练 | LoRA (<1% 可训练参数) |

---

## 系统架构

### 多智能体模型

系统在 AR 边缘计算场景中部署 **5 个协同智能体**，每个智能体对应一个 AR 用户：

```
                    ┌──────────────────────────────┐
                    │       边缘基站 (GPU 池)        │
                    │  ┌──────────────────────────┐ │
                    │  │    资源调度器 (Heuristic)  │ │
                    │  │    GPU 批处理: 1/2/4/.../64 │ │
                    │  └──────────────────────────┘ │
                    └──────┬───┬───┬───┬───┬────────┘
                           │   │   │   │   │
                    ┌──────┴─┐ ┌┴──┐ ┌┴──┐ ┌┴──┐ ┌┴──────┐
                    │Agent 0 │ │ 1 │ │ 2 │ │ 3 │ │Agent 4│
                    │User 0  │ │   │ │   │ │   │ │User 4 │
                    └────────┘ └───┘ └───┘ └───┘ └───────┘
```

**每个智能体的状态空间（24 维）：**

| 特征组 | 维度 | 说明 |
| :--- | :--- | :--- |
| 当前窗口处理时延 / 上传时延 | 2 | 本时间窗口内的平均延迟 |
| 上一窗口处理时延 / 上传时延 | 2 | 上一时间窗口的平均延迟 |
| 累积 QoS（当前 / 上一步） | 2 | 体验质量指标 |
| 上一动作 (one-hot) | 16 | 帧率 x 压缩率的离散动作 |
| 带宽估计（当前 / 上一步） | 2 | 可用带宽状态 |

**动作空间（16 个离散动作）：**

帧率 × 压缩率的组合：

```
帧率:     15 / 30 / 60 FPS
压缩率:   0.2 / 0.4 / 0.6 / 0.8 / 1.0
→ 3 × 5 = 15 种动作 + 1 种 "不动作"
```

**智能体交互机制：**

智能体之间无直接消息传递，而是通过共享基站 GPU 资源池形成隐式协作——一个智能体占用更多 GPU 批处理资源将导致其他智能体请求排队等待，从而在全局奖励信号的引导下实现自适应的资源竞争与协调。

### LLM 策略网络

核心模型 `OfflineRLPolicy`（`plm_special/models/rl_policy.py`）将预训练 LLM 作为策略骨干，通过多层嵌入将多智能体决策信息编码为序列，输入 LLM 生成动作预测。

```
输入序列构造:
┌──────────────────────────────────────────────────────────┐
│  时间步 t=1                                               │
│  ┌─────┐ ┌──────┐ ┌───────┐ ┌───────────────────────┐   │
│  │ R_t │ │id_m  │ │pre_r  │ │ 9维观测特征 (obs_1~9) │   │
│  └─────┘ └──────┘ └───────┘ └───────────────────────┘   │
│  ┌─────────┐                                              │
│  │ a_{t-1} │                                              │
│  └─────────┘                                              │
├──────────────────────────────────────────────────────────┤
│  时间步 t=2  (同结构)                                      │
├──────────────────────────────────────────────────────────┤
│  ...                                                     │
├──────────────────────────────────────────────────────────┤
│  时间步 t=K  (滑动窗口截断)                                │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  LayerNorm       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  预训练 LLM      │
              │  (LoRA 微调)     │
              │  自注意力建模     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Action Head     │
              │  Linear → 16类   │
              └──────────────────┘
                       │
                       ▼
                  动作预测 a_t
```

**9 维观测特征编码：**

每个智能体的局部观测通过 `UserObsEncoder`（`state_encoder.py`）编码为 256 维特征向量，再分别通过独立的线性层映射到 LLM 嵌入空间：

| 编号 | 特征 | 含义 |
| :--- | :--- | :--- |
| 1 | delay | 处理时延 |
| 2 | throughput | 吞吐量 |
| 3 | loss | 丢包率 |
| 4 | jitter | 抖动 |
| 5 | buffer | 缓冲区状态 |
| 6 | playtime | 播放时长 |
| 7 | action_feat | 动作特征 |
| 8 | band1 | 当前带宽估计 |
| 9 | band2 | 上一带宽估计 |

### 序列化决策建模

每个决策单元由 **13 个令牌** 组成：

```python
# rl_policy.py:136-151
stacked_input = torch.cat((
    returns_embeddings,      # 1. 目标回报 R_t
    agent_id_embeddings,     # 2. 智能体标识 id_m
    pre_r_embeddings,        # 3. 增量贡献度 pre_r
    delay_emb,               # 4. 处理时延
    throughput_emb,          # 5. 吞吐量
    loss_emb,                # 6. 丢包率
    jitter_emb,              # 7. 抖动
    buffer_emb,              # 8. 缓冲区
    playtime_emb,            # 9. 播放时长
    action_feat_emb,         # 10. 动作特征
    band1_emb,               # 11. 带宽 (当前)
    band2_emb,               # 12. 带宽 (上一步)
    action_embeddings        # 13. 上一动作 a_{t-1}
), dim=0)
```

多智能体的决策序列按时间步交错排列：

```
[R^1_t, id^1, pre_r^1, obs^1, a^1] [R^2_t, id^2, pre_r^2, obs^2, a^2] ... [R^N_t, id^N, pre_r^N, obs^N, a^N]
```

LLM 通过自注意力机制同时捕获：
- **时间维度依赖**：历史决策对当前的影响
- **智能体间交互**：不同智能体在同一时间步的协作/竞争关系

### 增量贡献度机制

增量贡献度（Incremental Contribution, `pre_r`）是本框架解决多智能体 **信用分配问题** 的关键机制。

**问题背景：** 在部分可观测的多智能体环境中，全局奖励信号无法准确反映单个智能体的贡献。传统方法（如 COMA 的反事实基线）需要额外的评论家网络。

**MADLLM 的方案：** 将增量贡献度作为输入令牌直接编码到序列中，让 LLM 隐式学习个体贡献与全局奖励之间的关系。

```python
# AR_env/AR_env_rl.py - 增量贡献度计算
def get_tp_pre_add_r(self, t_ep, id):
    """
    基于在线 Welford 算法计算每个智能体的增量贡献度
    pre_r_i = mean(QoS[0:i+1]) - penalty * std(QoS[0:i+1])
    """
    for i in range(self.max_user_num):
        r2 = np.std(r[0:i+1]) if len(r[0:i+1]) > 1 else 0
        r_pre[i] = sum(r[0:i+1]) / (i+1) - self.penalty * r2
    return r_pre[id]
```

**计算逻辑：** 按智能体索引顺序，逐步累加计算部分团队奖励，第 `i` 个智能体的增量贡献度即为加入该智能体后团队奖励的变化量。

---

## 训练流程

MADLLM 采用 **离线强化学习** 范式，训练分为三个阶段：

```
阶段 1: 经验收集              阶段 2: LLM 监督微调            阶段 3: 在线评估
┌────────────────┐          ┌──────────────────┐          ┌────────────────┐
│ COMA 算法与环境 │          │   加载预训练 LLM  │          │  加载微调模型   │
│ 交互收集轨迹    │ ──────▶  │   + LoRA 适配器   │ ──────▶  │  AR 环境在线   │
│ 计算增量贡献度  │          │   SFT 训练       │          │  决策评估       │
└────────────────┘          └──────────────────┘          └────────────────┘
    make_exp_pool.py            run_plm.py              evaluate.py / test.py
```

**阶段 1 — 经验池构建：**
- 运行传统 MARL 算法（COMA）与环境交互
- 收集状态、动作、奖励轨迹
- 在线计算每个智能体每步的增量贡献度
- 保存为 `ExperiencePool` 格式

**阶段 2 — LLM 监督微调（SFT）：**
- 加载预训练 LLM（如 Llama-7B），冻结原始参数
- 注入 LoRA 适配器（rank=128），仅训练约 0.1% 参数
- 将经验池数据构造成结构化序列
- 以交叉熵损失训练模型预测最优动作
- 梯度累积（32 步）+ Warmup（2000 步）+ 余弦衰减

**阶段 3 — 在线评估：**
- 加载微调后的模型权重
- 在 AR 环境中进行在线决策
- 支持跨场景测试（如 5 智能体训练 → 3/4 智能体评估）

---

## 支持的大语言模型

| 模型 | 可用规模 | 嵌入维度 | Transformer 层数 |
| :--- | :--- | :--- | :--- |
| **Llama** | base (7B) | 4096 | 32 |
| **Mistral** | base (7B) | 4096 | 32 |
| **GPT-2** | small / base / large / xl | 768 / 1024 / 1280 / 1600 | 12 / 24 / 36 / 48 |
| **OPT** | xxs / xs / small / base / large | 512 / 2048 / 2560 / 4096 / 5120 | 16 / 32 / 32 / 32 / 40 |
| **T5-LM** | small / base / large / xl | 512 / 768 / 4096 / 2048 | 6 / 12 / 24 / 24 |

所有模型均通过 HuggingFace Transformers 加载，支持 LoRA 参数高效微调。Llama 和 Mistral 的实现额外支持 **Early Stopping**（提前退出中间层），可用于推理加速和消融实验。

---

## 项目结构

```
AR_MALLM/
├── config.json                   # RL 环境、智能体、训练全局配置
├── config.py                     # LLM 训练超参数与模型规格定义
├── rw_config.py                  # 配置文件读写工具
├── run_plm.py                    # 主入口：LLM 微调启动脚本
├── make_exp_pool.py              # 阶段 1：经验池生成脚本
│
├── AR_env/                       # 多智能体 AR 资源分配环境
│   ├── AR_env_rl.py              # 核心环境类（状态转移、奖励计算、增量贡献度）
│   ├── User.py                   # 智能体实现（状态空间、动作空间、观测生成）
│   ├── BaseStation.py            # 基站模拟（GPU 批处理调度）
│   ├── DQN_Agent.py              # DQN 智能体（用于经验收集）
│   ├── episode_runner_rl.py      # Episode 执行器
│   ├── episode_batch.py          # 批量数据采集
│   ├── mix_dqn_runner.py         # 混合 DQN 运行器
│   ├── DemostrateModule.py       # 演示模块
│   ├── eva_model.py              # 模型评估工具
│   ├── test_90_model.py          # 90 步测试脚本
│   └── assist_func.py            # 辅助函数
│
├── plm_special/                  # LLM 训练与推理模块
│   ├── trainer.py                # 训练循环（梯度累积、学习率调度、检查点保存）
│   ├── evaluate.py               # 模型评估脚本
│   ├── test.py                   # 测试脚本
│   ├── models/
│   │   ├── rl_policy.py          # 核心：LLM 策略网络 (OfflineRLPolicy)
│   │   ├── state_encoder.py      # 智能体观测编码器 (UserObsEncoder)
│   │   ├── low_rank.py           # LoRA 参数高效微调配置
│   │   ├── gpt2.py               # GPT-2 模型封装
│   │   ├── llama.py              # Llama 模型封装（含 Early Stopping）
│   │   ├── mistral.py            # Mistral 模型封装
│   │   ├── opt.py                # OPT 模型封装
│   │   └── t5.py                 # T5-LM 模型封装
│   ├── data/
│   │   ├── dataset.py            # 经验数据集（滑动窗口采样、奖励归一化）
│   │   └── exp_pool.py           # 经验池数据结构
│   └── utils/
│       ├── plm_utils.py          # LLM 加载工具（多模型统一接口）
│       ├── utils.py              # 通用工具函数
│       └── console_logger.py     # 控制台日志
│
└── baseline_special/             # 基线算法对比
    ├── a3c.py                    # A3C 算法（TensorFlow 实现）
    ├── env.py                    # 基线环境
    ├── trace_generator.py        # 带宽轨迹生成
    └── utils/
        ├── constants.py          # 常量与配置
        └── utils.py              # 工具函数
```

---

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.7+（GPU 训练）
- PyTorch 2.0+

### 安装

```bash
# 克隆仓库
git clone https://github.com/senyuLian/MADLLM.git
cd MADLLM/AR_MALLM

# 安装依赖
pip install -r requirements.txt
```

### 下载预训练模型

将 HuggingFace 格式的预训练模型放入 `downloaded_plms/` 目录：

```bash
mkdir -p ../downloaded_plms
# 下载 Llama-7B、GPT-2、Mistral-7B 等模型到该目录
```

### 运行

**第一步：生成经验池**

```bash
python make_exp_pool.py
```

**第二步：LLM 监督微调**

```bash
python run_plm.py --adapt --grad-accum-steps 32 \
    --plm-type llama --plm-size base --rank 128 \
    --device cuda:0 --lr 0.0001 --warmup-steps 2000 \
    --num-epochs 10 --eval-per-epoch 2
```

**关键参数说明：**

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--plm-type` | llama | LLM 类型：llama / gpt2 / mistral / opt / t5-lm |
| `--plm-size` | base | 模型规模：见上方模型支持表 |
| `--rank` | 128 | LoRA 秩（越大可训练参数越多） |
| `--lr` | 1e-4 | 学习率 |
| `--num-epochs` | 80 | 训练轮数 |
| `--grad-accum-steps` | 32 | 梯度累积步数（等效大 batch size） |
| `--warmup-steps` | 2000 | 学习率预热步数 |
| `--device` | cuda:0 | 训练设备 |

**第三步：评估与测试**

```bash
# 标准评估
python plm_special/evaluate.py

# 跨场景泛化测试
python plm_special/test.py
```

### 使用预构建数据集

经验池数据也可直接下载：

- 夸克网盘：`AR_exp_pool_coma_90_0std_dta_2000.pkl`
- 链接：https://pan.quark.cn/s/83d038fdc923
- 提取码：zXFN

---

## 实验结果

### 主实验（5 智能体）

| 方法 | 总和 QoS (↑) | 公平性指标 Std (↓) |
| :--- | :---: | :---: |
| **MADLLM (Llama-7B + LoRA)** | **4075.15** | **165.25** |
| COMA (Baseline) | 4904.83 | 161.18 |

MADLLM 在 QoS 总量上较 COMA 提升 **31.2%**。

### 跨场景泛化

| 训练场景 | 测试场景 | MADLLM 性能衰减 | COMA 性能衰减 |
| :--- | :--- | :---: | :---: |
| 5 智能体 | 4 智能体 | **< 8%** | ~25% |
| 5 智能体 | 3 智能体 | **< 15%** | ~45% |

MADLLM 在未见过的智能体数量场景中展现出显著的泛化优势，证明 LLM 的上下文学习能力可有效适应多智能体规模变化。

---

## 技术特点

### 1. LLM 作为多智能体策略骨干

利用预训练 LLM 的强大序列建模能力，通过自注意力机制统一建模多智能体间的协作与竞争关系，无需设计复杂的信息共享机制。

### 2. 增量贡献度解决信用分配

引入基于 Welford 在线算法计算的增量贡献度令牌（`pre_r`），将每个智能体对团队目标的边际贡献显式编码为输入，使 LLM 能够学习个体行为与全局奖励之间的因果关系。

### 3. 参数高效微调 (LoRA)

冻结预训练 LLM 的全部参数，仅通过 LoRA 注入极少量的可训练参数（约 0.1%），在保持 LLM 原有知识的同时实现策略适配。

### 4. 滑动窗口推理

采用固定长度 $K$ 的滑动窗口截断输入序列，将自注意力计算复杂度从 $O(T^2)$ 降低至 $O(K^2)$，支持长时间 episode 的实时决策。

### 5. Early Stopping 层级退出

Llama/Mistral 模型封装支持在中间 Transformer 层提前退出推理（`stop_layer_idx` 参数），可用于：
- 推理加速：在精度损失可控的前提下减少计算量
- 消融实验：分析不同层级特征对决策质量的影响

### 6. 多模型后端支持

统一接口支持 GPT-2、Llama、Mistral、OPT、T5-LM 五种模型家族，可灵活选择不同规模（512 维 ~ 5120 维嵌入）的基座模型进行实验对比。

---

## 配置说明

### 环境配置 (`config.json`)

```json
{
    "env_config": {
        "n_state": 130,      // 全局状态维度
        "n_agent": 5         // 智能体数量
    },
    "agent_config": {
        "n_action": 16,      // 离散动作数量
        "n_obs": 24          // 单智能体观测维度
    },
    "AR_env_config": {
        "max_user_num": 5,                    // 最大用户数
        "user_arrive_time": [0, 10000, 20000, 30000, 40000],  // 用户到达时间 (ms)
        "user_active_time": 50,               // 用户活跃时长 (episodes)
        "penalty": 1.0,                       // 公平性惩罚系数
        "BS_alg": "new_heur_inter"            // 基站调度算法
    }
}
```

### 训练超参数 (`config.py`)

```python
lr = 1e-4              # 学习率
weight_decay = 1e-4    # 权重衰减
warmup_steps = 2000    # 学习率预热步数
num_epochs = 80        # 训练轮数
grad_accum_steps = 32  # 梯度累积步数
context_window = 20    # 滑动窗口长度
state_feature_dim = 256  # 状态编码器输出维度
penalty = 1.0          # 公平性惩罚系数
```

---

## 相关信息

- **专利**：一种基于大语言模型 SFT 的多智能体决策方法（人工智能与计算机网络交叉技术领域）
- **联系方式**：752384946@qq.com
- **问题反馈**：欢迎提交 Issue 或 Pull Request

---

## 许可证

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。
