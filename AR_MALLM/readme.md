# AR-MADLLM: Multi-Agent Decision Large Language Model for AR Resource Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: OpenPrompt](https://img.shields.io/badge/Framework-OpenPrompt-red.svg)](https://github.com/thunlp/OpenPrompt)

**AR-MADLLM** (Multi-Agent Decision LLM) 是一个基于大语言模型监督微调（SFT）的多智能体协同决策框架。本项目针对移动边缘计算（MEC）场景下的增强现实（AR）资源分配问题，将多智能体决策建模为序列生成任务，利用 LLM 强大的上下文理解能力实现高效、鲁棒且具备极强泛化性的协同策略。

## 🌟 核心技术特性 

本项目实现了 **MADLLM** 架构，相比传统多智能体强化学习（MARL），具备以下核心优势：

1.  **序列化决策建模**：借鉴 Decision Transformer (DT) 思想，将多智能体交互轨迹转化为结构化序列，直接学习从历史上下文到最优动作的映射。
2.  **增量贡献度机制 (Incremental Contribution)**：引入基于 **Welford 迭代算法** 在线计算的增量贡献度令牌。该机制量化了单个智能体对团队全局目标的即时边际影响，有效解决了部分可观测环境下的**信用分配 (Credit Assignment)** 难题。
3.  **参数高效微调 (PEFT)**：采用 **LoRA (Low-Rank Adaptation)** 技术，在冻结预训练 LLM（如 Llama, GPT-2）参数的基础上注入极少量可训练参数，大幅降低计算成本。
4.  **滑动窗口推理机制**：采用 $K$ 时间步滑动窗口控制输入序列长度（总长度为 $K \times N$ 令牌），将自注意力复杂度从 $O(T^2)$ 降低至 $O(K^2)$，支持大规模智能体实时决策。
5.  **卓越的跨场景泛化**：实验证明，在 5 个智能体规模下训练的模型可直接迁移至 3 或 4 个智能体的未见场景，性能优于传统 COMA 方法。

## 🛠 系统架构

### 输入序列结构
模型接收的每个决策单元包含以下结构化令牌：
`[时间步 t, 全局目标回报 Rt, 智能体标识 idm, 局部观测 otm, 历史增量贡献度 rt1:m-1, 上一动作 atm-1]`

### 业务逻辑
- **智能体侧**：调整视频帧率（15/30/60 FPS）与图像压缩率（0.2-1.0）。
- **服务器侧**：基于 GPU 资源竞争状态，通过优化体验质量（QoS）和长期公平性指标进行资源调度。

---

## 🚀 安装指南

### 1. 克隆仓库
```bash
git clone <repository-url>
cd AR_MALLM
```

### 2. 环境配置
建议使用 Python 3.8+ 和 CUDA 11.7+。
```bash
pip install -r requirements.txt
```

### 3. 下载预训练模型 (PLM)
本项目支持多种后端，请将 HuggingFace 格式的模型放入 `downloaded_plms` 目录：
- Llama-7B / Mistral-7B
- GPT-2 / GPT-J
- OPT 系列


### 4. 准备数据：
带宽轨迹：参考 constants.py。
确保配置正确。

---

## 💻 使用说明

### 第一步：生成经验池 (Data Collection)
运行 RL 代理与 AR 环境交互，收集离线决策轨迹，计算在线计算的增量贡献度数据：
```bash
python make_exp_pool.py
```
*轨迹数据将存储在 `ExperiencePool` 中，包含状态、动作及基于 Welford 算法计算的奖励。*

### 第二步：模型监督微调 (SFT)
使用 LoRA 对基座模型进行微调：
```bash
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 10 --eval-per-epoch 2
```

### 第三步：性能评估与泛化测试
在测试集或未见规模（如 3 智能体）环境下评估：
```bash
# 评估
python plm_special/evaluate.py

# 测试
python plm_special/test.py
```

---

## 📂 项目结构

```text
AR_MALLM/
├── config.json              # RL和环境配置
├── config.py                # PLM训练和全局配置
├── run_plm.py               # 主LLM微调脚本
├── make_exp_pool.py         # 经验池创建
├── AR_env/                  # AR RL环境
│   ├── AR_env_rl.py         # 主环境类（资源分配）
│   ├── User.py              # 用户模拟（帧生成）
│   ├── BaseStation.py       # 基站逻辑（GPU批处理）
│   └── ... (DQN_Agent.py, episode_runner_rl.py 等)
├── plm_special/             # LLM训练和模型
│── │── utils/               # 共享工具
│   ├── trainer.py           # 训练逻辑
│   ├── evaluate.py          # 评估
│   ├── test.py              # 测试
│   ├── models/              # PLM模型 (rl_policy.py, state_encoder.py, low_rank.py 等)
│   └── data/                # 数据集      
└── baseline_special/        # RL基线
```

---

## 📊 实验结果

| 方法 | 平均 QoS (↑) | 公平性指标 (↓) | 泛化性能 (3-Agent) |
| :--- | :--- | :--- | :--- |
| **AR-MADLLM (Ours)** | **1192.70** | **163.85** | **较好 (降幅 < 15%)** |
| COMA (Baseline) | 908.66 | 142.03 | 较差(降幅约为45%) |

---

## 📄 实验数据集
夸克网盘「AR_exp_pool_coma_90_0std_dta_2000.pkl」，点击链接即可保存。
链接：https://pan.quark.cn/s/83d038fdc923
提取码：zXFN
**相关专利信息：**
*名称：一种基于大语言模型 SFT 的多智能体决策方法*
*技术领域：人工智能与计算机网络交叉技术*

---

## 🤝 贡献与反馈
如有任何问题或改进建议，欢迎提交 Issue 或 Pull Request。
联系方式：752384946@qq.com
