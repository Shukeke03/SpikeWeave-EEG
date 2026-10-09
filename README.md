# SpikeWeave-EEG

**Dual-Branch Spiking EEG and Vision–Language Alignment for Montage-Constrained Wavelet Diffusion**

中文：**脉冲脑电与图文对齐的导联约束小波扩散模型**。

SpikeWeave-EEG 是医学 EEG 条件生成与患者独立少样本增强的私有研究项目。正式模型固定两条顶层表征分支：EEG SNN 分支 E，以及共享医学图文条件分支 C。C 接收文本、数值 CWT / 草图与已核实属性，延续 ViLT 的联合 token 交互思想；MCWave 生成器在导联测量一致的小波空间进行连续值条件扩散。

## 当前状态

2026-10-09：v0.3 保留双分支、固定对比和 MCWave 扩散，并接入七张数据侧表、16 项物理属性、严格 CWT 掩码、患者/事件采样、局部观测控制及 G0–G2 开发评估。G3 保留按效果解锁的最终测试入口。按用户要求，legion 阶段 A 表征训练与昼夜调度已安全停止，保存完整step600；尚无训练完成的生成模型或生成有效性结果。[停止训练报告](docs/training-stop-20261009.md)。

[最新实现与运行指南](docs/implementation-v0.3.md)说明数据迁移、条件来源、训练和评估命令；[M/L 容量与训练顺序](docs/capacity-versions.md)说明当前版本选择；[v0.2 架构实现记录](docs/implementation-v0.2.md)保留历史实现细节。

- EEG 分支：STAtten-8-512 的 EEG 适配版，PLIF、真实物理时间推进、脉冲与膜电位联合读出。
- 图文分支：BiomedCLIP 前端和六层共享 Transformer；文本、图像及其联合输入共享同一 C 分支。
- 固定对比学习：语义与结构两个读出头，多正例逐正例 log-softmax、双向平均、未知关系掩码。
- 生成器：先训练 M（12 层、512 维、8 heads），资源充足后训练 L（24 层、1024 维、16 heads）；两版均保留 clean prediction、VP diffusion、导联子空间约束和六类软约束。
- 研究评价：JET、WaveletDiff、同信息简单条件生成、生成后投影、患者独立真实增强及近邻记忆检查。

两版本使用同一代码、E/C 架构、数据划分和训练协议，扩散权重与运行目录分别保存。M 是当前验证起点，尚未证明容量最优；L 是后续容量比较，不要求同时启动两个训练作业。

已停止的legion作业采用M路线的A阶段昼夜配方：日间batch20×累积13，夜间batch8×累积32，保留Adam/RNG和全局优化进度。调度已退出，不会在下个时间边界自动恢复。[每日规则与监控](docs/daily-training.md)保留历史入口；[80%配方实测](docs/legion-m-training-80.md)保留探测记录。

| 版本 | 主配置 | MCWave 参数 | 完整模型参数 |
|---|---|---:|---:|
| SpikeWeave-EEG-M | [research-m.yaml](configs/research-m.yaml) | 105,769,728 | 378,734,353 |
| SpikeWeave-EEG-L | [research-l.yaml](configs/research-l.yaml) | 799,622,976 | 1,072,587,601 |

完整参数包括冻结的医学预训练前端，不能直接视为当前训练参数量。A/B 不构造扩散生成器；5,376 参数的导联几何投影仅在阶段 C 优化。

## 环境和路径

本机项目：`/Users/shuxing/AIlearning/SpikeWeave-EEG`。

legion 项目：`/home/shuxing/workspace/SpikeWeave-EEG`。

Miniforge 环境：`spikeweave-eeg`；legion 下载工具环境：`spikeweave-download`。

```bash
conda activate spikeweave-eeg
cd /Users/shuxing/AIlearning/SpikeWeave-EEG
spikeweave doctor --config configs/tiny.yaml --device cpu
```

正式类为`SpikeWeaveModel`，公共API为`from spikeweave import ...`。`triwave`模块/命令与`TriWaveModel`保留兼容别名。v0.1 检查点不能直接加载为新目标；当前协议的 E/C 表征检查点可初始化 M/L，扩散检查点须匹配各自容量。物理分片和`triwave-morlet-v1`数值变换继续兼容；新归一化采用不减均值的源RMS标量。旧目录软链接继续保留。

## 设计与记录

从 [项目总方案](docs/project-plan.md) 开始阅读：汇总最新架构、双分支数据处理和 G0–G3 分阶段评估。未达强基线时只运行必要审计与小规模开发筛选，候选 SOTA 阶段才展开完整验证。

| 文件 | 用途 |
|---|---|
| [project-plan.md](docs/project-plan.md) | 最新总方案、实现状态与按效果解锁的评估顺序 |
| [architecture.md](docs/architecture.md) | 已确立的双分支与 MCWave 架构 |
| [capacity-versions.md](docs/capacity-versions.md) | M/L 容量、共享表征与独立生成器训练 |
| [daily-training.md](docs/daily-training.md) | legion 昼夜显存规则、安全切换与监控 |
| [training-time-and-cvpr.md](docs/training-time-and-cvpr.md) | 当前训练时间估算与 CVPR 研究定位 |
| [contrastive-protocol.md](docs/contrastive-protocol.md) | 固定对比学习数学定义与关系规则 |
| [name-migration.md](docs/name-migration.md) | 本机、服务器、GitHub与环境的更名映射 |
| [implementation-v0.2.md](docs/implementation-v0.2.md) | 当前实现、运行方法和验证边界 |
| [implementation-v0.3.md](docs/implementation-v0.3.md) | 最新数据/条件/评估实现与兼容边界 |
| [engineering-v0.3.md](docs/engineering-v0.3.md) | 本机204项、legion206项测试与GPU恢复实测 |
| [development.md](docs/development.md) | 历史v0.1运行记录 |
| [dataset-chbmit-v1.md](docs/dataset-chbmit-v1.md) | 数据下载、处理及监控命令 |
| [data-protocol.md](docs/data-protocol.md) | 数据来源、物理单位、划分和合法条件 |
| [experiments.md](docs/experiments.md) | 固定实验与反证标准 |
| [engineering-v0.2.md](docs/engineering-v0.2.md) | 本机与legion的新模型工程验证 |
| [engineering-validation.md](docs/engineering-validation.md) | 历史原型的实测工程记录 |
| [architecture-v0.1-legacy.md](docs/architecture-v0.1-legacy.md) | 已取代的历史三分支设计 |

Sketch2CT 与 MSADN 保留为准确归属的研究灵感。模块组合和项目名称不构成已证明的新颖性。原始临床数据、报告、权重与凭据不进入 Git。
