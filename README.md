# SpikeWeave-EEG

**Dual-Branch Spiking EEG and Vision–Language Alignment for Montage-Constrained Wavelet Diffusion**

中文：**脉冲脑电与图文对齐的导联约束小波扩散模型**。

SpikeWeave-EEG 是医学 EEG 条件生成与患者独立少样本增强的私有研究项目。正式模型固定两条顶层表征分支：EEG SNN 分支 E，以及共享医学图文条件分支 C。C 接收文本、数值 CWT / 草图与已核实属性，延续 ViLT 的联合 token 交互思想；MCWave 生成器在导联测量一致的小波空间进行连续值条件扩散。

## 当前状态

2026-10-07：完成项目名称和运行路径迁移，确立双分支架构与多正例双向对比协议。此次未修改模型、损失或数据处理算法。当前 Python 实现仍是 v0.1 三分支工程原型；SNN、共享图文分支和 MCWave 新目标将在下一阶段实现。工程验证与临床性能分别记录，不能把旧原型测试当作新模型结果。

- EEG 分支：STAtten-8-512 的 EEG 适配版，PLIF、真实物理时间推进、脉冲与膜电位联合读出。
- 图文分支：BiomedCLIP 前端和六层共享 Transformer；文本、图像及其联合输入共享同一 C 分支。
- 固定对比学习：语义与结构两个读出头，多正例逐正例 log-softmax、双向平均、未知关系掩码。
- 生成器：DiT-L 尺度，24 层、1024 维、16 heads，clean prediction、VP diffusion、导联子空间约束和六类软约束。
- 研究评价：JET、WaveletDiff、同信息简单条件生成、生成后投影、患者独立真实增强及近邻记忆检查。

## 环境和路径

本机项目：`/Users/shuxing/AIlearning/SpikeWeave-EEG`。

legion 项目：`/home/shuxing/workspace/SpikeWeave-EEG`。

Miniforge 环境：`spikeweave-eeg`；legion 下载工具环境：`spikeweave-download`。

```bash
conda activate spikeweave-eeg
cd /Users/shuxing/AIlearning/SpikeWeave-EEG
spikeweave doctor --config configs/tiny.yaml --device cpu
```

`spikeweave` 暂时调用已有 `triwave.cli` 实现，旧 `triwave` 命令继续兼容。`src/triwave`、旧检查点、`triwave-morlet-v1` 数据协议和 v0.1 算法配置本轮不改。旧绝对目录由兼容软链接保留，已有数据索引无需因品牌更名重新生成。

## 设计与记录

| 文件 | 用途 |
|---|---|
| [architecture.md](docs/architecture.md) | 已确立的双分支与 MCWave 架构 |
| [contrastive-protocol.md](docs/contrastive-protocol.md) | 固定对比学习数学定义与关系规则 |
| [name-migration.md](docs/name-migration.md) | 本机、服务器、GitHub与环境的更名映射 |
| [development.md](docs/development.md) | 当前兼容原型运行方法 |
| [dataset-chbmit-v1.md](docs/dataset-chbmit-v1.md) | 数据下载、处理及监控命令 |
| [data-protocol.md](docs/data-protocol.md) | 数据来源、物理单位、划分和合法条件 |
| [experiments.md](docs/experiments.md) | 固定实验与反证标准 |
| [engineering-validation.md](docs/engineering-validation.md) | 历史原型的实测工程记录 |
| [architecture-v0.1-legacy.md](docs/architecture-v0.1-legacy.md) | 已取代的历史三分支设计 |

Sketch2CT 与 MSADN 保留为准确归属的研究灵感。模块组合和项目名称不构成已证明的新颖性。原始临床数据、报告、权重与凭据不进入 Git。
