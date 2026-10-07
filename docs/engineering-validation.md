# 工程验证记录

历史记录说明：以下验证针对 v0.1 三分支原型。此次品牌和路径迁移不构成双分支 SNN / 图文模型已实现或已通过临床验证的证据。名称迁移的检查另见 [name-migration.md](name-migration.md)。

日期：2026-10-07。代码验证版本：`fab06ce`。本记录只报告工程行为；没有真实临床数据训练、诊断评价或少样本增强实验。

## 已配置环境

| 项目 | 本机 | legion |
|---|---|---|
| 平台 | macOS ARM | Linux x86_64 / WSL2 |
| Python | 3.11.17 | 3.11.17 |
| 环境 | `/Users/shuxing/miniforge3/envs/spikeweave-eeg` | `/home/shuxing/miniforge3/envs/spikeweave-eeg` |
| PyTorch | 2.8.0 | 2.8.0+cu128 |
| 测试设备 | CPU | CPU + CUDA 变换测试 |
| GPU | 未用于本次检查 | RTX 3060 Laptop，6GB |
| 工作目录 | `/Users/shuxing/AIlearning/SpikeWeave-EEG` | `/home/shuxing/workspace/SpikeWeave-EEG` |

两端 `pip check` 均通过，依赖快照分别存于根目录 `requirements-lock-macos.txt` 和 `requirements-lock-cuda.txt`。

## 测试结果

- 本机 `python -m pytest -q`：**52 passed, 1 skipped**；跳过 CUDA 变换测试，因为本机没有 CUDA。
- legion `python -m pytest -q`：**53 passed**，包括 CUDA 变换往返和梯度。
- 本机 `python -m ruff check .`：通过。
- [GitHub Actions CPU verification](https://github.com/Shukeke03/SpikeWeave-EEG/actions/runs/37569055684)：通过。
- 完整 signed DWT 与 PyWavelets 系数一致；往返、梯度、不同长度和有效样本支持传播已检查。
- CWT 检查频率峰值、幅度平方变化、功率池化、边界/缺失 mask、全缺失数值安全和梯度。
- 三分支检查 shifted-window mask、缺失/NaN 条件隔离、完整模态 dropout、文本 FiLM 丢弃路径、独立对齐和完整频带输出梯度。
- 随机本地小型 BERT 验证 ClinicalBERT 后端加载接口、冻结/eval 行为、投影层梯度。**没有下载或验证真实 Bio_ClinicalBERT 权重的医学能力。**
- 用临时随机 µV 文件与合成 EDF 检查数据全链路、物理单位/反向导联、事件截断、患者锁定、训练统计与条件来源。query 幅度从 `1e8` 改为 `1e12`，训练归一化统计保持不变。
- 检查 DDPM v 的解析 clean 恢复、DDIM 最后 noiseless 状态、CFM 方向与反向积分、重复文本/缺失模态损失行为。
- 训练、优化器恢复、条件采样、输出 µV 恢复、reference 自动 CWT 和目标/teacher 字段不影响条件均有端到端检查。

## 实际训练与采样

本机已跑 tiny 三步检查，以及 `legion.yaml` 的两步完整数据尺寸检查，均保存检查点并生成有限数值输出。

legion 命令：

```bash
triwave smoke --config configs/legion.yaml --device cuda --amp --steps 3 --batch-size 2 --sampling-steps 4 --output runs/legion-smoke
```

- 三个优化步和四步 DDIM 采样完成，输出形状 `[2,16,1024]`，256Hz，4 秒。
- 各步 loss、梯度均有限；该检查每步使用不同合成 batch，不用于宣称 loss 单调下降。
- 文本为 offline 小型 Transformer，尚未医学预训练。
- PyTorch `max_memory_allocated`：**213.54 MiB**，覆盖本进程训练与采样。它不是整机/驱动显存，也不是正式大模型或 ClinicalBERT 配置的资源估算。
- `legion.yaml` 共 **3,761,372** 个可训练参数；`research-5090.yaml` 共 **75,416,736** 个可训练参数，后者只统计参数，未在 5090 验证运行。

同一代码还完成 CUDA 检查点恢复：从 step 3 续训到 step 4，随后两步 DDIM 生成有限输出。另起新模型做 CFM 比较，两步训练和三步反向 Euler 采样完成；这些结果仅检查训练/恢复/积分路径，不比较两种生成方法的临床效果。

合成检查的波形单位为任意单位，不解释为真实患者 EEG。运行产物在两端 `runs/` 目录中，未进入 Git。

## 联调修正

重复模板在 EEG/text 与 structure/text 对齐中使用多正样本分组，EEG/structure 保留样本配对。参考采样补齐与训练一致的 CWT。缺失模态的无效位置不会成为对比负样本；FiLM 和多尺度路径同时参与 modality dropout。

归一化支持 NumPy/Torch 并保留 Tensor 设备；统计仅拟合 train/source。物理辅助损失按每个样本自己的噪声可靠度加权。Git 的数据目录忽略规则改为项目根路径，确保 `src/triwave/data` 上传。

当前生成器尚未传播目标测量 mask 到所有注意力层，因而**明确拒绝缺失目标导联/采样的生成训练**；参考条件缺失仍可使用。外部 CWT 采样必须声明转换协议与归一化尺度。完整属性词典、其他医学数据 adapter、目标测量 mask、下游评估与跨来源泛化是后续工作。
