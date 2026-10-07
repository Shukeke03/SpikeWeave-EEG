> 本文命令是历史v0.1记录，新代码请使用[implementation-v0.2.md](implementation-v0.2.md)。旧flow与三分支入口不再是主训练流程。

# 开发与运行

状态：项目与环境已更名为 SpikeWeave-EEG。以下运行命令仍对应 v0.1 三分支原型；已确立的 SNN / 图文双分支与新损失尚未实现。新的 `spikeweave` CLI 与原 `triwave` CLI 暂时指向同一兼容实现，内部 Python namespace 和 `triwave-morlet-v1` 数据协议保持不变。当前架构见 [architecture.md](architecture.md)，固定对比协议见 [contrastive-protocol.md](contrastive-protocol.md)。

目前提供可运行的研究实现：EDF/CHB-MIT 处理、三分支对齐、条件融合、完整 signed DWT 上的 WaveDiT、DDPM v-prediction、DDIM 采样，以及需独立训练的 CFM 比较。尚未完成真实临床数据训练和泛化评估。合成信号只验证软件行为。

## 本机 Miniforge

本机为 macOS ARM，环境位于 `/Users/shuxing/miniforge3/envs/spikeweave-eeg`。CPU 支持完整验证；当前 Morlet CWT 的复数 FFT 实现不支持 MPS。

```bash
source /Users/shuxing/miniforge3/etc/profile.d/conda.sh
conda activate spikeweave-eeg
cd /Users/shuxing/AIlearning/SpikeWeave-EEG
python -m pip install -r requirements-macos.txt
python -m pip install -e . --no-deps
python -m ruff check .
python -m pytest -q
spikeweave doctor --config configs/legion.yaml --device cpu
spikeweave smoke --config configs/tiny.yaml --device cpu --output runs/local-smoke
```

新机器可先用 `conda env create -f environment.yml` 建立 Python 3.11 环境，再按平台安装上面的依赖。已有环境无需重新创建。

## legion 服务器

当前 `ssh legion` 指向 NVIDIA RTX 3060 Laptop 6GB 的 WSL2/Linux 主机。环境使用 `/home/shuxing/miniforge3/envs/spikeweave-eeg`，项目使用 `/home/shuxing/workspace/SpikeWeave-EEG`。

```bash
ssh legion
source /home/shuxing/miniforge3/etc/profile.d/conda.sh
conda activate spikeweave-eeg
cd /home/shuxing/workspace/SpikeWeave-EEG
git pull --ff-only
python -m pip install -r requirements-cuda.txt
python -m pip install -e . --no-deps
spikeweave doctor --config configs/legion.yaml --device cuda
python -m pytest -q
spikeweave smoke --config configs/legion.yaml --device cuda --amp --batch-size 2 --output runs/legion-smoke
```

CUDA 环境固定 PyTorch 2.8.0/cu128。WSL 下 `nvidia-smi` 的实际位置为 `/usr/lib/wsl/lib/nvidia-smi`。Miniforge 自身不提供显卡驱动；当前主机已有可用驱动。

## 配置与训练阶段

| 配置 | 用途 | EEG 形状 | 文本 |
|---|---|---|---|
| `tiny.yaml` | 快速数学/软件验证 | 4×128，128Hz | 未预训练离线 Transformer |
| `legion.yaml` | 6GB GPU 试运行 | 16×1024，256Hz | 未预训练离线 Transformer |
| `legion-clinical.yaml` | 临床文本适配 | 16×1024，256Hz | 冻结 Bio_ClinicalBERT，需要本地缓存 |
| `research-5090.yaml` | 后续规模实验候选 | 16×1024，256Hz | 离线占位，正式实验需选择文本后端 |

`research-5090.yaml` 的显存与吞吐尚未在 5090 测量。离线文本后端与稳定 hash 分词只供工程验证，不能作为预训练医学语言理解的证据。临床后端使用 `emilyalsentzer/Bio_ClinicalBERT`；默认 `text_local_files_only: true`，缺少缓存会显式报错。需要使用时再按模型许可准备权重和 tokenizer，并保持训练与采样的文本后端一致。

```bash
# 单独预训练三分支，生成主干此时尚未训练
spikeweave train --manifest /absolute/path/manifest.jsonl --config configs/legion-clinical.yaml --alignment-only --steps 1000 --output runs/alignment
# 载入对齐阶段，然后联合训练条件扩散
spikeweave train --manifest /absolute/path/manifest.jsonl --resume runs/alignment/last.pt --steps 1000 --output runs/diffusion
# CFM 使用新模型/新检查点，不能将 diffusion 检查点直接用于 flow
spikeweave train-synthetic --config configs/legion.yaml --objective flow --device cuda --steps 10 --output runs/flow-check
```

当前 CLI 的 steps 是优化步数，记录于 `metrics.jsonl`；`last.pt` 包含模型、优化器、配置、训练步和训练集归一化。每次命令结束原子保存，`--resume` 恢复优化器与 CPU Torch RNG，数据加载顺序和 CUDA RNG 尚未保存，恢复不保证与不中断训练逐位相同。采样参数 `--steps` 表示 DDIM/流积分步数；生成时间与物理 EEG 时间分开。

## 公开医学数据接入

当前已实现 CHB-MIT；先自行按照其访问条件准备原始 EDF 和官方 summary 文件。其他数据源需要对应 adapter，不能将病例诊断、事件标签和正常人混为同一任务。

```bash
spikeweave prepare-chbmit --root /absolute/path/chbmit --output /absolute/path/processed-chbmit --split-map /absolute/path/patient-splits.json
```

split-map 是 `{ "chb01": "train", "chb02": "validation", "chb03": "test" }` 形式，实际必须覆盖所有准备导入的病例；这里仅展示格式，不是论文划分。`chb21` 与 `chb01` 按同一患者锁定。流程先锁定患者、检查源文件哈希，再对整条记录滤波/重采样并切窗。保留 µV、16 条双极导联的两个电极坐标、标签来源和完整处理上下文；允许缺失导联的 Python API 会显式保存 mask。

生成条件文本由官方事件标注模板构造，表示“窗口与已标注发作重叠”或“窗口没有发作标注”；它们不是临床自由文本报告，“没有发作标注”也不是健康对照。模板没有额外的诊断语义信息，因此必须与 class-label 基线比较。

默认准备流程不创建 reference，三分支可以处理缺失模态。要做 reference/structure 控制实验，需要在 manifest 的 `conditions.reference` 指定独立 source/support 样本 `origin_id`，并声明 `inference_available: true`、`target_derived: false`。加载器会检查患者、事件、窗口和滤波上下文隔离；整记录滤波后的 support/query 不允许来自同一文件。训练归一化只拟合 train/source，保存可逆单一全局仿射尺度。完整目标 CWT 只用于对齐教师，不作为生成条件。

```bash
# 可选：在同一目录另存带审计 source references 的 manifest；不存在合法候选时保持缺失
spikeweave attach-references --manifest /absolute/path/processed-chbmit/manifest.jsonl
```

该命令优先匹配已指定事件类别，从训练 source 选择参考，使用 sample ID 与 seed 确定轮转顺序。它不从 query 构造支持集，属于 source-only 条件生成；显式的少样本 support 协议需要单独准备。持有参考类别标签是条件预算的一部分，必须与同条件基线公平比较。

当前 manifest loader 尚未加载独立外部 structure 文件，遇到该字段会报错；可以通过独立 reference EEG 计算 CWT，或在采样条件 NPZ 中显式提供结构。新数据源和低样本适配实验应在此限制解决后开展。

## 条件生成

```bash
spikeweave sample --checkpoint runs/diffusion/last.pt --conditions /absolute/path/conditions.npz --prompt 'This window overlaps an annotated seizure.' --device cuda --steps 30 --guidance 1.5 --output outputs/generated.npz
```

条件 NPZ 必须有 `lead_geometry`，形状 `[B,C,6]`，单位米，端点与训练导联一致。可选 `reference_waveform` `[B,C,L]`、布尔 `reference_valid` `[B,C]` 或 `[B,C,L]`、`structure`/`structure_valid` `[B,C,F,T]`、`attribute_values`/`attribute_valid` `[B,8]`。参考 EEG 在有临床归一化检查点时使用原始 µV，CLI 会应用训练尺度并计算 CWT；直接提供 structure 时必须已经是同一归一化尺度和本文 Morlet 实现的 log-power，并声明 `cwt_protocol="triwave-morlet-v1"` 及 `structure_scale="normalized_uV"`（合成检查点使用 `"synthetic_arbitrary_units"`）。未知属性使用 False mask；当前八维属性接口还没有通用临床语义词典，应由后续协议定义名称、单位和合法范围。

支持 text-only、reference+text、structure+text 或无可选条件生成；geometry 始终保留。CFG 的无条件分支同时丢弃 reference、structure、text、attributes 及其 FiLM 路径。输出 NPZ 包含 waveform、采样率、单位和生成来源。合成检查点输出 arbitrary units；医学数据检查点用保存的训练尺度恢复 µV。

## 实现边界

- DWT 使用 PyWavelets 在初始化时构造固定稠密矩阵，前后向均为 Torch，可微且包含所有 signed 系数；这是精度参考实现，没有压缩系数数目或计算复杂度。
- CWT 为 ω₀=6 的离散复 Morlet，零填充线性 FFT 卷积，COI/缺失 mask，在线性功率域池化后 log；不声称数值等同 PyWavelets CWT，也不是校准 PSD。
- Swin 分支包含真实局部窗口、shift mask、层级合并；ViLT 风格融合、文本 FiLM 和多尺度结构注意力是明确实现的机制。
- PSD、导联相关和逐 DWT 频带误差是正则项，不能保证生理真实性、诊断正确性或跨数据集泛化。事件结构、疾病状态、少样本收益和隐私记忆仍需独立临床评估。
- 尚未实现 GAN 蒸馏、四步采样、通用数据源全集、下游 TSTR 分类评估、独立外部结构训练和完整属性词典；不把这些设计目标报告成已测结果。
- 当前条件生成训练要求目标导联和时间采样完整，检测到缺失会报错；允许参考模态缺失。三分支对齐可使用目标有效 mask。输出测量 mask 到生成器注意力的传播是后续可变 montage 扩展。

依赖版本按平台记录在 requirements 文件。模型与数据代码为本仓库原创实现，论文借鉴和许可信息见 [related-work.md](related-work.md)。

`requirements-lock-macos.txt` 和 `requirements-lock-cuda.txt` 保存本次实际环境的依赖快照；需要精确复现时使用对应平台快照，再安装本项目。验证记录见 [engineering-validation.md](engineering-validation.md)。
