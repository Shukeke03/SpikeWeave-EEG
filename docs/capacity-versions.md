# SpikeWeave-EEG 的 M / L 容量版本

确定日期：2026-10-08。先训练 **SpikeWeave-EEG-M**，资源充足后再训练 **SpikeWeave-EEG-L**。两版在同一仓库维护，共享代码、数据协议和 E/C 表征，分别训练、保存生成器。这里的并行版本指两套可复现配置；当前仅有一个共享 Stage A 作业在运行。

后续用户将3060显存预算提高到约80%；旧A作业已停止并保留断点，改用M路线batch20×累积13的新A配方。最新记录见[legion-m-training-80.md](legion-m-training-80.md)。下文一半显存的batch8×32为历史启动配方。

## 容量与保留机制

| 项目 | M：当前主力 | L：后续扩容 |
|---|---:|---:|
| 配置 | `configs/research-m.yaml` | `configs/research-l.yaml` |
| MCWave blocks | 12 | 24 |
| 扩散隐藏维度 | 512 | 1024 |
| attention heads | 8 | 16 |
| FFN 隐藏维度 | 2048 | 4096 |
| MCWave 参数 | 105,769,728 | 799,622,976 |
| 完整模型参数 | 378,734,353 | 1,072,587,601 |

两版模型配置只有 `denoiser_depth`、`denoiser_dim`、`denoiser_heads` 三项不同。参数量通过实际 MCWave 模块在 PyTorch meta device 上实例化逐项计数，非估算。完整模型计数包含官方医学前端、E/C 与生成器，不含 EMA 的第二份生成器权重。

E 保持 8×512 PLIF EEG 编码器；C 保持官方 BiomedCLIP 图文前端与 6×768 ViLT 式共享交互。E/C 合计 272,959,249 参数。6→768 的 geometry projection 另有 5,376 参数，在 Stage C 训练，每版独立初始化。M 的 512 维生成状态通过投影接收 C 的 768 维条件，无需压缩医学表征。

两版均保留双分支、固定多正样本双向对比、完整 signed db4/J3 小波、时间/导联/条件三类注意力、clean VP 扩散、montage 测量审计及六类软约束。生成器参数降幅约 86.8%；完整模型参数降幅约 64.7%。容量选择仍需实验验证，不能预先宣称 L 效果更好或 M 已达到基线。

## 共用训练与独立训练

```text
同一 source-only 数据和 RMS 尺度
        ↓
共享 A：信号重构与数值 CWT 适配
        ↓
共享 B：E/C 固定双分支对比与局部保真
        ↓ 同一个已对齐表征 checkpoint
        ├── 优先：M 的 C 阶段，12×512
        └── 延后：L 的 C 阶段，24×1024
```

当前 A 使用 `configs/legion-pretrain.yaml`，运行目录为 `runs/legion-pretrain-20261008`。实际 microbatch8、累积32、有效 batch256。A/B 不构造 MCWave 或 EMA，故旧 A 断点中记录的 D24×1024 配置只是未使用的元数据。改变扩散容量不会减少当前 A 的每步耗时；也无需重启、改写断点或重新处理数据。

严格续训现有 A 时加载其 `last.pt`，**不额外传入新配置**。恢复遵守保存的配置、优化器与 RNG；更换生成容量通过后续新 Trainer 的 `--init-representation` 完成。A→B 的新阶段也使用初始化，而非把 A 的 optimizer 接着当 B 使用。

临床 C 从已训练的 B（`align`）checkpoint 初始化 E/C。初始化验证导联顺序、采样率、SNN heads/chunks、数值 CWT、属性与尺度协议等兼容性；允许生成容量及执行性参数不同。来源 SHA256、源阶段、配置和训练来源记录进入 lineage。两个 C 作业各自从 step0 开始，各有随机初始化的 D、geometry projection、optimizer 和 EMA。L 不通过裁剪、插值或强制加载 M 的 D 权重启动。

## 使用入口

先仅验证两套配置，不加载医学大权重：

```bash
spikeweave doctor --config configs/research-m.yaml --config-only
spikeweave doctor --config configs/research-l.yaml --config-only
```

下列命令是 **B 完成且资源/显存探测通过后** 的生成训练入口，未自动执行。`runs/shared-align/last.pt` 为将来 B 的输出路径；现有 A 的断点不能直接代替它。

```bash
spikeweave --seed 0 train \
  --manifest /home/shuxing/workspace/datasets/SpikeWeave-EEG/chbmit-1.0.0/processed/dualbranch-v2 \
  --config configs/research-m.yaml --init-representation runs/shared-align/last.pt \
  --device cuda --batch-size 4 --steps 100000 --balanced \
  --output runs/m/diffusion-seed0
```

后续 L 使用同一个表征 checkpoint，将配置改为 `configs/research-l.yaml`，输出改为 `runs/l/diffusion-seed0`。两个模板的 microbatch4×accumulation64=256 是当前初始配方；依据实际设备测得的稳态显存调整 microbatch/累积，保持有效 batch，并将配方写入独立运行记录。新服务器上应先完成真实数据短跑与 optimizer/EMA 分配后的显存检查。

100000 次优化更新是预算上限，非收敛保证。先用 M 推进 G1 基线筛选；未达基线时继续排查模型和数据，不启动完整 G3。L 的容量比较复用同一 E/C、训练来源/划分、尺度、条件配方、种子和验证集；同时报告优化更新、GPU 小时、显存和 NFE，容量增长带来的额外计算须记账。

## 当前资源边界

M 降低的是 C 阶段的生成器容量，当前 3060 Laptop 的 6144 MiB 总显存和约一半占用目标仍约束训练。M 的 FP32 生成器权重、EMA、梯度、Adam 状态加冻结 E/C 已约 3 GiB，尚未计入激活与 CUDA 开销，不能保证 C 能沿用 A 的 0.47 allocator fraction。当前保持已实测通过的共享 A 作业，B/C 的显存及吞吐分别实测。

`configs/research-5090.yaml` 与 `configs/legion-clinical.yaml` 现与 M 对齐，供旧命令兼容使用；新实验优先使用明确的 M/L 配置名。`research-xl.yaml` 仅保留历史扩容候选，不属于当前两版训练计划。

数据沿用完成的 dualbranch-v2：837,181 个窗口，训练519,047、验证66,418、测试251,716；无需为 D 的容量变化复制 raw/shards、CWT 缓存或改动 source RMS。数据、权重和训练产物仍在 Git 外。

相关细则：[架构](architecture.md)、[项目总方案](project-plan.md)、[legion 当前预训练与人工监控](legion-training.md)。

## 本次工程验证

2026-10-08：本机与legion的完整CPU测试均为248 passed、2 CUDA skips，ruff与diff检查通过。legion验证时显式隐藏CUDA，避免占用正在预训练的GPU；两版`doctor --config-only`均确认正式E/C和对应D容量，仓库仍为private。

验证覆盖精确MCWave参数计数、正式M/L配置差异、A/B不分配D、不同条件/生成宽度的缩微前反传和采样、旧A配置→共享B→独立生成器的转移/恢复、语义与尺度冲突拒绝，以及源`last.pt`被原子更新时的哈希一致性。它证明当前实现和转移边界可运行；正式M/L的完整尺寸C训练、吞吐、显存及效果仍需在后续资源上测量。
