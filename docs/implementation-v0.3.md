# SpikeWeave-EEG v0.3 实现与运行

更新：2026-10-08。此版本把最新的数据、条件控制和开发评估方案接入代码；工程验证不能替代临床效果实验。模型保持两条顶层分支 E/C、固定双向多正例对比、ViLT 联合交互、有符号小波和 MCWave clean VP 扩散，正式模型尺寸沿用既定配置。

## 本次实现

| 部分 | 实际行为 |
|---|---|
| 双分支数据 | 七张流式 Parquet 侧表、紧凑窗口索引、完成标志与摘要；复用 v1 连续物理分片 |
| 教师和条件 | full target 教师、独立参考条件、局部观测分别声明；query 标签不选择参考 |
| 数值属性 | 物理 µV 下计算固定 16 项属性，教师、生成条件、评估真值分开存储和使用 |
| 有效性 | CWT 池化格和视觉 patch 要求全部贡献位置有效；保留缺失和 COI 掩码 |
| 采样 | source train 按类别→患者→事件/背景块→窗口均衡；未标注源窗只进入表征重构 |
| 控制损失 | 独立参考只提供风格条件；partial observation 只约束可见区域；属性仅约束本次实际保留的条件 |
| 开发评估 | G0 数值/身份检查、G1 小 CNN 同预算比较、G2 三种子证据、G3 最终测试入口登记 |

属性算法 `hann-energy-v2` 使用 Hann rFFT 能量，支持区间为 `[0.5,80)`，五频段为 `[0.5,4)`、`[4,8)`、`[8,13)`、`[13,30)`、`[30,80)`。这是固定能量估计，不能称为已经校准的一侧 PSD。原生带宽不足的频段标未知；不完整时间观测不制造完整频谱标签。

非空样本的属性条件以 50% 概率为空，其余均匀选择 1–4 项有效属性。全空条件清除图文、参考和属性；其他 six-family 目标仍依照扩散时间门控。硬蒙太奇模式必须有绑定当前源库和测量定义的全 source 审计通过证明；单 batch 通过不能授权 hard。

## 环境与数据迁移

```bash
conda activate spikeweave-eeg
pip install -e '.[dev,medical]'
spikeweave doctor --config configs/research-5090.yaml --config-only
```

新增 `pyarrow`。本机和 legion 继续使用 `spikeweave-eeg` 环境与 `SpikeWeave-EEG` 目录；Python `triwave` 命名空间保留兼容。

在 legion 的 v1 处理完成、来源审计通过后执行以下一次性迁移：

```bash
cd /home/shuxing/workspace/SpikeWeave-EEG
spikeweave build-dualbranch \
  --root /home/shuxing/workspace/datasets/SpikeWeave-EEG/chbmit-1.0.0/processed/v1 \
  --output /home/shuxing/workspace/datasets/SpikeWeave-EEG/chbmit-1.0.0/processed/dualbranch-v2 \
  --config configs/legion.yaml --batch-size 128

spikeweave audit-dataset \
  --root /home/shuxing/workspace/datasets/SpikeWeave-EEG/chbmit-1.0.0/processed/dualbranch-v2 \
  --config configs/legion.yaml --numerical-windows 16
```

输出 `records/windows/texts/views/attributes/relations/condition_specs.parquet`，以及 `window_index_v2.npy`、`train_scale_v2.json`、`montage_audit.json`、`dataset_audit_v2.json`、`dualbranch_manifest.json`。完成目录原子发布；不完整构建不作为可训练数据。已存在但配方不符的目录拒绝覆盖。

默认只读取源训练 EEG 计算教师属性，不读取 query EEG 数值。`--include-query-attributes` 是显式离线教师导出选项，不能把这些属性喂给 query 条件；最终测试数值读取必须依照 G3 方案另行安排。`--include-unlabelled-source` 只纳入唯一隔离原因是未知注释、其余物理/QC 合格的源窗。

不全量缓存 CWT，不复制 EEG 分片。实际新增磁盘体量以构建后的文件统计为准；本次代码交付并不表示 legion 上全量临床 v2 资产已经生成。

## 训练和外部条件

训练入口 `--manifest` 接收 v2 目录。阶段 A 默认 `representation`，B/C 默认 `training`；`--data-purpose partial_observation` 单独声明局部观测实验。已有正式模型配置、学习率、块数、累积和 EMA 保持既定方案。

```bash
# 示例；正式临床长训练需另行启动。
spikeweave train --config configs/research-align.yaml \
  --manifest /path/to/processed/dualbranch-v2 \
  --init-representation runs/pretrain/last.pt \
  --batch-size 256 --steps 100000 --output runs/align
```

`sample` 条件 NPZ 必须包含 `lead_geometry[B,C,6]`；审计/hard 模式还要求一致的 `lead_names`。

- 文本：`--prompt` 或 NPZ 的 `prompts[B]`；来源是用户给定请求。
- 参考：物理 µV `reference_waveform`、布尔 `reference_valid`、JSON 字符串 `reference_provenance`，含 `origin_id`、`target_derived:false`、`inference_available:true`。模型只归一化一次并现场计算 CWT。
- 外部结构：`structure`、布尔 `structure_valid`、`structure_role`、JSON `structure_provenance`、`cwt_protocol=triwave-morlet-v1`、`cwt_mask_protocol=cwt-all-contribution-v2`，以及 checkpoint 一致的尺度/摘要字段。partial observation 另有 `structure_visible` 和 `protocol=observation-conditioned`，必须隐藏完整有效区域的一部分。
- 属性：`attribute_values`、布尔 `attribute_valid`、`attribute_protocol=hann-energy-v2`、JSON `attribute_origin`。来源必须可部署；源目标教师属性不能进入生成接口。

独立参考图不视为新输出的逐像素目标。全目标 CWT 不允许作为条件。仅部分观测实验允许声明已经观测到的目标区域，未观测值在编码前屏蔽。

新 checkpoint 保存条件协议。旧 CWT mask/属性协议 checkpoint 默认拒绝恢复；显式 `allow_protocol_migration=True` 仅迁移兼容数值权重，清空优化器/步数和 hard 许可，不能作为原实验的续训。不同条件有效性语义必须重新验证。

## 轻量评估与闸门

先导出固定 train/source 和自然 validation/query 的物理评估快照。选择数值子集不按标签采样，完整患者/记录/hash 身份仍审计；子集偏差在报告中声明。

```bash
spikeweave export-evaluation --root /path/to/dualbranch-v2 \
  --config configs/legion.yaml --split train --output runs/eval/train.npz
spikeweave export-evaluation --root /path/to/dualbranch-v2 \
  --config configs/legion.yaml --split validation --output runs/eval/validation.npz

spikeweave evaluate --stage G0 --train runs/eval/train.npz \
  --validation runs/eval/validation.npz --ledger runs/eval/gates.json --output runs/eval/g0.json
```

NPZ 载入固定物理单位、导联顺序、split/role、患者/记录/样本 ID 与标签本体。自然验证分布不能用类别平衡质量审查库替代。主指标为患者等权 AP；无正类患者的 AP 为不可定义，并报告覆盖率与误报，而不写成零。

生成评估文件支持 NPZ 中 `requested_classes[B]`，或 `--requested-class`。类别会映射成对应文本真正输入 C；CHB 使用两句固定注释模板，其他本体必须在源快照 `class_prompts` 中声明映射。已有文本与类别冲突时拒绝导出。这些标签是请求条件，不能视为患者诊断。`--evaluation-source runs/eval/train.npz` 把源摘要、尺度、checkpoint、代码身份和实际采样预算写入导出。独立条件还需声明 `source_role:train`；局部观测不混入 strict source-only 评估。

`sample --batch-size` 默认为4，在设备传输和参考 CWT 计算之前同步切分张量与逐样本来源。分块噪声种子为 `request_seed + start_index`，分块尺寸与总实际网络调用数有记录；改变它要重新生成配对实验输出。

```bash
spikeweave sample --checkpoint runs/diffusion/last.pt \
  --conditions requests.npz --evaluation-source runs/eval/train.npz \
  --steps 50 --output runs/eval/spikeweave.npz

spikeweave evaluate --stage G1 --train runs/eval/train.npz \
  --validation runs/eval/validation.npz \
  --generated spikeweave=runs/eval/spikeweave.npz \
  --generated jet=runs/eval/jet.npz --generated conditional_vp=runs/eval/conditional_vp.npz \
  --updates 200 --batch-size 32 --ledger runs/eval/gates.json --output runs/eval/g1.json
```

所有方法运行同一独立小 CNN、更新次数、样本暴露预算和同一自然 validation。内置 real-only、重复采样、常规增强；JET 和简单 conditional VP 接受有明确来源和预算的真实运行导出。文件命名不证明忠实复现，缺失基线会标 `incomplete` 且不打开下一闸门。

目录数值导出/开发评估默认每个 split 最多4096窗，选择种子 `--selection-seed 0` 固定，独立于模型/CNN种子。完整身份审计不受此上限影响。报告明确数值子集与观察时长，不把子集 AP 视为完整连续记录结果；需要扩大时显式提高 `--max-windows` 并冻结新的快照，旧闸门不能沿用。

TS-FID、独立物理特征 MMD、近邻和质量检查只用于开发诊断，不用它们单独宣称临床真实。G2 接受三个不同真实生成器训练种子、不同 checkpoint、同数据/代码/配方/训练与采样预算且质量合格的完整 G1 报告；换采样或 CNN 种子不算三次训练。均值胜过最强基线且至少 2/3 配对改善才通过。失败重跑会使后续闸门失效。

G3 当前只登记具名最终测试操作并校验闸门；完整消融、外部医院、统计推断和效率矩阵尚未自动实现或执行。只有达候选 SOTA 后才展开这些科研验证。

正式 BiomedCLIP 权重、正式模型显存、真实临床训练和控制/增强收益仍需实测。CHB 文本是事件注释模板，未提供真实诊断报告/MIL，也不能支持“所有 EEG 泛化”的结论。
