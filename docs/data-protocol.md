# 双分支数据与条件协议 v2

更新：2026-10-08。本文记录已确认的数据设计，实施顺序和“未达到 baseline 不启动完整评估”的门槛以 [project-plan.md](project-plan.md) 为权威主方案。历史下载和构建测量见 [dataset-chbmit-v1.md](dataset-chbmit-v1.md)；本次文档更新未执行新的服务器核查、下载或处理。

模型固定两条表征分支：`E(X)` 是 hybrid SNN EEG encoder；`C(T,I,A)` 是共享医学文本、数值时频结构与属性条件 encoder。`C(T)`、`C(I)` 和联合输入复用 C 参数，属性 A 进入同一条件分支。semantic/structure projection heads 是读出，不新增 encoder 分支。EEG派生CWT是结构视图，文本模板是已有注释的表达，不构成独立采集的新医学模态或新的临床证据。对比关系遵循 [contrastive-protocol.md](contrastive-protocol.md)。

| 状态 | 范围 |
|---|---|
| 已有实现，实际文件完成度仍需审计报告 | 可信16双极导联转换、256 Hz连续物理µV shards、4秒窗口索引、患者split lock、两类官方标注模板、训练患者参考索引、`source_rms_v2`、完整signed DWT与数值CWT在线计算 |
| 已有数值实现，掩码待升级 | `triwave-morlet-v1` pooled mask目前是至少一个贡献有效；v2要求全部贡献有效并单独记录mask版本 |
| 已确认设计，尚未生成 | 七张Parquet侧表、16项teacher属性、独立request bank、部分观测条件及属性采样入口 |
| 待统一 | 属性提取与可微控制loss的共同Hann rFFT能量定义、五带及总分母的`[0.5,80)`口径 |
| 尚未获得，不启用 | 自由临床报告及相应使用授权；模板不能改称医生报告 |

## 1. 首版数据安排

| 数据源 | 已有监督 | 作用 | 不能推出的结论 |
|---|---|---|---|
| CHB-MIT | 发作起止、montage、部分人口信息 | 首先完成的原型；来源内未见患者测试 | 非发作片段不等于健康对照；不能伪造发作亚型 |
| TUSZ | 发作起止、类型、通道标注 | 第二阶段，获准访问后的主要事件生成队列 | 记录中有发作不代表所有窗口发作 |
| TUEV | 片段级病理/背景/伪迹事件类别 | 第三阶段，扩展事件类型和困难负例 | 波形事件类别不等于完整疾病诊断 |
| TUEP | 经神经科医师判断的癫痫/非癫痫及相关 metadata | 后续患者级疾病条件 | 非癫痫不等于健康；诊断不等于当前窗口发作 |
| TUAB | 记录级正常/异常 | 辅助记录异常性评价 | 不能作为特定疾病或每窗口异常的真值 |

来源：[CHB-MIT](https://physionet.org/content/chbmit/1.0.0/)、[TUH 官方](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)。

TUH 当前不随语料分发临床报告。官方提供报告检索代运行服务，不等于向新申请者提供全文。首版以官方标注生成模板文本；真实报告只有实际取得并明确使用权限后才启用。

公开可申请与可重新分发不同。TUH 访问表约束数据再分发；只发布自身流程、协议与允许的派生资料，不把原始临床数据重新打包。[访问表](https://isip.piconepress.com/projects/nedc/forms/tuh_eeg.pdf)

## 2. 资产、关系与样本

底层保存三个层次：

1. 资产：EEG 原记录/片段、文本、结构视图。
2. 关系：真实配对、标注作用范围、派生关系和参考关系。
3. 样本：在某一训练/推理协议下选出的资产及有效 masks。

概念字段如下，不是目前已经存在的实际数据文件：

| 字段组 | 字段 |
|---|---|
| Provenance | dataset/version、source_file_id/hash、subject/session/run/record/event、access terms |
| EEG | start/end、sample rate、unit、lead names、positive/negative endpoints、reference、valid masks、quality flags |
| Annotation | raw label、canonical namespace、scope、onset/offset、channel coverage、negation、confidence、truncation |
| Text | content、role、source、template/model version、supported labels、scope、review status |
| Wavelet | parent EEG interval、wavelet、frequencies/scales、boundary mode、time/frequency coordinates、normalization、valid region |
| Relation | source/target asset IDs、relation type、positive group、evidence、scope、weight |
| Condition | role、origin ID、target-derived flag、inference availability、visible mask、desired vs observed attribute |
| Evaluation | patient/event/record split、source/support/query role、protocol version |

角色至少包含：`target_waveform`、`target_view`、`reference_waveform`、`structure_condition`、`annotation_template`、`clinical_report`、`physiology_attribute`。特别区分以下三种视图：

| 角色 | 来源和允许用途 | 信息边界 |
|---|---|---|
| `target_teacher` | 合法训练目标完整CWT、DWT、属性、clean embedding；用于表示配对和监督 | 完整目标CWT/clean embedding不进入盲生成条件；目标训练属性须显式声明 |
| `reference_condition` | 独立合法source或计入预算的support EEG及其CWT/属性 | donor与query、同一事件近邻和处理上下文隔离；参考不是完整输出复制真值 |
| `partial_observation` | 明确开放的目标局部波形/结构 | 只用于observation-conditioned协议，记录可见预算；不归入盲生成结果 |

新增目录为 `processed/dualbranch-v2/`，引用符合既定参数的v1物理shards，不复制整库EEG。七张待生成侧表固定如下，**当前不表示文件已经存在**：

| 文件 | 主键及内容 |
|---|---|
| `records.parquet` | 来源版本、EDF/hash、原case与canonical patient、record、split、处理区间、单位/导联/参考/极性/带宽/版本 |
| `windows.parquet` | window ID、record、offset/1024点、患者/事件、QC、交集与截断、source/support/query及允许上下文 |
| `texts.parquet` | text ID、原标签/canonical namespace、语义组、scope、否定/未知、时间范围、来源与模板版本 |
| `views.parquet` | view ID、父窗口、teacher/reference/observation角色、数值算子、坐标、normalizer摘要、有效/可见mask |
| `attributes.parquet` | window ID、teacher属性、可选逐导联值、单位/有效性/范围、算法与聚合、源波形与作用域 |
| `relations.parquet` | 资产配对与患者/母记录/事件分组证据；batch内动态P/N/U，不存全局N²矩阵 |
| `condition_specs.parquet` | request ID、协议、文本/结构/属性/参考来源、desired/observed、target-derived、推理可用性、可见预算与采样seed |

患者身份嵌入records/windows，原始标注及scope嵌入windows/texts并保留溯源；不另增替代views/relations的侧表。另保存 `train_scale_v2.json`、`montage_audit.json`、`dataset_audit_v2.json`。物理处理、尺度和条件策略分别指纹化，避免改变参考策略就重滤整库。缺失字段由有效mask表示，不能以全零冒充实测零。

## 3. 标注粒度与文本

统一 ontology 先保留原标签，再映射为 event type、patient diagnosis、record abnormality、measurement/context。避免将所有内容塞进同一个疾病类别。

患者级病史文本与记录/窗口事件文本不同。TUEP患者诊断、TUAB记录异常标签不广播为每窗真值。长报告需分角色与片段集合聚合或bag/MIL；当前没有已取得的自由报告及可验证聚合入口。报告未提到一个现象，不等于每个片段没有该现象。

初版短模板可以表达“本窗口含已标注发作，覆盖区间为……”。类型、涉及导联等字段仅在原标注提供时启用。模板需有多种表述，测试描述改写；但不能将多个改写当作新的独立临床信息。

由 EEG 计算的属性注明 `computed_from_signal`、单位、算法与有效范围，分开保存 `teacher_attributes`、`condition_attributes` 和 `evaluation_attributes`。训练条件可以由source训练目标的属性构造，需声明 `target_derived=true`、仅训练可用及其来源；部署/测试时相同属性必须来自用户指定、合法参考或支持集，不能依赖未观察的查询目标。

目标测得的属性可作为独立评价真值。若专门评价属性条件生成，说明提供了哪些属性及其来源，不把它称为完全无目标信息的生成。

当前普通manifest loader阻止target-derived属性直接进入条件，sharded loader属性valid全部为空；v2显式训练属性入口尚待实现。不能把来源flag改成false来绕过检查。

## 4. 处理流水线

### 4.1 数据库存与划分

先检查版本、文件、patient IDs、事件及子库重叠，再锁定 train/validation/test。TUH 各子库来自共同母库，需要跨子库排重。CHB-MIT 不同 case 可能属于同一人，不能把 case ID 一律当患者 ID。

若CHB-MIT用于原型开发，应在读取开发EEG前保留锁定患者，后续只能将其称为该来源的未见患者测试。CHB来源已参与设计/调参时，不再称为完全未见数据来源；严格unseen-source需另一个从未参与开发的合法队列。

当前CHB划分为16训练、3验证、4测试独立患者，测试为chb04/06/18/24；大量窗口和重复采样不增加独立患者数。source-only下测试患者EEG不参加表示预训练、生成器预训练、统计拟合或support。

生成器预训练、表示预训练、downstream few-shot、support adaptation 和 query test 分别记账。不得先用全量目标患者记录预训练，再把 detector 的少量标签预算作为整个系统的少样本预算。

### 4.2 物理波形与导联

保留原 EDF 和全部解析记录到 Git 外受控目录。解析单位、原导联极性、参考和 sampling rate。标准别名只在明确对应时转换，例如 T3/T7；不能只用字符串相似性猜导联。

主原型复用可核实的16标准双极导联、256 Hz、float32 µV连续shards和4秒/1024点不重叠索引。来自共参考信号的差分计算与已有双极导联选择各记录来源。若存在不同参考、硬件/滤波或导联缺失，保存对应masks/protocol，不能假设来源转换完全消除了domain shift。

### 4.3 滤波、重采样与质量

沿用整记录四阶Butterworth `0.1–80 Hz`、`sosfiltfilt`，必要时 `resample_poly`。在允许的连续记录/时间块内执行可追溯滤波/抗混叠再重采样，避免逐窗口滤波引入边界。支持与查询优先按完整记录隔离；必须同记录划分时，先锁允许时间块并分别处理。零相位IIR无严格有限支持，保护间隔按响应尾部容差审计，并覆盖重采样和全部CWT上下文。支持侧不得读取查询，即使只是无标签上下文。保留source bandwidth，80 Hz附近已有滤波衰减，不声称原始全频段无损。工频处理与源地区/采集说明匹配，不默认同时去50与60 Hz。

质量标记包括 flatline、clipping、坏导联、缺失区间和强非脑电干扰。病理尖波、发作瞬变不可仅因高振幅/高频被自动删除。无临床专家新标注时不将算法 QC 结果冒充专家判断。

尺度固定为已实现的 `source_rms_v2`：只用合格train/source有效样本拟合 `s=sqrt(sum(x_uV**2)/N)`，`x_model=x_uV/s`，反变换为 `x_uV=s*x_model`。`mean=0`，不减均值；历史 `std` 字段承载RMS尺度，不是中心化标准差。保存有效计数、训练区间、索引/来源摘要与µV单位。初版不逐窗z-score，不跨source/query混合白化。

v2明确尺度文件为 `processed/dualbranch-v2/train_scale_v2.json`。历史 `train_normalizer.json` 先核实 `scale_protocol` 与source摘要；旧 `legacy_affine_v1` 要重拟合，不能只改字段名。物理shards可复用；旧尺度的小波缓存/条件编码与checkpoint不可直接混用。

montage保留明确端点矩阵B及列空间Q；16导联主配置理想rank14，Q坐标是测量模式，不是电极。原始/处理后都审计 `e=X-QQᵀX` 的µV RMS、相对值、分位数、最大值和全source覆盖率。当前ModelConfig相对阈值0.01与类默认1e-3尚有区别，1%仅是待实测确认的工程闸门；默认audit，不静默投影删残差。近零波形兼查物理/量化误差，全体候选数据审计完成后才考虑hard，单批通过不代表整库通过。

### 4.4 窗口与事件

4 秒窗口是原型；记录 event overlap、within-window onset/offset、是否左/右截断、上下文与相邻关系。相邻/重叠窗口按患者、记录和事件分组；不能随机散入不同 split。

若主张完整事件持续时间，必须选能覆盖事件的长度与标注，或单独长时生成协议。短窗口内覆盖时间不能写成整个发作持续时间。

下游检测实验保留正常/非发作连续背景，使用真实测试持续时间计算每小时误报。只在平衡事件小切片分类的结果不足以支持临床检测效用。

### 4.5 CWT/DWT

CWT首版固定独立4秒窗，每导联64×128数值张量：`triwave-morlet-v1`，Morlet `omega0=6`，64个对数频率2–80 Hz，128时间格，离散核截断±4尺度/去DC/单位L2归一化，线性零填充FFT卷积；先池化线性 `|W|²`，再 `ln(power+1e-12)`。在源RMS尺度上计算模型视图并绑定尺度指纹，不逐图MinMax、不JPEG、不任意resize224×224。彩色图只用于查看。

保存COI、内部缺失核支持、原带宽和Hz/time/端点几何。v2严格 `mask_protocol=cwt-all-contribution-v2`：池化格要求全部贡献位置有效，16×16patch要求全部覆盖格有效。**这是待实现升级**，当前pooled mask是至少一个贡献有效；数值核未变时保留 `triwave-morlet-v1`，mask协议单独版本化，旧mask不能仅改metadata复用。

teacher、reference与生成预测均遵守同一4秒算子/边界，不读隐式邻窗上下文，也不宣称低频整窗可靠。未来加上下文另建协议：2 Hz完整截断核约需左右各1.91秒，在允许区间先算、裁剪中心4秒、再池化；不能把更长区间直接压成同一128格。上下文计入实际可用预算，不能通过CWT/滤波读入query。

DWT 输出所有 signed 子带，保存 extension 和长度。1024 点、db4、J=3、periodization 的 128/128/256/512 长度只属于该配置。测试 IDWT 数值重建及端点事件。

部分观测协议仅从全部有效的16×16数值CWT patches中开放约20%，记录实际比例、seed及 `teacher_valid/visible_mask`；未开放和无效位置填中性值并传mask。这是结构patch观察预算，不等于只读取20%原始EEG：CWT格可能依赖宽时域。限制原始波形预算时须从合法波形及核支持重算结构。该入口尚待实现，盲生成不开放目标patch。

### 4.6 合法生成条件

- 文本来自实际标签/允许报告或用户指定条件。
- 参考 EEG 来自允许支持集，与 query/目标及其预处理上下文隔离；同患者近邻也需排除同一事件并按处理支持范围设置保护间隔。
- 部分结构来自用户编辑、参考/支持集或明确的部分观测任务。
- full target CWT 只存为 teacher/target view；实际 code path 不允许它出现在外部条件列表。
- waveform sampling frequency、montage、单位尺度属于必要测量定义，不作为可随意丢弃的语义条件。

当前v1 `ref_index` 在独立训练患者中优先按目标真值标签选donor，只适用于声明类别已知的训练/条件实验。v2在 `condition_specs.parquet` 中另建独立request bank：先由用户/实验协议给出请求类、文本、属性及结构可用性，再检索合法source/support。无请求类别时用与隐藏query标签无关的固定策略；不能读取query波形、属性、完整CWT或真值类别挑donor。保存请求来源、donor patient/record/event、seed和预算。query标签只作独立真值及结果分层。

### 4.7 16项属性的共同算子（待统一）

属性从处理后的物理µV波形逐导联计算，再在有效导联等权平均；可保留逐导联值。谱估计临时减去各导联时间均值仅用于测量，不改变模型输入。固定1024点周期Hann：`energy[k]=abs(rFFT((x-mean(x))*h))[k]**2/sum(h**2)`；沿用现有能量口径，**不是校准的单边PSD密度**，不新增单边2因子或Hz分母。

频带均为左闭右开：δ[0.5,4)、θ[4,8)、α[8,13)、β[13,30)、γ[30,80) Hz；总功率/能量分母统一 `[0.5,80)`，同一离散频格求和。提取器、控制loss和核查共用算子。当前loss总support含80 Hz、band不含80 Hz，需要统一；本次文档不表示代码已改。

| schema字段（按固定顺序） | 定义 |
|---|---|
| `spectral_centroid_hz` | `[0.5,80)`能量加权频率中心 |
| `rms_uV` | `sqrt(mean(x²))` |
| `std_uV` | 原x的总体标准差，分母N |
| `mean_uV` | 原x时间均值 |
| `delta_fraction`、`theta_fraction`、`alpha_fraction`、`beta_fraction`、`gamma_fraction` | 各带能量/共同总能量，依次占5个slot |
| `line_length_uV` | 相邻样本绝对差均值，分母N−1；不是差分总和 |
| `delta_log_power_uV2`、`theta_log_power_uV2`、`alpha_log_power_uV2`、`beta_log_power_uV2`、`gamma_log_power_uV2` | 对应带 `ln(energy+1e-8)`，依次占5个slot；保留历史字段名与算法口径，不冒称校准PSD |
| `peak_abs_uV` | 每导联`max(abs(x))`后导联等权平均，不是跨导联最大值 |

谱分母近零/源带宽不足/没有有效导联时置无效，保存有效范围、保护阈值、单位、聚合和算法版本，不把clip后的比例冒作可靠实测。属性不推断无标注疾病。

待实现的非空生成条件采样：50%不给属性；其余从有效属性随机提供1–4项（均匀选择合法数量及字段），无有效项则空。值保留物理单位，编码尺度只在source拟合；全空CFG样本清空属性和参考。此策略不是当前trainer已有行为。

## 5. 三套评估协议

| 协议 | Target EEG 可用性 | 能声明什么 |
|---|---|---|
| Strict source-only | 锁定目标患者/队列任何 EEG 不参与训练、归一化、support | 源域训练到新患者/来源的迁移 |
| Few-shot support | 明确 K 条目标支持记录/事件；query 全部锁定 | 指定支持预算下适配和增强 |
| Observation-conditioned | 明确观察目标的部分波形/结构 | 重建/inpainting；不能替代盲生成结论 |

普通域预训练无标签目标 EEG 也属于使用目标数据，单独记录 transductive setting。不同协议不可混为同一个 zero-shot 结果。

CHB-MIT→TUSZ 或反向的外部测试同时改变人群、设备、montage 和采集策略。可以检验跨来源迁移，但不能单凭两队列结果把改善归因于医院差异或声称所有患者/疾病适用。

## 6. 数据质量验收

实施后需要实际产出的统计：患者/记录/事件数、每类分布、来源与年龄等可用混杂分布、缺失导联、质量淘汰理由、标签作用范围、可用结构频率、重复率和 split overlap。

数值验收包括单位/导联极性检查、波形反归一化、DWT 重建、结构 masks、窗口—标注时间对齐及实际条件来源审计。没有这些统计前，不声称已经建立高质量临床三模态数据集。

## 7. 复用、缓存和实施门槛

| 资产 | v2安排 |
|---|---|
| 原EDF/hash、库存、患者split与可信导联证据 | 复用并核对源版本，不因双分支重新下载 |
| 符合既定参数的物理µV连续shards/窗口/隔离索引 | 复用，历史参考索引另标类别已知边界；不因新增侧表重滤 |
| normalizer | 验证source_rms_v2与摘要，否则重拟合并写v2明确文件 |
| 七张侧表、teacher属性、独立request bank | 待首次生成与验收 |
| 小波缓存 | DWT在线；CWT在线或限额按需缓存；mask/尺度变更后重建相关派生缓存 |

每4秒窗EEG16×1024 float32为64 KiB，完整DWT同为64 KiB；CWT16×64×128 float32为512 KiB，逐元素bool mask另128 KiB。100万窗仅CWT+mask约655.36 GB/610.35 GiB，首版不全量持久缓存。16个float32属性+16bool每窗约80 bytes，不含ID/Parquet元数据。

连续shards加索引避免波形复制；参考CWT按母窗口复用，缓存key包含source SHA、区间/角色/上下文、滤波、小波数值及mask版本、normalizer摘要。mask可以位压缩，float16缓存须验证误差后采用。最终体量与可用窗口数必须实测，不能以理论估计冒充已建库统计。

未达到 [project-plan.md](project-plan.md) 的baseline门槛前，不启动完整评估。本次只更新文档，不启动付费训练、服务器数据处理或持续监控；既有会话授权按用户指示执行。CHB模板与派生CWT的原型不能宣称自由报告理解、广泛疾病泛化、完整发作生成或临床有效性。
