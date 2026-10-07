# 数据与条件协议 v0.1

当前仅定义数据规范，未下载或加工任何临床数据。

## 1. 首版数据安排

| 数据源 | 已有监督 | 作用 | 不能推出的结论 |
|---|---|---|---|
| CHB-MIT | 发作起止、montage、部分人口信息 | 可直接启动数据/模型原型；候选外部验证 | 非发作片段不等于健康对照；不能伪造发作亚型 |
| TUSZ | 发作起止、类型、通道标注 | 获准访问后的主要事件生成队列 | 记录中有发作不代表所有窗口发作 |
| TUEV | 片段级病理/背景/伪迹事件类别 | 扩展事件类型和困难负例 | 波形事件类别不等于完整疾病诊断 |
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

角色至少包含：`target_waveform`、`target_view`、`reference_waveform`、`structure_condition`、`annotation_template`、`clinical_report`、`physiology_attribute`。

## 3. 标注粒度与文本

统一 ontology 先保留原标签，再映射为 event type、patient diagnosis、record abnormality、measurement/context。避免将所有内容塞进同一个疾病类别。

患者级病史文本与记录/窗口事件文本不同。长报告需分角色与片段集合聚合；报告未提到一个现象，不等于每个片段没有该现象。

初版短模板可以表达“本窗口含已标注发作，覆盖区间为……”。类型、涉及导联等字段仅在原标注提供时启用。模板需有多种表述，测试描述改写；但不能将多个改写当作新的独立临床信息。

由 EEG 计算的属性注明 `computed_from_signal`、单位、算法与有效范围。训练条件可以由训练目标的属性构造；部署/测试时相同属性必须来自用户指定、合法参考或支持集，不能依赖未观察的查询目标。

目标测得的属性可作为独立评价真值。若专门评价属性条件生成，说明提供了哪些属性及其来源，不把它称为完全无目标信息的生成。

## 4. 处理流水线

### 4.1 数据库存与划分

先检查版本、文件、patient IDs、事件及子库重叠，再锁定 train/validation/test。TUH 各子库来自共同母库，需要跨子库排重。CHB-MIT 不同 case 可能属于同一人，不能把 case ID 一律当患者 ID。

若CHB-MIT用于原型开发，应在读取开发EEG前保留锁定患者，后续只能将其称为该来源的未见患者测试。CHB来源已参与设计/调参时，不再称为完全未见数据来源；严格unseen-source需另一个从未参与开发的合法队列。

生成器预训练、表示预训练、downstream few-shot、support adaptation 和 query test 分别记账。不得先用全量目标患者记录预训练，再把 detector 的少量标签预算作为整个系统的少样本预算。

### 4.2 物理波形与导联

保留原 EDF 和全部解析记录到 Git 外受控目录。解析单位、原导联极性、参考和 sampling rate。标准别名只在明确对应时转换，例如 T3/T7；不能只用字符串相似性猜导联。

主原型采用可核实的 16 标准双极导联。来自共参考信号的差分计算与已有双极导联选择各记录来源。若存在不同参考、硬件/滤波或导联缺失，保存对应 masks/protocol，不能假设来源转换完全消除了 domain shift。

### 4.3 滤波、重采样与质量

在允许的连续记录/时间块内执行可追溯的滤波/抗混叠再重采样，避免逐窗口滤波引入边界。支持与查询优先按完整记录隔离；必须同记录划分时，保护间隔覆盖滤波器支持范围、重采样和全部CWT上下文。支持侧处理不得读取查询区间，即使只是无标签上下文。频率范围按任务和原数据确定，保留 source bandwidth。工频处理与源地区/采集说明匹配，不默认每条信号同时去 50 和 60 Hz。

质量标记包括 flatline、clipping、坏导联、缺失区间和强非脑电干扰。病理尖波、发作瞬变不可仅因高振幅/高频被自动删除。无临床专家新标注时不将算法 QC 结果冒充专家判断。

global scale/任何统计参数只在训练集拟合。物理版本保留 µV；模型版本能明确反归一化。初版不进行逐窗口 z-score，也不跨 source/query 混合估计白化。

### 4.4 窗口与事件

4 秒窗口是原型；记录 event overlap、within-window onset/offset、是否左/右截断、上下文与相邻关系。相邻/重叠窗口按患者、记录和事件分组；不能随机散入不同 split。

若主张完整事件持续时间，必须选能覆盖事件的长度与标注，或单独长时生成协议。短窗口内覆盖时间不能写成整个发作持续时间。

下游检测实验保留正常/非发作连续背景，使用真实测试持续时间计算每小时误报。只在平衡事件小切片分类的结果不足以支持临床检测效用。

### 4.5 CWT/DWT

CWT 在允许的带上下文区间计算，数值张量和有效边界 mask 为主，彩色图用于查看。上下文计入实际数据可用预算，支持窗口即使不与查询窗口重叠，也不能通过CWT/滤波读入查询。查询评估只在其所属锁定时间块内预处理。保存母小波、频率、log/线性功率、尺度及物理坐标。

DWT 输出所有 signed 子带，保存 extension 和长度。1024 点、db4、J=3、periodization 的 128/128/256/512 长度只属于该配置。测试 IDWT 数值重建及端点事件。

### 4.6 合法生成条件

- 文本来自实际标签/允许报告或用户指定条件。
- 参考 EEG 来自允许支持集，与 query/目标及其预处理上下文隔离；同患者近邻也需排除同一事件并按处理支持范围设置保护间隔。
- 部分结构来自用户编辑、参考/支持集或明确的部分观测任务。
- full target CWT 只存为 teacher/target view；实际 code path 不允许它出现在外部条件列表。
- waveform sampling frequency、montage、单位尺度属于必要测量定义，不作为可随意丢弃的语义条件。

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
