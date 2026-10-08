# SpikeWeave-EEG：固定双分支架构 v0.2

日期：2026-10-07。状态：固定研究设计已实现为v0.2并进行工程验证，临床效果尚未测量。此文取代历史三分支设计；实际入口与边界见implementation-v0.2.md。

2026-10-08：最新数据迁移和按效果分阶段评估汇总于 [project-plan.md](project-plan.md)。低预算开发不改变本文固定双分支和六约束。

## 1. 名称与信息流

SpikeWeave-EEG: Dual-Branch Spiking EEG and Vision–Language Alignment for Montage-Constrained Wavelet Diffusion。

中文：脉冲脑电与图文对齐的导联约束小波扩散模型。

两条顶层表征分支为 E（EEG）与 C（图文条件）。C 内部包含文本/图像预训练前端和共享融合 Transformer，不声称全模型只有两个 encoder。训练目标 EEG 可以进入 E 进行对齐与重构监督；生成器的 EEG 条件只能来自独立 source/support。完整 target CWT 不进入 externally-conditioned generator。

推理：外部文本/期望结构/属性 → C；可选独立 reference → E；条件与随机噪声 → MCWave → signed coefficients → IDWT → EEG。

## 2. 数据与物理表示

主接口为 16 个经核实双极导联、256 Hz、4 秒、1024 点，保留 µV、正负电极、极性、采样/滤波来源与 masks。4 秒仅用于窗口事件和属性生成，不能承载整份患者诊断报告的真值。

数值 CWT 每导联 64×128，频率范围及有效边界服从源带宽和 COI；保留线性功率/对数功率定义、Hz/time坐标与观察mask。禁止频率/时间翻转、随意旋转等改变含义的图像增强。完整 DWT 使用 db4、J=3、periodization，保留 A3/D3/D2/D1 的全部 signed 系数。

归一化只拟合 train/source，采用已实现的 `source_rms_v2` 单一 RMS 标量，模型输入为物理 µV/s，不减均值。新统计写入 `train_scale_v2.json`；v0.1 affine 文件仅供历史逆变换，不能直接混入新生成器。真实完整数据的尺度拟合与验收仍以实际构建报告为准。

## 3. E：STAtten-8-512 EEG 适配版

| 项目 | 固定配置 |
|---|---|
| patch | 每导联连续16个原始样本完整线性投影，不平均池化 |
| 物理推进 | 64步，每步62.5ms；不是整段信号重复4次 |
| 主干 | 8 blocks，512维，8 heads，FFN2048 |
| STAtten chunk | 前4层4步（0.25s），后4层8步（0.5s） |
| 神经元 | Q/K/V和FFN等脉冲位置统一PLIF，膜电位残差 |
| PLIF初值 | 离散tau=2/4/8/16，各128特征；0<beta<1 |
| 时间常数解释 | beta=1-1/tau时，tau_phys=-62.5ms/log(beta)，约90/217/468/969ms |
| surrogate/state | ATan、detach-reset；膜状态和阈值比较FP32 |
| 输出 | spike与膜电位联合读出，保留局部token |
| reference tokens | 每导联4个时间摘要，合计64×768 |
| 对比读出 | semantic/structure两个头，分别512→1024→512，L2 normalize |
| 正则化 | spike普通dropout=0；连续读出dropout=0.1；连续残差drop-path从0到0.1，全时间步共享掩码 |

每条独立记录/reference开始时重置膜状态。保留masked patch重构头，比较patch4/8/16；PLIF与SE-adLIF为神经元对照。STAtten官方设置来自视觉任务；本适配没有已证明的EEG最优性或能耗优势。统一PLIF和有效DropPath须审计实现，不能仅依赖原库配置字段。

来源：[STAtten](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.html)、[官方8-512配置](https://github.com/Intelligent-Computing-Lab-Panda/STAtten/blob/main/conf/imagenet/8_512_200E_t4.yml)、[Meta-SpikeFormer](https://github.com/BICLab/Spike-Driven-Transformer-V2)。

## 4. C：医学图文共享分支

| 部分 | 固定配置 |
|---|---|
| 文本前端 | BiomedCLIP BiomedBERT-base-uncased-abstract，12层/768维/12heads，每段256token |
| 视觉前端 | BiomedCLIP ViT-B/16，12层/768维/12heads；数值CWT patch stem与位置编码适配 |
| 输入适配 | 每导联独立、共享视觉权重；单通道数值patch，保留Hz/time/导联标识及mask |
| 图像局部读出 | 每导联8个时频摘要，合计128×768 |
| 数值属性 | 最多16 tokens，值/单位/范围/有效mask |
| 联合Transformer | 6层/768维/12heads/FFN3072，pre-norm、GELU，attention/FFN dropout0.1 |
| 全局读出 | ALIGN/结构读出分别投影768→1024→512，L2 normalize |
| 生成读出 | 保留局部条件tokens与全局向量，不仅一个CLS |

C(T,I)、C(T,null)、C(null,I)共享同一套C参数，模态存在mask明确。null不等于低功率。初始冻结预训练前端，先训练数值stem/位置适配和联合Transformer，随后小学习率解冻高层。BiomedCLIP医学图文预训练不构成原生EEG CWT能力；验证容量匹配的直接数值patch+共享ViLT对照。

来源：[BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)、[官方配置](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/raw/main/open_clip_config.json)、[ViLT](https://proceedings.mlr.press/v139/kim21k.html)。

## 5. 固定对比学习

完整数学定义、P/N/U关系、正例平均、双向矩阵转置及模式归一化见 [contrastive-protocol.md](contrastive-protocol.md)。

固定语义对齐E↔C(T,I)和E↔C(T,null)，结构对齐E↔C(null,I)。两组projection heads避免把同事件类别等同于同一波形结构。温度0.07；对齐三项权重1/0.5/0.5，按实际有效项权重归一化。

L_representation=L_alignment+1.0 L_masked_patch+0.1 L_spectral。时间patch遮挡率40%，只监督被遮挡有效位置；不加入强脉冲稀疏惩罚。报告对齐使用记录级bag/MIL，不能将整份报告无条件复制为每窗真值。

## 6. MCWave连续值生成器

主力采用DiT-L容量24层/1024维/16heads/FFN4096；XL scaling为28层/1152维/16heads。SNN仅用于表征与独立reference条件；每个扩散状态由连续Transformer处理。

设X∈R^(C×T)，montage矩阵B_m，Q_m∈R^(C×r)为col(B_m)正交基，时间正交DWT矩阵W∈R^(T×T)。

Z0=Q_m^T X W^T；Zs=alpha_s Z0+sigma_s epsilon；alpha_s²+sigma_s²=1。

输入tokenizer前用Q_m Zs回到真实导联的小波系数，再添加正负端电极、子带、真实时间与支持范围。Q的r个坐标是测量模式而非电极。每子带16系数一patch，全部数值进入投影。block采用导联内时序/尺度注意力、跨导联注意力、外部条件cross-attention、adaLN-Zero和FFN。

head先输出C×T系数H_hat，再令Z0_hat=Q_m^T H_hat；X_hat=Q_m Z0_hat W。head无tanh/默认幅值裁剪。attention/FFN/drop-path主配置均0，独立条件缺失训练承担条件正则化。

对16双极导联候选，理想图rank14；必须先核实真实closure residual与gain/filter/reference。QQ^T投影会删除残差，不能称全流程无损或恢复脑源。周期边界与瞬变保真必须评估。

来源：[DiT官方容量](https://github.com/facebookresearch/DiT/blob/main/models.py)、[Min-SNR](https://arxiv.org/abs/2303.09556)、[probability-flow ODE](https://arxiv.org/abs/2011.13456)。

## 7. 生成目标与约束

基本clean-MSE权重w(s)=min(SNR_s,5)。六类软约束全部作用于同一个clean估计，乘g(s)=SNR/(1+SNR)=alpha_s²，按有效元素平均并按source尺度校准。

| 约束族 | 定义 | 初始权重 |
|---|---|---|
| 重建 | clean波形L1，Laplace残差 | 0.10 |
| 统计 | 导联全窗与局部窗均值/标准差 | 各0.05 |
| 动力学 | 完整时间轴Delta X_hat与Delta X的L1差，包含patch边界 | 0.05 |
| 时空相关 | 每导联Pearson及跨导联相关矩阵，稳健分母与有效mask | 各0.05 |
| 小波 | signed details L1与level log-energy差 | 各0.05 |
| 控制 | 提供且有效的局部CWT/数值/区间条件误差 | 0.10 |

L_generation=E[w(s) MSE(Z0_hat,Z0)+g(s) sum(lambda_k L_k)]。软约束不替代疾病状态独立评价。JET原TV作为消融，主方案用差分匹配；JET原实现按原设置复现。类别采样使用train/source计数，inverse-frequency指数1、有放回，固定采样器不称动态adaptive。

## 8. 训练与采样起点

| 阶段 | 新模块lr | 预训练前端lr | 有效batch |
|---|---|---|---|
| A信号重构/视觉适配 | 1e-4 | 先冻结 | 256 |
| B双分支对齐 | 1e-4 | 解冻高层时1e-5 | 256 |
| C生成 | 5e-5 | 初始冻结表征主干 | 256 |

AdamW betas=(0.9,0.95)，表征weight decay0.05、生成0.01；bias/norm/神经元时间常数不decay。warmup5000updates、cosine decay、gradient clip1、generator EMA0.9999、矩阵运算BF16。病例独立验证决定checkpoint与容量，不按3060压缩设计。23个CHB患者不能因切出许多窗口就视为大独立样本库。

生成条件模式：图文40%、仅文本40%、仅图像10%、全语义空10%；非空模式独立reference提供概率25%。全空时同时清除text/image/attributes/reference，必要montage/fs/unit元数据保留。没有真实图文配对不凭空制造新语义。

主采样VP probability-flow ODE+Heun，50积分区间、有限噪声端点后返回clean预测；DDIM同实际NFE对照。CFG初值2，验证扫描1/1.5/2/3；比较控制与多样性。Heun区间数不等于网络评估次数。

## 9. 研究来源与验收

Sketch2CT贡献局部文本-结构引导及全局融合思想；MSADN贡献多尺度条件注入思想。JET贡献clean重建、统计/Pearson/时序约束及类别采样参考；WaveletDiff已有DWT diffusion与能量约束；ELM/CLEF已有临床EEG语言对齐。普通子空间diffusion与ViLT本身不是新贡献。

必须证明SNN相比等规模连续encoder的瞬变/少样本价值，ViLT共享融合相比同信息简单条件的控制收益，以及全过程测量约束相比仅最终投影的效用。实验固定见 [experiments.md](experiments.md)。模型命名不构成创新或临床有效性的证据。
