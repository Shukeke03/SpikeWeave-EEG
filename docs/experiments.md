# SpikeWeave-EEG 固定实验协议 v0.2

状态：设计已确定，真实临床效果未测。双分支定义与loss遵循architecture.md和contrastive-protocol.md；旧三分支仅作历史原型/架构对照。

## 固定主张与反证

| 假设 | 必要比较 | 推翻或削弱结论的结果 |
|---|---|---|
| SNN有额外表征价值 | 预算匹配CNN/连续时空Transformer、无spike但有leaky state、spike-only/混合读出 | 收益仅来自连续旁路/参数量，或瞬变保真无改善 |
| 图文共享交互提高控制 | 同条件数值FiLM、直接patch共享ViLT、BiomedCLIP前端+共享融合 | 增加条件信息有效，但融合不优于简单方法 |
| 固定对比loss有价值 | 无对比、semantic/structure逐项消融、模态打乱、文本改写与否定 | 仅自身相似度提高，独立控制无改善或多样性下降 |
| 全过程测量约束有价值 | 无约束、软closure、仅最终QQ^T投影、全过程子空间 | 最终投影同样有效，或真实残差不支持严格子空间 |
| 生成有真实增强价值 | real-only、常规增强、JET、WaveletDiff、同条件diffusion | 独立真实query无收益，或重复/记忆训练事件 |

## 预算与划分

患者/事件/记录分组先于切窗与统计拟合。CHB chb01/chb21为同一患者。source-only、few-shot support和partial-observation协议分别报告；锁定query不得作为结构条件、对比预训练数据或归一化来源。

基线获取相同文本/标签/属性、相同support和真实数据；信息增加与架构收益分开。比较采样成本时记录实际NFE、CFG双分支计算与wall time，不只列steps。

## 评价

- 测量/结构：closure residual、µV幅度、差分/尖波、全时间轴边界、signed DWT、PSD、跨导联相关/相位与coherence。
- 控制：期望属性误差、局部时间/频率与导联范围、独立事件classifier、文本-only与image-only表现。
- 表征：患者独立检索、报告bag匹配、同义模板与未知关系审计；不能以训练头相似度独证效果。
- 多样性/记忆：同条件重复生成、训练样本最近邻、事件级近邻与成员风险检查。
- 实用性：锁定真实检测/分类；连续事件检测报告误报/小时，不能只用平衡切片分类替代。

## 必要容量与机制消融

SNN patch4/8/16与PLIF/SE-adLIF；共享融合与简单融合；对比三项权重固定主方案，移除为显式消融；raw/DWT域；约束逐组开关；DiT中/大/XL容量与患者独立验证。首版不将RAE、REPA或对抗蒸馏全部默认堆入。

原始JET按其公布loss/采样设置复现，单列代码与论文不一致之处。已有WaveletDiff、ELM/CLEF与SpikeWavformer贡献准确归属，不将模块组合称首次或预先承诺顶会。
