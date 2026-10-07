# SpikeWeave-EEG 固定双分支对比协议 v0.2

状态：2026-10-07正式确定，下一阶段按此实现；本轮未修改loss代码。

## 1. 分支、读出与运行模式

两条分支为 E_phi(X) 和 C_theta(T,I)。C(T,I)、C(T,null)、C(null,I)严格共享同一套文本前端、视觉前端和联合Transformer参数。null与presence mask明确，三种输入模式不构成三分支。

E与C各有semantic和structure两个512维投影头，输出L2 normalize。semantic投影对应可靠临床事件/描述语义；structure投影对应真实配对的局部时频实例。投影头不新增encoder分支。

## 2. 固定关系矩阵

为每个head分别建立P（已知正例）、N（已知有效负例）、U（未知/不适用）三值关系。不能只按batch对角线默认正例，其余默认负例。

- 语义P：标签作用范围一致、可靠并且规范化条件等价的事件/文本组；重复同义模板属于同组。CHB未标注发作不解释为健康或疾病阴性。
- 语义N：有可靠证据、在相同作用范围内可区分或互斥的语义条件。报告未提及属性、标签不完整或scope不匹配的关系归U。
- 结构P：真实EEG–CWT实例配对、同实例的有效视图或经核实结构等价组。相同seizure类别不自动成为结构P。
- 结构N：允许的不同结构实例，排除同源复制、重叠窗及未经核实的强相关近邻；后者归组或归U。
- P/N组、模板ID、parent EEG、患者/事件/记录/上下文和作用范围必须可审计。患者/医院ID用于划分与审计，不作为语义输入。

仅文本候选按规范化语义组去重，每组保留一个描述；联合图文候选保留不同图像实例，完全重复的图文资产对去重。损失按anchor组等权汇总以控制anchor频率偏置；不同实例的候选数量仍影响softmax分母，不声称完全复制不变。记录级报告只通过bag/MIL对齐，不当作每个4秒窗的真值。

## 3. 固定数学目标

对给定head与输入模式r，a_i、b_j^r为归一化表征，s_ij^r=a_i^T b_j^r/tau；tau固定初值0.07。M_ij^r=1仅当关系为P或N，U在分子和分母中均排除。

采用各正例分别取log-softmax后平均的多正例目标：

\[
\ell_i^r=-\frac1{|P_i^r|}\sum_{j\in P_i^r}
\log\frac{\exp(s_{ij}^r)}{\sum_{k:M_{ik}^r=1}\exp(s_{ik}^r)},
\qquad
L_{E\to C}^{r}=\frac1{|G_r|}\sum_{g\in G_r}\frac1{|A_{r,g}|}\sum_{i\in A_{r,g}}\ell_i^r.
\]

A_r仅含至少一个正例和一个有效负例的anchor。G_r为至少含一个有效anchor的组集合；A_{r,g}为组g的有效anchor集合。semantic按规范化语义组分组，structure按配对实例组分组；先组内平均，再组间等权。每个anchor的有效正例等权。反向损失转置P/N/M矩阵，并独立构造条件anchor组，重新计算组内与组间平均，不能复用正向行计数。

\[
L_{\mathrm{sym}}^r=\tfrac12(L_{E\to C}^r+L_{C\to E}^r).
\]

禁止以“将所有正例exp先求和、再取一次log”的positive-set objective代替此目标。该形式只需命中容易正例，与本协议逐正例对齐的目的不同。

## 4. 固定组合与缺失处理

三个固定项：

| 项 | head | 权重 |
|---|---|---:|
| E↔C(T,I) | semantic | 1.0 |
| E↔C(T,null) | semantic | 0.5 |
| E↔C(null,I) | structure | 0.5 |

仅两个方向均有有效anchor时，该双向项进入R_valid；任一方向为空则跳过整项并记录原因，不把空方向设为0后继续除以2。各项先按上述有效组规则平均，再按当前批次实际有效项的权重归一化：

\[
L_{align}=\frac{\sum_{r\in R_{valid}}\lambda_r L_{sym}^{r}}{\sum_{r\in R_{valid}}\lambda_r}.
\]

无有效项时对比loss为0，并记录有效anchor/正负关系数；不能从全空输入生成伪对比样本。全空C仅用于扩散CFG训练。EEG reference为空时，训练target仍可作为对比anchor，其clean编码不进入生成条件。

每次训练记录各模式loss、有效anchor、P/N/U数量、语义组/结构实例数量和temperature。不能仅报告自身embedding相似度；text-only检索、图像结构检索、文本改写/否定、模式打乱以及真实生成控制分别评价。

## 5. 重构与泄漏边界

固定表征目标：L_representation=L_align+1.0 L_masked_patch+0.1 L_spectral，时间patch遮挡率40%。masked reconstruction只监督被遮挡且有效的位置；spectral损失使用明确源尺度和有效区域。全局semantic对齐不替代局部特征保真。

训练源EEG及自身full CWT可用于表征正对/教师。生成器只读取外部描述、用户指定结构/属性或独立合法support；full target CWT、target clean编码和锁定query统计不可作为生成条件。

对比主协议已固定；是否移除某个项仅作为明示消融，不能在主结果中悄然更换目标。实际临床性能待实现和实验验证。
