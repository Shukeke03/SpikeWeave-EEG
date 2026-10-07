# SpikeWeave-EEG 推进顺序 v0.2

2026-10-07：已确立双分支架构与对比协议；本轮完成名称、GitHub、环境与路径迁移。旧v0.1代码和工程验证保留，新算法尚未实现。

## 当前步骤：名称与协议冻结

固定SpikeWeave-EEG名称、E/C两分支、共享图文Transformer、多正例双向语义/结构读出。迁移项目/数据目录、Miniforge环境和remote，保留旧绝对路径兼容。

验收：仓库private；本机和服务器相同commit；新环境可import原实现、启动新CLI；src/scripts/tests/算法configs与迁移前一致；数据未因更名复制或重新处理。

## 下一步骤：用户授权后修改算法代码

按architecture.md实现STAtten-PLIF EEG适配、BiomedCLIP数值图文适配、ViLT式共享分支与固定对比loss。随后实现montage子空间、clean预测VP与六类约束。旧namespace/CLI/checkpoint compatibility单独处理，数据协议ID不能仅因branding随意修改。

验收：独立source/support与target路径隔离、masked多正例关系正确、物理时间推进、spike/membrane读出、全模态空路径、DWT round-trip、真实closure residual、单位与mask一致。

## 数据验收与真实基线

核查CHB-MIT加工报告、患者/事件划分、冲突导联隔离、source-only统计与实际容量；不启动持续监控。TUSZ/TUEV等仅在实际可访问后接入；模板事件文本不是自由临床报告。

复现JET、WaveletDiff与同条件普通diffusion，匹配真实数据/属性/support预算和NFE。测SNN必要性、图文融合收益和最终投影反证。临床效果及付费服务器训练另行执行。

## 后续扩展

以独立队列、疾病和montage证据逐项扩展，不从单一4秒窗/23患者结果推出所有EEG泛化。生成后真实检测/分类提升、多样性和记忆风险同时验证；投稿贡献由结果决定。
