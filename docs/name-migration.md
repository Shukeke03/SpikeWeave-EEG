# SpikeWeave-EEG 名称迁移记录

日期：2026-10-07。本轮范围为品牌、目录、环境、包装元数据和设计文档，模型/数据算法下一步再改。

| 对象 | 新名称或路径 |
|---|---|
| 项目与模型 | SpikeWeave-EEG |
| GitHub（private） | https://github.com/Shukeke03/SpikeWeave-EEG |
| 本机目录 | /Users/shuxing/AIlearning/SpikeWeave-EEG |
| legion目录 | /home/shuxing/workspace/SpikeWeave-EEG |
| legion数据根 | /home/shuxing/workspace/datasets/SpikeWeave-EEG/chbmit-1.0.0 |
| 模型/处理环境 | spikeweave-eeg |
| legion下载工具环境 | spikeweave-download |
| Python distribution | spikeweave-eeg |
| 新CLI | spikeweave |

双分支：E为hybrid SNN EEG，C为共享医学图文条件。固定多正例双向对比协议见contrastive-protocol.md。

## 兼容范围

旧项目/数据/工具环境绝对路径保留软链接，已有索引、历史启动参数和用户的watch命令保持可达；这些链接不是第二份数据。原始EDF、derived shards、split、normalizer、hash/audit和已有检查点不因品牌更名重写。

src/triwave、TriWaveModel、triwave CLI和triwave-morlet-v1继续作为v0.1兼容标识。新CLI指向同一旧实现；不声称更名已经完成SNN双分支实现。configs/tiny、legion及research-5090仍属于原型算法配置。

旧三分支架构保存在architecture-v0.1-legacy.md。历史工程验证在engineering-validation.md，不作为新模型效果。

## 迁移验收

已核实 GitHub private；本机新前缀 Python 3.11.17 / Torch 2.8.0、editable distribution与新CLI、pip check通过。legion模型环境使用原有27个已提取Conda缓存包经官方clone/link流程迁移，Python 3.11 / Torch 2.8.0+cu128与CUDA可用。此迁移仅在一次进程中绕过失败的fetch步骤，没有修改Conda源码，没有重新下载PyTorch。旧前缀保留软链接以兼容历史脚本。

迁移前commit为9badba58c146454db56fbbbff40e6f9c33ebb1be；src/scripts/tests/算法configs/依赖文件与该commit保持一致。文档本地链接与git diff --check通过。服务器editable安装指向新目录，旧distribution元数据已归档；新CLI的CUDA doctor与pip check通过。下载工具环境经原有15个缓存包迁移，aria2 1.37.0可执行，无网络重下载。两机conda run按新环境名启动成功；旧项目/数据/环境路径samefile检查通过。

本机与legion的origin均指向新私有仓库；名称与设计提交37cb666已在两机同步。其后的本文件验收记录提交也通过push与服务器fast-forward同步核验。
