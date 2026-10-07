# 文献依据与迁移边界

核查截止：2026-10-07。已核实的论文结果与本项目待验证设计分开。论文公开、代码可见、代码许可和数据许可是不同事项。

## 1. 两篇原始灵感：保留具体机制

### Sketch2CT

An, Delin; Wang, Chaoli. **Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation**. CVPR 2026, pp.37600–37610。

- [CVF论文](https://openaccess.thecvf.com/content/CVPR2026/html/An_Sketch2CT_Multimodal_Diffusion_for_Structure-Aware_3D_Medical_Volume_Generation_CVPR_2026_paper.html)
- [全文](https://arxiv.org/html/2603.22509v1)
- [作者实现](https://github.com/adlsn/Sketch2CT)

已报告：草图与文本联合引导结构分割生成，再由结构引导医学体数据生成。TSFE使用文本驱动FiLM，CGFM结合cross/self-attention。

本项目迁移：文本调制局部小波结构、全局/局部条件融合、结构先行的生成接口。完整“文本→结构分布→EEG”两阶段规划器仅为后续候选。

不可直接迁移：EEG功率图不是器官分割；它不唯一决定相位或波形。原文几何描述不等于临床诊断语义，需要实际临床监督。

许可：[MIT文件](https://raw.githubusercontent.com/adlsn/Sketch2CT/master/LICENSE)已核实。此许可不赋予源医学数据的新权限。

### MSADN

Shi, Yanli et al. **Multi-scale adversarial diffusion network for image super-resolution**. Scientific Reports 15:11690, 2025-04-05, DOI 10.1038/s41598-025-96185-2。

- [Nature原文](https://www.nature.com/articles/s41598-025-96185-2)
- [作者代码](https://github.com/zxxxxxxh/MSADN)

已报告：MSGG提取低分辨率多尺度条件，结合style相关引导与通道注意力；时间相关判别器支持少步生成；DWT高频残差损失保留细节，实验包含4步采样。

本项目迁移：小波层级特征与多尺度条件注入、细节/频带约束。判别器和少步加速留作后续实验。

不可直接迁移：原文DWT主要用于细节损失，不是“完整小波域扩散生成EEG”；图像四步结果不能成为EEG速度承诺。纯噪声到完整EEG也不是低分辨率图像超分任务。

许可：作者公开仓库当前未核到明确LICENSE。可参考论文思想，复制/修改代码前另核授权。

## 2. 最接近的临床EEG工作

| 工作 | 已有能力 | 可借鉴点 | 本项目需要另证的内容 |
|---|---|---|---|
| [EEG-CLIP，2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1625731/full) | 临床EEG与报告对比学习，少样本/文本零样本解码 | 独立编码器、投影、医学文本初始化 | 它不是波形生成；本项目对齐是否改善生成而非只分类 |
| [ELM，ICML2025](https://proceedings.mlr.press/v267/gijsen25a.html) | 临床报告监督、弱配对与MIL、标签效率 | 记录集合—报告对齐，处理局部错配 | 模板文本与真实报告分别处理；没有报告时不复制其证据范围 |
| [GenEEG，CBM2026](https://www.sciencedirect.com/science/article/pii/S0010482525017524) | 患者适配latent diffusion，临床状态与12项生理属性条件 | 患者独立协议、折内统计、持续适配设计 | 小波/三分支相比已有生理条件的增益，及压缩后细节问题 |
| [JET，ICML2026](https://arxiv.org/abs/2605.21280) | 原波形Transformer、状态条件CFM、结构约束，TUAB/TUEV/TUSZ | 现代EEG骨干、条件调制、类别不平衡处理 | partial-wavelet条件与融合的额外控制/效用，不能再claim条件CFM+频谱loss本身 |

JET 是首版必须纳入的强基线。[作者代码](https://github.com/Y-Research-SBU/JET)及[MIT许可](https://github.com/Y-Research-SBU/JET/blob/main/LICENSE)已核实，可在保留许可和来源的情况下评估作为后续实现起点。当前仓库尚未复制其代码。

代码边界：

- [EEG-CLIP实现](https://github.com/tidiane-camaret/EEGClip)：README称MIT，但当前未核到所指完整LICENSE文件；报告CSV的授权不能从代码声明推出。
- [ELM实现](https://github.com/SamGijsen/ELM)：当前未核到明确LICENSE文件。
- [GenEEG实现](https://github.com/soinikghosh9/GenEEG)：README明确All rights reserved；不能按宽松开源代码复制/修改或再分发。
- GenEEG正式卷期为2026-01-15，DOI/online年份为2025；文献版本不要混写。

## 3. Transformer与融合设计

| 工作与来源 | 本项目采用/候选的思想 | 边界 |
|---|---|---|
| [ViT，ICLR2021](https://arxiv.org/abs/2010.11929) | patch token与注意力表示 | EEG物理时间、导联身份需要额外定义；普通ViT不自动理解时序 |
| [Swin，ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html) | 局部窗口、shift、层级多尺度 | 本项目轴为频率/时间，导联拓扑不能依赖图像排版 |
| [ViLT，ICML2021](https://proceedings.mlr.press/v139/kim21k.html) | 简洁token接口、轻量联合交互 | 本项目仍有独立编码器，只借鉴融合，不称完整ViLT复现 |
| [SD3/MMDiT，2024](https://arxiv.org/abs/2403.03206) | 模态专属参数和跨模态token交互 | 首版不加载或训练完整SD3；双流是后续融合对照 |

[Swin作者代码](https://github.com/microsoft/Swin-Transformer)的MIT许可已核实；具体代码与预训练权重需按实际版本核查。[ViLT作者代码](https://github.com/dandelin/vilt)是结构参考，当前未在本仓库引入。

## 4. 现代扩散：基础与2026进展分开

| 工作 | 时间/确认状态 | 优先级与可迁移机制 |
|---|---|---|
| [DiT](https://arxiv.org/abs/2212.09748) | ICCV2023基础工作 | patch生成骨干、adaLN；不能称2026最新 |
| [SiT](https://arxiv.org/abs/2401.08740) | 2024 | 相同Transformer上比较路径、目标和采样；[MIT代码](https://github.com/willisma/SiT/blob/main/LICENSE.txt)可作为后续实现参考 |
| [JiT: Back to Basics](https://arxiv.org/abs/2511.13720) | 2025预印本，CVPR2026 | 无有损tokenizer、clean-data预测；用于参数化对照，不保证EEG收益 |
| [JET](https://arxiv.org/abs/2605.21280) | ICML2026 | 与临床EEG最直接的现代近邻，优先强基线/可复用主干 |
| [MPDiT](https://arxiv.org/abs/2603.26357) | CVPR2026 | 粗到细patch，全局结构与局部细节；token瓶颈明确后才加入 |
| [ReGen](https://proceedings.mlr.press/v306/lee26f.html) | ICML2026 | 层级表示与波形联合生成；验证于音频，后续研究启发 |
| [PixelREPA](https://arxiv.org/abs/2603.14366) | 2026预印本 | 强语义对齐可能损害直接数据生成/多样性；风险依据而非必加组件 |
| [MeanFlow](https://arxiv.org/abs/2505.13447) | NeurIPS2025 | 少步/一步生成候选；先证明增强效用再加速 |

这些工作不必进入同一个模型。主方案只保留适合当前问题的路径；后续升级由数据和实测瓶颈决定。

原始DiT [LICENSE](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt)实际文本为CC BY-NC 4.0；GitHub标为NOASSERTION，不能简单宣称它为MIT。SiT/JET等明确许可证的代码也需保留第三方声明。模型权重、数据和训练依赖的许可另核。

## 5. 最近工作的共同覆盖与候选差异

已被覆盖：EEG文本对齐、类别/生理属性条件生成、Transformer/flow matching、频谱结构约束。

候选差异不是“三分支更多”，而是：在同信息预算下，通过语义与可编辑的部分时频/导联结构交互，增加可检验的条件控制能力，同时保留对真实患者增强有用的变化。

还需要证明：直接属性FiLM是否已经足够；小波域与多尺度条件的作用是否独立于容量/预训练；对比学习是否使条件表示更好而非压缩样本多样性。贡献随实验结果修订，不能预先写“首次”或“必然顶会”。
