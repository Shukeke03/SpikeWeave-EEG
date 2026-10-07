# TriWave-EEG

**Tri-Branch Wavelet-Conditioned Diffusion Transformers for Clinical EEG Generation**

中文：**面向临床脑电生成的三分支小波条件扩散 Transformer**。

TriWave-EEG 是一个处于实现与验证阶段的私有项目。首篇聚焦临床病理事件条件生成与跨患者少样本增强。三分支分别建模 EEG 波形、小波结构视图和临床文本；生成器学习多通道 EEG 的条件分布。

当前已实现数据处理、三分支对齐、条件融合、WaveDiT 训练与采样，支持 CPU 和 CUDA 验证。真实临床数据训练与效果评估尚未完成；工程测试不构成泛化或疾病生成有效性的证据。项目命名不代表方法的新颖性已经成立。

## 运行入口

详见 [开发与运行](docs/development.md)。本机与 legion 均使用专用 Miniforge 环境 `triwave-eeg`。

```bash
conda activate triwave-eeg
python -m pytest -q
triwave smoke --config configs/tiny.yaml --device cpu
```

`configs/legion.yaml` 保持 16 导联、4 秒、256Hz，缩小 Transformer 宽度以供当前 RTX 3060 Laptop 6GB 运行。离线文本后端用于工程验证，医学语言实验使用单独的 ClinicalBERT 配置。

## 研究问题

在相同真实数据预算下，结构条件相比粗类别条件能提供多少控制信息；在相同属性输入下，三分支融合能否进一步提高结构控制，同时保持样本多样性，并改善未见患者真实记录上的检测？

## 必须保留的设计

- 三分支表示与对比学习：波形、数值小波结构视图、临床文本。
- ViT 类时间/导联建模、Swin 式频率/时间层级建模及 ViLT 式 token 融合。
- 显式小波条件扩散；完整可逆 DWT 域是候选主路径，原始波形域是必要对照。
- Sketch2CT 的文本—结构协同与结构引导生成启发。
- MSADN 的多尺度条件注入及 DWT 细节约束启发。
- 现代 DiT/JET 主干思想；DDPM 条件扩散与 conditional flow matching 分别定义，分别验证。

## 文档入口

| 文档 | 内容 |
|---|---|
| [开发与运行](docs/development.md) | Miniforge、平台依赖、CLI、数据准备、训练/恢复/采样与实现边界 |
| [模型架构](docs/architecture.md) | 张量接口、三分支、融合、生成器、损失、训练与推理、资源估算 |
| [数据协议](docs/data-protocol.md) | 配对证据、标注粒度、导联、预处理、支持集与划分 |
| [文献与迁移依据](docs/related-work.md) | 原始灵感、最近工作、可迁移机制、代码授权核查 |
| [实验与反证](docs/experiments.md) | 强基线、核心消融、控制指标、跨患者增强与失败条件 |
| [推进顺序](docs/roadmap.md) | 首版边界、里程碑与继续投入条件 |

## 数据与证据边界

首版不使用 ImageNet 或自然视觉刺激队列。优先在 CHB-MIT 开发子集建立原型，读取开发EEG前预留测试患者；获准访问后以 TUSZ 做主实验。CHB若已参与开发，只能承担该来源的未见患者测试，严格未见来源需另一个未参与开发的队列。TUEV/TUEP 是后续事件/诊断扩展。

派生小波图不是独立采集的医学图像。标注模板不是医生报告。患者诊断、整记录结论与窗口事件不能混为同一级标签。完整目标 EEG 或其小波图只能用于监督或明确的重建任务，不能成为文本/草图生成测试的隐藏输入。

原始 EEG、临床报告、患者信息、模型缓存和训练产物不进入 Git。源数据按各自访问条款取得；本仓库不授予第三方源代码、数据或权重的新许可。

实现版本：v0.1；文献核查截止：2026-10-07。
