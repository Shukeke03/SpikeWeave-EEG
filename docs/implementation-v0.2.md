# SpikeWeave-EEG v0.2 implementation

2026-10-07。双分支和MCWave已实现；本文记录实现边界与可执行入口，不是临床生成质量结果。

## 实现与固定设计

- `SpikingEEGEncoder`：完整16样本patch，PLIF Q/K/V与FFN，ATan代理梯度、detach reset、FP32膜态、物理时间4/8 chunk交互、膜电位残差、按记录清空状态、共享时间DropPath、两个独立投影头、40%遮挡重构。STAtten思想的原创建模适配，不加载视觉SNN权重，不声称完全复现或节能。
- `SharedVisionLanguageEncoder`：三种输入模式共享同一C及六层联合Transformer。正式前端严格读取本地官方BiomedCLIP配置/权重及BiomedBERT配置；单通道stem以RGB权重求和初始化，插值视觉位置，重用原ViT权重并显式屏蔽缺失patch。频率、时间、正负导联端点和覆盖率进入token。数值CWT不采用RGB自然图像归一化。
- 固定对比目标：P/N/U、逐正例log-softmax、组等权、独立反向分组、文本语义候选去重、重复图文实例去重、任一方向无有效anchor则跳过整项、按有效项权重归一化。没有可靠语义标签时不制造诊断正负对；记录报告拒绝按窗口复制。
- `MCWave`：有符号完整DWT、真实导联内时间/跨尺度交互、导联间交互、条件cross attention、adaLN-Zero与连续clean head。Q坐标先lift为真实导联，输出再reduce；不是把线性模式当成电极。
- VP clean目标、min(SNR,5)权重、alpha²辅助门控，六类约束包含L1重构、局部/全局mean/std、完整样本差分、Pearson/跨导联相关、有符号detail/能量、已提供的CWT和数值属性。
- 数值属性顺序由checkpoint的`attribute_schema`明确保存。频率为Hz，幅度为µV，功率比例为[0,1]，带能量是自然对数；生成波形经源尺度恢复后测量。窗口全局控制不等于局部诊断或疾病保证。CHB现无这些真实属性标签，mask保持未知。
- 阶段A另以权重1执行40%数值CWT patch遮挡重构，适配视觉stem/位置/解码头；阶段B的固定表征目标仍为alignment + EEG masked reconstruction + 0.1 spectral。
- 阶段A/B/C、AdamW(.9,.95)、无bias/norm/neuron decay、warmup/cosine、clip、BF16/FP32膜态、梯度累积、生成器EMA、格式2检查点及恢复。B解冻预训练末两层，使用新模块学习率的0.1；C冻结E/C并训练生成器及导联适配投影。
- 条件按每个样本joint40%、text40%、image10%、all-null10%；nonnull参考保留25%。all-null清除属性和参考，保留测量几何。
- Heun50区间=101次预测器调用；CFG2时实际网络调用202次。DDIM比较可按相同实际调用数配置。有限噪声终点、最终clean预测及真实NFE分别记录。

## 本机与服务器入口

```bash
conda activate spikeweave-eeg
pip install -e '.[dev,medical]'
spikeweave doctor --config configs/tiny.yaml --device cpu
spikeweave doctor --config configs/research-5090.yaml --config-only
spikeweave prepare-biomedclip --output weights/BiomedCLIP
```

权重准备是独立显式步骤，构造器不隐式下载且不随机回退。Hugging Face下载支持8并发；下载失败应先检查网络/镜像，不能把随机初始化称为医学预训练。代码的实际OpenCLIP/HF/timm加载接口用明确标注的随机小checkpoint验证；官方大权重和正式大模型尚需准备后验证。

```bash
# CPU / legion CUDA：只验证工程路径，不评价临床效果。
spikeweave smoke --config configs/tiny.yaml --device cpu --steps 2 --batch-size 3 --sampling-steps 2 --output runs/v02-cpu
spikeweave smoke --config configs/tiny.yaml --device cuda --amp --steps 2 --batch-size 3 --sampling-steps 2 --output runs/v02-cuda
spikeweave smoke --config configs/tiny.yaml --stage align --device cuda --amp --steps 2 --batch-size 3 --sampling-steps 2 --output runs/v02-align
```

正式训练的三个配置为`research-pretrain.yaml`、`research-align.yaml`、`research-5090.yaml`。L版本生成器24×1024，XL版本28×1152。文件名称不代表已经测定5090显存可行性；正式大模型尚未做显存实测。阶段A/B的直接batch=256、accumulation=1；阶段C默认microbatch=4、accumulation=64。对比候选属于当前microbatch，不能把累积的优化器batch误称为256个对比候选。缩小A/B的microbatch会改变候选池，需作为明示工程配置/消融记录。

```bash
# 仅示例，不在此实现任务中自动启动临床长训练。
spikeweave train --config configs/research-pretrain.yaml --manifest /path/to/processed --stage pretrain --batch-size 256 --steps 100000 --output runs/pretrain
spikeweave train --config configs/research-align.yaml --manifest /path/to/processed --init-representation runs/pretrain/last.pt --batch-size 256 --steps 100000 --output runs/align
spikeweave train --config configs/research-5090.yaml --manifest /path/to/processed --init-representation runs/align/last.pt --batch-size 4 --steps 100000 --output runs/diffusion
```

默认按source/train实际类别计数进行逆频率replacement采样；`--no-balanced`仅用于消融。CLI的`steps`指优化器更新次数，自动执行对应累积microsteps。阶段转移严格读取E/C参数，恢复训练使用`--resume`。

## 数据与兼容边界

数据管线继续使用`triwave-morlet-v1`数值CWT协议，原物理µV分片无需重新生成。新`source_rms_v2`归一化只拟合source/train单个RMS标量，不减均值。分片训练新拟合文件为`train_scale_v2.json`，旧affine文件保留供历史逆变换，不能直接训练新生成器。

显式导联顺序与原CHB管线一致并在加载时验证。默认montage_mode=audit，先记录真实残差而不删除信号分量。改为hard后，每个训练目标必须通过残差门槛；无已通过训练残差记录的checkpoint不能hard采样。不支持未知参考、异质缺失导联的硬生成，避免错误Q假设。矩阵秩随montage变化；标准16导联rank14已做数学测试。

生成条件NPZ的审计/硬模式需包含`lead_names`。外部CWT包含协议、尺度，并在物理checkpoint下明确`structure_normalizer_protocol=source_rms_v2`、`structure_fit_digest`、`structure_source_scale`，与checkpoint匹配。未观测CWT区间使用False mask，不能用未知当0真值。full target CWT只用于表征教师；生成读取合法独立参考或外部条件。

正式模型名`SpikeWeaveModel`，公共入口`from spikeweave import ...`；`triwave`模块/命令及`TriWaveModel`为兼容别名。v0.1检查点明确拒绝加载，需重新训练，避免不同head目标混用。

## 未完成的科研验证

尚未用真实临床数据训练正式模型、未验证真实疾病/文本控制、跨患者增强效果、医院泛化及baseline优劣。公共CHB文本是官方事件注释模板，不是真实诊断报告。当前record-report输入守卫不能替代尚待真实配对数据验证的bag/MIL路线。稀疏数值结构条件已经支持；任意手绘图像仍需单独定义数值/频率/区间校准协议。

灵感与主干来源：[Sketch2CT](https://openaccess.thecvf.com/content/CVPR2026/html/An_Sketch2CT_Multimodal_Diffusion_for_Structure-Aware_3D_Medical_Volume_Generation_CVPR_2026_paper.html)、[MSADN](https://www.nature.com/articles/s41598-025-96185-2)、[STAtten](https://github.com/Intelligent-Computing-Lab-Panda/STAtten)、[BiomedCLIP官方配置](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/raw/main/open_clip_config.json)、[JET](https://github.com/Y-Research-SBU/JET)、[WaveletDiff](https://github.com/GarlicWang/WaveletDiff)。实现为原创模块及第三方库接口，不复制这些项目源码；模块组合不等于已证实创新。
