# SpikeWeave-EEG v0.2 工程验证

日期：2026-10-07。代码主提交：`cc66be3`。使用公开软件接口与人工工程fixture，临床有效性尚未测量。

## 本机

- Miniforge `spikeweave-eeg`，Python3.11.17，torch2.8.0；OpenCLIP3.3.0、timm1.0.30、torchvision0.23.0，pip check通过。
- ruff与git diff --check通过。完整pytest：138 passed、1 skipped（本机无CUDA）；重复导联fixture的MNE警告是刻意构造。
- 新测试覆盖实际多正例公式、独立反向组平衡、U排除、候选去重、双向为空跳过、PLIF梯度/FP32状态/记录隔离、遮挡值不泄漏、共享C模式、导联动态秩和残差门槛、VP schedule/实际NFE、六类约束、物理属性、源RMS闭环、阶段转移/冻结、累积和恢复、旧检查点/affine拒绝。
- 实际OpenCLIP/HF/timm检查点接口以随机小checkpoint验证：strict加载、CWT stem RGB求和、预训练模块冻结、缺失mask与可训练适配梯度。该fixture明确不宣称医学预训练。
- CPU两步扩散与两步alignment CLI均完成优化器更新、保存及有限采样，形状[3,4,128]。alignment固定三项均有效，六类约束分别记录；两区间Heun=5预测器/CFG2时10网络调用。
- 研究配置校验：E8×512/8heads，C6×768/12heads，MCWave24×1024/16heads；XL28×1152。未实例化/训练完整正式医学前端和大模型。

## legion

- Miniforge同名环境、Python3.11.17、torch2.8.0+cu128、CUDA12.8、RTX3060 Laptop；OpenCLIP3.3.0、timm1.0.30、torchvision0.23.0+cu128，pip check与ruff通过。
- 从private GitHub拉取`cc66be3`并重装editable v0.2.0；完整pytest：139 passed（含CUDA信号变换测试）。同一刻意重复导联fixture有MNE警告，无测试失败。
- 实际CUDA BF16训练：pretrain、align、diffusion分别完成两次优化器更新、保存和有限采样，tiny输入[3,4,128]；固定三项alignment与numeric-visual adaptation分别记录。
- 16导联1024点工程配置完成两次扩散更新和采样，输出[2,16,1024]；峰值torch allocated显存100.36MiB。tiny扩散23.78MiB、alignment25.67MiB、pretrain22.29MiB。这些是缩小工程配置的torch分配量，排除CUDA上下文/缓存，不代表正式大模型显存。
- 四次CLI均记录两区间Heun的5预测器调用、CFG2时10实际网络调用；训练检查点及输出位于ignored的`runs/v02-*-cuda-final`目录。
- 没有自动启动临床长训练、重处理真实数据或持续下载监控。官方大医学权重尚未准备并运行完整验证。

## 解释边界

数学/工程测试不能证明医学真实性、疾病控制、泛化或增强收益。原物理数据未重新生成；新RMS只在显式管线或train入口源统计拟合时生成，旧affine文件与旧历史测试不能替代新临床证据。
