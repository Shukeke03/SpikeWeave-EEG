# legion 正式表征预训练

2026-10-08：用户授权在现有 3060 上慢速训练，显存占用约为一半，启动后不持续监控。

## 阶段与配置

实际 GPU 是 RTX 3060 Laptop，6144 MiB；目标总占用约 3072 MiB。正式 MCWave 为 799,622,976 参数，FP32 权重加 EMA 就约 5.96 GiB，无法仅减小 batch 达到这个预算。因此本次启动 **Stage A 信号重构与数值视觉适配**，使用 `configs/legion-pretrain.yaml`，保留正式 8×512 PLIF EEG encoder、官方 BiomedCLIP 前端及 6×768 共享 C 配置。不在 Stage A/B 构建尚未参与训练的生成器或 EMA；它们会在 Stage C 按正式配置构建。

开启 SNN/医学 ViT block 激活重计算，保留 RNG 与 PLIF 梯度；冻结医学 ViT 仍向数值 stem 反传。BF16 矩阵运算、FP32 神经元状态、AdamW、有效 batch256、lr1e-4、warmup5000、最多100000次优化更新保持阶段 A 配置。实际 microbatch 由真实数据显存探测决定，梯度累积到256。Stage A 没有对比损失；之后 Stage B 的对比候选数量不能用累积伪造。本次不会自动进入 B/C，也不把表征 checkpoint 当作能生成 EEG 的模型。

探测用 `scripts/probe_training_memory.py`，每个候选独立进程，以 accumulation=1 完成两次真实 optimizer 更新，包含 Adam 状态分配。CUDA allocator fraction 初值0.43，为驱动/context留空间；总占用另由 `nvidia-smi` 核实。这是显存上限，不能保证其他 GPU 应用不会另占显存。

## 数据与权重

真实 CHB-MIT v1 共837181个可用窗口，train519047、validation66418、test251716。完整 EDF 已下载/处理；52个不兼容记录继续隔离。v2 已在 legion 构建七张 Parquet 表、source RMS 标量、来源属性及独立 reference bank，额外文件共180659334字节，复用原连续信号分片。原始文件45.759 GB，v1处理文件54.959 GB；v2额外约180.66 MB，没有缓存整库 CWT，也没有为训练数值读取 query EEG。

官方权重通过 `HF_ENDPOINT=https://hf-mirror.com` 准备。BiomedCLIP 的 bin 文件 SHA256 必须匹配 `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`。两个官方模型卡标注MIT，BiomedCLIP注明研究用途；当前仅做研究预训练。[BiomedCLIP 模型卡](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)，[BiomedBERT 模型卡](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)。权重和临床数据均不提交 Git。

## 断点与人工监控

后台运行目录为 `runs/legion-pretrain-20261008`。`train.log`/`metrics.jsonl`提供实时微步损失和显存；`run.json`记录源码版本、数据身份、配置和额外优化更新预算。`last.pt`在每100次更新或600秒后的首个完整累积边界原子保存；不在未完成累积时保存，恢复后数据迭代器会重启，不能声称逐样本轨迹完全复现。

人工查看命令：

```bash
ssh -t legion 'cd /home/shuxing/workspace/SpikeWeave-EEG && watch -n 2 "python3 scripts/monitor_training.py --run runs/legion-pretrain-20261008"'
```

Ctrl+C只关闭监控，训练独立于 SSH 会话继续。不会部署自动持续监控任务。实际启动显存和 batch 记录见本文件后续实测条目；完成表征预训练不等于完成扩散训练或证明临床性能。
