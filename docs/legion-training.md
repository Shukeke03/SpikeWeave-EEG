# legion 正式表征预训练

2026-10-08：用户授权在现有 3060 上慢速训练，显存占用约为一半，启动后不持续监控。

## 阶段与配置

实际 GPU 是 RTX 3060 Laptop，6144 MiB；目标总占用约 3072 MiB。正式 MCWave 为 799,622,976 参数，FP32 权重加 EMA 就约 5.96 GiB，无法仅减小 batch 达到这个预算。因此本次启动 **Stage A 信号重构与数值视觉适配**，使用 `configs/legion-pretrain.yaml`，保留正式 8×512 PLIF EEG encoder、官方 BiomedCLIP 前端及 6×768 共享 C 配置。不在 Stage A/B 构建尚未参与训练的生成器或 EMA；它们会在 Stage C 按正式配置构建。

开启 SNN/医学 ViT block 激活重计算，保留 RNG 与 PLIF 梯度；冻结医学 ViT 仍向数值 stem 反传。BF16 矩阵运算、FP32 神经元状态、AdamW、有效 batch256、lr1e-4、warmup5000、最多100000次优化更新保持阶段 A 配置。实际 microbatch 由真实数据显存探测决定，梯度累积到256。Stage A 没有对比损失；之后 Stage B 的对比候选数量不能用累积伪造。本次不会自动进入 B/C，也不把表征 checkpoint 当作能生成 EEG 的模型。

探测用 `scripts/probe_training_memory.py`，每个候选独立进程，以 accumulation=1 完成两次真实 optimizer 更新，包含 Adam 状态分配，再测试两次累积 microstep，检查已有 Adam 状态及保留梯度时的峰值。CUDA allocator fraction 最终设0.47，为驱动/context留空间；总占用另由 `nvidia-smi` 核实。这是 allocator 上限，不能保证其他 GPU 应用不会另占显存。

实测最终配置：**microbatch8 × accumulation32 = effective batch256**。batch8在0.43的保守限制下触及上限；根据约166 MiB驱动/context开销调整为0.47后通过累积稳态测试。峰值 allocated2563.37 MiB、reserved2834 MiB，`nvidia-smi`总占用3000/6144 MiB（48.8%）。4个真实microstep/3次optimizer更新约14.42秒，不能直接当作长跑吞吐。完整E/C参数272964625，官方医学权重严格加载；未降低主干宽度或层数。

## 数据与权重

真实 CHB-MIT v1 共837181个可用窗口，train519047、validation66418、test251716。完整 EDF 已下载/处理；52个不兼容记录继续隔离。v2 已在 legion 构建七张 Parquet 表、source RMS 标量、来源属性及独立 reference bank，额外文件共180659334字节，复用原连续信号分片。原始文件45.759 GB，v1处理文件54.959 GB；v2额外约180.66 MB，没有缓存整库 CWT，也没有为训练数值读取 query EEG。

官方权重通过 `HF_ENDPOINT=https://hf-mirror.com` 准备。BiomedCLIP 的 bin 文件 SHA256 必须匹配 `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`。两个官方模型卡标注MIT，BiomedCLIP注明研究用途；当前仅做研究预训练。[BiomedCLIP 模型卡](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)，[BiomedBERT 模型卡](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)。权重和临床数据均不提交 Git。

## 断点与人工监控

后台运行目录为 `runs/legion-pretrain-20261008`。先完成1次正式累积更新并保存初始断点，再从该断点后台续训99999次更新，总预算100000次。`train.log`/`metrics.jsonl`提供实时微步损失和显存；`run.json`记录源码版本、数据身份、配置和额外优化更新预算。`last.pt`在每100次更新或600秒后的首个完整累积边界原子保存；不在未完成累积时保存，恢复后数据迭代器会重启，不能声称逐样本轨迹完全复现。

人工查看命令：

```bash
ssh -t legion 'cd /home/shuxing/workspace/SpikeWeave-EEG && watch -n 2 "python3 scripts/monitor_training.py --run runs/legion-pretrain-20261008"'
```

Ctrl+C只关闭监控，训练独立于 SSH 会话继续。不会部署自动持续监控任务。完成表征预训练不等于完成扩散训练或证明临床性能。

本机完整测试215 passed、2 CUDA skips；legion完整测试217 passed、无skip。真实v2数值审计16个source窗口通过，DWT最大误差4.7684e-6，属性/CWT masks核对通过，query数值读取0。仅运行必要数据与训练工程验收，未开始G1–G3效果评估。

实际启动发现 `set_per_process_memory_fraction` 必须使用含 index 的 CUDA device；CLI已解析当前 GPU 编号并增加回归断言，本机/legion各10项CLI测试通过。初次失败没有优化更新，其日志保留于 `runs/legion-setup-20261008/failed-launch-cuda-index.log`。
