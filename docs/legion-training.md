# legion 正式表征预训练

2026-10-08：用户授权在现有 3060 上慢速训练，显存占用约为一半，启动后不持续监控。

## 阶段与配置

实际 GPU 是 RTX 3060 Laptop，6144 MiB；目标总占用约 3072 MiB。启动时的24×1024生成器为799,622,976参数，FP32权重加EMA约5.96 GiB，无法仅减小batch达到这个预算。因此已启动 **Stage A 信号重构与数值视觉适配**，使用 `configs/legion-pretrain.yaml`，保留正式8×512 PLIF EEG encoder、官方BiomedCLIP前端及6×768共享C。不在Stage A/B构建生成器或EMA。

现确立两个容量版本：M12×512（105,769,728生成器参数）优先，L24×1024延后，见 [capacity-versions.md](capacity-versions.md)。A/B供两版共用；现有运行继续按旧断点保存的配置，不重启或改写元数据。Stage C将分别构建M/L，在独立目录保存生成器、geometry projection、optimizer与EMA。M的C阶段显存仍需实测，不能把当前一半显存目标直接套用。

开启 SNN/医学 ViT block 激活重计算，保留 RNG 与 PLIF 梯度；冻结医学 ViT 仍向数值 stem 反传。BF16 矩阵运算、FP32 神经元状态、AdamW、有效 batch256、lr1e-4、warmup5000、最多100000次优化更新保持阶段 A 配置。实际 microbatch 由真实数据显存探测决定，梯度累积到256。Stage A 没有对比损失；之后 Stage B 的对比候选数量不能用累积伪造。本次不会自动进入 B/C，也不把表征 checkpoint 当作能生成 EEG 的模型。

探测用 `scripts/probe_training_memory.py`，每个候选独立进程，以 accumulation=1 完成两次真实 optimizer 更新，包含 Adam 状态分配，再测试两次累积 microstep，检查已有 Adam 状态及保留梯度时的峰值。CUDA allocator fraction 最终设0.47，为驱动/context留空间；总占用另由 `nvidia-smi` 核实。这是 allocator 上限，不能保证其他 GPU 应用不会另占显存。

实测最终配置：**microbatch8 × accumulation32 = effective batch256**。batch8在0.43的保守限制下触及上限；根据约166 MiB驱动/context开销调整为0.47后通过累积稳态测试。峰值allocated2563.37 MiB、reserved2834 MiB，`nvidia-smi`总占用3000/6144 MiB（48.8%）。4个真实microstep/3次optimizer更新约14.42秒，不能直接当作长跑吞吐。无D模型总参数272,964,625，其中E/C为272,959,249，另有5,376参数的geometry projection未在A/B训练；官方医学权重严格加载，未降低表征宽度或层数。

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

现有A后续续训使用`--resume runs/legion-pretrain-20261008/last.pt`，不额外传`--config`；它保留启动时D24元数据，虽然A没有构造D。新M默认不会追溯改变运行的Python进程或断点。后续B使用新的`--init-representation`入口，再将已训练的B交给M/L；不能将A的optimizer或随机D跨容量恢复。

本机完整测试215 passed、2 CUDA skips；legion完整测试217 passed、无skip。真实v2数值审计16个source窗口通过，DWT最大误差4.7684e-6，属性/CWT masks核对通过，query数值读取0。仅运行必要数据与训练工程验收，未开始G1–G3效果评估。

实际启动发现 `set_per_process_memory_fraction` 必须使用含 index 的 CUDA device；CLI已解析当前 GPU 编号并增加回归断言，本机/legion各10项CLI测试通过。初次失败没有优化更新，其日志保留于 `runs/legion-setup-20261008/failed-launch-cuda-index.log`。

## 实际启动与预计时长

正式初始更新已完成：32个microstep、1个optimizer update，训练循环106.74秒，含保存共107.68秒，`last.pt`已写入。随后于2026-10-08约15:35 CST从该断点启动独立后台进程，PID117152，运行源码版本`96d6949`；目标总更新100000、额外99999。一次性启动确认中，进程已推进到microstep39，损失有限，GPU2980/6144 MiB（48.5%），峰值allocated2563.49 MiB、reserved2814 MiB。之后停止代理持续监控，用户可用上述watch命令自行查看。

按约107秒/optimizer update估算：1000步约30小时，10000步约12.4天，100000步约124天。当前约3.3秒/microstep；梯度累积32次才是一次更新，不能把一个microstep的时间当作一步预算。该估算只依据一次完整更新及启动时的微步吞吐，机器负载/暂停/写盘会改变实际时长；不是训练收敛或效果达标的预计时间。这里只计Stage A，后续B对齐与C扩散需另计；迁移到更快服务器后可加载已有表征断点，但若改batch/训练协议，应另记运行配置。
