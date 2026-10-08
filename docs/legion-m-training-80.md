# legion 的 M 路线与约 80% 显存训练

2026-10-08：用户要求停止旧作业，优先小容量 M，将现有3060的显存预算提高到约80%。旧进程117152已停止，最近完整断点保留为`runs/legion-pretrain-20261008/stopped-checkpoint-20261008T115256Z.pt`，记录到133次optimizer更新。未落盘的微步不保留。

旧作业实际是共享A表征预训练，没有分配24×1024的D。新路线的生成器固定M12×512，但当前仍先完成它所需的A/B；不能把带未使用D元数据的A checkpoint称为已训练生成器。E/C主干和物理数据协议保持相同。

## 实测配方

GPU：RTX3060 Laptop，6144 MiB。配置：`configs/legion-m-pretrain-80.yaml`。

| 配方 | 稳态实测 / 结果 |
|---|---|
| batch16，allocator fraction0.77 | 总占用4304 MiB，约70%；通过 |
| batch22，fraction0.77 | 默认allocator与expandable segments均触发预算内OOM |
| **batch20，fraction0.77，expandable segments** | **总占用4850 MiB，约78.9%；通过** |

每个候选为独立进程，使用真实source窗口，完成两次Adam更新，再进行保留梯度和已有Adam状态的累积测试。选定batch20的峰值allocated4295.65 MiB、reserved4684 MiB；4个微步/3次优化更新耗时28.52秒，仅用于显存验收，不能当作正式长期吞吐。PyTorch allocator设置0.77，另为CUDA context留空间；总占用由`nvidia-smi`核实，80%是目标预算而非逐秒固定值。

训练使用**microbatch20 × accumulation13 = effective batch260**，相比旧256增加1.56%。保持AdamW、lr1e-4、weight decay0.05、BF16矩阵/FP32膜态、warmup5000、总更新预算100000、激活重计算和全部源数据。批量变化明确记为新配方，后续容量比较须记账，不声称与旧作业逐位相同。

## 断点续接与停机

新增显式`--fork-resume`，只适用于无D/EMA的Stage A：复用完整E/C、Adam状态、RNG、normalization、source数据身份和optimizer step。它只允许批量/累积、未使用的D容量及执行/定位参数变化；更改阶段、学习率、schedule或物理语义会拒绝。常规`--resume`仍严格遵守保存的配置。

旧microstep=4256不整除新accumulation13，因此操作计数器重基为133×13=1729，源历史计数与这次重基完整记在`run_provenance.recipe_fork`；优化器进度133不重置。后续新微步完整累积13次才更新。数据迭代器重新开始，来源断点同inode读取/哈希，旧文件不改写。

新运行目录约定为`runs/legion-m-pretrain-80pct-20261008`。先执行1次完整新配方更新并保存，再独立后台续训至总更新100000。run.json记录M容量元数据、批量配方、allocator设置、源SHA和recipe fork。当前启动确认见后续追加记录。

新CLI接收SIGINT/SIGTERM后，完成当前累积周期，再原子保存checkpoint并写入`status=stopped`；不会在半个累积周期把残缺梯度当作完整更新。该行为已有CLI回归验证。

## 人工查看

```bash
ssh -t legion 'cd /home/shuxing/workspace/SpikeWeave-EEG && watch -n 2 "python3 scripts/monitor_training.py --run runs/legion-m-pretrain-80pct-20261008"'
```

Ctrl+C仅关闭watch，后台训练继续。启动完成后代理不持续监控。

严格续训新作业时使用其`last.pt`及batch20、fraction0.77，不额外传配置；保留`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。M的C扩散阶段须在B对齐完成后另行实测显存与启动，L保持延后。

本机工程验证：260 passed、2 CUDA skips；新增测试覆盖Adam/RNG/进度保留、显式counter重基、非法配方拒绝、源文件不修改、严格resume不放宽，以及信号停止时的完整累积与原子保存。正式临床生成和效果尚未验证。

## 实际启动确认

legion的CPU工程验证同为260 passed、2 CUDA skips，验证时显式隐藏CUDA。服务器访问GitHub SSH/443超时，改由SSH传输已推送的Git bundle，校验base与commit后快进到`e6941ec`，保持与本机/私有GitHub的源码一致。

真实新配方的首个完整更新已完成：step133→134，microstep1729→1742，13个microbatch，循环89.56秒，损失16.4231且梯度有限。`last.pt`原子保存成功；峰值allocated4295.65 MiB、reserved4722 MiB，加启动测得约166 MiB的CUDA context约占总显存79.6%。run.json的generator version为M、depth12/dim512/heads8，`active=false`确认当前A没有加载生成器。

随后独立后台严格恢复新`last.pt`，执行额外99866次optimizer更新，目标总计100000。`bootstrap-summary.json`保留第一次更新的运行记录，`launcher.json`记录实际后台PID和命令，显存探测日志保存在`runs/legion-m-80pct-setup/`。启动确认后停止代理持续监控。
