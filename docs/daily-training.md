# legion 每日显存规则

2026-10-08，按用户授权，使用北京时间 `Asia/Shanghai`。目标比例指总显存占用；GPU计算利用率会随运算变化。当前执行M路线共享Stage A表征预训练，生成器12×512尚未进入训练。

| 时间 | 配方 | 实测总显存 |
|---|---|---|
| 09:00 ≤ 时间 < 23:00 | microbatch20 × accumulation13 = effective260；allocator fraction0.77 | 独立探测4850/6144 MiB，78.9%；完整累积峰值reserved4722 MiB加context约79.6% |
| 23:00 ≤ 时间或时间 < 09:00 | microbatch8 × accumulation32 = effective256；allocator fraction0.47 | 独立探测2934/6144 MiB，47.8%；peak allocated2559.41、reserved2768 MiB |

两档固定E/C、BiomedCLIP权重、学习率1e-4、AdamW、warmup5000与100000 optimizer updates预算，仅更改批量与累积；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。夜间独立测试包含Adam状态与保留梯度的累积路径，4个微步14.37秒；短显存探测不能视为完整更新吞吐。早先与日间作业重叠的探测不用于独立显存/时间结论。

控制器为`scripts/train_daily_schedule.py`，提前180秒请求SIGTERM。CLI完成当前累积、原子保存`last.pt`后退出；等待09:00/23:00正式边界再启动另一档，可能短暂空闲。极慢累积超过提前量时仍优先保存，不能承诺绝对秒级切换。全局step、Adam、RNG和源数据身份保留；显式recipe fork及计数器重基有审计记录，数据迭代重新开始，effective batch260/256不同，因此不等价于固定配方的逐样本轨迹。

单实例文件锁与遗留活进程检查阻止同一调度目录重复作业。子进程失败则停止调度并保留已有断点，不循环重试；控制器收到SIGINT/SIGTERM时也等子进程安全保存。控制器只读取时钟及进程状态，不承担持续效果评估或Codex通知。

启动父断点是`runs/legion-m-pretrain-80pct-20261008/last.pt`，其作业已安全停止于step138；旧step133断点继续保留。新的调度根目录：`runs/legion-m-daily-20261008`。

```bash
ssh legion
cd /home/shuxing/workspace/SpikeWeave-EEG
/home/shuxing/miniforge3/envs/spikeweave-eeg/bin/python scripts/train_daily_schedule.py \
  --manifest /home/shuxing/workspace/datasets/SpikeWeave-EEG/chbmit-1.0.0/processed/dualbranch-v2 \
  --checkpoint runs/legion-m-pretrain-80pct-20261008/last.pt \
  --output runs/legion-m-daily-20261008
```

加`--plan-only`只校验并打印规则，无CUDA分配。正式部署以独立后台进程运行，避免关闭SSH中断训练。`controller.json`记录控制器/子进程PID、状态、当前档位、下次边界和最近段结束断点；`schedule.json`记录固定规则与配置SHA。每段有独立`run.json`、`metrics.jsonl`、`train.log`、`launcher.json`、`last.pt`；`current`原子软链接指向当前段，避免人工追踪新目录。

实时查看：

```bash
ssh -t legion 'cd /home/shuxing/workspace/SpikeWeave-EEG && watch -n 2 "python3 scripts/monitor_training.py --run runs/legion-m-daily-20261008/current"'
```

Ctrl+C仅退出监控。切换等待期可查看`controller.json`；其中`completed_step`在段退出时更新，段内最新step见`current/metrics.jsonl`。`sampling_epoch`在每个新段重新计数，不可直接当作跨段全局数据轮数。

工程验证：本机280 passed、2 CUDA skips；包含20项调度测试，覆盖09/23边界、UTC时区转换、提前停机、时间跳变、完整保存、互斥锁、失败停调度与等待取消。真实两档显存分别验证；后续B/C须单独建立适合其模型的配方，不复用A的显存结论。

## 实际部署记录

legion同样280 passed、2 CUDA skips（测试时隐藏CUDA），plan-only确认北京时间20:50属于day80、下一边界23:00。源码`cab5c98`已推送私有GitHub；服务器GitHub连接超时期间通过校验Git bundle快进同步。

2026-10-08 20:50北京时间后台控制器PID151456，训练子进程PID151458，从完整step138恢复，目标100000。首段为`0001-day80-20261008T205044`，已输出有限损失的真实微步；启动观测峰值allocated4295.65 MiB、reserved4702 MiB，配合context约79%显存。run.json标记`generator_capacity.active=false`，当前为共享A。后续PID/段目录会随规则切换变化，使用`current`监控。完成启动确认后代理停止持续监控，首次23:00实际切换结果尚未观察，边界行为已有进程控制测试覆盖。
