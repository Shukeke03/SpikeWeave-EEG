# CHB-MIT 分片数据集 v1

2026-10-07：完整下载与增量构建已启动；最终可用窗口数、事件保留数与派生数据体量等待服务器验收。本文区分官方库存与实际完成量。

## 官方库存与体量

来源：[CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/)，镜像 `s3://physionet-open/chbmit/1.0.0/`。完整 S3 对象清单、`RECORDS`、`RECORDS-WITH-SEIZURES`、`SHA256SUMS.txt` 与各病例 summary 联合核对：

| 项目 | 官方库存 |
|---|---:|
| 对象 | 857 |
| EDF | 686 |
| 含发作 EDF | 141 |
| summary 发作区间 | 198 |
| 总字节 | 45,759,184,625 |
| 原始体量 | 45.759 GB / 42.617 GiB |
| case / 独立患者 | 24 / 23 |

网页旧正文中的部分计数与当前附带 chb24 的完整库存不同；以版本文件清单为构建依据。10 个 chb24 背景记录未出现在 summary 中，通过完整含发作文件清单与 `.edf.seizures` 清单共同确认其无已标注发作，不能称为健康。

服务器核查发现含发作列表将 chb07_18 列为阳性、遗漏 chb07_19，而 summary 明确前者0次、后者1次，且仅后者存在 `.seizures` 侧车。以区间摘要与侧车库存的一致证据协调这两条记录；原清单不修改，差异进入逐记录和库存审计。摘要与侧车也不一致时必须隔离。

## 患者划分与处理定义

`configs/chbmit-patients-v1.json` 在读取 EEG 前固定患者身份；chb01/chb21 为同一患者，合并到训练组。独立患者数为训练16、验证3、测试4；未处理前对应 EDF 数494/74/118、标注事件数144/18/36。该划分检验来源内未见患者，不能称为未见数据来源。

- 保留16个标准双极导联与两个电极端点几何，不将双极导联当单电极。
- 输出256 Hz、float32、物理 µV。0.1–80 Hz 四阶 Butterworth 零相位连续记录滤波；需要时以 `resample_poly` 重采样。
- 4秒、不重叠窗口；不完整尾段保留在连续分片并计数，不能直接作为固定长度训练窗口。
- 只根据官方区间构造窗口发作/无已标注发作标签与两类文本模板。模板不是医生报告或疾病诊断；没有真实诊断和自由报告的条件保持缺失。
- 非有限输入整记录隔离，避免滤波传播；平线窗口排除。高幅值只标记，单独不排除；不会凭数值阈值判断病理。
- 标准导联不完整进入隔离目录与索引。chb12_27 仅依据显式 CS2 共同参考通道差分；chb12_28/29 的裸电极通道参考未声明，保留原始文件、头证据及标注，禁止猜测转换。
- 一个 EDF 对应一个连续 `.npy` 分片；窗口索引不复制波形。DWT、带有效性掩码的数值 CWT 在训练读取时计算，避免大量图片缓存与量化损失。
- 全局 µV 仿射统计只拟合合格 train/source 窗口，保留通道之间相对幅值；摘要绑定训练索引、协议、split lock 与源 SHA。
- 每个参考只来自不同独立患者的训练组，优先匹配指定事件类别。完整目标 CWT 仅作训练监督。真实标签作为文本/参考类别的条件适用于明确的类别条件评估，不能据此声称未知类别的盲生成。

## legion 路径与加速

服务器根目录：`/home/shuxing/workspace/datasets/TriWave-EEG/chbmit-1.0.0`。

| 目录/文件 | 用途 |
|---|---|
| `raw/` | 原始 EDF、官方摘要/标注/校验清单 |
| `metadata/official-inventory.json` | 对象大小库存 |
| `metadata/aria2-download.log` | 分段下载进度 |
| `metadata/download-process.json` | 下载 PID 与参数 |
| `metadata/aria2-unfinished.txt` | 断点会话 |
| `metadata/dataset-status.json` | 持久预处理状态 |
| `processed/v1/records/` | 连续物理分片、局部索引、完成账本 |
| `processed/v1/window_index.npy` | 可用窗口索引和独立参考索引 |
| `processed/v1/isolated_index.npy` | 隔离窗口，不能进入正常训练 |
| `processed/v1/train_normalizer.json` | 仅训练集拟合的统计与溯源 |
| `processed/v1/build_report.json` | 构建状态与丢失原因 |
| `processed/v1/dataset_audit.json` | 最终验收、事件覆盖、QC、体量 |

下载使用官方 AWS HTTPS 镜像，8文件并发、每文件4段、断点续传、SHA-256、低速重试。实测原站约0.03 MiB/s，AWS单文件约2.73 MiB/s，并发总速约10 MiB/s；测量不是持续带宽保证。`triwave-download` 为独立 aria2 工具环境，`triwave-eeg` 为模型/处理环境。

低速阈值作用于分段连接，不能把总速度目标直接施加到每段。初轮256KiB/s阈值造成连接频繁重试并留下125个未完成文件；已调整为32KiB/s、60秒超时、持续重试，恢复未完成段并监测总吞吐。

在服务器仓库目录运行以下命令；已经运行的下载不会重复启动：

```bash
python3 src/triwave/data/download.py \
  --base /home/shuxing/workspace/datasets/TriWave-EEG/chbmit-1.0.0 \
  --aria2 /home/shuxing/miniforge3/envs/triwave-download/bin/aria2c \
  --concurrent-files 8 --segments 4 --background

/home/shuxing/miniforge3/envs/triwave-eeg/bin/python -m triwave.data.pipeline \
  --base /home/shuxing/workspace/datasets/TriWave-EEG/chbmit-1.0.0 \
  --config configs/legion.yaml --workers 2 --interval 120 --device cpu
```

预处理进程须通过 supervisor 或 `start_new_session=True` 持久启动以独立于 SSH 会话。流水线锁禁止双写；正在下载的 `.aria2` 文件标记 pending，不预先哈希；完成文件核验 SHA 后处理，后续轮次恢复已完成记录。只有所有源文件校验通过且无未解决处理失败，才拟合训练统计并最终验收。临时索引是 preview，正常训练入口拒绝不完整数据。

验收检查所有索引、患者身份、标注和分片头信息；数值 DWT/IDWT、CWT 检查只读训练组16个窗口，保留锁定查询的数值评估。它不构成模型效果或临床有效性证据。原始与派生数据、报告、权重均不进入 Git。

## 本机实时监控

```bash
ssh -t legion 'watch -n 2 python3 /home/shuxing/workspace/TriWave-EEG/scripts/monitor_legion.py'
```

每2秒显示完整文件体量/数量、最近下载速率、处理账本与验收状态、CPU/RAM/GPU与可用磁盘。下载完成量仅计完整文件，部分分段另有可恢复字节；最终校验以构建账本和验收为准。Ctrl+C 只退出监控，后台工作继续。正常训练读取已拟合统计并核对来源，避免重复扫描训练波形。
