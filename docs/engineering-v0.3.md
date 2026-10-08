# v0.3 工程验证记录

日期：2026-10-08。代码提交：`255f1e1`。本记录只描述实际执行的工程检查；真实临床训练、JET/VP 对照和 SOTA 评价没有在本次运行中执行。

## 实测结果

| 环境 | 检查 | 结果 |
|---|---|---|
| 本机 Miniforge `spikeweave-eeg` | 全套 pytest | 204 passed、2 CUDA skipped，5.96 s |
| legion 同名环境、Torch 2.8.0+cu128 | 全套 pytest | 206 passed，无 skip，10.83 s |
| 本机 / legion | Ruff | 全部通过 |
| 本机 | `git diff --check` | 通过 |
| 本机 | 正式 research-5090 配置解析 | E=8×512/8 heads，C=6×768/12 heads，MCWave=24×1024/16 heads；BiomedCLIP、montage audit |

pytest 唯一 warning 来自故意构造重复 T8-P8 名称的 MNE 隔离测试；该测试通过。正式配置检查没有实例化完整模型、加载官方大权重或测量正式显存。

legion 用清华 PyPI 镜像安装 PyArrow 24.0.0：48.8 MB wheel，工具报告约9.7 MB/s；随后 editable 安装项目 v0.3.0。现有 Torch/OpenCLIP/timm 依赖保持兼容版本。

## GPU 端到端与恢复

```bash
spikeweave --seed 41 smoke --config configs/tiny.yaml \
  --stage joint_smoke --device cuda --amp --steps 2 --batch-size 3 \
  --sampling-steps 2 --output runs/v03-cuda-engineering-255f1e1

spikeweave --seed 41 smoke --device cuda \
  --resume runs/v03-cuda-engineering-255f1e1/last.pt \
  --steps 1 --batch-size 3 --sampling-steps 2 \
  --output runs/v03-cuda-resume-255f1e1
```

两次 status=ok；训练至 step2，恢复后 step3，输出形状 `[3,4,128]` 且有限。数据为明确声明的工程合成信号、任意幅度单位，text backend 为 offline；不构成医学预训练或临床效果。

Heun2 区间实际5次 predictor、CFG2实际10次网络调用。两次报告 `peak_cuda_mib=32.71728515625`，这是 tiny 配置的 Torch 峰值已分配显存，不能外推到24层正式生成器，也不包含全部驱动上下文开销。

原始工程日志、checkpoint 和合成 NPZ 保存在 legion 的上述 ignored `runs/` 目录，没有提交到 GitHub。

## 关键回归覆盖

- 七表原子迁移、源 RMS、元数据/配置/索引指纹、条件策略与尺度指纹分离，以及来源身份审计先于 EEG 数值读取。
- CHB01/21 同患者、跨 split 同文件内容、query 不能拟合统计或训练；未标注源窗不伪造负标签。
- 原生128 Hz 数据的64 Hz频率支持、完整 γ 带未知、共享物理属性与可微控制损失一致。
- CWT 全贡献/完整 patch 掩码、未观察 NaN 屏蔽、局部可见控制和独立参考不作逐像素目标。
- 图像全空也不能绕过参考 EEG 来源；构造/推理分别复查，少样本支持需明确预算和上下文隔离。
- 属性跟随实际丢弃掩码、50%为空/其余1–4项；教师与部署条件隔离。
- 类别请求真实进入 C 文本、冲突文本拒绝、分批生成时保留逐样本来源。
- 同预算小 CNN、患者 AP 无正类不可定义、缺失/冒名基线不升级；三独立生成器训练 seed/checkpoint、固定代码/配方/预算，以及失败使后续闸门失效。
- 格式与条件协议校验、显式旧协议权重迁移、优化器恢复、CUDA 控制梯度和工程 checkpoint 恢复。

完整真实数据迁移、正式模型显存、真实诊断文本/报告、疾病控制及跨患者增强收益仍需后续实验。G3 只实现具名操作登记；完整科研矩阵依照达到基线后再解锁的方案开展。
