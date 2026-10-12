# DepthTrack · 官方 RGB-D 权重评测

本次使用 SEATrack 官方 `rgbd/SEATrack_ep0025.pth.tar`，完成 50 个测试序列、76,373 帧。北京时间 2026-10-11 01:32:49 完成指标分析，按论文精度取整与论文一致。这是官方权重评测，不是自训练实验。

| 指标 | 本次实测 | 论文值 |
|---|---:|---:|
| PR | 0.6290959840516169 | 0.629 |
| RE | 0.6354595519787544 | 0.635 |
| F | 0.6322617564618732 | 0.632 |

## 文件目录

```text
2026-10-11_official-rgbd/
├── README.md
├── manifest.sha256                     # 归档文件 SHA-256 校验清单
├── reports/
│   ├── depthtrack_report.html          # 可直接打开的指标表和曲线
│   └── depthtrack_metrics.json         # 完整精度指标和序列元数据
├── trajectories/
│   └── rgbd-unsupervised/
│       └── <序列名>/
│           ├── <序列名>_001.txt
│           ├── <序列名>_001_confidence.value
│           └── <序列名>_001_time.value
├── logs/
│   ├── evaluation.log                 # 跟踪及首次分析日志，含历史错误
│   ├── analysis.log                   # 修复兼容性后的成功分析日志
│   └── tracker_initial_failure.log    # 首次模块导入失败的诊断日志
└── reproduction/                      # 服务器实际配置和脚本快照
    ├── config.yaml
    ├── trackers.ini
    ├── list.txt
    ├── rgbd.yaml
    ├── seatrack_baseline.py
    ├── run_depthtrack_eval.sh
    ├── run_depthtrack_analysis.sh
    └── check_depthtrack_results.py
```

`001` 是 VOT 的第 1 次运行编号；保留原名以便工具读取。轨迹、置信度和耗时文件均逐序列与标注行数核对。首帧是初始化帧，置信度可以为空。

## 来源与复现环境

- 原始结果：服务器 `/nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/results/rgbd/rgbd-unsupervised/`。
- 报告：同一 workspace 的 `analysis/depthtrack_reproduction/`，不是作者自带的 `analysis/rgbd/` 报告。
- 环境：Python 3.8.20、PyTorch 2.2.2、vot-toolkit 0.5.3、vot-trax 3.0.3；物理 GPU 4，RTX 4090。
- 协议：`votrgbd2021` stack、`rgbd-unsupervised` experiment、1 次运行，分析使用单 worker。
- 修复：启动脚本继承外部 GPU 设置；项目路径优先导入；补齐 `sequences/list.txt`；VOT `analysis/tpr.py` 中 `astype(np.int)` 改为 `astype(int)`。

`reproduction/` 是运行时快照，含服务器绝对路径，迁移环境前需调整。`run_depthtrack_eval.sh` 的旧分析阶段曾因 NumPy 别名兼容问题失败，随后使用 `run_depthtrack_analysis.sh` 成功生成此次报告；完整经过见根目录 `复现日志.md`。数据集、权重及 Python 环境不包含在此归档中。

在本目录执行 `shasum -a 256 -c manifest.sha256` 可检查归档文件完整性。
