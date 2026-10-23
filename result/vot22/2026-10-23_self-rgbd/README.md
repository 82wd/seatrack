# VOT2022 RGB-D 自训练权重评测

2026-10-23 用户回传 VOT 全量跟踪与 JSON/HTML 分析均成功、退出码均为 0。报告 tracker 为 `rgbd_self_ep0025`，包含 127 个序列，实验名为 `baseline`。

| 指标 | 报告原始值 | ×100 展示 | 论文（×100） | 与论文展示值之差 |
|---|---:|---:|---:|---:|
| EAO | 0.7375977345234603 | 73.759773 | 73.6 | +0.159773 |
| Accuracy | 0.8206357885293973 | 82.063579 | 82.1 | −0.036421 |
| Robustness | 0.8859998200859379 | 88.599982 | 88.4 | +0.199982 |

论文只报告一位小数，差值以其展示值计算，不能视作与论文未舍入原始结果的精确差异。当前结果数值接近论文，不足以证明统计显著改善或独立训练复现成功。

## 用户回传的分析项

- `vot.analysis.multistart.EAOScore`：`[[0.7375977345234603]]`。
- `vot.analysis.multistart.AverageAccuracyRobustness`：`[[0.8206357885293973, 0.8859998200859379, [0.8859998200859379, 0.8206357885293973], 1162375, 80741]]`。
- `EAOCurve` 为曲线数据，不作为额外主指标；此前用户回传为截断摘要；完整曲线现已随本地JSON归档。

## 服务器产物与验证范围

- 指定权重：`/nas/liziyan103/seatrack/SEATrack/SEATrack-main/models_rgbd_self/checkpoints/rgbd/SEATrack_ep0025.pth.tar`。
- 独立工作区：`/nas/liziyan103/seatrack/evaluation_runs/self_vot22_ep0025/full`。
- 报告：`/nas/liziyan103/seatrack/evaluation_runs/self_vot22_ep0025/full/analysis/self_vot22_ep0025/results.json`。
- 分析日志：`/nas/liziyan103/seatrack/logs/self_vot22_analysis_20261023_104102.log`。
- 单序列日志：`/nas/liziyan103/seatrack/logs/self_vot22_smoke_20261022_003123.log`。
- 权重使用清单（已回收并解析）：`/nas/liziyan103/seatrack/evaluation_runs/self_vot22_ep0025/checkpoint_used.json`。

入口为 [test_vot22_self.py](../../../scripts/test_vot22_self.py)，在 VOT 子进程内覆盖 checkpoint，使用 `vot2022/rgbd` stack；分析禁用缓存。用户操作服务器，助手未独立连接核验。完整 JSON、HTML 与权重使用清单现已本地归档核验；运行日志、轨迹和权重文件仍未在本次回收。

此前 RGB-D 自训练文件与官方路径文件参数相同的来源疑点尚未解决；本次评测完成不消除这一限制。相关证据与经过见 [复现日志](../../../复现日志.md)。

## 本地核验结果

2026-10-23补下载后，已实际解析[results.json](reports/results.json)，确认tracker=rgbd_self_ep0025、baseline实验、127序列；名单与[权重使用清单](provenance/checkpoint_used.json)一致，三项指标与上表一致。[report.html](reports/report.html)已保存且非空，未逐页做视觉验收。

清单SHA256为`f019674f7f50b1fbf867eade32649d5bf18e120a10e606fa10e30359a3afd1e7`，与历史自训练文件记录一致；这不是本次读取远端权重重算哈希。首次下载后目录缺失及补下载过程保留在[日志](../../../复现日志.md#log-2026-10-23)，当前已不存在报告缺失问题。
