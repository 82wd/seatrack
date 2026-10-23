# DepthTrack 自训练路径权重评测

2026-10-23已回收并解析[JSON](reports/results.json)，同时保存[HTML](reports/report.html)。实验发生于2026-10-11；tracker为rgbd_self，50个序列，VOT PrecisionRecall分析输出如下（×100）：

| PR | RE | F-score |
|---:|---:|---:|
| 62.909598 | 63.545955 | 63.226176 |

当前报告数值与本地官方报告相同。历史证据显示自训练第5至第25轮有参数更新，但最终模型与当时官方路径net参数逐张量相同，权重来源问题未解决。因此保留为自训练路径评测报告，不据此宣称独立训练精确复现。证据演变见[复现日志](../../../复现日志.md)。

报告源目录：`/nas/liziyan103/seatrack/SEATrack/SEATrack-main/Depthtrack_self_workspace/analysis/depthtrack_self`。本次只核验本地报告，未重新连接服务器或检查全部轨迹。
