# RGBT234 自训练 RGB-T 权重评测

2026-10-22 用户回传全量跟踪与指标计算结果：234 个序列、116,649 帧，跟踪退出码为 0。本地仅归档回传摘要，尚未下载完整报告或轨迹，也未连接服务器独立复查。

| 评测口径 | MPR (%) | MSR (%) |
|---|---:|---:|
| 参考工具预测取整口径（主结果） | 86.141946 | 63.733215 |
| 未取整诊断 | 86.183550 | 63.670792 |

与此前官方权重同脚本主结果 87.814611 / 63.908269 相比，分别降低 1.672665 / 0.175054 个百分点。仅据本次结果不能确定差异来自训练方式、随机性或其他因素。沿用项目 Python 参考协议，尚未与官方 MATLAB 工具交叉验证。

## 服务器文件

- 权重：`/nas/liziyan103/seatrack/training_runs/20261011_remaining/rgbt/models/checkpoints/rgbt/SEATrack_ep0060.pth.tar`
- 结果：`/nas/liziyan103/seatrack/SEATrack/RGBT_workspace/self_rgbt_ep0060_rgbt234/RGBT234`
- 报告：`/nas/liziyan103/seatrack/SEATrack/RGBT_workspace/self_rgbt_ep0060_rgbt234/metrics_20261022_002510.json`
- 权重使用记录（入口设计保存，尚未回收核查）：`/nas/liziyan103/seatrack/SEATrack/RGBT_workspace/self_rgbt_ep0060_rgbt234/checkpoint_used.json`
- 跟踪日志：`/nas/liziyan103/seatrack/logs/self_rgbt234_full_20261021_135732.log`

运行位置为 cp-2；此前回传硬件为 RTX 3090，提供的启动指令选用 GPU 6。程序报告 elapsed 8355 秒，含检查和跳过已完成序列等开销，不能视为全量纯推理时间；yellowcar 的 11.27048 FPS 仅为该序列的脚本计时。

跟踪入口：[test_rgbt234_self.py](../../../scripts/test_rgbt234_self.py)。指标脚本：[evaluate_rgbt234.py](../../../scripts/evaluate_rgbt234.py)。官方权重对照：[历史结果](../2026-10-13_official-rgbt/README.md)。
