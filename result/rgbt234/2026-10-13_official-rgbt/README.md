# RGBT234 官方 RGB-T 权重测试

2026-10-13 用户回传服务器输出：234 个序列、116,649 帧轨迹验收通过，异常 0，额外 txt 文件 0；指标计算完成。

| 评估口径 | MPR (%) | MSR (%) |
|---|---:|---:|
| 参考工具预测取整口径 | 87.814611 | 63.908269 |
| 未取整诊断 | 87.826440 | 63.850523 |

主结果保留一位小数为 87.8 / 63.9，与本仓库论文译文表中的数值一致。这是数值精度上的一致，不代表已经使用官方 MATLAB 工具交叉核验。

使用 [evaluate_rgbt234.py](../../../scripts/evaluate_rgbt234.py)，参考第三方 [RGBT_toolkit](https://github.com/opacity-black/RGBT_toolkit) 的评估口径。每帧取两模态最小中心误差与最大 IoU，MPR 使用 <=20 像素，MSR 使用严格大于 0:0.05:1 的成功率均值，序列等权；保留首帧。float32 加载、预测取整后以 float64 计算几何。未取整分数仅作诊断。

## 服务器文件

- 任务：`official_rgbt234_full_20261013_074307`
- 结果：`/nas/liziyan103/seatrack/evaluation_runs/20261013_remaining/code/RGBT_workspace/official_rgbt234_full_20261013_074307/RGBT234`
- 指标：`/nas/liziyan103/seatrack/evaluation_runs/20261013_remaining/rgbt234/official_rgbt234_full_20261013_074307_metrics.json`
- 日志：`/nas/liziyan103/seatrack/logs/official_rgbt234_full_20261013_074307.log`

2026-10-23已下载并解析[完整JSON](reports/official_rgbt234_full_20261013_074307_metrics.json)，指标及234序列/116649帧与回传一致；完整轨迹及运行日志尚未下载归档。未连接服务器独立复查，未运行官方 MATLAB 评估。换行兼容修复及各步骤见 [复现日志](../../../复现日志.md)。
