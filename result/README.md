# 评测结果索引

**本地核验日期：2026-10-23。** 本表区分论文数值、官方路径权重、自训练路径权重；原始JSON是指标依据，README是摘要。报告本地可读不代表已重新运行实验或查清权重来源。

## 五个基准对照

所有数值按百分数或原始分数×100展示；VOT三项均为原始分数×100。`待确认`不是0分。

| 数据集 | 指标顺序 | 论文 | 官方路径权重 | 自训练路径权重 |
|---|---|---|---|---|
| LasHeR | PR / NPR / SR | 71.6 / 67.5 / 57.3 | 71.5961 / 67.4782 / 57.2631 | 待确认 |
| RGBT234 | MPR / MSR | 87.8 / 63.9 | 87.814611 / 63.908269 | 86.141946 / 63.733215 |
| DepthTrack | PR / RE / F | 62.9 / 63.5 / 63.2 | 62.909598 / 63.545955 / 63.226176 | 62.909598 / 63.545955 / 63.226176† |
| VOT2022 | EAO / Acc / Rob | 73.6 / 82.1 / 88.4 | 待确认 | 73.759773 / 82.063579 / 88.599982† |
| VisEvent | PR / SR | 77.1 / 60.3 | 77.095041 / 60.260964 | 77.095041 / 60.260964† |

† RGB-D/RGB-E历史权重来源疑点仍存在，不能据这些数值声称独立训练已完全复现作者模型。DepthTrack自训练本地JSON现已回收，精度由报告确认，不再仅依赖早期四位小数回传。论文只保留一位小数，不能据细小差异推断显著提升。

## 已在本地的报告

| 实验 | 序列数 | 报告 | 附加文件与核验范围 |
|---|---:|---|---|
| LasHeR官方（2026-10-11） | 245 | [JSON](lasher/2026-10-11_official-rgbt/reports/lasher_metrics_python.json) · [说明](lasher/2026-10-11_official-rgbt/README.md) | 220703帧；已有轨迹目录，本次未重新逐条验收 |
| RGBT234官方（2026-10-13） | 234 | [JSON](rgbt234/2026-10-13_official-rgbt/reports/official_rgbt234_full_20261013_074307_metrics.json) · [说明](rgbt234/2026-10-13_official-rgbt/README.md) | 116649帧；完整轨迹待回收 |
| RGBT234自训练（2026-10-22） | 234 | [JSON](rgbt234/2026-10-22_self-rgbt/reports/metrics_20261022_002510.json) · [说明](rgbt234/2026-10-22_self-rgbt/README.md) | [权重使用清单](rgbt234/2026-10-22_self-rgbt/provenance/checkpoint_used.json)已核对；轨迹/运行日志待回收 |
| DepthTrack官方（2026-10-11） | 50 | [JSON](depthtrack/2026-10-11_official-rgbd/reports/depthtrack_metrics.json) · [HTML](depthtrack/2026-10-11_official-rgbd/reports/depthtrack_report.html) · [说明](depthtrack/2026-10-11_official-rgbd/README.md) | 76373帧；历史归档含日志及复现配置 |
| DepthTrack自训练（2026-10-11） | 50 | [JSON](depthtrack/2026-10-11_self-rgbd/reports/results.json) · [HTML](depthtrack/2026-10-11_self-rgbd/reports/report.html) · [说明](depthtrack/2026-10-11_self-rgbd/README.md) | tracker=rgbd_self；与官方报告数值相同，来源待核查 |
| VisEvent官方（2026-10-13） | 320 | [JSON](visevent/2026-10-13_official-rgbe/metrics.json) · [说明](visevent/2026-10-13_official-rgbe/README.md) | 已有轨迹压缩包及运行日志；保留标注差异限制 |
| VisEvent自训练（2026-10-14） | 320 | [JSON](visevent/2026-10-14_self-rgbe/reports/metrics_reference_20261014_141237_849005.json) · [说明](visevent/2026-10-14_self-rgbe/README.md) | 指标与官方报告相同；历史参数/轨迹相同问题未解决 |
| VOT2022自训练（2026-10-23） | 127 | [JSON](vot22/2026-10-23_self-rgbd/reports/results.json) · [HTML](vot22/2026-10-23_self-rgbd/reports/report.html) · [说明](vot22/2026-10-23_self-rgbd/README.md) | [权重清单](vot22/2026-10-23_self-rgbd/provenance/checkpoint_used.json)名单与报告一致；轨迹/运行日志待回收 |

本轮下载的“八项”指六组报告加两份权重使用清单；加上此前已归档的DepthTrack/VisEvent官方报告，目前上表共有八组实验报告，二者计数口径不同。

## 验证边界

- **权重来源：** 历史检查发现RGB-D/RGB-E自训练末轮与当时官方路径的net参数相同，也有早末轮参数更新证据。文件来源和覆盖/恢复链路尚未查明；指定checkpoint的评测成功不等于独立训练可重复性已验证。
- **指标实现：** LasHeR、RGBT234、VisEvent使用Python参考实现，尚未与MATLAB交叉验证。RGBT234主结果固定为预测取整口径，不按诊断分数高低选口径。
- **VisEvent对齐：** 一条序列预测332行、官方标注306行，按已核对的参考规则截断前缀；完整320序列是主结果，319序列仅诊断。详见[VisEvent说明](visevent/2026-10-13_official-rgbe/README.md)。
- **VOT报告：** 本地核对了JSON中的tracker、序列与指标，并确认HTML非空；没有在此次整理中逐页验收HTML或重新运行GPU任务。
- **缺失状态：** LasHeR自训练指标、官方VOT2022最终报告未取得。先查已有服务器产物，不因为本地缺失就重跑。

## 目录约定

```text
result/<数据集>/<日期_实验>/
├── README.md       # 指标摘要、协议、来源和局限
├── reports/        # JSON、HTML及网页资源
├── provenance/     # checkpoint路径和SHA等使用清单（如有）
├── trajectories/   # 预测轨迹（如有）
└── logs/           # 运行日志及退出码（如有）
```

部分早期归档把metrics.json或轨迹压缩包放在实验根目录，保持已有路径，不为形式统一移动报告。文件日期按实验记录归类，服务器时间戳原样保留。

[下载脚本](../scripts/download_result_reports.sh)在本地运行，只下载已列出的报告/清单；已有目标跳过，不能把跳过视为内容重新验证。更新指标后同步[文本汇总](../复现结果汇总.txt)和[操作日志](../复现日志.md)，待办统一放在[plan.md](../plan.md)。
