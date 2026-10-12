# 评测结果归档

按 `数据集/评测日期_实验名称/` 整理，每次独立评测使用单独目录。日期采用北京时间；原始逐序列文件保留评测工具的命名，便于重新分析。

| 数据集 | 实验 | 状态 | 结果 |
|---|---|---|---|
| DepthTrack | 2026-10-11，官方 RGB-D 权重 | 完成，50 序列 / 76,373 帧 | PR / RE / F = 62.9 / 63.5 / 63.2 |
| LasHeR | 2026-10-11，官方 RGB-T 权重 | 用户回传：Python 指标完成，245 序列 / 220,703 帧；完整文件待归档 | PR / NPR / SR = 71.5961 / 67.4782 / 57.2631 |

- [本次 DepthTrack 归档说明](depthtrack/2026-10-11_official-rgbd/README.md)
- [HTML 指标报告](depthtrack/2026-10-11_official-rgbd/reports/depthtrack_report.html)
- [完整操作记录](../复现日志.md)
- [LasHeR 实测摘要与验证范围](lasher/2026-10-11_official-rgbt/README.md)

此目录仅归档实际评测产物，不存放数据集、模型权重或作者提供的参考结果。
