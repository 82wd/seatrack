# LasHeR 官方 RGB-T 权重评测

日期：2026-10-11（北京时间）。来源：用户回传服务器终端输出；完整 JSON 和原始预测尚未下载到本目录。本文件为实测摘要，不是完整报告。

| 指标 | 实测百分数 | 论文百分数 |
|---|---:|---:|
| PR | 71.5961 | 71.6 |
| NPR | 67.4782 | 67.5 |
| SR | 57.2631 | 57.3 |

三项值取一位小数与论文一致。共 245 个序列、220703 帧，全量名称/行数/预测有限性检查已由用户确认 PASS。

## 方法与验证范围

- 使用官方 RGB-T checkpoint，属于权重评测，不是自训练复现。
- 使用项目 [Python 指标脚本](../../../scripts/evaluate_lasher.py)；其 [公式和来源说明](../../../scripts/lasher_metrics.md) 已保存。
- 无效 GT 帧计数 1260，按参考工具 `xywh` 所有分量均需大于 0 的规则统计；其中心误差和 IoU 均为 -1，分别计为距离成功/重叠失败，不剔除。这不是缺失预测的计数。
- 沿用前一帧预测的修补次数为 0。
- 尚未执行 MATLAB/Python 数值对照，数据集 visible.txt 与参考工具 annos 的全量一致性尚未验证。

## 服务器文件

```text
报告：/nas/liziyan103/seatrack/result/lasher/2026-10-11_official-rgbt/reports/lasher_metrics_python.json
预测：/nas/liziyan103/seatrack/SEATrack/RGBT_workspace/LasHeR/
标注：/nas/liziyan103/seatrack/datasets/lasher/testingset/
权重：/nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbt/SEATrack_ep0060.pth.tar
```

后续归档完整 JSON、原始轨迹、运行日志与配置快照，再记录校验值；当前不把未下载文件列为本地交付。

## 2026-10-23 本地归档核验更新

完整指标JSON已下载至本目录reports/并成功解析，指标与此前回传一致；以上“报告尚未下载”的历史说明由本条更新。此次未重新运行评测，未据报告检查所有轨迹。
