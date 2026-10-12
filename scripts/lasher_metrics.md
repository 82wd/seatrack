# LasHeR 指标计算说明

`evaluate_lasher.py` 使用 CPU 和 NumPy，兼容服务器 Python 3.8，不依赖 PyTorch、MATLAB 或 Octave。读取已生成的预测文件，不会重新跟踪或修改输入数据。报告包含汇总指标、曲线、逐序列计数和输入文件 SHA-256。

## 来源与验证范围

参考的是公开维护的 [RGBT-Tracking / LasHeR_Toolkit](https://github.com/xuboyue1999/RGBT-Tracking/tree/3613a3cbc9d0626a54c7755a549241f658ef6d46/LasHeR_Toolkit)，固定提交 `3613a3cbc9d0626a54c7755a549241f658ef6d46`。这不是已核验与 LasHeR 官方网盘版本逐字一致的工具；本脚本也不是官方发布的 Python 评测器。

逐项核对的文件：

- `readme.md`、`run_tracker_performance_evaluation.m`：PR/NPR 使用 threshold 排名，MATLAB 下标 21；SR 使用 AUC 排名。
- `utils/calc_seq_err_robust.m`：中心、归一化、首帧和无效标注规则。
- `utils/calc_rect_int.m`：xywh 矩形重叠。
- `utils/eval_tracker.m`：阈值集合、比较运算和帧数分母。
- `utils/plot_draw_save.m`：序列等权汇总，排除全零曲线。

已用合成样例验证边界行为；尚未运行 MATLAB/Python 数值对照，也未在此本地电脑取得用户的 LasHeR 全量预测。现阶段生成的指标应称为“公开工具规则的 Python 计算结果”，待对照参考标注/作者原始结果后再作最终复现结论。

## 具体口径

| 项目 | 实现 |
|---|---|
| 输入 | N×4，xywh；预测支持空格或逗号分隔 |
| 标注 | 默认读取每个序列的 `visible.txt`，不取两模态最好值 |
| PR | 中心误差 ≤ 20 像素，序列等权平均 |
| NPR | 中心坐标分别除以 GT 宽高，再求距离；阈值 ≤ 0.20 |
| SR | IoU 严格大于 0:0.05:1 的成功曲线，21 点算术均值 |
| 首帧 | 内存中将第一帧预测换成第一帧 GT；不删除首帧 |
| 非正预测宽高 | 第二帧起按参考工具沿用前一帧预测，并记录次数 |
| 无效 GT | 任意 xywh 值 ≤ 0，中心误差和 IoU 设为 -1；这会使其计入 PR/NPR 成功、SR 失败，保留参考工具行为并记录数量 |
| NaN/Inf、长度不同 | 明确报错；不偷偷截断、补齐或掩盖不完整输入 |
| 汇总 | 每种曲线分别排除全零序列，再等权平均；报告记录被排除名称 |

注意：NPR 的 0.20 阈值不同于“归一化精度曲线 AUC”；SR 的 21 点均值不同于直接取平均 IoU或梯形积分。完美预测在该 SR 口径下为 20/21，因为 IoU > 1 永远为假。浮点阈值附近保留参考公式的运算顺序，不自行添加容差。

默认数据集 `visible.txt` 与工具 `annos/<sequence>.txt` 尚未全量对照。拿到工具标注后，可以传 `--annotation-root <annos目录>` 使用参考标注重新计算；不要为匹配论文数字切换规则而不记录。

## 用户在服务器运行

先将本地 `scripts/evaluate_lasher.py` 上传为 `/nas/liziyan103/seatrack/scripts/evaluate_lasher.py`（目录不存在先创建）。在已激活的 `seatrack` 环境中执行：

```bash
python /nas/liziyan103/seatrack/scripts/evaluate_lasher.py \
  --data-root /nas/liziyan103/seatrack/datasets/lasher/testingset \
  --results /nas/liziyan103/seatrack/SEATrack/RGBT_workspace/LasHeR \
  --output /nas/liziyan103/seatrack/result/lasher/2026-10-11_official-rgbt/reports/lasher_metrics_python.json
```

输出文件若已存在会拒绝覆盖。需要重算时使用新文件名。控制台输出百分数 PR/NPR/SR、帧数和异常标注计数；JSON 中 `scores` 为 0–1 值，`percent` 为百分数。

本地测试：`python3 -m unittest discover -s scripts -p 'test_evaluate_lasher.py' -v`。
