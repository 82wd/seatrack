# 脚本使用说明

核对日期：2026-10-23。此处说明当前本地脚本的用途和参数，不代表服务器实时状态。服务器任务由用户执行；已完成的实验不需要为查看结果重新运行。

## 先分清四类动作

| 动作 | 入口 | 运行位置与产物 |
|---|---|---|
| 自训练权重逐帧跟踪 | `test_*_self.py` | Linux服务器/GPU；输出轨迹和权重使用清单，不更新模型 |
| 计算指标 | `evaluate_*.py`、VOT `--stage analyze` | CPU；读取已有轨迹，输出报告 |
| 下载报告 | [download_result_reports.sh](download_result_reports.sh) | 本地Mac；下载六组报告和两份清单到`result/` |
| 训练或监控 | `remaining_training/`、`remaining_official/` | 历史训练调度、数据准备与进度脚本，不能作为当前运行状态 |

`test_*_self.py`名称中的test指数据集评测，不是单元测试。`test_evaluate_lasher.py`和`remaining_training/test_accumulation.py`才是实现验证脚本。

## 自训练权重评测

| 入口 | 参数 | 选定权重与结果 |
|---|---|---|
| [test_lasher_self.py](test_lasher_self.py) | `--gpu N`，可选`--smoke` | RGB-T ep0060；`RGBT_workspace/self_rgbt_ep0060/LasHeR` |
| [test_rgbt234_self.py](test_rgbt234_self.py) | `--gpu N`，可选`--smoke` | RGB-T ep0060；`RGBT_workspace/self_rgbt_ep0060_rgbt234/RGBT234` |
| [test_visevent_self.py](test_visevent_self.py) | `--gpu N`，可选`--smoke` | RGB-E ep0045；`RGBE_workspace/self_rgbe_ep0045/VisEvent` |
| [test_vot22_self.py](test_vot22_self.py) | `--stage prepare/smoke/full/analyze`；smoke/full必须传`--gpu N` | 内层RGB-D ep0025；`evaluation_runs/self_vot22_ep0025/{smoke,full}` |

上表前三项结果路径相对服务器外层`SEATrack/`；VOT结果路径相对服务器`/nas/liziyan103/seatrack/`。前三项去掉`--smoke`即全量；VOT默认stage为smoke，单序列与全量使用不同工作区。

### 服务器路径约定

- RGB-T权重：`/nas/liziyan103/seatrack/training_runs/20261011_remaining/rgbt/models/checkpoints/rgbt/SEATrack_ep0060.pth.tar`。
- RGB-E权重：同一training_runs目录中的`rgbe/models/checkpoints/rgbe/SEATrack_ep0045.pth.tar`。
- RGB-D权重：`/nas/liziyan103/seatrack/SEATrack/SEATrack-main/models_rgbd_self/checkpoints/rgbd/SEATrack_ep0025.pth.tar`。
- LasHeR数据：`datasets/lasher/testingset`；RGBT234：`datasets/rgbt234`；VisEvent入口实际使用`datasets/visevent/test_subset`及官方320序列清单；VOT：`datasets/vot2022_rgbd`。

脚本当前绑定这些服务器路径，不是通用路径参数化工具。换机器时先适配脚本和代码里的配置。评测前查看当前GPU状态、确认`Actual checkpoint`，避免默认参数读取官方路径模型。脚本保存checkpoint路径和SHA清单，但这不能消除历史权重来源疑点。

### VOT阶段说明

`prepare`只建立目录、数据链接、VOT桥接入口与使用清单；`smoke`优先跑adapter01_indoor_1；`full`跑127序列；`analyze`以禁用缓存、单worker方式生成JSON/HTML。smoke/full启动前要求所选GPU至少4096MiB空闲，这是初筛而非运行期间显存保证。

VOT多起点会多次启动跟踪器。判断完成时同时保留VOT结束提示、退出码和完整报告；分析成功后再核对tracker ID、序列名单与指标类型，不能用EAOCurve当作EAOScore。

## 指标脚本

| 入口 | 必要参数或依赖 | 口径 |
|---|---|---|
| [evaluate_lasher.py](evaluate_lasher.py) | `--data-root`、`--results`、`--output`；可选`--annotation-root` | PR/NPR/SR；细节见[LasHeR指标说明](lasher_metrics.md) |
| [evaluate_rgbt234.py](evaluate_rgbt234.py) | `--dataset`、`--results`、`--output` | 主结果为rounded_reference_protocol；raw_float_diagnostic仅诊断 |
| [evaluate_visevent.py](evaluate_visevent.py) | 可选`--result-dir`；其他参考文件路径见脚本 | 官方320序列参考规则；319序列排除诊断不是主结果 |
| [test_vot22_self.py](test_vot22_self.py) | `--stage analyze` | VOT2022 EAO、Accuracy、Robustness；无需GPU |

这些入口不是同一套CLI。应先读对应脚本参数，不能把`--results`与`--result-dir`混用。Python指标实现与MATLAB交叉验证、VisEvent对齐限制见[结果索引](../result/README.md#验证边界)。

## 报告下载

在Mac项目根目录运行：

```bash
bash scripts/download_result_reports.sh
```

连接参数已按用户配置写在脚本中；脚本只调用scp下载，不执行远端训练或分析。先存入临时目录，再放置最终文件，已有目标跳过；失败会保留暂存路径并统计。末尾检查目标存在、报告目录有非空results.json，但并不验证所有指标或网页显示。完成后仍需解析JSON并核对tracker、序列和清单。

## 历史运行与辅助脚本

| 目录/文件 | 用途及使用边界 |
|---|---|
| [remaining_training/README.md](remaining_training/README.md) | RGB-T/RGB-E训练方案变更记录；早期双卡与后期单卡方案并存，不能按文件名“remaining”推断仍在等待 |
| [remaining_training/](remaining_training/) | 配置、解压、排队、梯度累积、训练核验及进度显示；不要为读报告启动prepare或run_task |
| [remaining_official/](remaining_official/) | 官方权重阶段的数据准备、GPU观察和进度工具；部分路径/trackers指向旧官方工作区，不能直接监控新的self工作区 |
| [dl.sh](dl.sh) | 下载辅助工具；不是报告归档入口 |

报告、日志、轨迹的实际位置按[复现日志](../复现日志.md)中的最近证据查找。旧GPU分配、tmux存在状态和队列计划均需重新核实。
