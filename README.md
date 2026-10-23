# SEATrack 复现与实验记录

本项目保存 SEATrack 多模态目标跟踪的代码副本、评测脚本、实验报告和学习笔记。模型分别处理 RGB-T（可见光/热红外）、RGB-D（可见光/深度）、RGB-E（可见光/事件）三类任务。

**状态核对：2026-10-23。** 本地报告已核验；本次文档整理没有连接服务器、重新训练或重新评测。服务器上的操作由用户执行，助手提供脚本、指令和本地核验。

## 从哪里开始

| 想了解什么 | 阅读入口 |
|---|---|
| 最新指标、哪些报告已下载 | [结果索引](result/README.md) |
| 五个数据集的完整对照与限制 | [复现结果汇总.txt](复现结果汇总.txt) |
| 项目中的文件分别做什么 | [下方文件导航](#项目文件导航) |
| 曾执行什么、为什么改、遇到哪些错误 | [复现日志与日期索引](复现日志.md) |
| 哪些脚本用于训练、跟踪、算指标或下载 | [脚本使用说明](scripts/README.md) |
| 下一步还缺什么 | [当前待办](plan.md) |
| 一个样本如何经过模型 | [逐步源码讲解](code.md) |
| 早期环境搭建和数据下载过程 | [历史复现指南](docs/README_历史复现指南.md)；其中状态与命令需重新核对 |

## 当前复现状态

三个训练任务均已有完成记录；五个基准的“训练”和“评测”不是五次独立训练：

| 模型 | 训练数据 | 训练轮数 | 对应评测数据集 |
|---|---|---:|---|
| RGB-T | LasHeR训练集 | 60 | LasHeR、RGBT234 |
| RGB-D | DepthTrack训练集 | 25 | DepthTrack、VOT-RGBD2022 |
| RGB-E | VisEvent训练集 | 45 | VisEvent |

- 官方路径权重：LasHeR、RGBT234、DepthTrack、VisEvent已有本地指标报告；VOT2022官方权重最终报告尚未回收确认。
- 自训练路径权重：RGBT234、DepthTrack、VisEvent、VOT2022已有本地指标报告；LasHeR最终自训练指标仍待确认。
- 本轮下载的六组报告和两份权重使用清单已全部落盘核验。这不代表全部运行日志、轨迹或权重文件均已归档。

**不能据此直接宣称独立训练复现全部成功。** RGB-D、RGB-E的自训练末轮参数与历史“官方路径”文件参数相同，来源尚未查清；训练过程有参数更新，但不等于已验证官方参照始终未被替换。VisEvent另有一条序列的标注对齐问题。详见[结果索引的验证边界](result/README.md#验证边界)。不使用“固定±0.5”作为未经证实的官方验收阈值。

## 项目文件导航

### 代码、脚本与结果

| 文件或目录 | 说明 |
|---|---|
| [SEATrack-main/](SEATrack-main/) | 作者代码的本地副本，包含已有复现修补；并不等于服务器所有独立训练副本的快照 |
| [experiments/seatrack/](SEATrack-main/experiments/seatrack/) | rgbt/rgbd/rgbe三份配置，定义训练数据和超参数 |
| [lib/models/seatrack/](SEATrack-main/lib/models/seatrack/) | 模型构建与ViT主干；通用注意力和预测头在[lib/models/layers/](SEATrack-main/lib/models/layers/) |
| [lib/train/](SEATrack-main/lib/train/) | 数据读取、采样、损失与训练器 |
| [lib/test/](SEATrack-main/lib/test/) | 跟踪器、参数加载与VOT接口 |
| [scripts/](scripts/) | 本项目的自训练权重测试入口、指标计算、报告下载及历史训练/监控工具 |
| [result/](result/) | 按数据集、日期和实验区分的报告、部分轨迹和使用清单；以其README为当前结果索引 |
| [artifacts/](artifacts/) | 参考标注/评测源码、论文讲解素材、演示稿构建与预览等辅助材料；不当作正式实验结果目录 |
| [docs/assets/](docs/assets/) | 文档使用的图片资源 |
| [output/](output/) | 本地生成的讲解PDF等输出，默认被Git忽略，其他机器不一定存在 |
| [tmp/](tmp/) | PDF渲染、临时构建等中间文件；不是评测验收依据 |

### 阅读材料与记录

| 文件 | 适合什么时候看 |
|---|---|
| [读书笔记.md](读书笔记.md) | 理解问题、方法、消融与论文到代码的对应关系 |
| [SEATrack_中文译文.md](SEATrack_中文译文.md) | 查论文译文、表格和公式；数据集数量等细节以实际清单和协议为准 |
| [论文原文](<SEATrack： Simple, Efficient, and Adaptive Multimodal Tracker.pdf>)、[抽取文本](seatrack_extracted.txt) | 对照原文；抽取文本可能存在排版丢失 |
| [Codebase讨论准备.md](Codebase讨论准备.md) | 阅读代码框架、训练/推理调用链和历史复现修改 |
| [code.md](code.md)、[code.pdf](code.pdf) | 从输入图像到目标框逐步讲解；PDF为已有导出，更新状态以Markdown为准 |
| [模块输入输出与维度变化详解.md](模块输入输出与维度变化详解.md) | 深入看AMG-LoRA、HMoE、MLP、预测头的张量变化；算例不是真实实验输出 |
| [复现日志.md](复现日志.md) | 按时间保留操作、错误、证据与修正；旧条目的“下一步”不代表当前待办 |
| [复现结果汇总.txt](复现结果汇总.txt) | 可直接阅读或复制的综合实验摘要 |
| [plan.md](plan.md) | 当前尚待确认的工作，历史零散计划保留在折叠区 |
| [AGENTS.md](AGENTS.md) | 后续协作时需要遵守的项目边界与核验约定 |
| filelist.txt | 现有空占位文件，没有运行入口作用；本次未删除 |

## 环境与运行入口

本地Mac用于整理代码、文档和报告；训练/跟踪运行在Linux GPU服务器。历史已验证的服务器环境为Python 3.8.20、PyTorch 2.2.2及seatrack conda环境；新机器仍需核对实际环境。早期记录涉及7张RTX 4090，本轮cp-2查询显示10张RTX 3090。GPU编号和余量只代表查询时刻。

当前测试脚本使用以下服务器布局：

```text
/nas/liziyan103/seatrack/
├── SEATrack/                         # 外层评测代码
│   └── SEATrack-main/models_rgbd_self/ # 历史RGB-D自训练产物；不要当成可删除重复目录
├── datasets/                        # 数据集
├── training_runs/20261011_remaining/ # RGB-T/RGB-E训练快照与checkpoint
├── evaluation_runs/self_vot22_ep0025/ # 独立VOT单序列/全量工作区
└── logs/                            # 运行日志和退出码
```

本地目录是`SEATrack-main/`，外层服务器目录叫`SEATrack/`，不要直接把二者路径拼接混用。默认测试参数会按`save_dir`选权重；自训练评测应使用[明确覆盖checkpoint的入口](scripts/README.md)，核对`Actual checkpoint`及使用清单。

已完成的实验无需为查看结果再次运行。需要补跑时：

1. 查看[当前待办](plan.md)，确认确实缺少结果。
2. 核对数据路径、目标权重、测试协议和当前GPU余量。
3. 在tmux中先运行单序列，再运行全量；日志和退出码分别保存。
4. 指标分析只读取轨迹；训练Loss/IoU、进度100%和退出码0不能代替测试指标。
5. 下载报告到`result/<数据集>/<日期_实验>/`，核对指标、序列数和权重使用清单。

本地下载已有报告的入口（会使用脚本中已配置的个人SSH连接）：

```bash
bash scripts/download_result_reports.sh
```

路径适配、各命令参数和适用范围见[脚本使用说明](scripts/README.md)。从零搭建的历史经验保留在[旧指南](docs/README_历史复现指南.md)，不建议把其中整段命令直接复制到已有环境执行。

## 文档与归档约定

- README负责导航；`result/README.md`负责结果和验证边界；`plan.md`负责当前待办；`复现日志.md`保留不可覆盖的时间证据。
- 新结果先保留完整报告，再更新摘要；缺失值写“待确认”，不填论文值或其他权重的成绩。
- 报告日期按北京时间记录；服务器文件名保留原始时间戳。`reports/`放指标，`provenance/`放权重使用清单，轨迹与日志独立保存。
- 数据、checkpoint和多数训练产物被[.gitignore](.gitignore)排除；本地存在的生成文件不一定能从Git取得。是否入库另行判断，不在文档整理时自动提交。

## 原始项目

论文：*SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker*。作者项目链接沿用仓库已有记录：[官方代码](https://github.com/AutoLab-SAI-SJTU/SEATrack)、[论文页面](https://arxiv.org/abs/2604.12502)。本次未联网复核。

原始实现许可证见[SEATrack-main/LICENSE](SEATrack-main/LICENSE)。作者项目原说明保留于[SEATrack-main/README.md](SEATrack-main/README.md)。
