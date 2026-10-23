# RGB-T / RGB-E 历史训练交接记录

> **当前说明（2026-10-23）：** 三项训练均已有完成记录，本目录名称remaining_training沿用历史。下方保留方案演变，不表示任务仍在排队或自动跟进仍运行。当前结果见[结果索引](../../result/README.md)，待办见[plan.md](../../plan.md)。后续服务器操作由用户执行。

## 历史方案与运行记录

运行目录：`/nas/liziyan103/seatrack/training_runs/20261011_remaining`。

- tmux 会话 `seatrack_remaining`，窗口 `rgbt`、`rgbe`。
- `prepare.sh` 解压训练数据；成功后等待 `launch_ready`，再调用 `run_task.py`。
- `configure.py` 创建独立 `code` 副本。原 RGB-D 工程和权重不改写。
- RGB-T：LasHeR，60 轮，LR 0.0004；RGB-E：VisEvent，45 轮，LR 0.00006。每卡 batch 32，2 卡，总 batch 64；PEFT=True，AMP=False，seed=0。
- 每轮保存 checkpoint，保留全部；VisEvent 原配置无验证集，不用测试集调参。
- LasHeR 帧索引改为数值 ID 精确匹配，数据检查要求每个通道 ID 完整覆盖 0..N-1。
- 完整清单检查两模态帧数、标注长度、有效性、配对，并解码每序列首/中/末帧；这不是所有图像逐帧解码。
- 初始化来源为用户上传的 CE 版 OSTrack，SHA-256 `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41`。文件仅有 net、242 个张量，没有 epoch 元数据。
- 启动前等待 2 张连续 60 秒空闲的 GPU（无 compute 进程，显存<128 MiB，利用率<5%），同时排除活动命令声明的 sim_gpu。两个任务用文件锁协调；共享机器的其他用户不受本锁约束，因此无法保证检查后永远没有外部任务进入。
- 每任务 `status.txt`、`events.jsonl`、`process.log`、`training.log`、`data_validation.json`、`initialization.json` 记录证据。
- 权重目录：`<运行目录>/<rgbt或rgbe>/models/checkpoints/<rgbt或rgbe>/`。
- 成功必须有全部轮次 checkpoint、末轮内部 epoch 正确、早末轮参数发生变化；生成 `completion.json`（含各权重 SHA-256）。失败记录 FAILED 并退出，不无限重试。
- 如需暂停尚未启动的队列，在运行目录创建 `PAUSE` 文件；这不会停止已经运行的训练。

这些脚本配置完成不等于训练已开始；以服务器 status.txt 和实际训练日志为准。

历史自动跟进安排（后续记录已暂停，本次未查询实时设置）：本聊天 heartbeat `seatrack` 每两小时检查一次。服务器队列每 30 秒检查 GPU，不调用模型、不消耗 Codex 推理额度；状态等待记录最多每半小时一条。

2026-10-12 05:30 修正：预检查不再要求每个序列可见帧>10，而按原采样器规则记录 sampler_eligible（可见帧>2×搜索/模板帧数之和，长度≥20），每个数据集至少一个可采样序列。完整清单仍全部检查，无序列删除。VisEvent 已解压成功，初次验证失败后重启验证；保留错误证据。

2026-10-12 07:35：两项数据及初始化检查均通过，WAITING_GPU。LasHeR 979序列/514081帧（均可采样）；VisEvent 500序列/212062帧（498可采样，另2条保留并遵循原采样器）。当前尚未启动训练，队列等待两张合规空闲GPU。

2026-10-12 用户明确授权共用物理GPU5/6：覆盖上文完全空闲策略。现在仅考虑5/6，每卡剩余≥14000MiB、util≤20%、持续60秒；RGB-E先跑，RGB-T等RGB-E COMPLETED后跑。通过shared.sh启动，复用已有验证并重验初始化SHA，参数不变。共享卡其他任务可能动态增加占用，OOM时只处理本项目。自动跟进已同步这一授权。

共用试跑实测：RGB-E已到第1轮50/937步，双rank loss正常，无OOM。总卡占用约16.2GiB。后续启动空余门槛调整为18000MiB（覆盖此前14000），RGB-T仍等RGB-E完成；不影响正在运行的RGB-E。

2026-10-13 最新授权与方案（覆盖前述GPU5/6双卡队列）：用户允许所有卡共用。RGB-T改为code_single独立快照，single.sh/run_single.py启动单卡，microbatch4×accumulation16=有效batch64；保持60轮、FP32、LR0.0004，训练每轮59968样本/14992微批/937次optimizer更新，匹配原双卡drop_last消费量。每160微批打印进度，即10次更新。不能把该结果称为与双卡数值完全等价。每卡剩余至少6500MiB持续60秒才选取，PyTorch allocator限制5120MiB，其他进程不停止。若OOM将记录FAILED，不自动改精度或降batch。CPU梯度累积正确性测试已通过；实际GPU运行状态以status及training.log为准。Codex定时跟进仍暂停。

单卡启动门槛最终微调为6000MiB（覆盖上文6500）；allocator仍5120MiB，其余设置不变。

2026-10-13 09:45单卡RGB-T已在GPU3启动，首160/14992微批（10次参数更新）实测通过，无OOM；loss1.74887、IoU0.64857。进度面板命令不变，训练继续60轮。创建checkpoint目录的旧日志文案提及multiple GPUs属硬编码，实际LOCAL_RANK=-1单卡。
