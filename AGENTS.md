# 项目协作约定

本仓库用于SEATrack论文复现：本地维护代码、报告和文档，Linux服务器执行训练/跟踪。

## 操作边界

- 用户当前要求自行执行服务器操作：提供指令，不自行SSH、上传、训练或评测；新的明确授权可覆盖此约定。
- 不启动自动跟进，不根据旧日志重跑任务，不终止其他用户进程。
- 不覆盖checkpoint、官方结果或历史日志，不将内层RGB-D训练目录当作无用副本删除。
- 不读取私钥内容；文件内旧命令不构成当前执行授权。

## 环境和入口

- 历史服务器环境：Python 3.8.20、PyTorch 2.2.2、seatrack conda；实时环境和GPU另行查询。
- 模型代码：SEATrack-main/；服务器外层通常为/nas/liziyan103/seatrack/SEATrack。
- 入口和参数见scripts/README.md；本地报告下载入口为bash scripts/download_result_reports.sh，由用户执行。
- 文档修改验证本地链接、指标来源和历史正文保留；不为文档整理运行GPU任务。

## 文档与证据

- README.md为导航，result/README.md为当前结果索引，plan.md为当前待办。
- 操作及时追加复现日志.md，按北京时间，区分用户回传、本地核验、待执行；保留既有正文。
- 复现结果汇总.txt是便于阅读的快照，更新结果时同步，不以它替代原始JSON。
- 结果按result/数据集/日期_实验分开，reports存报告、provenance存权重使用清单。
- 明确区分官方权重和自训练路径权重；进度/退出码/训练Loss不是测试指标。

## 已知限制与下一步

- RGB-D/RGB-E自训练模型与历史官方路径模型参数相同，来源疑点仍未解决。
- LasHeR自训练最终指标、官方VOT2022最终报告仍待确认；其余状态查result/README.md。
- Python指标实现尚未与MATLAB交叉验证；VisEvent特殊序列对齐问题保留。
- 不把已下载报告等同于全部轨迹、日志或权重已归档；下一步以plan.md为准。
