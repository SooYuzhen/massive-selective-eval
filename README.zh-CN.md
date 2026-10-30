# 多语言选择性预测与可靠性评测

这是MASSIVE 1.1中英日意图分类的可复现工程作品：支持本地数据加载、冻结模型预测、dev阈值弃权、风险—覆盖率、同源ID配对和聚类不确定性分析。

**工程GO；MASSIVE独立论文DROP。** 本项目不自称已经验证的LLM-as-a-Judge系统，不把工程成功改写为科研创新。另行执行的真实Judge最小实验不混入此公开包。

真实历史实验包含17,082行预测与108项冻结阈值，两系统各语言的test分母均为2,847个共同源ID。字符模型英/中/日准确率83.81%/82.09%/82.40%，冻结MiniLM共享头84.37%/80.65%/80.72%。这些数字只表示与原intent标签一致，不能证明本地化或关键参数正确。完整结果及限制见[英文README](README.md)。

无需完整语料即可验证最小工程功能：

```bash
PYTHONPATH=src python -m massive_eval demo --out local_outputs/demo
PYTHONPATH=src python -m unittest discover -s tests -v
```

环境需已有NumPy和SciPy；也可以按[安装运行指南](docs/RUNNING.md)安装本地包。demo是合成软件测试，不是新增人工金标准，不用于性能主张。

公开候选只包含代码、汇总结果、许可说明和验证证据，不包含原始语料、模型权重、个人路径或凭据。完整冻结结果重算需要在本机另行提供原第九/十轮资产。原历史文件保持只读，本轮没有自动创建GitHub仓库或推送。

发布前请查看[安全与许可审核](docs/LICENSE_AND_SECURITY.md)和[发布清单](docs/PREPUBLICATION_CHECKLIST.md)。本轮后不默认开启新的论文审计。
