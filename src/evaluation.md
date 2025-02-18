# 使用 SWE-bench 进行评估
John Yang &bull; 2023年11月6日

本教程将介绍如何使用 SWE-bench 评估模型和方法。

## 🤖 生成预测结果
对于 SWE-bench 数据集中的每个任务实例,给定一个问题(`problem_statement`)和代码库(`repo` + `base_commit`),你的模型需要尝试生成一个差异补丁预测。关于 SWE-bench 任务的完整详细信息,请参阅主论文的第2节。

每个预测必须按以下格式进行格式化:
```json
{
    "instance_id": "<唯一的任务实例ID>",
    "model_patch": "<.patch文件内容字符串>",
    "model_name_or_path": "<模型名称(例如 SWE-Llama-13b)>",
}
```

将多个预测存储在一个 `.json` 文件中,格式为 `[<预测1>, <预测2>,... <预测n>]`。不需要为每个任务实例生成预测。

如果你更想专门运行评估来了解其工作原理,可以下载并使用这组[预测结果](https://drive.google.com/uc?export=download&id=11a8mtuX6cafsdVDJAo-Bjw8X7zHS8EYx),这是你的预测应该是什么样子的示例。

## 🔄 运行评估
要运行评估,请修改并运行 `harness/run_evaluation.sh` 脚本,该脚本会调用 `run_evaluation.py` 脚本。需要以下参数:

```bash
python run_evaluation.py \
    --predictions_path <预测.json文件的路径> \
    --swe_bench_tasks <swe-bench.json文件的路径> \
    --log_dir <用于写入每个任务实例日志的文件夹路径> \
    --testbed <用于执行每个任务实例的临时目录路径>
```

其他参数在 `run_evaluation.py` 中定义。下图从高层次上展示了 `run_evaluation.py` 的功能。更多细节请参见 `harness/` 和主论文的附录。

<div align="center">
    <img style="width:70%" src="../assets/evaluation.png">
</div>

## 📈 指标

成功完成 `./run_evaluation.sh` 后,应该已经为每个预测创建了一个日志文件并存储在 `log_dir` 中,日志文件的命名格式如下: `<instance_id>.<model>.eval.log`。

要获取模型的评估结果,请使用 `metrics/report.py` 中的 `get_model_report` 函数。它使用与 `harness/run_evaluation.sh` 相同的参数集。

以下是演示其正确用法的代码片段:

```python
model = "模型名称(与预测中使用的名称相同)"
predictions_path = "预测.json文件的路径"
swe_bench_tasks = "swe-bench.json文件的路径"
log_dir = "包含每个任务实例日志的文件夹路径(与上面的log_dir相同)"

report = get_model_report(model, predictions_path, swe_bench_tasks, log_dir)

for k, v in report.items():
    print(f"- {k}: {len(v)}")
```

给定模型名称,`get_model_report` 函数返回一个按如下格式的字典:
```json
{
    "no_generation": ["实例ID列表"],
    "generated": ["实例ID列表"],
    "with_logs": ["实例ID列表"],
    "install_fail": ["实例ID列表"],
    "reset_failed": ["实例ID列表"],
    "no_apply": ["实例ID列表"],
    "applied": ["实例ID列表"],
    "test_errored": ["实例ID列表"],
    "test_timeout": ["实例ID列表"],
    "resolved": ["实例ID列表"],
}
```

每个键值条目都是一个仓库与每个预测结果的配对,通过 `instance_id` 标识:
* `no_generation`: 预测为 `None`。
* `generated`: 预测非空。
* `with_logs`: 为此预测创建了日志文件(应该 `=` `generated` 的数量)。
* `install_fail`: 执行环境安装失败。
* `reset_failed`: 执行环境的 GitHub 仓库无法正确检出。
* `no_apply`: 预测未能成功应用为补丁。
* `applied`: 预测成功应用为补丁。
* `test_errored`: 测试命令出错。
* `test_timeout`: 测试命令超时。
* `resolved`: 预测通过所有测试。

关于理解报告数字的一些注意事项:
* `no_generation` + `generated` = 预测总数。
* `generated` >= `applied` >= `resolved`。
* 解决率 % =
    * SWE-bench lite: `resolved` / 300
    * SWE-bench test: `resolved` / 2294
    * 通常为: `resolved` / (`no_generation` + `generated`)
