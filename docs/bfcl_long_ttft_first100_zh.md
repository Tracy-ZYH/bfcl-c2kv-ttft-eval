# BFCL Long 前 100 条 TTFT 实验结果

## 实验设置

实验使用 BFCL `multi_turn_long_context` 固定前 100 条任务（`multi_turn_long_context_0` 至 `multi_turn_long_context_99`），模型为 Qwen3-4B-Instruct-2507，C2KV/Racer checkpoint 为 `/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000`。生成配置采用确定性解码，`temperature=0`、固定 seed，最大生成长度为 4096 tokens。

对比方法包括完整历史冷启动（Full Cold）、完整历史持久缓存（Full Cached）、仅 C2KV 压缩（C2KV Only）以及带 T02 检测与恢复的 Racer。Full 结果来自 GPU7，C2KV Only 来自 GPU4，Racer 来自 GPU7；GPU 型号相同。所有表格只统计前 100 条已完成任务，不包含第 101 条及之后的失败或补跑结果。

### 对比方法

**Full Cold** 在每个决策点都使用完整的 System、Tool Definitions、全部历史和当前轮输入，不进行 C2KV 压缩，也不执行检索或恢复。SGLang 的 Radix Cache 被显式关闭，因此每次请求都需要重新处理完整 prompt。这一方法表示没有历史压缩、没有跨请求前缀 KV 复用时的长文本 prefill 基线。

**Full Cached** 与 Full Cold 使用相同的完整消息、工具定义和生成配置，模型实际可见的上下文没有删减或压缩。区别是保留 SGLang Radix Cache，使后续请求可以复用已经计算过的公共前缀 KV，只对新增或发生变化的后缀执行 prefill。因此它测量的是理想的完整上下文前缀缓存收益，而不是压缩算法收益。由于任务轨迹可能在早期输出后发生分叉，Full Cached 与 Full Cold 即使采用确定性生成，也不保证最终任务 Accuracy 完全相同。

**C2KV Only** 使用训练后的 C2KV checkpoint，以 `dynamic-interleave` 和 Q/K/V gist 投影将已完成历史压缩为较短的 gist KV，本实验请求压缩率为 8x。System、Tool Definitions 和当前轮输入始终保持原始完整形式；只有已经完成的历史事件进入压缩路径。历史按语义事件切分，运行配置使用最多 768 tokens 的 chunk、64 tokens overlap 和最多 48 个 chunk。当前 S0 控制器还会在生成前根据当前任务做 lexical 排序，最多保留两个候选历史事件作为 raw evidence，其余已完成历史由 gist 表示。该方法关闭 post-draft recovery，每个决策只生成一次，因此这里的“Only”表示仅使用固定压缩视图生成，不表示完全没有预生成检索。

**Racer** 与 C2KV Only 使用相同的 checkpoint、8x C2KV 压缩视图、S0 预生成选择和首次 draft generation。在 draft 生成后，冻结的 T02 detector 使用统一特征计算风险分数，阈值固定为 0.5。如果风险未超过阈值，直接提交原 draft；如果触发，则使用当前 query、draft tool name/arguments 以及历史 witness 信息，从同一语义历史候选库中进行 lexical/semantic 检索和候选准入，选择可恢复的 raw history evidence。恢复证据被加入当前压缩上下文后，对同一决策重新生成，最多允许一次 recovery generation，BFCL 环境只看到最终提交的动作。若没有可行候选，或恢复结果没有通过提交条件，则保留原 draft。该流程旨在用少量按需恢复降低压缩造成的 action drift，但会增加 detector、检索、证据注入和第二次模型调用的时延。

## 评测指标

BFCL Accuracy 使用官方 checker 的二值任务正确率。Native TTFT 是 SGLang 从请求进入原生生成路径到首 token 的时间；Prefill 是模型 prefill 路径时间；Final Action Ready 是从控制器开始处理到最终可提交动作就绪的时间。时延分位数按全部已记录 generation call 或 committed decision 汇总，不按任务平均。

相对变化均以 Full Cold 的 P50 或 Accuracy 为基准。正的 TTFT 降低率代表加速，负值代表变慢；性能衰减为相对 Accuracy 降幅。

## 实验结果

| 方法 | 正确数 / 100 | BFCL Accuracy | Native TTFT P50 | Native TTFT P95 | 相对 Full Cold TTFT | Prefill P50 | Final Action Ready P50 | 相对性能衰减 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Cold | 17/100 | 17% | 297.54 ms | 836.75 ms | 基准 | 296.81 ms | 863.63 ms | 基准 |
| Full Cached | 14/100 | 14% | 40.15 ms | 257.01 ms | 降低 86.51% | 38.88 ms | 592.01 ms | 17.65% |
| C2KV Only | 8/100 | 8% | 366.97 ms | 694.90 ms | 增加 23.34% | 339.34 ms | 920.78 ms | 52.94% |
| Racer | 11/100 | 11% | 379.88 ms | 738.96 ms | 增加 27.67% | 355.59 ms | 2938.75 ms | 35.29% |

| 方法 | Native Generation Calls | Committed Decisions | Detector Calls | Recovery Count |
|---|---:|---:|---:|---:|
| Full Cold | 1063 | 1063 | 0 | 0 |
| Full Cached | 1075 | 1075 | 0 | 0 |
| C2KV Only | 1433 | 1433 | 0 | 0 |
| Racer | 2053 | 1489 | 1180 | 564 |

## 当前结论

Full Cached 显著降低了 TTFT，但 Accuracy 相对 Full Cold 下降 17.65%，略高于 15% 目标。当前 C2KV Only 与 Racer 均未实现相对 Full Cold 的 TTFT 加速，且任务性能衰减分别为 52.94% 和 35.29%，尚未达到“prefill 时延降低至少 50%、下游性能衰减不超过 15%”的目标。

Racer 相比 C2KV Only 将 Accuracy 从 8% 提升到 11%，但 564 次 recovery 增加了 generation call，并将 Final Action Ready P50 提高到 2.94 秒。当前瓶颈主要是压缩注入路径本身没有降低 prefill，以及 recovery 频率和恢复开销过高。

本结果是当前代码与现有运行产物的阶段性观测。C2KV/Racer run 在完成前 100 条后于后续任务失败，因此这里仅使用前 100 条已有完整官方判分和计时的结果，不将整个 200 条 run 标记为完成。

## 数据来源

- Full Cold / Full Cached：`bfcl_long_ttft_gpu7_full200_fullcontext_20261026_224715`
- C2KV Only / Racer：`bfcl_long_ttft_c2kv_gpu4_racer_gpu7_20261027_150348`
