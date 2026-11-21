# BFCL Long 冻结基线记录

更新时间：2026-11-11

## Full 200 正式基线

后续 BFCL `multi_turn_long_context` 实验统一使用 2026-11-09 的 Full 200 结果，不再重复运行 Full 200。

权威结果文件：

`results/bfcl_long_joint_scope_gpu6_full200_20261109_230337/runs/full/summary_full.json`

SHA256：

`5f80cf558f7808bd0578122a2de48532fe4bf29c15d21d106f4c4328c203c5f4`

| 指标 | 数值 |
| --- | ---: |
| Tasks / Scored | 200 / 200 |
| Correct | 63 |
| BFCL Accuracy | 31.50% |
| 成功计时请求 | 2041 |
| 请求错误 | 3 |
| Total Online TTFT | 2,574,047.38 ms |
| Mean Online TTFT | 1,261.17 ms |
| P50 Online TTFT | 267.41 ms |
| P95 Online TTFT | 5,539.24 ms |

运行使用模型与 tokenizer：

`/home/zhuyuhan/models/b_history/arm-C/seed-42/checkpoint-1000`

服务配置确认 `disable_radix_cache=true`、`random_seed=0`、`max_running_requests=1`。虽然服务初始化了 C2KV 模块，但 Full arm 的 `compressed_requests=0`、`gist_tokens_total=0`，因此实际为原始完整上下文 Prefill。3 个 upstream request error 已保留在产物中，没有静默删除；官方分母仍为完整 200 条任务。

2026-11-07 的独立重复结果同样为 `63/200`，P50/P95 为 `270.20/5552.73 ms`，可作为复现性旁证，但不作为主表数据源。

后续方法可以复用该 Full 结果，但必须保持相同 BFCL 类别与 200 个 task IDs、模型/checkpoint、tokenizer、prompt/checker、请求计时口径和推理配置。若闭环轨迹导致请求集合不同，应将 TTFT 标为系统级 workload 对比，不能称为严格 same-input paired TTFT。任何改变实验语义的新实验另建基线，不覆盖本记录。

## Fixed 20K 已有结果

### 197 条离线配对估算

结果文件：

`results/bcrs_stage_a_v2_final_20261110_210050/comparison.csv`

SHA256：

`fcf9a788e242f953382442e697a79b9ddc669861924a7ff4ddc9ffbaee78e201`

| 指标 | Fixed 20K |
| --- | ---: |
| 总任务数 | 197 |
| 选择压缩 | 39 / 197 |
| 压缩选择比例 | 19.80% |
| 估算 Total Online TTFT 降幅 | 73.59% |
| Held-out 选择压缩 | 34 / 177 |
| Held-out 估算 TTFT 降幅 | 67.06% |

这是基于已有配对 Profiling 数据的 Stage A 离线重放估算，不是真实 197/200 条闭环执行，也没有可用于正式结论的 Fixed 20K Closed-loop Accuracy。可用于说明路由分布和预估速度，不应写成正式 200 条结果。

### 2026-11-09 的 20 条闭环 smoke

结果文件：

`results/bfcl_long_gp_sjcs_smoke_gpu6_20261109_212729/summary/gp_sjcs_summary.json`

| 指标 | Fixed 20K |
| --- | ---: |
| Correct / Tasks | 4 / 20 |
| Accuracy | 20.00% |
| Decisions | 250 |
| Total Online TTFT | 43,163.98 ms |
| Mean / P50 / P95 | 172.66 / 147.80 / 297.45 ms |
| 系统级 TTFT 降幅 vs Full | 18.97% |
| Same-input paired 降幅 | 19.71% |
| 实际配置选择 | A0 Full 250 次 |
| 实际压缩次数 | 0 |

该结果完成了闭环，但 250 个 decision 全部选择 Full，Tool/History retention 均为 1.0。因此它只能验证 Fixed 20K 路由管线，不能证明阈值触发后的压缩精度或加速。

### 2026-11-11 的 20 条 Stage B 闭环

结果文件：

`results/bfcl_long_bcrs_v2_stage_b_runlevel_fix_gpu0_20261111_003838/summary/bcrs_smoke.json`

SHA256：

`73a4e6f9707a7c31747ddd4fbfbf60bed7879852262dd19e8837eb8a851093c0`

| 指标 | Full | Fixed 20K |
| --- | ---: | ---: |
| Correct / Tasks | 5 / 20 | 5 / 20 |
| Accuracy | 25.00% | 25.00% |
| Decisions | 230 | 230 |
| Total Online TTFT | 104,607.35 ms | 106,696.26 ms |
| Mean Online TTFT | 454.81 ms | 463.90 ms |
| P50 Online TTFT | 433.31 ms | 442.29 ms |
| P95 Online TTFT | 735.50 ms | 749.44 ms |
| TTFT 降幅 vs Full | Baseline | -2.00% |
| Full / Racer 路由次数 | - | 230 / 0 |

这份结果同样没有触发压缩，前 20 条任务的逐轮可观测上下文均未达到 20K。它证明当前 Fixed 20K 实现不会错误压缩短上下文，但不是 Fixed 20K 压缩效果结果。

## 当前结论

Full 200 已正式冻结为 2026-11-09 的 `63/200` 结果，后续不再重跑。

Fixed 20K 已有可信的 197 条离线配对速度估算：39/197 个任务选择压缩，估算总 TTFT 降低 73.59%。已有 20 条真实闭环 smoke，但均未达到阈值，实际压缩次数为 0。当前尚无一份同时包含真实阈值触发、完整 200 条 Closed-loop Accuracy 和 Online TTFT 的 Fixed 20K 正式结果。
