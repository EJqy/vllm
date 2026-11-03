# 语法感知投机解码：进度记录

## 当前状态

- **阶段 0 已完成（2026-11-03）。** 用户随后要求直接实现混合语法起草插件，进行中。
- 阶段 0 最重要的结论：**报告第四节写的路线 A（只改调度器）在当前 main 上不成立**，原因有三：V2 runner 下调度器改不了草稿内容；不另加补丁的话，路线 A 只能在 V1 + 同步调度下工作；上游有一个草稿 PR #63373 正在删除 V1 runner。详见 a 节。

## 环境

| 项目 | 实际情况 | 与报告的差别 |
| --- | --- | --- |
| GPU | 1 × H100 NVL，94 GB（93.09 GiB 可用） | 报告假设 H100 80GB；KV cache 能多出约 14 GB |
| vLLM | commit ffdf26702（`grammar-spec` 分支的父提交），预编译 wheel `0.33.0rc1.dev592` | 一致 |
| xgrammar / torch / transformers | 0.2.8 / 2.14.0+cu132 / 5.22.0 | 一致 |
| 目标模型 | `Qwen/Qwen3-32B` BF16 权重，`--quantization fp8_per_tensor` 在线量化 | 一致；该参数合法（`vllm/model_executor/layers/quantization/__init__.py:43`） |
| 草稿头 | `RedHatAI/Qwen3-32B-speculator.eagle3`，**BF16，不跟随在线量化** | 加载草稿头后日志会再打一次 `Quantizing 256 layers`，这是用目标模型 config 重复打印的，有误导性；草稿头的量化配置实际取自 `draft_model_config`（`vllm/model_executor/models/utils.py:1167`），默认 None |
| BFCL 数据 | gorilla commit `6ea5797`，用 `scripts/fetch_data.sh` 拉取 | — |

## 已完成：阶段 0（2026-11-03）

| 子任务 | 产出 |
| --- | --- |
| a. 核对报告引用的源码 | 本文 a 节 |
| b. 用真实分词器复跑离线分析 | `scripts/offline_upper_bound.py`，结果在 `results/offline/` |
| c. 启动 Qwen3-32B FP8 + EAGLE-3 | 日志 `results/logs/stage0_serve_eagle3_k5_compact.log` |
| d. 基线压测脚本 | `scripts/server.py`、`bench_structured.py`、`run_matrix.py`、`fetch_data.sh`；小矩阵结果在 `results/baseline/stage0_smoke/` |
| e. GPU 小时估算 | `scripts/estimate_gpu_hours.py`，本文 e 节 |

---

### a. 报告与当前源码的核对

#### a1. 会改变实施路线的出入

**1. V2 runner 下，调度器只能决定草稿的长度，改不了草稿的内容。**
V2 runner 前向时，草稿 token 的值从 worker 自己的 GPU 状态 `req_states.draft_tokens` 读取（`vllm/v1/worker/gpu/model_runner.py:1569-1576`，`combine_sampled_and_draft_tokens`）。`scheduler_output.scheduled_spec_decode_tokens` 在 V2 里只用来取每个请求的草稿个数（同文件 `:1382`、`:1471`），整个 `vllm/v1/worker/gpu/` 下没有任何地方把调度器给的草稿值写回 GPU。V1 runner 则不同：同步模式下会把调度器的草稿值写进 `token_ids_cpu`（`vllm/v1/worker/gpu_model_runner.py:1482`，`update_req_spec_token_ids`）。
**影响**：报告说"路线 A 只能走同步"，这还不够，准确的说法是**路线 A 只能走 V1 runner + 同步调度**（`VLLM_USE_V2_MODEL_RUNNER=0 --no-async-scheduling`）。

**2. 默认配置下，几乎所有方法都走 V2 + 异步。**
V2 内置了 eagle / eagle3 / ngram / ngram_gpu / suffix / draft_model / mtp / dflash / dspark 的 speculator（`vllm/v1/worker/gpu/spec_decode/speculator_config.py:287-300`），只有 `custom_class` 是 V1 专属（同文件 `:304`）。我用与服务相同的参数在 CPU 上解析了配置，结果如下：

| 方法 | model runner | async_scheduling |
| --- | --- | --- |
| 不开投机 / eagle3 / ngram / suffix | V2 | True |
| custom_class | V1（打印 `Model Runner V2 does not yet support speculative method 'custom_class'`） | False（打印 `Async scheduling is disabled because the V1 model runner does not support it with custom_class-based speculative decoding.`） |

报告 4.1 说"CPU n-gram、suffix、custom_class 会自动退回同步"，前提是 V1 runner；而默认配置下 n-gram 和 suffix 走的是 V2，并且开着异步调度。
**影响**：基线里的 n-gram 和 suffix 也是异步的；"纯语法起草"消融如果用 custom_class 实现，会被强制切到 V1 + 同步，和 EAGLE-3 默认基线不在同一条执行路径上。

**3. 上游正在删除 V1 runner。**
草稿 PR [#63373 "[Core] Remove Model Runner V1"](https://github.com/vllm-project/vllm/pull/63373)（2026-10-23 开，11-02 仍在更新，+905/−37651 行）写明 "MRV2 is already the default … Features that only worked on MRV1 are dropped rather than ported"。
**影响**：路线 A 和 custom_class 消融都依赖 V1，一旦这个 PR 合入就没有上游价值。路线 A 只适合在本地做一次性的收益验证。

**4. 与路线 B 重叠的上游工作（报告写的"无 PR"已过时）。**

| PR | 状态 | 内容 | 与本项目的关系 |
| --- | --- | --- | --- |
| [#62873](https://github.com/vllm-project/vllm/pull/62873) | 2026-10-28 关闭，未合入 | #62130 作者的实现：`grammar_mask_draft_logits` | reviewer 指出的主要阻塞问题：异步调度下草稿 bitmask 与语法状态错位。作者已转去做其他 PR |
| [#64445](https://github.com/vllm-project/vllm/pull/64445) | 草稿，2026-10-29 开（AMD） | MRV2 "grammar-aware draft proposal bridge"：在下一次起草前，把已验证 token 和结构化输出元数据交给 speculator，由 speculator 自己维护 matcher 和设备侧 mask | 基本就是路线 B 需要的基础设施；本项目可以在它之上实现 EAGLE-3 的语法起草，或者与作者协作 |
| [#47885](https://github.com/vllm-project/vllm/pull/47885) | 开放，8 月后无更新 | V1 `ShadowGrammarManager`，在 worker 里维护 matcher，起草时过滤非法 token | 只针对 V1 |
| [#64844](https://github.com/vllm-project/vllm/pull/64844) | 开放，+11462 行 | V2 custom speculator 的 "speculative round contract" | 可能改变 `BaseSpeculator` 的接口 |

**5. "异步调度是否开启"没法靠 grep 日志判断。**
异步开启时 INFO 级别没有任何日志（只有 DEBUG 级的 `Batch queue is enabled with size %d`，`vllm/v1/engine/core.py:226`），被禁用时才打印 warning。V1 runner 同样不打印"Using V1"，只有回退时才有 warning。
**做法**：`scripts/server.py` 以 `VLLM_SERVER_DEV_MODE=1` 启动服务（只绑 127.0.0.1），通过 `/server_info?config_format=json` 读出实际的 `async_scheduling`，并把结果写进每次运行的 `*.meta.json`。

#### a2. 其他出入（不改变路线，但实现时要注意）

| # | 报告说法 | 实际 | 位置 |
| --- | --- | --- | --- |
| 6 | `trim_spec_tokens()` 截掉语法不允许的后缀 | 会**多保留第一个非法 token**（`+ 1`），它在验证时被 bitmask 拒掉，白占一个验证位。另外遇到终止 token 后，后面的草稿也会多留一个 | `vllm/v1/structured_output/__init__.py:386` |
| 7 | （未提及）异步模式下调度器如何处理草稿 | `update_draft_token_ids_in_output()` 不做语法截断，原样保留 worker 草稿（注释写着 "Preserve draft slots; rejection sampling handles grammar violations"），非法草稿全靠验证时拒绝 | `vllm/v1/core/sched/scheduler.py:2910` |
| 8 | 4.3/风险表：CPU 填 bitmask 用线程池批量 | 现有代码**只在不开投机时**才走并行填充，开投机时逐请求、逐位串行 | `vllm/v1/structured_output/__init__.py:424` |
| 9 | matcher 用 `fork()` 试探 | 两种做法都可行。vLLM 创建 matcher 时传了 `max_rollback_tokens=K`，但 xgrammar 0.2.8 已废弃这个参数，回滚步数不受限（会打 DeprecationWarning）。**更正**：本文早先版本把它写成了"硬约束"，那是错的，是在写插件单元测试时发现的 | `vllm/v1/structured_output/backend_xgrammar.py:202`；`xgrammar/matcher.py:262` |
| 10 | 无损性依据：`vllm/v1/sample/rejection_sampler.py` 的 `NO_DRAFT_PROBS` | V1 的描述属实（`:938`、`:1099`），但 EAGLE-3 默认走 V2，用的是另一份实现。V2 在没有草稿 logits 时同样按 one-hot 处理，结论成立 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:200` |
| 11 | 4.3：worker 的 `CachedRequestState.sampling_params.structured_outputs` 能拿到 schema | `CachedRequestState` 是 V1 的类。V2 没有对应结构，要在 `add_requests()` 里从 `new_req_data.sampling_params.structured_outputs` 自己存一份（这个字段确实会传到 worker） | `vllm/v1/worker/gpu/model_runner.py:1272` |
| 12 | custom_class 的 `propose()` 只收到 3 个参数 | 还会收到 `slot_mappings` 关键字参数。结论（拿不到 schema 和语法状态）不变 | `vllm/v1/worker/gpu_model_runner.py:4734-4740` |
| 13 | Prometheus 指标 `vllm:spec_decode_num_drafts` 等 | 实际暴露的名字带 `_total` 后缀，例如 `vllm:spec_decode_num_drafts_total`；逐位指标是 `vllm:spec_decode_num_accepted_tokens_per_pos_total` | `vllm/v1/spec_decode/metrics.py:229` |
| 14 | 6 节：xgrammar 0.2.8 紧凑模式下会在逗号后放出空格 | 实测只在**没有 `properties` 的自由对象**上复现（`{"type":"object"}`，即 `json_object` 模式）；显式列出属性的 schema（含嵌套对象、数组、anyOf）都不放行逗号后空格 | 测试在 c 节 |
| 15 | 3.3：用 `disable_any_whitespace` 控制空白 | 只有服务级配置生效。请求级字段 `structured_outputs.disable_any_whitespace` 存在（`vllm/sampling_params.py:110`），但 xgrammar 后端从不读取，会被静默忽略。所以每换一次空白设置都必须重启服务 | `backend_xgrammar.py:40` |
| 16 | 5.2：`benchmark_serving_structured_output.py` 的命令示例 | 脚本没有 `enable_thinking` 选项，xgrammar_bench 用默认模板（思考开启）后走 `/v1/completions`。另外报告说不关思考"测到的大部分是思考段"，这只在服务端加了 `--reasoning-parser` 时成立；报告的启动命令没有加，这种情况下语法从第一个 token 就生效（`<think>` 在 Qwen3 里不是特殊 token，但 JSON 语法开头不允许 `<`），思考段会被语法直接压掉，模型处于偏离训练分布的状态。结论（压测要关思考）仍然对，理由不同 | `benchmarks/benchmark_serving_structured_output.py:281` |
| 17 | 风险表："RFC #48197 正在重构这块" | #48197 已于 2026-09-22 以 completed 关闭 | — |
| 18 | 6 节 `--quantization fp8`\_per\_tensor | 排版错误，应为 `--quantization fp8_per_tensor` | — |

#### a3. 核对属实的内容

- 草稿流向：`gpu_model_runner.propose_draft_token_ids()`（V1，`:4704`）。同步模式下引擎在 `post_step()` 里调 `take_draft_token_ids()` → `Scheduler.update_draft_token_ids()` → `trim_spec_tokens()` → `validate_tokens()`（`core.py:741-748`、`scheduler.py:2866-2884`）。
- 同步 `step()` 先发起 `execute_model` 再算 bitmask，此时用的是上一步调度器修剪过的草稿（`core.py:709-721`）。异步 `step_with_batch_queue()` 用 worker 自己的草稿发起前向，之后才 `take_draft_token_ids()` → `update_draft_token_ids_in_output()` → 计算 bitmask（`core.py:750-862`）。4.1 的"关键约束"属实。
- 同步模式下逐请求草稿长度可以不同，调度器按 `num_tokens_with_spec` 排槽（`scheduler.py:703-707`、`:1009-1020`）。
- `grammar_bitmask()` 沿草稿逐位 `accept_tokens` 推进，为每个草稿位和 bonus 位各填一行，最后 `rollback`（`__init__.py:388-510`）。
- `XgrammarGrammar` 只有一条指向 `find_jump_forward_string` 的注释，整个 `vllm/` 下没有任何调用（`backend_xgrammar.py:253`）。
- `compile_grammar`：默认 `any_whitespace=True`；`disable_any_whitespace` 时设 `any_whitespace=False` 加 `separators=(",", ":")`；0.2.8 空格 bug 的注释和"10 月 8 日已修复"的说法都在（`backend_xgrammar.py:120-156`）。
- V2 默认启用，不支持时回退并打印 `Model Runner V2 does not yet support ...`（`vllm/config/vllm.py:735-760`）；V1 下的异步兼容列表是 EAGLE 系 / GPU n-gram / draft_model / dspark / dflash（`vllm.py:1691-1700`）。
- `register_speculative_method()` 是实验性接口，要求 V2，从 general plugin 注册（`speculator_config.py:310-327`）；V2 `BaseSpeculator.propose()` 接收 `num_sampled`、`num_rejected` 等 GPU 张量（`vllm/v1/worker/gpu/spec_decode/speculator.py:95-123`）。
- `draft_sample_method` 默认 `greedy`，可选 `probabilistic`（`vllm/config/speculative.py:694`）。
- `num_speculative_tokens_per_batch_size`（`speculative.py:578`）、`VLLM_BATCH_INVARIANT`、`--no-async-scheduling`、`tests/v1/spec_decode/`、`tests/v1/structured_output/` 都存在。
- xgrammar 0.2.8 上 `fill_next_token_bitmask`、`find_jump_forward_string`、`accept_token`、`fork`、`is_terminated`、`rollback`、`traverse_draft_tree`（`GrammarMatcher` 的方法）、`apply_token_bitmask_inplace` 都存在。
- 外部引用：#62130（open，2026-10-17）、#61024（Q4 路线图，open）、Apple coreai-models #391（2026-11-02 合入）都属实。

---

### b. 离线上限分析复跑

**做法**：先把附录脚本原样跑一遍，只把 `TIK` 换成 HF `Qwen/Qwen-7B` 里的 `qwen.tiktoken`，结果**与报告 5.0 节完全一致**（633/631 个样本，跳过 25/27 个）。然后用 `scripts/offline_upper_bound.py` 换成 HF `Qwen/Qwen3-32B` 分词器，TokenizerInfo 按 vLLM 的方式构建（`from_huggingface(vocab_size=151936)` 再过滤特殊 token），其余逻辑不变。

**分词器本身没有带来差异**：tiktoken 和 Qwen3-32B 分词器在全部 2632 段 BFCL 文本上逐 token 一致。**唯一的差异是 stop token**：vLLM 给 grammar 用的是 `SamplingParams.all_stop_token_ids`，对 Qwen3-32B 是 `{151645 <|im_end|>, 151643 <|endoftext|>}`（来自 generation_config），而附录脚本只设了一个 EOS。

| Schema 形式 | 空白 | 平均长度 | 第 1 档：报告 | 第 1 档：vLLM 实际 stop | 第 1+2 档：报告 | 第 1+2 档：vLLM 实际 | 前向减少 K=8（报告值不变） | 前向减少 K=5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| name + arguments | 默认 | 35.6 | 3.6% | **0.8%** | 29.8% | **27.0%** | 27.0% | 26.7% |
| name + arguments | 紧凑 | 28.5 | 4.5% | **1.0%** | 60.2% | **56.7%** | 53.2% | 51.1% |
| 只有 arguments | 默认 | 23.8 | 4.8% | **0.6%** | 20.5% | **16.3%** | 16.3% | 16.3% |
| 只有 arguments | 紧凑 | 19.7 | 6.2% | **1.1%** | 42.4% | **37.4%** | 37.3% | 37.2% |

结论：

- 报告里第 1 档的 4%–6%，大约 80% 是**每个样本末尾的 EOS 位**。在 vLLM 的实际配置下，结尾位放行 2 个 stop token，不算"单 token 强制"。真正的单 token 强制只有 **0.6%–1.1%**，第 1 档单独几乎没有价值，收益全靠第 2 档。
- "目标前向次数减少"不受影响：最后一个 token 总能作为目标模型的 bonus token 产出，它是否可确定不改变步数。所以报告的核心收益上限（紧凑 + name/arguments 约 53%）成立。
- K=5（当前 EAGLE-3 配置）下紧凑 name+arguments 的上限是 51.1%，和 K=8 差别不大，说明 BFCL 这类短调用里可确定的连续段很少超过 5 个 token。
- 结果文件：`results/offline/qwen3_32b_stop_vllm.json`、`qwen3_32b_stop_single.json`。

---

### c. Qwen3-32B FP8 + EAGLE-3 启动与结构化输出

启动命令就是报告 5.2 的服务端命令（K=5、xgrammar、紧凑 JSON），日志 `results/logs/stage0_serve_eagle3_k5_compact.log`。

| 项目 | 结果 | 依据 |
| --- | --- | --- |
| model runner | **V2** | 日志 `Using V2 Model Runner` |
| 异步调度 | **开启** | 没有 `Async scheduling is disabled` 日志；CPU 上解析同一份配置得到 `async_scheduling=True, max_concurrent_batches=2`；之后用 `server.py` 启动的服务，`/server_info` 返回 `async_scheduling: true` |
| 启动耗时 | 69–138 s | `results/logs/serve_*.meta.json` 的 `startup_s`；KV cache 194,048 tokens（`stage0_serve_eagle3_k5_compact.log`） |
| 结构化输出 | 正常 | 小矩阵 3 个配置 × 4 格共 312 条请求全部通过 `jsonschema` 校验，且全部是紧凑 JSON（各格 JSON 里的 `schema_compliance_rate`、`compact_rate`） |

EAGLE-3（K=5，V2 异步）在小矩阵上的接受长度。每格只有 20–32 条，只用于验证脚本，不是结果：

| 场景 | 平均接受长度 | 原始文件 |
| --- | --- | --- |
| json-mode-eval，c=1 / c=16 | 2.59 / 2.73 | `results/baseline/stage0_smoke/eagle3_k5_compact/jsonmode_c{1,16}.json` |
| BFCL simple，c=1 / c=16 | 2.26 / 2.28 | `results/baseline/stage0_smoke/eagle3_k5_compact/bfcl_c{1,16}.json` |
| 普通对话（无语法约束） | 未完成：阶段 0 交互式测过一次，但输出没有落盘，按规则 4 不引用；待用 `stage0_checks.py server-checks` 补录 | — |

xgrammar 紧凑模式空白 bug 的抽查（`any_whitespace=False, separators=(",", ":")`）：`{"type":"object"}` 会放行 `{"a":1, "b":2}`；显式属性的对象、嵌套对象、数组、anyOf 都拒绝逗号后空格。原始结果见 `results/stage0/xgrammar_probe.json`。runner 与异步的解析结果见 `results/stage0/config_resolution.json`，分词一致性见 `results/stage0/tokenizer_identity.json`，附录原样复跑见 `results/offline/appendix_verbatim_tiktoken.txt`。

---

### d. 基线压测脚本

没有直接用 `benchmarks/benchmark_serving_structured_output.py`，原因：它不能关闭思考，不能逐请求换 schema（BFCL），也不读投机解码指标、不做 schema 校验。

| 脚本 | 作用 |
| --- | --- |
| `scripts/fetch_data.sh` | 拉取固定 commit 的 BFCL 数据和 `qwen.tiktoken`，放在仓库外的 `~/.cache/grammar_spec` |
| `scripts/server.py` | 按配置（方法 / K / 空白 / 同步 / runner）启动服务，等待健康检查，从 `/server_info` 和日志记录实际的 runner 与异步状态，写 `results/logs/serve_<id>.meta.json` |
| `scripts/bench_structured.py` | 客户端压测：chat 接口 + `structured_outputs.json` + `enable_thinking=False`，流式统计 TTFT/TPOT/ITL/吞吐；前后各抓一次 `/metrics` 算接受长度和逐位接受率；`jsonschema` 校验和紧凑检查；可保存每条输出用于贪心一致性对比 |
| `scripts/run_matrix.py` | 编排：配置 × 数据集 × 并发，已完成的格子跳过，汇总写 `summary.csv` |
| `scripts/stage0_checks.py` | 阶段 0 的各项核对，结果写 `results/stage0/` |
| `scripts/estimate_gpu_hours.py` | GPU 小时估算 |

**小矩阵验证**（`results/baseline/stage0_smoke/`，配置：不开投机 / EAGLE-3 K=5 异步 / EAGLE-3 K=5 同步，紧凑 JSON，json-mode-eval 和 BFCL，c=1 和 c=16）：12 格全部跑通，0 错误；`/server_info` 正确区分了异步（True）和 `--no-async-scheduling`（False）。这些 TPOT 只用于验证脚本，不作为结果引用。

**贪心一致性预检**（`results/stage0/greedy_compare_stage0_smoke.json`）：c=1 时三种配置输出逐字一致（20/20、20/20）；c=16 时分别有 22/32、27/32 一致，这是批次组成带来的数值差异。所以正确性测试要用 batch=1，或者先确认 `VLLM_BATCH_INVARIANT` 在 FP8 在线量化下生效（代码里有 `quantization is not None` 的排除分支，还没验证）。

---

### e. 5.2 节完整基线矩阵的 GPU 小时估算

这是**计划估算，不是实验结果**。输入取自小矩阵的实测（TPOT、TTFT、输出长度、启动耗时）。c=4 用插值、c=64 用外推，JSONSchemaBench 和长 schema 的输出长度（200 / 600 token）是假设。脚本 `scripts/estimate_gpu_hours.py`，输出 `results/estimates/stage0_gpu_hours.txt`。

| 方案 | 服务启动次数 | 格子数 | GPU 小时（单次重复） |
| --- | --- | --- | --- |
| 5.2 完整矩阵，只算基线（不开投机 + n-gram/suffix/EAGLE-3 异步/EAGLE-3 同步，各扫 K=1–7 × 两种空白） | 58 | 928 | 27.3 |
| 5.2 完整矩阵，含 3 个自研方法 | 100 | 1600 | 46.0 |
| 裁剪方案（含主结果 3 次重复） | 27 | 304 | 6.9 |

裁剪方案：
1. **先选 K 再铺开**：只在两个短数据集、紧凑 JSON、c∈{1,16,64} 上扫 K∈{1,3,5,7}，挑出每个方法的最优 K。EAGLE-3 保留 2 个 K 进入主矩阵（不同并发下最优 K 可能不同）。
2. **空白设置只给主线做两遍**：不开投机和 EAGLE-3 异步跑默认空白和紧凑两种，n-gram/suffix 只跑紧凑。
3. **EAGLE-3 同步只用来隔离调度模式的影响**：最优 K、紧凑、两个短数据集。
4. **长 schema 数据集减量**：c=1 用 50 条，耗时主要来自它和 JSONSchemaBench。
5. **只重复主结果**：不开投机与 EAGLE-3 异步各再跑 2 次，其余跑 1 次。

完整矩阵超过 3 小时，按 CLAUDE.md 规则 6 需要先征得同意。

---

## 下一步

阶段 0 汇报中途，用户（11-03）要求直接实现"语法确定 token 当草稿 + EAGLE-3/n-gram 补位"的混合起草（以插件形式，尽量不改 vLLM 源码），并验收性能。实现方案见 `grammar_spec/project_report.md`。

## 失败的尝试及原因

- py-spy 无法 attach EngineCore（容器没有 SYS_PTRACE），改用 `/server_info` 读取实际配置。
- 第一次解析 Prometheus 指标时没有去掉 label，导致 KeyError；已在 `bench_structured.py` 里修正。
- 阶段 0 中有几条一次性的 JSON 查看命令用了系统 `python3`，违反 AGENTS.md；之后统一用 `.venv/bin/python`。

## 未解决的问题

- `VLLM_BATCH_INVARIANT=1` 在 FP8 在线量化下能否让 c>1 时贪心输出可复现，尚未验证。
- JSONSchemaBench 和长 schema 数据集的加载器还没写。
