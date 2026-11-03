# 语法感知投机解码插件：项目报告

## 结论

在 Qwen3-32B（FP8）+ 官方 EAGLE-3 草稿头、单卡 H100 NVL、紧凑 JSON、batch=1、K=5 的设置下，插件让 **TPOT 比 vLLM 官方默认配置（EAGLE-3，异步调度）降低 8%–27%**，平均接受长度从 2.2–2.6 提高到 2.9–3.1；贪心输出与不开投机时逐 token 一致。

| 数据集（条数） | 不开投机 | 官方 EAGLE-3（默认，异步） | 官方 EAGLE-3（同步） | **EAGLE-3 + 插件（同步）** | 相对官方默认 |
| --- | --- | --- | --- | --- | --- |
| json-mode-eval（40） | 15.06 ms | 7.67 ms / 2.57 | 8.16 ms / 2.57 | **7.05 ms / 2.87** | TPOT −8.1% |
| BFCL 工具调用（40） | 15.17 ms | 9.15 ms / 2.22 | 9.58 ms / 2.22 | **6.64 ms / 3.05** | TPOT −27.4% |
| 长 schema 抽取（16） | 15.25 ms | 8.36 ms / 2.24 | 8.85 ms / 2.24 | **6.44 ms / 3.07** | TPOT −23.0% |

格式为"平均 TPOT / 平均接受长度"。原始结果：`results/baseline/pilot/<配置>/<数据集>_c1.json`，配置目录分别是 `none_compact`、`eagle3_k5_compact`、`eagle3_k5_compact_sync`、`eagle3_k5_compact_sync_gd`。贪心一致性：`results/stage0/greedy_compare_pilot.2.json`，4 种配置在 96 条请求上逐字一致。

**这只是第一版验证，范围很窄**：只测了并发 1、K=5（官方的 K 没有扫）、每格 16–40 条、各跑一次。高并发下收益能保持多少、温度 1 下的分布检验，都还没做（见第 6 节）。

## 1. 做了什么

在 JSON schema 或工具调用这类结构化输出里，很多 token 是语法唯一确定的，例如必填 key 名、引号、冒号、逗号和括号。本项目实现了一个 vLLM 插件 `vllm-grammar-draft`：模型 drafter（EAGLE-3 或 n-gram）照常起草，插件在调度器里按语法改写它的草稿。

- 模型草稿 token 只要语法接受，就保留。
- 遇到语法拒绝的模型 token（或者模型草稿用完了），如果语法已经完全确定了接下来的字符（xgrammar 的 jump-forward 字符串），就把这串字符的规范分词当作草稿接上去。
- 两者都给不出合法 token 时停止，剩下的草稿位用一个此处语法必然拒绝的 token 补齐到 K 个。

改写后的草稿照常交给目标模型验证，验证时用的是同一份语法 bitmask，所以**输出分布不变**。

开工前在 GitHub 上搜过：主线没有"把语法确定的 token 当草稿"的功能。最接近的是 [#45400](https://github.com/vllm-project/vllm/pull/45400)，它是 mgoin 的草稿原型，只做纯语法起草（单 token 强制），用 guidance 后端，作者自己报告在 json-mode-eval 上"几乎不触发"，而且自 8 月起无更新。另有两个已关闭的 jump-forward PR：#36142 和 #37105。本项目与它们的区别：与模型 drafter 混合；用 xgrammar 的字符串级确定，而不是单 token 强制；以插件形式交付，不改 vLLM 源码。

## 2. 和需求、报告不一样的地方，以及原因

| 原计划 | 实际做法 | 原因 |
| --- | --- | --- |
| 用 `custom_class` proposer 接口写混合 proposer | 用 vLLM general plugin 在调度器里改写任意 drafter 的草稿 | `custom_class` 的 `propose()` 只拿到 `sampled_token_ids`、`token_ids_cpu` 等，**拿不到目标模型的 hidden states**，EAGLE-3 草稿头也不会被加载，所以没法在里面跑 EAGLE-3。它还会把引擎强制切到 V1 runner + 同步调度，而上游 [#63373](https://github.com/vllm-project/vllm/pull/63373) 正在删除 V1。在调度器里改写草稿则对 EAGLE-3、n-gram 等所有方法都适用 |
| 报告：路线 A 只需改调度器 | 另加了一个 worker 侧的小钩子 | V2 model runner（EAGLE-3 的默认 runner）前向时从自己的 GPU buffer 读草稿值，调度器只决定草稿个数；不加钩子的话，调度器改写的草稿根本进不了前向 |
| 报告：第 1 档（bitmask 只放行 1 个 token）是保底收益 | 只用第 2 档（jump-forward 字符串） | 用真实分词器和 vLLM 实际的 stop token 复跑离线分析后，单 token 强制只占 0.6%–1.1%（报告写的 4%–6% 主要来自结尾的 EOS 位）；每个位置都填一次 bitmask 的 CPU 开销不值得 |
| 报告：用分词器把确定字符串切成规范 token 当草稿 | 同上，但模型 token 只要合法就优先 | EAGLE-3 的 token 是对目标模型分词方式的预测；只在它出错时才换成规范分词，可以避免在两种分词方式都合法时，因为切分不同而浪费后续的模型草稿 |
| （未提及）草稿长度 | 每个请求都补齐到 K | 草稿长度不齐会让 decode 批次掉出 FULL uniform cudagraph（#45400 实测 ITL 回退 64%）。补位 token 用 `\x00`：JSON 在任何位置都不允许未转义的控制字符，所以目标概率为 0，必然被拒绝；它也是普通 token，被拒时 xgrammar 不打警告（特殊 token 会打）。每次补位前都会用 matcher 检查，确认语法确实拒绝它 |
| 异步调度 | 只支持同步调度（`--no-async-scheduling`），在异步配置下启用插件会直接报错 | 异步调度时前向在调度器拿到草稿之前就已经发起，调度器改写草稿来不及生效（阶段 0 a1 第 1 条）。要支持异步，就得把语法状态搬到 worker 里（报告的路线 B） |

## 3. 代码结构

插件在 `grammar_spec/plugin/`，用 `uv pip install -e grammar_spec/plugin` 安装，通过 entry point `vllm.general_plugins` 被 vLLM 自动加载。**没有改动任何 vLLM 源码。**

| 文件 | 作用 |
| --- | --- |
| `vllm_grammar_draft/__init__.py` | `register()` 入口。`GRAMMAR_DRAFT` 不为 1 时直接返回，什么都不打补丁。为 1 时用一个 import hook 在目标模块**被导入之后**再打补丁，所以 API server 进程不会因此导入 GPU 代码 |
| `vllm_grammar_draft/config.py` | 开关：`GRAMMAR_DRAFT`、`GRAMMAR_DRAFT_MODE`（`hybrid` / `grammar_only`）、`GRAMMAR_DRAFT_PAD`、`GRAMMAR_DRAFT_STATS` |
| `vllm_grammar_draft/drafter.py` | 核心算法 `GrammarDrafter.propose(matcher, model_draft, k)`。不依赖 vLLM，只用到 matcher 的 `accept_token`、`find_jump_forward_string`、`is_terminated`、`rollback` 四个接口；返回前把 matcher 回滚到原状态 |
| `vllm_grammar_draft/scheduler_hook.py` | 替换 `StructuredOutputManager.trim_spec_tokens`：对 xgrammar 请求调用 `GrammarDrafter`，其余情况交还 vLLM 原逻辑（非结构化请求、非 xgrammar 后端、推理段未结束、语法已终止）。同时旁路 `accept_tokens`，按来源统计接受数。初始化时检查配置：异步调度、`draft_sample_method != greedy`、非标准拒绝采样都会直接报错，因为这些情况下无法保证无损 |
| `vllm_grammar_draft/worker_hook.py` | 包装 V2 `GPUModelRunner.update_requests`：同步模式下把调度器给出的草稿值拷进 `req_states.draft_tokens`（一次 pinned H2D + `index_copy_`）。值为 -1 的占位行跳过。V1 runner 本来就用调度器的值，不需要这个钩子 |
| `vllm_grammar_draft/stats.py` | 各来源（model / grammar / pad）的起草数和接受数，以及 propose 的 CPU 耗时；由一个守护线程每 0.5 s 写一次 JSON |
| `tests/test_drafter.py` | 10 个单元测试，用真实的 Qwen3 分词器和 xgrammar matcher（构建方式与 vLLM 相同）：matcher 状态复原、合法模型 token 保留、非法 token 被语法续写替换、替换之后不再使用模型 token、预算截断、纯语法模式、补位 token 在宽松语法下自动跳过等 |
| `tests/test_switch.py` | 子进程里分别以关、开两种方式调用 `register()`：关闭时 vLLM 的方法一个都没被替换；开启时只有在模块被导入后才打补丁 |

单步数据流（同步调度）：

```text
worker:    目标前向 → 拒绝采样 → EAGLE-3 起草 K 个 → 草稿拷回 CPU
scheduler: update_from_output → accept_tokens（统计）
           post_step → update_draft_token_ids → trim_spec_tokens（插件）
                       模型 token 合法则保留 → 否则接 jump-forward 规范分词 → 补位到 K
下一步:    schedule → worker_hook 把草稿值写进 GPU buffer → 前向验证
           bitmask 由调度器按改写后的草稿逐位计算（vLLM 原逻辑）
```

压测与分析脚本在 `grammar_spec/scripts/`：`server.py`（启动服务并记录实际的 runner、异步状态、插件状态）、`bench_structured.py`（压测客户端）、`run_matrix.py`（编排）、`longschema.py`（长 schema 数据集）、`sampling_check.py`（温度 1 下的分布检验）。

## 4. 怎么用

```bash
uv pip install --python .venv/bin/python -e grammar_spec/plugin

GRAMMAR_DRAFT=1 .venv/bin/vllm serve Qwen/Qwen3-32B \
  --quantization fp8_per_tensor --max-model-len 8192 --no-async-scheduling \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-32B-speculator.eagle3","num_speculative_tokens":5}' \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
```

启动日志里出现 `Grammar draft enabled: mode=hybrid, K=5, pad_token_id=188` 即表示生效。请求端不需要任何改动（`structured_outputs` 或 `response_format` 照常传）。不设 `GRAMMAR_DRAFT=1` 时插件什么都不做，行为与原版 vLLM 完全一致。`GRAMMAR_DRAFT_MODE=grammar_only` 会忽略模型草稿，只用语法起草（消融用）。

## 5. 怎么测的

- **正确性**：
  - 单元测试 12 个（`.venv/bin/python -m pytest grammar_spec/plugin/tests`），覆盖起草算法和开关行为。
  - 端到端贪心一致性：temperature=0、batch=1，比较 4 种配置对同一批 96 个 prompt 的完整输出文本（`scripts/stage0_checks.py greedy-compare`）。
  - 所有输出都用 `jsonschema` 校验，并检查是否紧凑，合规率全部为 100%。
- **性能**：`scripts/run_matrix.py` 依次启动每个配置的服务（同一时间只有一个），用 `scripts/bench_structured.py` 走 OpenAI 兼容的 chat 接口发流式请求，关闭思考（`enable_thinking=False`），每条请求带自己的 JSON schema。
  - TPOT =（端到端延迟 − 首 token 延迟）/（输出 token 数 − 1），输出 token 数取自服务端返回的 usage。
  - 平均接受长度 = 1 + 接受的草稿 token 数 / 起草次数，取自 vLLM Prometheus 指标在测量前后的差值。
  - 每个配置先预热，再清前缀缓存，然后开始测量。
- **数据集**：
  - json-mode-eval：vLLM 自带压测用的 `NousResearch/json-mode-eval`，过滤掉 xgrammar 不支持的 schema。
  - BFCL v4：simple_python 和 live_simple，要求输出 `{"name", "arguments"}` 形式的工具调用 JSON。
  - 长 schema：`scripts/longschema.py` 合成的信息抽取类 schema，每个 20–35 个必填字段，key 名长，带枚举、嵌套对象和数组；平均输出约 430 token。
  - 三个数据集都用固定种子抽样，所有配置跑的是同一批 prompt。

复现上表：

```bash
grammar_spec/scripts/fetch_data.sh   # BFCL 数据
.venv/bin/python grammar_spec/scripts/run_matrix.py --tag <新目录名> \
  --configs none eagle3,k=5 eagle3,k=5,sync eagle3,k=5,sync,gd \
  --datasets jsonmode bfcl longschema --concurrency 1 \
  --bfcl-root ~/.cache/grammar_spec/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data
```

## 6. 插件内部统计与局限

插件统计（同一批 c=1 运行，原始数据在各结果文件的 `grammar_draft` 字段）：

| 数据集 | 含被语法拒绝的模型 token 的草稿占比 | 每次草稿的语法 token 数 | 语法 token 接受率 | 模型 token 接受率 | 每请求每步 CPU 耗时 |
| --- | --- | --- | --- | --- | --- |
| json-mode-eval | 44% | 0.77 | 41% | 45% | 94 µs |
| BFCL | 68% | 1.56 | 48% | 56% | 120 µs |
| 长 schema | 50% | 1.33 | 73% | 34% | 83 µs |

结论：EAGLE-3 的草稿里有一半左右会在某个位置出现语法不允许的 token（多在 key 名和结构符号处）；插件在这些位置换成语法确定的 token 后接着起草。这部分 CPU 开销在一步约 7 ms 的解码里可以忽略。

**没做的和已知限制**：
- **只验证了 batch=1**。插件要求同步调度；在 vLLM 当前代码里，同步比异步慢 4.7%–6.4%（上表中官方同步对异步）。插件的收益已经覆盖了这部分代价，但并发 4/16/64 下的表现**未测**。
- **官方 EAGLE-3 的 K 没有扫**，只用了 K=5。CLAUDE.md 规则 3 要求扫过取最优，正式结论前必须补上。
- **温度 1 下的分布检验未做**。`scripts/sampling_check.py` 已经写好，但还没跑。无损性目前的依据是：理论上草稿只经过标准拒绝采样验证、补位 token 被 bitmask 屏蔽；实测上贪心输出逐字一致。
- 每格样本量小（16–40 条），各跑一次，没有重复。
- 不支持异步调度：在异步配置下启用插件会直接报错。要支持异步，需要把语法状态搬到 worker 里（报告的路线 B）。

## 7. 是否动了 vLLM 源码

**没有。** 插件通过 vLLM 的 general plugin 机制加载，用 monkeypatch 替换了 3 个方法，全部只在 `GRAMMAR_DRAFT=1` 时生效：
- `StructuredOutputManager.trim_spec_tokens`：改写草稿；
- `StructuredOutputManager.accept_tokens`：只做统计，然后调用原方法；
- V2 `GPUModelRunner.update_requests`：先调用原方法，再把调度器的草稿值写进 GPU buffer。
