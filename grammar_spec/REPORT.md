# 语法感知投机解码：在结构化输出场景超越 vLLM

Nov 3, 2026 · @Eddy Ji

## 一、这是什么

一句话：在 JSON、工具调用这类结构化输出里，很多 token 是被语法直接决定的。把语法本身当成一个零 GPU 开销的 drafter，和 EAGLE-3 等模型 drafter 混合起草，就能在这类场景下比 vLLM 官方投机解码更快，而且输出分布完全不变。

**结构化输出怎么工作。** vLLM 每个解码步都让语法引擎（默认 XGrammar）算一张 bitmask，把语法不允许的 token 的 logits 屏蔽掉。比如 schema 规定对象依次有 `name` 和 `age` 两个字段，那么模型写完 name 的值之后，接下来的 ` ", "age":  ` 基本没有选择余地。

**两种"确定"，要分开对待。**

- **token 级强制**：bitmask 只放行 1 个 token。屏蔽后目标模型的分布只剩这一个 token，贪心和随机采样都必然选它，所以把它当 draft，接受率严格是 100%。
- **字符串级确定**：语法只允许一串固定字符（例如一个 key 名），但这串字符有多种切分方式，bitmask 会放行好几个 token。这时用分词器把它切成规范 token 当 draft，接受率很高但不是 100%，要靠验证兜底。

**收益从哪来。** 普通 drafter 在这些位置也常能猜对，但它会在长 key 名、枚举值、schema 特有的嵌套结构上出错，而且每步只有固定的 K 个草稿位。语法 drafter 把确定部分零成本填满，模型 drafter 只需负责值的部分。

**和 jump-forward 的区别，这是你面试时最该讲清楚的一点。** jump-forward 把确定字符串直接塞进上下文，等于替模型选定了分词方式，可能偏离模型自然的分词。投机版本把这串 token 交给目标模型验证，不匹配就拒绝，所以是无损的。可以把本项目概括为"无损版 jump-forward，并与模型 drafter 融合"。

## 二、vLLM 现状与相关工作

结论：vLLM 目前只把语法当"过滤器"，草稿由模型或 n-gram 生成后再被语法截断，没有任何路径让语法自己产出草稿。社区 10 月已有人提出"语法约束 EAGLE 起草"，但没有做"语法强制 token 当草稿"，也只测了 batch=1。这两点就是你的空位。

### vLLM 源码现状

以下基于 2026-11-03 的 [vLLM main 分支](https://github.com/vllm-project/vllm)（commit ffdf267）阅读所得。

- **草稿的流向**：proposer 在 worker 进程的 `gpu_model_runner.propose_draft_token_ids()` 里运行。同步调度时，引擎在 `post_step()` 调 `take_draft_token_ids()` 取回草稿，交给调度器的 `update_draft_token_ids()`。
- **语法只做截断**：调度器调用 `StructuredOutputManager.trim_spec_tokens()`，内部用 `grammar.validate_tokens()` 截掉语法不允许的后缀。
- **验证时的 bitmask**：`grammar_bitmask()` 沿着草稿逐位推进 matcher，为每个草稿位和 bonus 位各填一行 bitmask，再交给拒绝采样。
- **jump-forward 没实现**：`backend_xgrammar.py` 的 `XgrammarGrammar` 类里只有一条指向 xgrammar `find_jump_forward_string` 的注释，写着留给 jump-forward，没有任何调用。
- **custom proposer 拿不到语法**：`method: custom_class` 的 `propose()` 只收到 `sampled_token_ids`、`num_tokens_no_spec`、`token_ids_cpu`，看不到请求的 schema 和语法状态。这一点决定了第四节的实现路线。

### 社区相关工作

| 工作 | 做了什么 | 局限，也就是你的机会 |
| --- | --- | --- |
| [vLLM issue #62130](https://github.com/vllm-project/vllm/issues/62130)（2026-10-17 提出，仍 open，无 PR） | 在 EAGLE-3 草稿 logits 上加语法 mask。7B、A100、batch=1：不开投机 TPOT 11.09 ms，无 mask 的 EAGLE-3 11.47 ms（反而更慢），加 mask 9.61 ms | 没有把语法强制 token 当草稿；只测 batch=1、两个自制 schema、贪心解码 |
| [grammar-spec-decode 原型仓库](https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode) | 上一条的原型，运行时 patch `EagleProposer`；实测 n-gram 草稿 84.9% 被语法拒绝 | 非正式实现，结果未经第三方复现 |
| [vLLM 2026 Q4 路线图 #61024](https://github.com/vllm-project/vllm/issues/61024) | SIG Spec Decode 列了"提升投机解码在工具调用、推理、结构化输出下的配置易用性和鲁棒性" | 该条目没有指定负责人，适合做成上游贡献 |
| [RFC #48197](https://github.com/vllm-project/vllm/issues/48197) | 重构 StructuredOutputManager 与投机解码的接口；指出 `validate_tokens()` 只是尽力而为 | 不涉及 jump-forward 或语法起草；说明这块代码近期变动多，要尽早 rebase |
| [Apple coreai-models PR #391](https://github.com/apple/coreai-models/pull/391)（2026-11-02 合入） | "grammar proposals"：语法完全确定一段时，用这段代替草稿去验证 | 端侧单请求、仅贪心、没有给出加速数据 |
| [SGLang jump-forward](https://www.lmsys.org/blog/2024-02-05-compressed-fsm)（2024） | 压缩 FSM，确定字符串一步写入；llama-7B 上延迟最多降 2 倍 | 不同切分会导致不同的下一 token 分布，靠重新分词缓解，不是严格无损 |
| [DOMINO](https://proceedings.mlr.press/v235/beurer-kellner24a.html)（ICML 2024） | 子词对齐的约束解码，结合预计算和投机，部分场景接近 2 倍 | 不在 vLLM 里，不面向批量服务 |

**你的差异点**：把"语法强制 token 草稿"和"语法约束的 EAGLE-3 起草"融合到 vLLM 的批量服务路径里，支持随机采样，在 30B 级模型、单卡 H100、公开数据集上做完整的并发扫描。#62130 的数据还说明一个现象：结构化输出下，官方 EAGLE-3 可能比不开投机还慢，这正是你要打的基线。

## 三、核心设计

每个解码步，对每个结构化输出请求，让语法"走在前面"：能确定的 token 直接写进草稿，确定不了的位置交给模型 drafter，并且模型起草时也套上语法 mask。草稿总长不超过 K，验证仍由目标模型完成，所以输出分布不变。

### 3.1 草稿怎么拼

下面是单个请求一次起草的逻辑。matcher 是 xgrammar 的 `GrammarMatcher`，状态停在"已确认输出"之后。

```python
draft = []
while len(draft) < K:
    mask = matcher.fill_next_token_bitmask()
    if popcount(mask) == 1:                      # 第 1 档：token 级强制
        toks = [唯一允许的 token]                # 接受率 100%
    elif s := matcher.find_jump_forward_string(): # 第 2 档：字符串级确定
        toks = canonical_tokenize(s)             # 规范分词，可能被拒
    else:                                        # 第 3 档：交给模型
        toks = [drafter 在 mask 下的 argmax]     # 不合法的 token 不会被提出
    for t in toks[: K - len(draft)]:
        if not matcher.accept_token(t):
            break
        draft.append(t)
matcher.rollback(len(draft))                     # 起草不改变真实语法状态
```

三档的优先级就是成本从低到高：第 1 档白送，第 2 档几乎白送，第 3 档才动用 GPU 上的 drafter。

### 3.2 为什么是无损的

vLLM 的拒绝采样在草稿没有概率时（n-gram、suffix、custom 都是这样），把草稿当成 one-hot 分布：以目标概率 p(t) 接受，拒绝时从去掉 t 后重新归一化的目标分布里采样。这正是标准投机采样在草稿分布为 one-hot 时的特例，所以随机采样下也严格无损。源码见 `vllm/v1/sample/rejection_sampler.py` 里 `NO_DRAFT_PROBS` 分支，草稿概率直接取 1。

第 1 档为什么一定被接受：结构化输出会先用 bitmask 屏蔽目标 logits。只放行 1 个 token 时，屏蔽后它的概率就是 1，所以 p(t)=1，任何温度、top-p、top-k 下都必然接受。

### 3.3 三个要提前想清楚的坑

- **分词边界**：第 2 档的字符串单独分词，结果可能和模型在上下文里的分词不一样，比如引号和逗号会被合成一个 token。做法是把"已生成文本的尾部 + 确定字符串"一起分词，只取边界之后的 token；就算还不一致，验证也会兜底。SGLang 的 jump-forward 也遇到过同样的问题，靠重新分词缓解。
- **空白字符会吃掉强制 token**：vLLM 默认用 `any_whitespace=True` 编译 JSON schema，字段之间允许任意空白，很多位置就不再是单一 token。打开 `--structured-outputs-config '{"disable_any_whitespace": true}'` 生成紧凑 JSON，强制 token 会明显变多。这是重要的实验变量，基线必须用同样的设置。
- **草稿概率要对齐**：vLLM 的 `draft_sample_method` 默认是 `greedy`，不带草稿概率，混合起草最简单。如果打开 `probabilistic`，被语法替换的位置必须改成 one-hot 概率，否则接受率公式就错了。

### 3.4 两条实现路线

| 路线 | 改哪里 | 能覆盖的调度模式 | 难度 | 收益上限 |
| --- | --- | --- | --- | --- |
| A：调度器侧语法补全 | `StructuredOutputManager.trim_spec_tokens()`，拿到任意 drafter 的草稿后按 3.1 重写 | 仅同步调度 | 低，约 1 周 | 中：模型草稿在被替换位置之后不再可信，只能截断 |
| B：drafter 内语法起草 | 在 EAGLE-3 的起草循环里加 mask 和强制 token，worker 侧维护 matcher | 同步和异步都行 | 中高，约 2 周 | 高：drafter 每一步都基于正确的前缀 |

建议先做 A 验证收益，再做 B 作为最终版本。第四节解释为什么 A 只能走同步调度。

## 四、在 vLLM 里怎么实现

先在同步调度路径上做路线 A（只改调度器，一周内出第一组数），再在 worker 侧做路线 B（改 EAGLE-3 起草循环），最后补异步调度和批量优化。

&#91;embedded content: 草稿数据流 · 两个进程，两个改动点\]

草稿从 worker 出发，在调度器里被语法截断，再带着 bitmask 回到 worker 验证。路线 A 改调度器里的截断步骤，路线 B 改 worker 里的起草步骤。

### 4.1 先确认进程走哪条路径

- **两套 model runner**：V1 是 `vllm/v1/worker/gpu_model_runner.py`，V2 在 `vllm/v1/worker/gpu/`。V2 支持时默认启用，否则退回 V1，日志打印 `Model Runner V2 does not yet support ...`。
- **异步调度默认开**：但在 V1 runner 下，只有 EAGLE 系、GPU n-gram、draft\_model、dspark、dflash 能配异步。CPU n-gram、suffix、custom\_class 会自动退回同步，日志打印 `Async scheduling is disabled because ...`。
- **关键约束**：异步调度时，`core.py` 的 `step_with_batch_queue()` 先用 worker 自己的草稿发起前向，调度器之后才拿到草稿去算 bitmask。所以调度器改草稿对异步模式无效，路线 A 只能走同步。
- **实操**：每次启动都 grep 上面两条日志。开发路线 A 时加 `--no-async-scheduling`，但最终基线一定用 vLLM 默认配置，并单独报告"同步 vs 异步"本身的差距。

### 4.2 路线 A：调度器侧语法补全（第 1 周）

1. 在 `StructuredOutputManager` 里新增 `fill_spec_tokens(request, spec_tokens, k)`，按 3.1 的前两档生成语法 token。模型草稿只在和语法 token 一致时保留；第一次不一致之后，后面只用语法 token。
2. 在 `Scheduler.update_draft_token_ids()` 里，用它替换现在的 `trim_spec_tokens()` 调用，加一个开关，比如环境变量 `VLLM_GRAMMAR_DRAFT=1`。同步模式下 `request.spec_token_ids` 长度可以逐请求不同，调度器会按实际长度排槽。
3. matcher 不要用 `rollback()` 回退，改用 xgrammar 0.2.8 提供的 `GrammarMatcher.fork()` 复制一份来试探。#62130 的作者报告过 rollback 会和 xgrammar 状态不同步。
4. 做一个"纯语法"消融：写一个 `custom_class` proposer，所有请求都返回空草稿，草稿完全由第 1 步补全。它能单独回答"语法本身值多少"。
5. 给每个请求记录三档各贡献了多少 token、各自接受了多少，第五节的分析全靠这些计数。

### 4.3 路线 B：drafter 内语法起草（第 2–3 周）

1. **挂载点**：V2 runner 提供实验性的 `register_speculative_method()`，可以从 vLLM general plugin 注册自定义 speculator，见 `vllm/v1/worker/gpu/spec_decode/speculator_config.py`。也可以直接在 EAGLE 的 speculator 起草循环里加钩子，#62130 建议的位置是 `compute_logits` 之后、`argmax` 之前。
2. **worker 侧的语法状态**：worker 的 `CachedRequestState.sampling_params.structured_outputs` 能拿到 schema。用同一个 `GrammarCompiler` 编译（有缓存），每步用上一轮接受的 token 推进 matcher。
3. **每个起草步**：CPU 上填 bitmask，拷到 GPU，用 `xgr.apply_token_bitmask_inplace` 作用到草稿 logits，再 argmax。mask 只剩 1 个 token 或 jump-forward 有结果时，直接写入，不用 drafter 的输出。
4. **主要开销是 CPU 和 GPU 之间的同步**：#62130 的原型每个草稿位同步一次。优化方向：所有请求的 bitmask 用线程池批量填；先在 CPU 上算出前缀里的确定 token，只在第 3 档位置才同步；与上一步的 GPU 计算重叠。
5. **异步调度**：上一步接受了几个 token 在 GPU 上（`num_sampled`、`num_rejected`），推进 CPU matcher 需要一次 D2H 拷贝，这个同步点要仔细放。

### 4.4 源码阅读清单

| 文件 | 重点看 |
| --- | --- |
| `vllm/v1/structured_output/__init__.py` | `trim_spec_tokens`、`grammar_bitmask`、`_get_constraint_bounds`（推理模型的思考结束边界） |
| `vllm/v1/structured_output/backend_xgrammar.py` | `XgrammarGrammar` 的 accept / validate / rollback；`compile_grammar` 的空白设置 |
| `vllm/v1/core/sched/scheduler.py` | `update_draft_token_ids`、`update_draft_token_ids_in_output` |
| `vllm/v1/engine/core.py` | `step`（同步）与 `step_with_batch_queue`（异步） |
| `vllm/v1/worker/gpu_model_runner.py` | `propose_draft_token_ids`，各 proposer 的调用方式 |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | V2 的 `BaseSpeculator.propose` 接口 |
| `vllm/v1/sample/rejection_sampler.py` | `NO_DRAFT_PROBS` 分支，无损性的依据 |

xgrammar 侧要用到的接口：`fill_next_token_bitmask`、`find_jump_forward_string`、`accept_token`、`fork`、`is_terminated`。另外 0.2.8 还有一个 `traverse_draft_tree`，能一次为整棵草稿树的每个节点填 bitmask，以后做树状草稿时可以直接用。

## 五、怎么测试

测试分四层：先离线算收益上限（不需要 GPU，下面已经跑了一版），再证明无损，再测性能，最后做消融。离线结果显示，紧凑 JSON 的工具调用里约 60% 的 token 可以由语法确定，纯语法起草最多能省掉一半的目标前向。

### 5.0 离线上限分析（已跑通）

做法：取 [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) v4 的 live\_simple 和 simple\_python 两个子集的标注答案，拼成工具调用 JSON。用 Qwen 系列的 151,643 词 BPE 词表分词，在 xgrammar 0.2.8 里逐 token 回放，统计每一位是否由语法确定，并模拟 K=8 的纯语法起草。

| Schema 形式 | JSON 空白 | 平均长度（token） | 第 1 档：单 token 强制 | 第 1+2 档：语法可确定 | 目标前向次数减少 |
| --- | --- | --- | --- | --- | --- |
| name + arguments | 允许空白（vLLM 默认） | 35.6 | 3.6% | 29.8% | 27.0% |
| name + arguments | 紧凑 | 28.5 | 4.5% | 60.2% | 53.2% |
| 只有 arguments | 允许空白（vLLM 默认） | 23.8 | 4.8% | 20.5% | 16.3% |
| 只有 arguments | 紧凑 | 19.7 | 6.2% | 42.4% | 37.3% |

每行约 630 个样本，另有约 25 个因 schema 转换或字段顺序不符被跳过。三个结论：

- **大头在第 2 档，不在第 1 档**：单 token 强制只占 4% 到 6%，因为同一串字符往往有多种合法切分。所以第 2 档的分词边界处理是项目成败的关键。
- **紧凑 JSON 让收益翻倍**：关掉任意空白后，语法可确定比例从约 30% 升到约 60%。
- **这是上限，不是实测**：它假设模型总是输出规范分词，也没扣掉 EAGLE-3 本来就能猜对的部分；样本是短工具调用，长 schema 的比例需要你实测。脚本见文末附录。

### 5.1 正确性：必须先过

1. **贪心逐 token 一致**：temperature=0，开关你的方法各跑一遍，输出必须完全相同。批次组成会引入数值差异，可以先用 batch=1，或试试 vLLM 的 `VLLM_BATCH_INVARIANT=1`。
2. **采样分布一致**：temperature=1，同一个 prompt 各采样 2000 次以上，比较前若干位置的 token 频率（总变差距离或卡方检验），开关两种情况应无显著差异。
3. **Schema 合规率**：所有输出用 `jsonschema` 库校验，合规率不能低于不开投机时。#62130 发现无 mask 的 EAGLE-3 在难 schema 上合规率会下降，这一项值得单独报告。
4. **回归测试**：跑通 vLLM 仓库里 `tests/v1/spec_decode/` 和 `tests/v1/structured_output/` 下的已有测试。

### 5.2 性能测试

**指标**：TPOT（ms）、输出吞吐（tok/s）、平均接受长度（1 + 接受 token 数 / 起草次数，来自 Prometheus 指标 `vllm:spec_decode_num_drafts`、`vllm:spec_decode_num_accepted_tokens`）、三档草稿各自的接受率、CPU 起草耗时占每步耗时的比例。

**数据集**：

- vLLM 自带的 `benchmarks/benchmark_serving_structured_output.py`，数据集选项有 json、json-unique、grammar、regex、choice 和 xgrammar\_bench（来自 NousResearch/json-mode-eval）。
- BFCL v4 的 simple、multiple、parallel 等工具调用子集。
- [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)：约 1 万个真实 schema，按复杂度分层抽样。
- 自制"长 schema"集：20 到 50 个字段的信息抽取、长 key 名、大量枚举。这是你的方法最占优的场景，但短 schema 的结果也要如实报告。

**基线矩阵**（每组都扫 `num_speculative_tokens` 1 到 7 取最优，并发扫 1、4、16、64，空白设置默认和紧凑各跑一遍）：

| 配置 | 作用 |
| --- | --- |
| 不开投机 | 绝对基线 |
| 官方 n-gram、suffix | 无模型 drafter 的官方方案 |
| 官方 EAGLE-3（默认异步调度） | 最强官方基线，你要赢的对象 |
| 官方 EAGLE-3（同步调度） | 把调度模式的影响单独隔离出来 |
| 纯语法起草 | 语法本身值多少 |
| 路线 A：EAGLE-3 + 调度器侧语法补全 | 第一版结果 |
| 路线 B：语法约束 EAGLE-3 + 强制 token | 最终版本 |

**命令示例**：

```bash
# 服务端：官方 EAGLE-3 基线，紧凑 JSON
vllm serve Qwen/Qwen3-32B --quantization fp8_per_tensor --max-model-len 8192 \
  --speculative-config '{"method":"eagle3","model":"RedHatAI/Qwen3-32B-speculator.eagle3","num_speculative_tokens":5}' \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'

# 客户端
python benchmarks/benchmark_serving_structured_output.py --backend vllm \
  --model Qwen/Qwen3-32B --dataset xgrammar_bench --num-prompts 500 \
  --max-concurrency 16 --output-len 512 --save-results
```

注意：Qwen3 默认开思考模式，而语法约束要等思考结束才生效。压测时在 `apply_chat_template` 里传 `enable_thinking=False`，否则测到的大部分是思考段。

### 5.3 消融与分析

- 三档草稿逐一开关，看各自贡献。
- K 取 3、5、8、12，看语法草稿能否支撑比官方更大的 K。
- 按输出长度、字段数分桶报告加速比，画出"schema 越长，收益越大"的曲线。
- 失败分析：把第 2 档被拒的位置全部导出，确认是不是分词边界造成的。

## 六、实验环境与模型选型

主模型推荐 Qwen3-32B，用 FP8 跑在单张 H100 80GB 上，配 Red Hat 开源的 EAGLE-3 草稿头。

| 项目 | 选择 | 理由 |
| --- | --- | --- |
| 目标模型 | Qwen3-32B，FP8（`--quantization fp8`\_per\_tensor 在线量化，或现成 FP8 权重） | BF16 权重约 64 GB，加草稿头后 KV cache 不到 10 GB；FP8 权重约 32 GB，能留出约 30 GB 给 KV cache |
| 草稿头 | [RedHatAI/Qwen3-32B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3) | 官方基线直接可用；用 ShareGPT 和 UltraChat 训练，没专门见过工具调用 JSON，正是语法能补位的地方 |
| 备选模型 | Qwen3-30B-A3B-Instruct-2507，配 [DFlash 草稿](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.dflash) | MoE 激活参数小，可作为第二个数据点；DFlash 是块并行起草，接语法 mask 的方式不同，留作扩展 |
| vLLM | 2026-11 的 main 分支（本报告基于 commit ffdf267） | 结构化输出和投机解码这块代码变动频繁，固定一个 commit 做全部实验 |
| xgrammar | vLLM 当前锁定的 0.2.8 | 有 `fork()` 和 `traverse_draft_tree()`；注意下面的空白 bug |

三个环境细节：

- **草稿头的接受长度参考值**：Red Hat 在 2×A100、vLLM 0.11.0、k=3 下报告 2.15 到 2.49。FP8 目标模型配 BF16 训练的草稿头，接受率可能略降，要自己测。
- **xgrammar 的空白 bug**：vLLM 源码注释提到，xgrammar 0.2.8 在紧凑模式下仍会在逗号后放出空格，修复已于 2026-10-08 合入 xgrammar 主分支，等 0.2.9 发布。做紧凑 JSON 实验前，先抽查输出确认真的紧凑。
- **关掉思考模式**：Qwen3 的思考段不受语法约束，压测时统一用 `enable_thinking=False`，第八节会再提。

## 七、时间计划与里程碑

4 周拿到第一版完整结果，每周末用一道门槛决定是否进入下一阶段。

&#91;embedded content: 4 周计划 · 4 个阶段，3 道门槛\]

门槛没过就先定位原因，不要带着问题进入下一阶段。第 1 周的门槛最关键，它直接检验离线算出的上限在真实模型上是否成立。

## 八、风险与应对

最大的两个风险是被人抢先，以及 CPU 开销在高并发下吃掉收益；前者靠尽早公开和差异化，后者靠批量化和如实报告适用区间。

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| #62130 的作者先提 PR | 语法约束 EAGLE 起草这一半不再新 | 你的核心差异是强制 token 草稿、批量服务、30B 级模型和公开数据集。早点在 issue 下说明计划，争取协作而不是撞车 |
| CPU 填 bitmask 成为瓶颈 | 高并发下收益缩水甚至变负 | 线程池批量填、只在第 3 档位置同步；测出 CPU 耗时占比；必要时结合 vLLM 已有的按并发调 K 的动态投机配置 `num_speculative_tokens_per_batch_size` |
| 第 2 档分词不一致 | 第 2 档接受率偏低，而它是收益大头 | 带上下文重新分词；导出所有被拒位置逐个分析 |
| 短 schema 收益有限 | BFCL 平均只有 20 到 36 个 token，加速比不显眼 | 主战场放在长 schema 和信息抽取；按长度分桶报告，写清适用区间 |
| vLLM 代码变动快 | 补丁频繁冲突（RFC #48197 正在重构这块） | 固定 commit 做实验；改动集中在少数函数；补丁保持小 |
| 思考模型的边界 | 语法从思考结束才生效，边界可能落在草稿窗口里 | 先只支持非思考模式；思考模式作为扩展，复用 `_get_constraint_bounds` 的边界逻辑 |
| 大 batch 下投机本身负收益 | 高并发时所有投机方法都可能变慢 | 你的方法和基线一样扫并发，诚实报告拐点 |

## 九、简历写法与面试问题

简历上写清三件事：场景、相对谁快多少、为什么无损。下面的 X、a、b 全部换成你实测的数字，没测到的不要写。

**简历条目模板**：

- 在 vLLM 中实现语法感知投机解码：把 XGrammar 语法确定的 token 作为零成本草稿，并在 EAGLE-3 起草循环中加入语法约束。在 Qwen3-32B（单卡 H100）的工具调用和 JSON 抽取场景下，TPOT 相比 vLLM 官方 EAGLE-3 降低 X%，平均接受长度从 a 提升到 b，贪心输出逐 token 一致。
- 定位 vLLM 异步调度下草稿先于语法校验进入前向的问题，设计 worker 侧语法状态同步方案，使方法同时支持同步和异步调度。
- （如果合入）向 vLLM 提交 PR #xxxx，对应 2026 Q4 路线图中"投机解码在结构化输出下的鲁棒性"一项。

**面试常见追问和答题要点**：

| 问题 | 要点 |
| --- | --- |
| 为什么是无损的？ | 草稿当 one-hot 分布，接受概率是目标概率 p(t)，拒绝时从去掉 t 的目标分布重采样；强制 token 在 bitmask 后 p=1 |
| 和 jump-forward 有什么区别？ | jump-forward 替模型选定分词，可能改变分布；这里交给目标模型验证，不一致就拒绝 |
| 为什么单 token 强制只占 4% 左右？ | 同一串字符有多种合法切分，bitmask 会放行多个 token；收益主要来自字符串级确定 |
| 为什么异步调度下不能在调度器改草稿？ | 异步时前向先用 worker 自己的草稿启动，调度器之后才看到草稿，只能算 bitmask |
| 高并发下还有收益吗？ | 看你的并发扫描曲线；讲清 CPU 填 bitmask 的耗时占比和拐点 |
| 为什么不直接把 K 调大？ | 验证计算随 K 线性增长，大 batch 下计算受限；语法草稿的价值是让大 K 的位置大多被接受 |
| 打开 probabilistic 草稿采样怎么办？ | 被语法替换的位置要改成 one-hot 概率，否则接受率公式不成立 |
| 允许任意空白时收益掉多少？ | 离线分析里语法可确定比例从约 60% 降到约 30%，因为空白让很多位置不再唯一 |

## 参考资料

- [vLLM 源码 main 分支](https://github.com/vllm-project/vllm)（本报告读的是 2026-11-03 的 commit ffdf267）
- [vLLM 投机解码文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [vLLM 动态投机解码文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
- [vLLM 结构化解码介绍博客](https://vllm.ai/blog/struct-decode-intro)
- [vLLM issue #62130：语法约束采样与 EAGLE 投机解码结合](https://github.com/vllm-project/vllm/issues/62130)
- [grammar-spec-decode 原型仓库](https://github.com/YuvrajSinghBhadoria2/grammar-spec-decode)
- [vLLM 2026 Q4 路线图 #61024](https://github.com/vllm-project/vllm/issues/61024)
- [RFC #48197：StructuredOutputManager 与投机解码重构](https://github.com/vllm-project/vllm/issues/48197)
- [Apple coreai-models PR #391：JSON schema 下的投机解码](https://github.com/apple/coreai-models/pull/391)
- [SGLang 压缩 FSM 与 jump-forward 博客](https://www.lmsys.org/blog/2024-02-05-compressed-fsm)
- [DOMINO：Guiding LLMs The Right Way（ICML 2024）](https://proceedings.mlr.press/v235/beurer-kellner24a.html)
- [JSONSchemaBench 论文](https://arxiv.org/abs/2501.10868)
- [BFCL 数据（gorilla 仓库）](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [RedHatAI/Qwen3-32B-speculator.eagle3](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3)
- [RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.dflash](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.dflash)

## 附录：离线上限分析脚本

5.0 节的数字就是用这个脚本跑出来的，CPU 上约 10 秒。准备工作：`pip install xgrammar tiktoken dashscope`（Qwen 词表取自 dashscope 包内置的 `qwen.tiktoken`，把 `TIK` 改成你机器上的路径），再用 git sparse checkout 拉下 gorilla 仓库的 `berkeley-function-call-leaderboard/bfcl_eval/data`，把这个目录作为参数传给脚本。把第 67 行换成 `out.append((params, inst))` 就得到"只有 arguments"的两行结果。

```python
"""Offline upper-bound analysis: how many tokens of a tool-call JSON are
decided by the grammar alone (Qwen tokenizer, xgrammar), on BFCL data."""
import base64, json, sys, glob
import numpy as np
import tiktoken, torch
import xgrammar as xgr

TIK = "/usr/local/lib/python3.13/site-packages/dashscope/legacy_1x/tokenizers/qwen.tiktoken"
PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ranks = {}
for line in open(TIK, "rb"):
    if line.strip():
        tok, r = line.split()
        ranks[base64.b64decode(tok)] = int(r)
EOS = len(ranks)  # <|endoftext|>
enc = tiktoken.Encoding("qwen", pat_str=PAT, mergeable_ranks=ranks,
                        special_tokens={"<|endoftext|>": EOS})
vocab = [b""] * (EOS + 1)
for b, r in ranks.items():
    vocab[r] = b
info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=EOS + 1, stop_token_ids=[EOS])
compiler = xgr.GrammarCompiler(info, max_threads=8)

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": None}

def conv(s):
    s = dict(s)
    t = s.get("type")
    if t in TYPE_MAP:
        if TYPE_MAP[t] is None:
            s.pop("type")
        else:
            s["type"] = TYPE_MAP[t]
    for k in ("description", "default", "format", "optional"):
        s.pop(k, None)
    if "properties" in s:
        s["properties"] = {k: conv(v) for k, v in s["properties"].items()}
        s["additionalProperties"] = False
    if "items" in s:
        s["items"] = conv(s["items"])
    return s

def build(root, split):
    docs = [json.loads(l) for l in open(f"{root}/BFCL_v4_{split}.json")]
    ans = {json.loads(l)["id"]: json.loads(l) for l in open(f"{root}/possible_answer/BFCL_v4_{split}.json")}
    out = []
    for d in docs:
        if d["id"] not in ans:
            continue
        f = d["function"][0]
        gt = ans[d["id"]]["ground_truth"][0]
        name, args = next(iter(gt.items()))
        if name != f["name"]:
            continue
        params = conv(f["parameters"])
        inst = {}
        for k, opts in args.items():
            if not opts:
                continue
            v = opts[0]
            if v == "" and k not in params.get("required", []):
                continue
            inst[k] = v
        schema = {"type": "object",
                  "properties": {"name": {"const": f["name"]}, "arguments": params},
                  "required": ["name", "arguments"], "additionalProperties": False}
        out.append((schema, {"name": f["name"], "arguments": inst}))
    return out

def analyze(schema, obj, compact, K=8):
    sep = (",", ":") if compact else (", ", ": ")
    text = json.dumps(obj, separators=sep, ensure_ascii=False)
    toks = enc.encode(text) + [EOS]
    g = compiler.compile_json_schema(json.dumps(schema), any_whitespace=not compact,
                                     separators=sep if compact else None)
    m = xgr.GrammarMatcher(g)
    bm = xgr.allocate_token_bitmask(1, info.vocab_size)
    forced, jf_hit = [], []
    for t in toks:
        m.fill_next_token_bitmask(bm)
        bits = bm[0].numpy().view(np.uint32)
        n_allowed = int(np.unpackbits(bits.view(np.uint8)).sum())
        is_forced = n_allowed == 1
        jf = m.find_jump_forward_string() if not is_forced else ""
        jf_ok = bool(jf) and enc.encode(jf, allowed_special=set())[:1] == [t]
        forced.append(is_forced)
        jf_hit.append(jf_ok)
        if not m.accept_token(t):
            return None
    # simulate grammar-only drafting with K slots: tiers 1+2 propose, then verify
    n, i, steps = len(toks), 0, 0
    free = [a or b for a, b in zip(forced, jf_hit)]
    while i < n:
        acc = 0
        while acc < K and i + acc < n and free[i + acc]:
            acc += 1
        steps += 1
        i += acc + 1
    return dict(n=n, forced=sum(forced), jf=sum(jf_hit), free=sum(free), steps=steps)

root = sys.argv[1]
for compact in (False, True):
    tot = dict(n=0, forced=0, jf=0, free=0, steps=0); cnt = bad = 0
    for split in ("live_simple", "simple_python"):
        for schema, obj in build(root, split):
            try:
                r = analyze(schema, obj, compact)
            except Exception as e:
                r = None
            if r is None:
                bad += 1; continue
            cnt += 1
            for k in tot: tot[k] += r[k]
    print(f"compact={compact} samples={cnt} skipped={bad} tokens={tot['n']} "
          f"avg_len={tot['n']/cnt:.1f} forced={tot['forced']/tot['n']:.1%} "
          f"jump_fwd_extra={tot['jf']/tot['n']:.1%} grammar_free={tot['free']/tot['n']:.1%} "
          f"target_passes_saved={1-tot['steps']/tot['n']:.1%}")
```
