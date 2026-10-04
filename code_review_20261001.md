# Debate-ERC 代码审查修复契约（2026-10-01）

> **修复结果（2026-10-01 完成）**：57 条确认问题全部修复（1 Critical / 19 Major / 37 Minor）。
> A-D 四组并行 + E 组收尾，最终测试 **153 passed, 4 skipped**（修复前 76 passed）。
> 遗留说明：① importlinter 无法在当前环境安装（PyPI 受限），pyproject 中的分层配置为声明性交付，E 组已用 AST 静态扫描验证与实际 import 图零违例；② aux_task/evaluation 与 training 在 importlinter 配置中置同层（MultitaskScheduler 运行时持有 SFTTrainer 实例，严格分层需改为依赖注入，暂不做）。


本文件是 57 条已确认审查问题的修复依据。四组并行修复（A/B/C/D）+ 一组收尾（E），
**文件所有权互斥**，跨组行为以本文件「共享契约」为准，禁止修改他人所有权文件。

## 所有权划分

- **A组**：`src/debate_erc/config.py`、`configs/**/*.yaml`；例外许可：可改 `tests/test_config_invariants.py` 中 run_id/默认值相关断言
- **B组**：`src/debate_erc/training/*`、`models/*`、`aux_task/*`、`fewshot/*`
- **C组**：`src/debate_erc/pipeline/*`（runner/assembler/run/stages）；例外许可：可改 `tests/test_pipeline_assembly.py` 相关断言
- **D组**：`src/debate_erc/agents/*`、`debate/*`、`reasoning/*`、`routing/*`、`data/*`、`evaluation/*`、`utils/*`；例外许可：可改 `tests/test_debate_loop_mock.py` 相关断言
- **E组（A-D 完成后）**：`tests/*`（新增）、`scripts/*`、`pyproject.toml`

环境：`/home/lsy20252770/.conda/envs/instructerc/bin/python`（项目已 pip install -e）。
每组完成后运行 `cd /home/lsy20252770/Agent_Reason/Debate-ERC && ~/.conda/envs/instructerc/bin/python -m pytest tests/ -q`
（其他组改动可能引起无关失败，只需保证自己所有权文件相关测试通过、模块可 import）。

## 共享契约（所有组必须遵守）

### S1. run_id 编码（A 实现，E 测试）
`make_run_id(config, seed)` 新格式：
`{name}[_{dataset} 若非 meld]_{method}[_{parts 以 _ 连接}]_s{seed}`
- method 判定顺序不变：debate→`debate*`，routing→`adaptive*`，reasoning→`reason`，否则 `base`
- parts 新增：`dpo`（training.dpo_enabled）；aux 标签必须区分任务名（intensity→`auxi`，speaker_task→`auxs`，同时启用→`auxi_auxs` 按此序）
- 示例：`no_debate_reason_s42`、`debate_full_route_auxi_auxs_syn_focal_dpo_s42`、`base_sft_iemocap_s42`

### S2. 配置校验（A 实现）
`validate_config` 新增：
- `reask.mode != "conclude"` → 报错「未实现」；`debate.self_refine == true` → 报错「未实现」
- `reask.enabled` 时要求 `reask.low_confidence > debate.confidence_threshold`
- `training.dpo_enabled` 且 `lora.enabled == false` → 报错（7B 全参 DPO 必然 OOM）
- output_dir 禁 `/tmp`、`/var/tmp`、`/dev/shm` 前缀；删除冗余 `== "/tmp"` 分支
- `_from_dict`：dataclass 字段收到非 dict 值 / dataclass_seq 收到非 dict 元素 → ValueError
- 默认值变更：`reask.low_confidence: 0.75`、`training.eval_every_steps: 0`（0=关闭，避免默认开启拖慢训练）
- 同步修改 `configs/base.yaml` 及所有显式设 `low_confidence: 0.6` 的 YAML

### S3. DPO 修复（B/C 协作）
- C：runner 中 adapter 包装后重绑定 `model = adapter_manager.model`（若 adapter_manager 非 None），SFT/DPO/生成器统一用 PeftModel 引用
- B：`DPOTrainer.__init__` 对非 `PeftModel` 直接 `raise RuntimeError`（静默退化禁止）；epoch 落盘仅存 adapter（PeftModel.save_pretrained 默认行为，禁止落盘主干）
- B：偏好对 chosen/rejected 统一 `f" {label}"` 同构格式（推理文本只留在 `source_error_pattern`）

### S4. DebateOutcome 新字段（D 实现，C 消费）
`first_round_label: str | None = None`（辩论首轮 Agent1 标签）、
`reask_pre_label: str | None = None`（reask 前的 final_label）、
`generation_calls: int = 0`（本样本 LLM 生成调用次数，DebateLoop/reask_loop 各 infer +1）

### S5. SFTTrainer 接口（B 实现）
- `build_dataset(samples)` 公开；`train(..., dataset=None)` 复用预构建 dataset
- warmup：`get_linear_schedule_with_warmup`，`num_training_steps = steps_budget or 总步数`，warmup = `warmup_ratio × num_training_steps`（docstring 注明 chunk 模式下按 chunk 近似）
- `eval_every_steps > 0` 且传入 eval_fn 时按 global_step 触发；eval_fn 调用前 `model.eval()` 后恢复 `train()`
- MultitaskScheduler：各任务 dataset 只构建一次；`total_main = min(steps_per_epoch*epochs, max_steps)`（max_steps>0 时）；辅助任务预算 = `weight × total_main`
- focal_ce 接受已展平的 `[N, V]` float logits + 展平 targets（调用方传 flat，避免双份复制）

### S6. runner 落盘/统计（C 实现，E 消费）
- `results["smoke_samples"] = config.smoke_samples`
- dev 辩论前 `model.eval()`（DPO 偏好对不受 dropout 污染）
- 每 seed run 开头 `torch.cuda.reset_peak_memory_stats()`
- 单 Agent 基线 `single_labels` 用 `first_round_label`（缺失回退 proponent 末轮）
- oracle gap 同口径：`upper_bound_wf1 − wf1(actual_preds)`，结果同时记录 `actual_wf1`
- 多辅助任务 switch_steps 不一致时 warning 并取第一个
- 辩论汇总增加 `mean_generation_calls`
- `training.eval_every_steps > 0` 时 runner 构造 eval_fn（direct agent 对 dev 前 min(200, len) 条子集算 W-F1）

### S7. 辩论/推理行为（D 实现）
- 轮次注入兜底：未收敛且 critic critique_points 为空且标签不一致 → 注入 `[f"label disagreement: {a1.label} vs {a2.label}"]`
- DebateLoop 给 critic 传 `reasoning_hint`；`reask_loop` 签名加 `reasoning_hint: str | None = None` 并透传
- `CausalProponentAgent.infer_reask(context, disagreement_points, previous_label, previous_confidence, reasoning_hint=None)`：渲染 `reask.j2`（含 re-derive 指令），复用标签解析；`ReaskFallback.reask_loop` 优先调用它（hasattr 探测），否则回退 infer
- reask_loop 首次更新前写入 `outcome.reask_pre_label`
- `compute_reask_correction_rate`：分子 = `reask_pre_label != gold 且 final_label == gold`（pre 为 None 时回退旧口径）
- 置信度解析：捕获值 >1 且 ≤100 → 除以 100；>100 → 回退 0.5
- critic 解析：缺 `Critique:` 锚 → 返回 `[]`
- `TemplateRouter(schema, long_utterance_chars=150)`：assembler 注入 `config.routing.long_threshold`；MINORITY 模板仅当 `"fear" in schema.labels`，否则回退 TRIGGER
- `DialogueContext.render(max_chars=800)`：超限**保尾截断**（目标行必保留，从尾部往前累加行直到预算），删除 ×2 系数
- `IEMOCAP_SCHEMA` 改 4 类（neutral/joyful/sadness/anger），iemocap_loader docstring 同步
- significance：bootstrap 分位改线性插值；配对 t 常数非零差 → `(inf, 0.0)`（全零差仍 `(0.0, 1.0)`）；`compare_systems` 的 d 改配对口径 `d_z = mean(diffs)/sd(diffs)`
- `OracleUpperBound.gap(actual_preds)` 改同口径（S6）
- synthesis：组合去重（无放回）；docstring fear 268→482
- `_fallback_score` 实现程度词放大（booster 倍率作用于下一个情感词）
- `class_balanced_sampler_labels` 加 `rng: random.Random` 参数做随机欠采样（默认 None 时用模块级 Random(42)）
- `_reached_consensus` 改用 `outcome.converged`
- backbone：pad_token_id 断言改 `RuntimeError`；`label_confidence` 编码 `f" {cand}"`
- DPO 参考前向 `model.eval()` 包裹后恢复 train
- seed.py：PYTHONHASHSEED 注释标明仅对子进程生效
- utils/label_schema.py 注释指向 `tests/test_data_official_counts.py`（E 将创建）

## E 组任务（A-D 完成后）
1. `pyproject.toml`：dependencies 加 `accelerate>=0.34`；`[tool.importlinter]` layers（utils → data/models → agents/reasoning/routing/fewshot/aux_task → debate → training → pipeline；evaluation 只依赖低层）；dev extras 加 `importlinter`
2. `scripts/run_ablation_matrix.py`：`--smoke` 时向子进程传 `--output-dir outputs/smoke_runs`
3. `scripts/aggregate_results.py`：过滤 `smoke_samples > 0` 的 run；主方法 vs 基线补 bootstrap CI（读 predictions.jsonl 调 `significance.bootstrap_ci`）
4. 新增测试：`test_run_id_uniqueness.py`（全部 configs×{42,43,44} run_id 两两唯一 + S1 示例）、`test_preference_pair.py`（chosen/rejected 同构、不读收敛状态）、`test_significance_numerics.py`（常数差 p→0、CI 插值、d_z）、`test_data_official_counts.py`（伪造样本对拍 MELD_OFFICIAL_COUNTS）
5. 适配既有测试：`test_debate_loop_mock.py`（reask_correction_rate 新口径期望值、补 converged-but-wrong 样本使 error_consensus_rate 断言有区分度、first_round_label/reask_pre_label 断言）
6. 全量 pytest 通过（76+ 新增）

---

# 第二轮全面审查（2026-10-01）

> **修复结果（2026-10-01 完成）**：19 条确认问题全部修复（2 Major / 17 Minor），测试 **183 passed, 4 skipped**（修复前 153 passed，新增 30 个回归测试）。
> 功能性验证：19 个配置全部加载+校验通过（iemocap run_id 已无 syn 段）；reask.j2 空分歧点/有分歧点双分支渲染正确；render 目标行锁定实证通过；importlinter 分层契约按修正后的方向+同层互禁语义 AST 复核零违例。

## 确认问题（修复状态标注）

### Major（2 条）

| ID | 问题 | 位置 | 改进建议 |
|----|------|------|----------|
| R2-M1 | IEMOCAP synthesis 静默失效 + run_id 错误归因：IEMOCAP 4 类不含 fear/disgust → originals=0 → n_new=0 静默返回 `[]`（仅 info 级全 0 统计），但 `make_run_id` 仅凭 `synthesis_enabled` 开关即追加 syn 段 → run_id 表观"有合成增强"实际为零，消融归因错误 | `src/debate_erc/data/synthesis.py:78-82,128-134`；`src/debate_erc/config.py:373-374`；`configs/experiments/iemocap_debate_full.yaml:17-19`；`src/debate_erc/pipeline/runner.py:114-120` | ① `apply_fewshot_synthesis` 对目标类集 ∩ 少数类为空时 `raise ValueError`（或 validate_config 拒绝 dataset 非 MELD 且 synthesis.enabled）；② run_id 的 syn 段以"实际合成数 > 0"为准 |
| R2-M2 | 训练模块第一轮核心修复零回归测试：left_truncate_with_bos、warmup 调度、budget 截断跳收尾、flush 梯度补偿、per-epoch loss 重置、DPO PeftModel 强校验、参考前向 eval 包裹、adapter-only 落盘等约 20 处修复，tests/ 7 个文件无一 import SFTTrainer/DPOTrainer——静默回退将直接污染训练结果且测试全绿 | `src/debate_erc/training/sft_trainer.py:41-60,291-368`；`src/debate_erc/training/dpo_trainer.py:132-139,194-206,240-245` | 补训练模块回归测试：left_truncate_with_bos 单测（BOS 重前置/保尾）、warmup 调度数值单测、budget 截断跳过收尾、DPO 非 PeftModel raise + adapter-only 落盘（mock） |

### Minor（17 条）

| ID | 问题 | 位置 | 改进建议 |
|----|------|------|----------|
| R2-M3 | `DialogueContext.render` 保尾截断在目标行居中且其后内容超预算时丢失目标行（实证复现 max_chars=30 时输出仅 `'D: D'`，违反 docstring 契约）；当前 4 处生产构造点 target 恒为末位、不可触发，属潜在契约缺陷 | `src/debate_erc/data/schemas.py:40-64` | 先锁定目标行再从目标行向两侧分配预算；补 target 居中契约测试 |
| R2-M4 | importlinter layers 用 `\|` 声明 training/aux_task/evaluation 同层互相独立，与 `multitask_scheduler.py:26 from ..training.sft_trainer import SFTTrainer` 矛盾（官方文档：`\|` 分隔 = not allowed to import each other，装上 lint-imports 必报违例；第一轮"AST 零违例"结论基于错误语义） | `pyproject.toml:44-51`；`src/debate_erc/aux_task/multitask_scheduler.py:26` | `"\|"` 改列表形式 `["debate_erc.training", "debate_erc.aux_task", "debate_erc.evaluation"]`（同层允许互相 import） |
| R2-m1 | `_from_dict` 对 tuple 标量字段收字符串静默透传：YAML `seeds: "42"` → run.py `list("42")` 拆成单字符 seed，深层 `np.random.seed('4')` 抛难定位 TypeError | `src/debate_erc/config.py:207-210`；`src/debate_erc/pipeline/run.py:41` | `_from_dict` 对 tuple 字段元素做类型强转/校验，失败即 ValueError（配置期报错） |
| R2-m2 | `AuxTaskConfig.name` 注释列出 irony 但 runner 仅实现 intensity/speaker_task：错误在模型加载后才 raise；lora.enabled=false 时静默无操作跑完全程 | `src/debate_erc/config.py:58`；`src/debate_erc/pipeline/runner.py:275-283` | validate_config 增加辅助任务名白名单校验 |
| R2-m3 | long_threshold 边界语义不一致：difficulty_router 用 `>=`、template_router 用 `>`，恰等于阈值（150 字符）的样本两路由器结论矛盾 | `src/debate_erc/routing/difficulty_router.py:67`；`src/debate_erc/reasoning/template_router.py:68` | 统一为 `>=`（与 config.py:80 注释一致） |
| R2-m4 | tokenizer 先行右截断（truncation=True/max_length=2048）架空 left_truncate_with_bos 保尾语义：prompt 超长时尾部（含目标话语）先被截，自定义截断与 orig_len 告警名存实亡；当前 render 800 字符预算下触发罕见 | `src/debate_erc/training/sft_trainer.py:92-99`；`src/debate_erc/training/dpo_trainer.py:45-48` | prompt tokenize 改 `truncation=False`，截断全权交 left_truncate_with_bos |
| R2-m5 | aux 预算 `max(1, ...)` 兜底使 weight=0 仍训 1 步，违反 S5 契约"辅助任务总预算 = weight × total_main" | `src/debate_erc/aux_task/multitask_scheduler.py:124-127,150-152` | weight ≤ 0 时预算置 0 并跳过该任务 |
| R2-m6 | DPO max_steps 截断的 epoch 仍无条件落盘 `dpo_epoch{n}`，与 SFT 侧截断时显式跳过 epoch 收尾的语义不一致，checkpoint 易误读为完整 epoch | `src/debate_erc/training/dpo_trainer.py:240-245` | 与 SFT 同口径：截断时跳过 epoch 落盘 |
| R2-m7 | `_label_token_offset` 签名含 label 参数但函数体未使用（死参数） | `src/debate_erc/training/sft_trainer.py:63-69` | 删除该参数 |
| R2-m8 | 收敛低置信触发 reask 且 critic critique_points 为空时，reask.j2 渲染空分歧点列表却指令"逐条解决上述分歧"，提示自相矛盾（非必现：critic 认同标签时仍可给非空审查点） | `src/debate_erc/debate/debate_loop.py:80-83`；`src/debate_erc/agents/prompts/reask.j2:9-17` | reask.j2 对空分歧点加 Jinja 分支，改指令为"复核低置信度判断的依据" |
| R2-m9 | `_summarize_debate_local`（约 70 行）为不可达死代码：debate.consensus 导入恒成功 → try/except 永不触发；模块 docstring"consensus 尚未就绪"已过时 | `src/debate_erc/evaluation/debate_metrics.py:15-18,26-97,124-127` | 删除回退分支与本地实现，直接 import；更新 docstring |
| R2-m12 | paired_ttest 常数负差返回 +inf（数学极限应为 -inf），t 值符号错误；aggregate_results 直接打印 t，恒劣情形报告方向相反；现有测试仅锁定恒正差 | `src/debate_erc/evaluation/significance.py:76-79`；`scripts/aggregate_results.py:168,173` | 按差值符号返回 ±inf；补恒负差数值测试 |
| R2-m13 | `get_logger` 的 log_file 参数全项目零调用，run 级日志从不落盘（模块 docstring"run 级别落盘"承诺未兑现，长跑日志随终端丢失） | `src/debate_erc/utils/logging.py:10-27` | runner/run.py 装配时传 `log_file=outputs/{run_id}/run.log` |
| R2-m14 | MELD loader 对 labels/speakers/sentences 三并行列表长度零校验：不一致时静默 zip 截断 / 延迟 IndexError | `src/debate_erc/data/meld_loader.py:36-40` | `load_meld_split` 增加长度一致性断言 |
| R2-m15 | `import os` 位于双层 for 循环体内 | `scripts/run_ablation_matrix.py:89` | 提升至模块顶部 |
| R2-m16 | reask 条件 2（收敛低置信）集成路径无测试锁定：现有集成测试双方置信 0.3 < 0.6 实际走条件 1；条件 2 需 mock 置信度落在 [confidence_threshold, low_confidence) 区间 | `tests/test_pipeline_assembly.py:114-133` | 补条件 2 集成测试（mock 置信度落在阈值区间） |
| R2-m17 | 三个脚本核心行为零测试覆盖（矩阵展开与续跑判断、smoke 过滤与双口径聚合、数据准备核对） | `scripts/prepare_meld.py:29-52`；`scripts/run_ablation_matrix.py:45-106`；`scripts/aggregate_results.py:36-251` | 抽取纯函数补单测（矩阵展开、聚合口径） |

## 误报剔除（2 条，双验证代理 2/2 一致）

| ID | 原声称 | 剔除依据 |
|----|--------|----------|
| R2-m10 | IEMOCAP loader `.get(raw_label, "neutral")` 缺失字段静默回退 | `IEMOCAP_INDEX_TO_RAW` 值集 {happy,sad,neutral,angry,excited,frustrated} 与 `IEMOCAP_LABEL_MAP` 键集完全相等，默认分支不可达；索引越界/缺字段均显式 KeyError（`iemocap_loader.py:38-40`）——完备映射下的防御性写法 |
| R2-m11 | synthesis fear/disgust 同 seed 产生逐条完全相关样本 | 实证同 seed=42 下两类合成样本 utterances 逐条完全相同为 0/240：fear 路径每样本多消耗一次 `rng.choice` 使随机流从首轮分叉，仅约 1% 结构性随机碰撞（`synthesis.py:83-134`）——虚假相关不成立 |

## 修复优先级建议

1. **P0（实验有效性）**：R2-M1（归因错误）+ R2-m12（显著性方向）+ R2-m4（截断语义）——直接影响实验结论口径
2. **P1（防线）**：R2-M2（训练回归测试）+ R2-m16（条件 2 集成测试）——防止静默回退
3. **P2（一致性/健壮性）**：R2-M4、R2-m1/m2/m3/m5/m6/m14——配置期报错与口径统一
4. **P3（清理）**：R2-M3、R2-m7/m8/m9/m13/m15/m17——死代码、提示词鲁棒性、日志落盘
