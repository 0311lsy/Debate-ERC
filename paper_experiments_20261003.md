# 实验记录与论文素材：异构选择性辩论（Selective Heterogeneous Deliberation）

> 归档日期：2026-10-03
> 数据集：MELD（train 9989 / dev 1109 / test 2610），7 类
> 标签序：neutral, joyful, sadness, anger, surprise, disgust, fear
> test gold 分布：neutral 1256 / joyful 402 / anger 345 / surprise 281 / sadness 208 / disgust 68 / fear 50
> 硬件：单卡 48GB（双 7B 常驻实测 37.5GB）；环境 conda `instructerc`（torch 2.11+cu128, transformers 4.48.3, peft 0.14）
> 状态说明：**8-seed 主实验已完成并通过显著性检验（2026-10-04）**；第二异构家族 Mistral、Self-Consistency 对照、hint 消融均已完成。剩余：三分支 risk-coverage 整合、IEMOCAP 泛化。

---

## 1. 方法定义

**Proponent（主模型）**：LLaMA2-7B（NousResearch/Llama-2-7b-hf）+ LoRA 纯 SFT（m0_sft，r32，33.6M 可训练参数，3 epoch）。
**Critic（异构复审者）**：Qwen2.5-7B + LoRA SFT（r64/a128/dropout 0.05，q_proj+v_proj，3 epoch，lr 2e-4，bs4×ga8），与主模型**完全相同的训练数据/提示口径/超参预算**，仅骨干家族不同。

**置信度**：`score_labels()` 对完整 `" label"` token 序列 teacher-forcing logsum 后候选内 softmax。不剥离首 token（LLaMA2 `" neutral"` 为 [29871, 21104] 双 token，Qwen 为单 token，剥首 token 会破坏 Qwen 口径）。该置信度**仅作门控信号**，不替代贪心解码做预测。

**三分支（当前实现为前两支）**：
1. `p_a ≥ τ`：主模型直答（捷径，不调用 critic）；
2. `p_a < τ`：调用 critic，当 `y_b ≠ y_a` 且 `p_b > p_a + margin` 时翻案；
3. （规划）极低置信弃权（abstention），与第 4 节贡献线整合。

**固定超参（论文口径）**：τ=0.65，margin=0.05，由三 seed 合并 dev（3×1109）网格一次标定，test 全程零调参。贪心解码（do_sample=False, max_new_tokens=8）。

---

## 2. Layer 0：可信基线（统一框架下自复现）

| 系统 | test W-F1 | Macro-F1 | Acc | 备注 |
|---|---|---|---|---|
| **m0_sft（纯 SFT，主基线）** | **0.6860**（s42）；3-seed 68.78±0.19 | 0.5462 | 0.6946 | 贪心生成 |
| m1_sft（合成 1.8x + focal + cb + aux） | 0.6661 | — | — | 比 m0 低 ~2.4 点 |
| m1_dpo | 0.6621 | — | — | ECE 0.276，过自信，校准劣化 |

- 主基线跨 seed 极稳：s42/43/44 = 68.60 / 68.97 / 68.76（mean 68.78 ± 0.19）。
- 与历史 PRC-Emo 项目（default prompting，seed42 离线忠实重算 test W-F1=68.56）相互印证：LLaMA2-7B 裸 SFT 在 MELD 的真实水位为 **68.5–69.0**。
- 置信度分桶单调（m0）：p<0.5 桶 acc 0.483 → p≥0.9 桶 acc 0.904，支持其作为门控信号。

---

## 3. 对照阶梯：为什么必须"异构 + 同能力"

| Critic 配置 | 与主模型一致率 | critic 独立 Acc/W-F1 | test ΔW-F1 | 结论 |
|---|---|---|---|---|
| 同构 self-debate（LLaMA2 自己） | 91.3% | 0.707（与主模型相同） | **−0.64**（翻转改对/改错=48/49） | 错误高度相关，证伪 |
| 异构 Qwen-1.5B（能力不对等） | 56.1% | Acc 0.486（87% 退化为 neutral）；W-F1 0.486 | −0.05（8 次翻转 1:1） | 有独立性、无能力，证伪 |
| **异构 Qwen2.5-7B（能力对等）** | 79.9%（dev） | Acc 0.700 / W-F1 0.6916 | **+1.01（s42）** | 独立性+对等同时满足才有效 |

**核心论点**：复审增益严格依赖 critic 的独立性（非同构）与能力对等（非弱模型）两者同时成立。

---

## 4. Abstention 贡献线（m0_sft，`outputs/abstention/selective_prediction.json`）

| 指标 | 数值 |
|---|---|
| AURC（置信度排序） | **0.156** |
| AURC 随机弃权对照 | 0.307 |
| AURC oracle | 0.052 |
| 风险 ≤5% 的覆盖率 | 19.2% |
| 风险 ≤10% 的覆盖率 | 37.0% |
| 弃权 20% 后作答集 W-F1 | **72.38**（随机弃权 68.88） |
| 弃权 30% 后作答集 W-F1 | **74.69**（Acc 77.12） |
| 弃权 40% 后作答集 W-F1 | 77.69（Acc 80.65，Macro 降至 51.43） |

m1_dpo AURC=0.193，定量证明 DPO 破坏校准。
**论文用途**：三分支框架（直答/辩论/弃权）的动机与独立贡献线；弃权大幅提升 W-F1/Acc 但 Macro 随覆盖下降，需在论文中诚实交代。

---

## 5. 3-seed 主结果（test，n=2610）

### 5.1 各 seed 独立 dev 标定 τ

| seed | τ | 基线 W-F1 | 辩论 W-F1 | ΔW-F1 | bootstrap 95% CI | McNemar p | 翻转 对/错 |
|---|---|---|---|---|---|---|---|
| 42 | 0.70 | 68.60 | 69.61 | **+1.01** | [+0.23, +1.82] | 0.0199 | 66/41 |
| 43 | 0.50 | 68.97 | 68.85 | **−0.12** | [−0.72, +0.48] | 0.603 | 27/32 |
| 44 | 0.60 | 68.76 | 69.42 | **+0.66** | [−0.17, +1.53] | 0.073 | 73/52 |
| 均值 | — | 68.78±0.19 | 69.29±0.39 | **+0.52±0.58** | — | — | — |

Macro 54.67±0.09 → 55.17±0.95（Δ +0.50±1.02）；Acc 69.51±0.09 → 70.04±0.54。
跨 seed 配对 t 检验不显著（t≈1.55, p≈0.26）。

### 5.2 共享 τ*=0.65（pooled-dev 标定，test 零调参）——**8-seed 最终主结果**

| seed | 基线 | 辩论 | ΔW-F1 | 翻转 rescue/harm | critic 独立 W-F1 |
|---|---|---|---|---|---|
| 42 | 68.60 | 69.63 | **+1.04** | 64/38 | 69.16 |
| 43 | 68.97 | 68.70 | **−0.27** | 50/60 | 68.39 |
| 44 | 68.76 | 69.66 | **+0.90** | 81/54 | 69.00 |
| 45 | 69.14 | 69.57 | **+0.43** | 65/54 | — |
| 46 | 68.83 | 69.38 | **+0.56** | 68/51 | 68.37 |
| 47 | 69.78 | 69.68 | **−0.10** | 51/49 | 68.64 |
| 48 | 68.35 | 69.09 | **+0.74** | 78/54 | 68.53 |
| 49 | 68.61 | 69.25 | **+0.65** | 68/54 | 68.94 |

**显著性（N=8 配对 seed，统一固定 τ=0.65/margin=0.05，n=2610/seed，6 正 2 负；脚本 `scripts/aggregate_8seeds_uniform.py`）**：

| 检验 | 统计量 | 结论 |
|---|---|---|
| 均值 ± std | **+0.49 ± 0.46 pp** | critic 调用率 24.8–27.8%（8 seed 稳定） |
| paired t-test | t=3.030, **p=0.019** | ✅ 显著 |
| paired bootstrap 95% CI（10 万次） | **[+0.184, +0.771] pp** | ✅ 下界 > 0 |
| Wilcoxon signed-rank | **p=0.039** | ✅ 显著 |
| 配对标准化效应 Cohen's d_z | **1.07（大效应）** | 绝对量小但相对训练噪声大 |
| 剔除 s42 后 t 检验 | +0.42pp, t=2.517, **p=0.046** | ✅ 非单点拉动 |
| 符号检验 | p=0.29（6/8，N 小功效低，正常） | 辅助 |

> 口径修正记录（2026-10-04）：初版聚合误将 s42/43/44 的"独立标定 τ"值（+1.01/−0.12/+0.66）与 s45–49 的"固定 τ"值混合检验。终版一律以 τ=0.65 从逐条四元组离线重算（s42/43/44 用 cross/ 全量 critic 转储恢复，n=2610），s42/43/44 修正为 +1.04/−0.27/+0.90。结论方向不变，数值以本节为准。

**边界观察（写论文主动交代）**：两个非正 seed（43/47）不能剔除（选择性报告=学术红线）。s47 主模型为 8 seed 中最强（69.78），增益≈0——呈"**基线越强、可补空间越小**"的收益递减模式，属合理边界而非方法缺陷；s43 归因见第 6 节（双弱配对）。

### 5.3 根因分析（排除阈值/规则解释）
- pooled-dev 网格最优规则即 τ=0.65/margin=0.05；加 critic 置信度地板、加大 margin 均更差。三 seed **dev 全部为正**（+1.06/+1.43/+0.96），s43 dev 曾是三者最好，无法用 dev 预警其 test 失败 → 属配对级方差，非规则可修复。

---

## 6. 3×3 因子交叉（8/9 配对为正，关键归因证据）

τ=0.65/margin=0.05，3 proponent × 3 critic 在相同 test 上离线组合：

|  | critic42 | critic43 | critic44 | 行均（proponent） |
|---|---|---|---|---|
| **prop42** | +1.04 | +0.65 | +1.13 | +0.94 |
| **prop43** | +0.48 | **−0.27** | +0.69 | +0.30 |
| **prop44** | +0.52 | +0.05 | +0.90 | +0.49 |
| 列均（critic） | +0.68 | +0.14 | +0.90 | — |

- **9 格均值 +0.58±0.45；8/9 为正；6 个非对角格全部为正（+0.59±0.35）**。
- 唯一负格 (prop43, critic43) 恰为**最弱 proponent × 最弱 critic**：critic43 独立 W-F1 66.33（c42 67.58 / c44 67.49），prop43 行均最低。二者分别换配即回正。
- 结论：s43 失败是"双弱同 seed"低概率配对交互，非方法失效。实际部署中两模型独立训练，同 seed 双弱概率低。
- 数据文件：`outputs/selective_hetero/cross/{critic42,critic43,critic44}_all_test.jsonl`；结果 `cross_3x3_analysis.txt`。

---

## 7. 朴素集成基线对照（防"这不就是 ensemble 吗"）

相同两模型全部输出（critic 100% 调用）上的策略比较，3-seed 均值：

| 策略 | W-F1 | Macro | Acc | critic 调用率 |
|---|---|---|---|---|
| Proponent only | 68.78±0.19 | 54.67 | 69.51 | 0% |
| Critic only | 67.13±0.70 | 53.61 | 67.66 | 100% |
| Confidence-pick（谁置信高取谁） | 69.37±0.53 | 55.45 | 70.09 | 100% |
| 无门控按 margin 翻案 | 69.51±0.57 | 55.77 | 70.23 | 100% |
| **Ours（门控 τ=0.65）** | **69.33±0.54** | 55.39 | 70.06 | **26.7%** |
| Oracle（两答案二选一上界） | 76.08±0.52 | 64.75 | 76.70 | 100% |

**论点**：门控用 ~1/4 的 critic 调用取得与全量朴素集成几乎相同的增益（69.33 vs 69.37）；调用率 27%→100% 仅多买 0.18 W-F1。critic 单独使用比主模型低 1.65 点，却提供互补增益——排除"直接换强模型"解释。Oracle 76.08 显示互补空间大、方法只开采一部分（future work 动机）。

### 7.1 Self-Consistency 预算匹配对照（s42，`outputs/self_consistency/sc_k5_s42/`）

主模型同提示采样 k 次（T=0.7/top-p=0.9）多数投票，平票以标签序列后验破平：

| 策略 | W-F1 | Δ vs 贪心 | 前向预算/样本 |
|---|---|---|---|
| 贪心（基线） | 68.60 | — | 1× |
| SC@3 | 66.59 | **−2.00** | 3× |
| SC@5 | 67.04 | **−1.55** | 5× |
| **异构辩论（我们）** | **69.61** | **+1.01** | **~1.27×**（1 + 27% 触发） |

**论点**：短标签 ERC 上采样随机性的危害大于投票收益，SC 用 3–5 倍预算反而显著掉点；我们以约 1/4 的额外预算超过其 5 倍预算 2.6 个点——效率叙事闭环。

### 7.2 第二异构家族 Mistral-7B（防"Qwen 特例"，s42）

Mistral-7B-v0.1 以完全相同数据/LoRA SFT 协议训练 critic（3 epoch loss 0.449/0.310/0.144），固定 τ=0.65/margin=0.05：

| Critic 家族 | critic 独立 W-F1 | 辩论 test W-F1 | ΔW-F1 |
|---|---|---|---|
| Qwen2.5-7B | 69.16 | 69.61 | **+1.01** |
| **Mistral-7B** | 68.25 | 68.88 | **+0.29** |

两个独立模型族方向一致为正，"异构性"主张获双家族背书。Mistral 增益较小，与其 critic 独立能力较弱一致，符合"互补性 × 能力"的机制预期。

### 7.3 CRITIC_REASONING_HINT 消融（防"增益是 prompt 工程"，s42）

同一 Qwen critic adapter，仅在推理时去掉三类错误自检指引（`--no-critic-hint`），严格同 τ=0.65/margin=0.05：

| 配置 | ΔW-F1 | 翻转 rescue/harm | critic 独立 W-F1 |
|---|---|---|---|
| 有 hint（主实验） | **+1.04** | 64/38 | 69.16 |
| 无 hint（普通 classify 提示） | **+0.84** | 65/43 | 69.29 |

**论点**：移除 hint 后仍保留 **81%（0.84/1.04）** 的增益——增益主体来自异构复审 + 门控架构，非提示词工程；hint 的边际作用是提高翻案精度（误伤 43→38）。
代码护栏：`SelectiveDeliberator` 的 `critic_hint=None` 才用默认，显式空串不再被回退（修复了"无法关闭 hint"的 bug）。

---

## 8. τ 敏感性与成本-增益（宽平台，非挑尖峰）

| τ | dev ΔW-F1 | test ΔW-F1 | test 调用率 |
|---|---|---|---|
| 0.30 | +0.12 | +0.01 | 0.7% |
| 0.40 | +0.38 | +0.09 | 4.6% |
| 0.50 | +0.96 | +0.43 | 12.3% |
| 0.60 | +1.02 | +0.48 | 22.3% |
| **0.65** | **+1.15** | +0.56 | 26.7% |
| 0.70 | +1.09 | +0.60 | 31.4% |
| 0.80 | +0.97 | +0.68 | 41.6% |
| 0.90 | +1.01 | +0.74 | 55.4% |

dev 上 τ∈[0.50,0.90] 增益连片（+0.96~+1.15），0.65 落在平台区。test 增益随调用率单调缓升——天然的**可调节计算分配**曲线（论文配图）。

---

## 9. 逐类别 F1 与翻案机制（3-seed 合计 498 次翻转）

| 类别 | prop F1 | ours F1 | Δ |
|---|---|---|---|
| neutral | 81.00 | 81.47 | +0.46 |
| joyful | 67.40 | 66.85 | −0.54 |
| sadness | 45.47 | 46.69 | **+1.21** |
| anger | 56.20 | 57.95 | **+1.74** |
| surprise | 62.71 | 62.94 | +0.24 |
| disgust | 39.14 | 42.94 | **+3.80** |
| fear | 30.80 | 28.91 | **−1.89** |

翻转三分类：**rescue（错→对）195 / harm（对→错）152 / neutral（错→错）151，rescue/harm=1.28，净 +43**（与 ΔAcc 逐 seed 自洽：s42 +26 / s43 −10 / s44 +27）。
按 gold 类别 rescue/harm/neutral：

| gold | rescue | harm | neutral |
|---|---|---|---|
| neutral | 73 | 55 | 41 |
| joyful | 25 | 32 | 25 |
| sadness | 15 | 14 | 24 |
| anger | 49 | 14 | 28 |
| surprise | 16 | 27 | 16 |
| disgust | 15 | 2 | 5 |
| fear | 2 | 8 | 12 |

**机制结论**：增益集中于中等频次、易混情绪（disgust 最干净 15:2、anger 49:14、sadness）；fear（test 仅 50 条）翻案 2:8 不可靠，是第三分支"极低置信弃权而非翻案"的直接动机。

---

## 10. 成本与工程口径

- 显存：双 7B 推理 37.5GB；单模型 logprob 批打分安全 batch=32（batch=64 峰值 46.7GB，禁用）。
- 速度：双模型终审 0.4–0.7s/条；触发率 ~27%，即每样本期望 critic 调用 0.27 次（非双倍成本）。
- 训练（串行不共存）：m0_sft ~27.5 分钟/seed（skip-eval 快路径，1652s）；Qwen critic ~18 分钟/seed；终审 ~35 分钟/seed。
- 训练成本对等：proponent 33.6M LoRA 参数；critic 3 epoch loss 三 seed 轨迹重合（~0.53/0.40/0.27）。

---

## 11. 统计规范

- 配对设计：每个 seed 用自己的基线算 Δ；**8 seeds（42–44、45–49）全部完成**。
- 每个配置：W-F1/Macro/Acc，10 万次 paired bootstrap 95% CI，精确 McNemar（翻案方向），跨 seed 报告 mean±std、配对 t 检验与 Wilcoxon。
- dev 标定与 test 终审严格分离；共享 τ 仅从合并 dev 产生一次，全 8 seed test 零调参。
- 所有消融表必须 ≥3 seed 并带 CI，禁止只放点估计。（SC/hint/Mistral 当前为 s42 单 seed 对照，属"有没有"的存在性验证，正式表格中应注明或补 seed。）

## 12. 待办实验（按优先级）

1. **[完成]** 8-seed 显著性（+0.48±0.40，p=0.012，CI [+0.22,+0.72]）。
2. **[完成]** 第二异构家族 Mistral-7B（+0.29，双家族为正）。
3. **[完成]** hint 有/无消融（无 hint 保留 81% 增益）。
4. **[完成]** self-consistency 预算匹配对照（SC@3/5 反降 1.5–2.0 点）。
5. **[下一步·纯 CPU]** 三分支整合：直答/辩论/弃权统一 risk-coverage 曲线（"仅弃权" vs "弃权+辩论"），含 Pareto 主图（SC/集成/oracle/ours 按调用率布点）。
6. **[下一步·GPU 半天]** IEMOCAP 泛化（关闭 synthesis），验证框架跨数据集；通过后补 Mistral@IEMOCAP。
7. Case study：按 hint 三类错误（trigger mismatch / context reversal / sarcasm）归因 3–5 个救回案例 + fear 失败案例。
8. [附录级] M1 配方消融（syn1.2/focal/cb/auxi，配置已备，各 1 seed）；轻量 DPO 改造（仅答错样本构对 + 1 epoch + dev ECE 门槛）。
9. [可选/rebuttal] 更小对等 critic（3B 级）成本扫描；SC/hint/Mistral 补到 3 seed。

## 13. 建议论文故事线

单模型盲点不可自查（同构 self-debate 证伪，−0.64）→ 异构同能力模型提供独立视角（1.5B 对照 + 跨家族泛化）→ 全量咨询昂贵且冗余（朴素集成对照：27% 调用≈100% 收益）→ **不确定性驱动的自适应推理分配：有把握直答、拿不准请异构同行复审、极低置信弃权**（τ 成本平台 + risk-coverage + 类别机制）→ 多 seed 显著性 + 组件消融 + 误差归因。

## 14. 文件与结果索引

- 代码：`src/debate_erc/deliberation/selective.py`（SelectiveDeliberator）；`models/backbone.py::score_labels`；`models/adapter_manager.py`（同名加载护栏）。
- 脚本：`scripts/eval_selective.py`（同构）、`eval_selective_hetero.py`（异构，支持 `--fixed-tau/--fixed-margin`）、`train_critic_sft.py`、`analyze_abstention.py`、`dump_critic_all.py`、`cross_pair_analysis.py`、`rule_search_dev.py`、`shared_tau_analysis.py`、`ensemble_baselines_offline.py`、`aggregate_selective.py`。
- 结果：
  - m0 adapters：`outputs/runs/base_sft_s{42,43,44}/adapter_main/main/`
  - critic adapters：`outputs/hetero_critic/qwen7b_sft_s{42,43,44}/adapter_main/main/`
  - s42 终审：`outputs/selective_hetero/qwen7b/{dev_calibration.json,test_metrics.json,*_records.jsonl}`
  - s43/s44：`outputs/selective_hetero/qwen7b_s{43,44}{,_shared}/`
  - 交叉与集成：`outputs/selective_hetero/cross/{cross_3x3_analysis.txt,ensemble_baselines.txt,ensemble_baselines.json}`
  - 弃权：`outputs/abstention/selective_prediction.json`
  - 8-seed 终审：`outputs/selective_hetero/qwen7b_s{45..49}_shared/test_metrics.json`（42/43/44 见 `qwen7b/`、`qwen7b_s{43,44}/`）
  - Self-Consistency：`outputs/self_consistency/sc_k5_s42/{summary.json,records.jsonl}`
  - Mistral critic：adapter `outputs/hetero_critic/mistral7b_sft_s42/`，终审 `outputs/selective_hetero/mistral7b_s42/`
  - hint 消融：`outputs/selective_hetero/qwen7b_s42_nohint/`
  - 队列脚本：`scripts/post_chain_queue.sh`（SC→hint→Mistral）、`scripts/redo_queue.sh`；日志 `outputs/{post_chain_queue,redo_queue}.log`
- 测试：219 passed / 4 skipped（含 test_label_scoring / test_adapter_guard / test_selective_deliberation）。

## 15. 可直接引用的结论句（8-seed 终版）

- 中文（主结果）：在 MELD 上，门控异构复审以平均 26.7% 的额外复审调用（8 seed 区间 24.8–27.8%），使加权 F1 提升 **+0.49 个百分点（8 个配对随机种子，mean±std=+0.49±0.46，配对 t 检验 p=0.019，bootstrap 95% CI [+0.18,+0.77]，Wilcoxon p=0.039，配对标准化效应 d_z=1.07；6/8 种子为正）**；增益集中于 disgust/anger/sadness 等易混类别。同构自辩（−0.64）与能力不对等的 1.5B 复审者（−0.05）均无增益，且该增益在 Qwen2.5-7B 与 Mistral-7B 两个独立模型族上方向一致（+1.04/+0.29），证明增益同时依赖复审视角的独立性与能力对等性。
- 中文（效率）：在同等推理预算比较中，self-consistency 即便使用 3–5 倍前向调用反而使 W-F1 下降 1.5–2.0 点，而本方法以约 1.27 倍期望前向取得 +1.0；移除复审自检提示后仍保留 81% 的增益，表明提升来自异构复审架构而非提示工程。
- English (main result): On MELD, gated heterogeneous review improves weighted F1 by **+0.49 points across 8 paired seeds (mean ± std = +0.49 ± 0.46; paired t-test p=0.019; bootstrap 95% CI [+0.18, +0.77]; Wilcoxon p=0.039; paired standardized effect d_z=1.07; 6/8 seeds positive)** while invoking the reviewer on only 26.7% of utterances (24.8–27.8%). Both homogeneous self-debate (−0.64) and an under-capable 1.5B reviewer (−0.05) fail, and the gain replicates across two model families (Qwen2.5-7B +1.04, Mistral-7B +0.29), demonstrating that the benefit requires reviewer independence and capability parity jointly.
- English (efficiency): At matched inference budgets, self-consistency degrades W-F1 by 1.5–2.0 points despite 3–5× forward calls, whereas our method gains +1.01 at ~1.27× expected forwards; removing the reviewer's diagnostic hint retains 81% of the gain, attributing the improvement to the heterogeneous review architecture rather than prompt engineering.

---

# 第二部分：结果反映了什么（论文写作的解释层）

> 以下不是数字复述，而是从第 2–11 节结果中可合法推出的机制主张、边界条件与攻防口径。每条都标注证据链，写论文时可直接改写为段落；标 **[推断]** 者属于合理但需更谨慎措辞的解释，标 **[待验证]** 者尚未取得直接证据。

## 16. 六条核心机制推断（论文的"解释力"来源）

### M1. ERC 错误是系统性相关的，"同一模型多算几次"无法修复
- **证据链**：① 同构 self-debate −0.64，proponent/critic 一致率 91.3%、独立准确率完全相同（0.707）；② self-consistency 用 3–5 倍采样预算反降 1.5–2.0 点（第 7.1 节）。
- **反映的东西**：7B-SFT 模型在 MELD 上的错误主体不是随机噪声（aleatoric fluctuation），而是由参数化知识与训练分布决定的**相关盲点（correlated, epistemic errors）**。同一参数下的重复采样/自辩高度同质，投票无法把相互关联的错误"平均掉"；而且短标签任务采样（T=0.7）本身注入噪声，使 SC 劣于贪心。
- **论文用途**：这是全文 motivation 的理论支点——**为什么"辩"必须辩给一个独立训练的他模型，而不是让自己多想几遍**。把 self-debate 与 SC 两个负结果并置，可以写成一句有力的话：*"额外的同源计算（homogeneous compute）不提供新信息；错误去相关需要参数层面的独立性。"*

### M2. 复审增益 = 独立性 × 能力，是乘法关系而非加法
- **证据链**（对照阶梯）：同构（有能力、无独立性）−0.64；1.5B（有独立性、无能力）−0.05；Qwen-7B（两者俱备）+1.04；Mistral-7B（独立性与能力均居中）+0.29；3×3 交叉中唯一负格是"双弱配对"。
- **反映的东西**：两个必要条件任一缺失增益即消失，说明机制不是"多一个模型投票"，而是**"另一个在我会错的地方恰好不错的模型"**。critic 独立 W-F1 低于 proponent 却仍有正贡献（69.16 vs 68.60 量级相当、条件错误分布不同），证明价值来自**错误集合的错位（error decorrelation）而非平均能力**。
- **论文用途**：支撑方法的设计原则（heterogeneous + capability-parity peer），也给出选型标准：critic 不追求更强，追求"同能力、异家族"。
- **谨慎点 [推断]**：Mistral 增益小于 Qwen，有两个不互斥的解释——(a) Mistral 与 LLaMA2 同属 LLaMA 架构谱系（GQA、相近分词与预训练语料），独立性天然较弱；(b) Mistral critic 独立能力也更弱（68.25 vs 69.16）。现有实验无法分离二者，论文应**主动写明混淆**并把"能力匹配下的独立性扫描"列为 future work，不能直接主张 (a)。

### M3. 错误在置信度轴上"可路由"，门控的本质是按期望价值分配计算
- **证据链**：p_a 分桶准确率单调（p<0.5 桶 0.483 → p≥0.9 桶 0.904）；仅调用 26.7% 的 critic 即取得全量集成 97% 的收益；8 个 seed 调用率高度稳定（24.8–27.8%），说明路由行为不依赖特定训练实现；全量翻案（100% 调用）只多买 0.18 点。
- **反映的东西**：复审的期望收益在样本间极不均匀，且标签序列后验 p_a 是一个足够好的、零额外训练的路由信号。方法因此可表述为 **selective computation / EV-aware routing**：高置信样本上复审 EV≈0（还可能为负），低置信样本上 EV 为正。
- **论文用途**：把"省成本"从工程妥协改写成方法的**核心设计主张**。τ 不是凑出来的截断，而是计算-精度的可调旋钮（第 8 节宽平台），部署者按预算在 Pareto 前沿上选点。
- **诚实点**：门控相对全量翻案有 0.18 点精度代价——这是用 3.7 倍成本差换来的 Pareto 选择，正文需给出两个点而非只报自己赢的口径。

### M4. 类别级结果揭示了翻案成立的条件：集体有知 vs 集体无知
- **证据链**：disgust rescue/harm=15/2（+3.80）、anger=49/14、sadness 净正；fear=2/8（−1.89，test 仅 50 条）；oracle 二选一上界 76.08，而方法只到 69.3。
- **反映的东西**：翻案有效的微观条件是"**proponent 在该类易错、critic 在该类条件错误率更低**"——disgust/anger/sadness 属于中等频次、语义线索易混但可被独立视角纠偏；fear 属于**两模型集体不擅长**（训练样本极少），critic 的异议往往只是"用另一个错误替换这一个错误"。oracle 与实际值的 6.8 点差距说明：两个模型的集体知识是存在的，当前的 margin 仲裁规则只开采了其中一部分。
- **论文用途**：① 为第三分支"弃权"提供定量动机——集体无知区不该翻案、该弃权；② 为 future work（学习型仲裁器、类别条件门控、置信度校准）给出明确的上界空间。

### M5. 收益随基模型增强而递减——天花板效应界定了方法的适用区
- **证据链**：s47 proponent 最强（69.78）时 Δ≈0（rescue/harm=51/49）；s48 proponent 最弱（68.35）时 Δ=+0.74；8 seed 呈基线越高增益越小的趋势。
- **反映的东西**：可恢复错误池随基模型增强而收缩，方法的价值最大区间是"**中等强度基模型 + 固定训练预算**"。这与 M3 一致——路由应当只在复审期望收益为正的实例/模型上开启。
- **论文用途**：写进 boundary/limitations，反而强化框架叙事（adaptive allocation where EV>0）；不要把它藏起来。注意：8 个点的趋势是描述性的，正式主张需要相关性检验 [推断]。

### M6. 训练侧"偏好锐化"与推理侧"外部复审"是两条正交的轴，且前者会伤害路由
- **证据链**：m1_dpo W-F1 更低（66.21）、ECE 0.276（m0 为 0.116，logprob 口径）、AURC 0.193（m0 为 0.156）；m1 合成+focal+cb+aux 配方整体落后纯 SFT 约 2.4 点。
- **反映的东西**：为拉开 chosen/rejected margin 做的偏好优化会系统性地制造过自信，恶化置信度的排序质量——而选择性方法恰恰依赖这个排序。**推理时的 deliberation/abstention 与训练时的 likelihood shaping 是两个正交杠杆，且后者可能损害前者赖以工作的校准信号**。
- **论文用途**：在 related work/discussion 中定位本方法：它不与"更强的训练"竞争，而是为已训练模型提供错误处理机制；同时提示 selective system 应把校准作为一等目标。

## 17. 边界条件与失败模式总表（Limitations 的素材，主动写=加分）

| 边界 | 证据 | 论文中如何表述 |
|---|---|---|
| 绝对增益小（+0.49pp） | 8-seed 主结果 | 明确定位为"效率优先的选择性推理"而非性能突破；用 d_z=1.07、Pareto、相对 SC/self-debate 的优势框定贡献 |
| 双弱配对失效 | s43 −0.27，3×3 唯一负格 | 部署建议：独立训 2–3 个 critic 候选，按 dev 选配对（廉价实用）；多配对集成可作稳定性方案 [待验证] |
| 强基线饱和 | s47 ≈0 | 天花板效应，方法价值在中等基模型区 |
| 极小类集体无知 | fear 2/8 | 由弃权分支处理；类别条件门控是 future work |
| 第二家族增益较小 | Mistral +0.29 | 独立性与能力的混淆尚未分离（见 M2） |
| 单数据集 | 仅 MELD | IEMOCAP 泛化是必补实验，不补则全文加 scope 限定 |
| 超参单次标定 | τ/margin 仅 pooled-dev 一次 | 已保证 test 零调参；可补"每 seed 独立标定"作为附录对照（第 5.1 节已有 3 seed 数据） |
| SC/hint/Mistral 单 seed | 存在性验证 | 正式表格须注明或补 3 seed；主结论不依赖这些对照的点估计 |
| 双模型部署成本 | 37.5GB 显存 | 工程上 critic 可按需加载/卸载，不必常驻；报告期望调用率而非峰值显存作为主成本口径 |

## 18. 审稿攻防预案（预判意见 → 已备证据 → 建议措辞）

| # | 预判审稿意见 | 已有证据 | 应答要点 |
|---|---|---|---|
| Q1 | "+0.5 太小，没有意义" | d_z=1.07；CI 下界 +0.18；基线 68.6–69.8 已强 | 不拼绝对点，拼三件事：**效率**（1.27× 前向）、**方向稳健**（8 seed/2 家族）、**对照全负**（SC、self-debate、1.5B、DPO）。引用 oracle 76.08 说明上界在规则而非思路 |
| Q2 | "这不就是 ensemble / 为什么不概率平均" | 第 7 节集成表 | 全量集成 100% 调用仅 69.37–69.51，门控 26.7% 达 69.33；critic-only 反低 1.65 点。主张是选择性路由，不是融合精度 |
| Q3 | "τ 是不是在 test 上调的" | dev 平台 + test 零调参 + 8 seed 共用一个 τ | 第 8 节宽平台 + 标定流程写清；公布 pooled-dev 网格 |
| Q4 | "s43/s47 为负为什么还报" | 3×3 归因、收益递减 | 选择性报告是红线；主动给 failure analysis；6/8 正且统计显著本就允许个体为负 |
| Q5 | "Mistral 只 +0.29，泛化很弱吧" | 双家族同向 | 强调"方向复制"而非幅度复制；主动交代独立性/能力混淆与 future 实验 |
| Q6 | "为什么不用 self-consistency / 多采样" | SC@3/5 为负 | 同源计算不提供新信息（M1）；且我们更便宜 |
| Q7 | "增益来自那段 critic 提示词吧" | 无 hint 保留 81% | 架构为主、hint 仅微调翻案精度；hint 不含 gold/标签信息，无泄漏 |
| Q8 | "和 selective prediction / abstention 有何区别" | §4 + §21 三分支（已完成） | 弃权只减覆盖，辩论在不改覆盖的前提下改答案；统一框架显示：浅层弃权带内两者互补（5% 带 +0.79），深度弃权时增益被吸收，AURC 改善未显著——主动报告边界 |
| Q9 | "critic 自己更差，凭什么改我的答案" | critic 独立 W-F1 与 prop 同档、条件错误错位 | 互补性 ≠ 平均能力；给逐类 rescue/harm 与一致率 79.9% |
| Q10 | "test 上跑了全量 critic 是否泄漏" | 门控仅用 p_a | 全量 critic 仅用于离线分析/上界；在线流程中 critic 只在 p_a<τ 时被调用；论文需明确区分 |
| Q11 | "统计是否可靠" | 配对 t/bootstrap/Wilcoxon/剔除 s42 | 给全部四种检验、CI、效应量；全 8 seed 明细表放附录 |
| Q12 | "ECE/置信度是在哪个口径测的" | 第 2/4 节 | 明确 logprob logsum 仅作门控信号，预测口径始终是贪心生成；两个数字不混用 |
| Q13 | "双 7B 不实用" | 37.5GB、0.4–0.7s/条 | 按需加载 critic（73% 样本根本不加载）；期望成本口径；future 蒸馏 3B |

## 19. 论文写作映射（数字 → 位置 → 允许的主张强度）

### 正文图表规划
| 位置 | 内容 | 数据来源 |
|---|---|---|
| Fig.1（动机） | 错误相关性：一致率 91.3%（同构）vs 79.9%（异构）；同源多算（self-debate/SC）收益为负 | §3、§7.1 |
| Table 1（主结果） | 8-seed mean±std + CI + p；base/ours 两行 + Δ | §5.2 |
| Fig.2（主图，Pareto）✅已完成 | x=期望前向调用，y=W-F1：贪心(1×,68.60)、critic-only(2×,67.58)、conf-pick(2×,69.18)、always-consult+flip(2×,69.56)、oracle(2×,75.53)、SC@3/5、门控 τ 扫描曲线（0.4–0.9）。图：`outputs/figures/fig_pareto.{png,pdf}`；布点：`outputs/figures/pareto_points.json`；脚本 `scripts/make_paper_figures.py` | §7、§7.1、§8 |
| Table 2（对照阶梯） | 同构 / 1.5B / Qwen-7B / Mistral-7B | §3、§7.2 |
| Table 3（组件消融） | gate × hint × margin 全组合 | §7.3 + 离线 |
| Table 4（逐类） | 7 类 ΔF1 + rescue/harm/neutral | §9 |
| Fig.3（框架）✅已完成 | 三分支 risk-coverage：仅弃权 vs 弃权+辩论（含 oracle 虚线、高覆盖区放大插图）。图：`outputs/figures/fig_risk_coverage.{png,pdf}`；数值：`outputs/three_branch/three_branch_risk_coverage.json`、`curves.npz`；脚本 `scripts/three_branch_analysis.py`。**诚实结论：AURC 改善未达显著（见 §21）** | §21 |
| Fig.4（超参） | τ 的 dev/test 双曲线 | §8 |

### 附录
3×3 交叉全表、8 seed 逐行明细、dev 标定网格、训练 loss 曲线、成本与显存明细、s43/s47 失败案例、ECE/AURC 计算口径。

### 允许 / 不允许的主张
- ✅ 可以说："在固定训练预算与 ~1.27× 推理成本下，带来统计显著但幅度温和的提升（+0.49pp，CI [+0.18,+0.77]）"；"增益需要独立性与能力对等同时成立"；"门控取得全量集成约 97% 的收益"；"同源增算（自辩/采样）在 ERC 短标签任务上无效甚至有害"。
- ❌ 不能说："显著优于集成方法"（69.33 < 69.51）；"对任意异构 critic 有效"（1.5B 失败、fear 失败）；"超越 PRC-Emo/InstructERC"（协议不同）；把 s42 的 +1.04 当 headline；"Mistral 证明独立性越强增益越大"（有能力混淆）。

## 20. Limitations 与 Future Work 草拟（可直接改写）

**Limitations**：(1) 提升幅度温和，且在基模型很强时趋于消失（天花板效应）；(2) 增益存在配对级方差，8 seed 中 2 个非正，部署需按 dev 选择模型配对；(3) 对训练样本极少的类别（fear）复审倾向于以错易错，需弃权兜底；(4) 目前验证限于 MELD；(5) critic 的"独立性"与"能力"两个因素在 Mistral 对照中尚未完全解耦；(6) 仲裁规则是手工的置信度差，oracle 上界（76.1）表明仍有 6+ 点开采空间。

**Future work**：① 学习型/类别条件仲裁器与校准，逼近 oracle；② 三分支统一框架下联合标定 τ_直答/τ_弃权；③ critic 蒸馏（3B 或更小）与按需加载的部署优化；④ 独立性×能力的因子实验（同族不同规模、异族同能力）；⑤ 跨数据集（IEMOCAP）与跨任务（对话立场/关系抽取）泛化；⑥ 训练时保持校准的偏好优化，避免 DPO 式过自信损伤路由。

# 第三部分：三分支统一框架与论文主图（2026-10-04 完成）

## 21. 三分支（直答/复审/弃权）与 Pareto 主图

口径：s42 全量 2610 条四元组（gold, y_a, p_a, y_b, p_b），统一固定 τ=0.65/margin=0.05。弃权置信度统一用部署口径 `p_a`（`score_labels` 的完整标签序列 teacher-forcing logsum + 候选内 softmax，不剥首 token）；预测口径始终为贪心生成。脚本 `scripts/three_branch_analysis.py`，结果 `outputs/three_branch/`。

### 21.1 AURC（风险-覆盖率曲线下面积，越低越好）

| 方案 | AURC | 说明 |
|---|---|---|
| A：仅按 p_a 弃权 | 0.1368 | 经典 selective prediction |
| DA：门控复审 + 弃权 | 0.1359 | 复审后再以原 p_a 排序弃权 |
| DAf：以终判置信排序 | 0.1368 | 复审后用终判置信排序（不更好，故主分析不用） |
| oracle-A | 0.0523 | 知道谁错、直接弃权 |
| oracle-DA | 0.0487 | oracle 复审 + 弃权 |

- DA − A = **−0.00089**，配对 bootstrap 95% CI **[−0.00202, +0.00019]，跨 0，未达显著**。
- cov@risk≤5% = 24.1%；cov@risk≤10% = 39.9%（A 与 DA 相同——曲线左半段由弃权主导）。
- 图 `outputs/figures/fig_risk_coverage.{png,pdf}`：两条实线几乎重合，高覆盖区（0.70–1.0）放大插图可见红线下穿蓝线。

### 21.2 各弃权比例作答集 W-F1（A vs DA）

| 弃权比例 | 仅弃权 A | 复审+弃权 DA | Δ |
|---|---|---|---|
| 5%（作答 95%） | 70.75 | **71.54** | **+0.79** |
| 10% | 72.40 | **73.01** | **+0.61** |
| 20% | 75.35 | 75.43 | +0.08 |
| 30% | 79.09 | 79.09 | 0 |
| 40% | 82.30 | 82.30 | 0 |

### 21.3 机制解释（为什么 AURC 几乎不动、全量点却有 +1.04）

- **91% 的 rescue 翻案落在 p_a 最低 20% 弃权带内，30% 弃权带覆盖 100% 的 rescue**：一旦深度弃权，被复审救回的样本本就会被弃权机制丢掉，两条机制作用在同一批低置信样本上，增益被吸收。
- 复审的价值集中在**必须全量作答（高覆盖、不能大量弃权）的部署区间**：5–10% 弃权带内仍有 +0.6~0.8pp，且不牺牲覆盖；oracle-DA < oracle-A（0.0487 vs 0.0523）说明"能改答案"在理论上确实优于"只会拒答"，但当前手工仲裁规则只兑现了很小一部分（oracle 上界 75.5）。
- 与 §5.2 主结果不矛盾：主结果测覆盖 100% 时的点指标（+1.04），AURC 测整条风险-覆盖曲线的积分；复审是"高覆盖区修正器"，不是"弃权替代物"。

### 21.4 Pareto 成本-精度主图（Fig.2）

`outputs/figures/fig_pareto.{png,pdf}`（布点明细 `pareto_points.json`，s42）：
- 门控 τ 扫描（0.4→0.9）构成 Pareto 前沿，成本 1.04–1.56 前向/条、W-F1 68.97–69.64，**τ=0.65 居峰（1.26×，69.64）**且 0.5–0.7 为宽平台。
- 同成本 2× 的全量 critic 策略：critic-only 67.58（更差）、confidence-pick 69.18、always-consult+flip 69.56——门控以约 1/4 的 critic 调用拿到前沿点。
- SC@3=66.59、SC@5=67.04（成本 3×/5×，精度反低于贪心 68.60）；oracle 75.53 标示上界缺口。
- 论文读法：门控点（69.64 @1.26×）对全量 critic 三策略（均 @2×）同时**更便宜**，对其中两者（critic-only 67.58、conf-pick 69.18）也**更准**；对最强对照 always-flip（69.56）精度仅高 0.08pp，故主张口径为"成本-精度 Pareto 占优"，不得写成"显著优于集成"（见 §19 ❌）。

### 21.5 置信度口径统一声明（论文必须注明）

早期 abstention 分析（`outputs/eval_logprob_layer0_details.jsonl` 的 `m0_sft_p`）与现行 `p_a` 排序不一致（Spearman 0.80；最低 20% 集合 Jaccard 仅 0.30；旧均值 0.771 vs 新 0.793）。旧打分脚本已不在仓内。**所有选择性数字（三分支、门控、ECE/AURC）一律以 `p_a` 为准，两套数字禁止混用**（旧 abstention 弃权 20% = 72.38 vs 新口径 75.35 即源于此）。论文附录需一句话说明打分函数在终版被统一。

### 21.6 本节允许 / 不允许的主张

- ✅ "在覆盖不变（100% 作答）的前提下提升精度；在允许少量弃权的区间（5–10%）与弃权互补；复审定位为高覆盖区修正机制"。
- ✅ "门控点位于成本-精度 Pareto 前沿：1.26× 期望前向下 W-F1 69.64，优于同成本带所有对照"。
- ❌ 不得宣称"改善 risk-coverage 曲线/AURC"——差值 CI 跨 0，未显著；写为"未观察到 AURC 显著改善，增益集中于高覆盖区"。
- ❌ 不得用旧 abstention 数字与新数字直接对比下结论。

## 22. IEMOCAP 跨数据集泛化（2026-10-04 完成，s42）——回应"单数据集质疑"

与 MELD 完全同配方（LLaMA2-7B m0 + Qwen2.5-7B critic，LoRA r64/a128，3 epoch，synthesis 关闭；[base_sft_iemocap.yaml](configs/experiments/base_sft_iemocap.yaml)，链脚本 `scripts/iemocap_chain_s42.sh`）。IEMOCAP 4 类口径（neutral/joyful/sadness/anger），train 5163 / dev 647 / test 1623，**不与 MELD 跨集比较绝对值**。

### 22.1 两种参数口径（test W-F1，n=1623）

| 口径 | 参数来源 | base → debate | Δ(pp) | 触发率 | 翻转(对/错/净) |
|---|---|---|---|---|---|
| 自身标定 | IEMOCAP dev 网格（τ=0.6, margin=0.15） | 79.93 → 80.33 | **+0.39** | 11.2% | 43（23/15/+8） |
| **零调参迁移** | **冻结 MELD pooled-dev 的 τ=0.65/margin=0.05** | 79.93 → **80.86** | **+0.92** | 15.3% | 78（43/26/+17） |

- proponent 独立 79.93；critic 独立 80.4~80.65（与 proponent 同档，满足 M2"能力对等"前提）。
- 迁移口径下触发桶（n=248）内 proponent 准确率仅 49.6%，复审后 56.5%（桶内 +6.9pp）——复审精准命中低置信困难样本，**与 MELD 的机制完全一致（M3 可路由性跨数据集成立）**。
- 逐类 ΔF1（迁移口径）：neutral +0.87、joyful +1.06、sadness +0.83、anger −0.27——anger（样本最多类）轻微受损，其余三类全正。

### 22.2 值得注意的意外发现

**冻结 MELD 参数（+0.92）优于 IEMOCAP 自身 dev 标定（+0.39）**。IEMOCAP dev 仅 647 条，标定网格在小 dev 上噪声大、易过拟合（dev 上 τ=0.6/m=0.15 的 W-F1 76.47 仅比 base 75.98 高 0.49）；而 MELD pooled-dev 标定的 τ=0.65 来自 3×1109 条，更稳健。这给论文一个额外论点：**门控参数有良好的跨数据集迁移性，pooled-dev 大样本标定优于目标数据集的小 dev 标定**。

### 22.3 本节允许 / 不允许的主张

- ✅ "方法在两个标签体系、对话风格、类别分布均不同的数据集上同向为正（MELD 8-seed 均值 +0.49，IEMOCAP +0.39~+0.92）"。
- ✅ "MELD 上标定的门控参数可直接迁移到 IEMOCAP 且效果不损反增（零 test 调参）"。
- ❌ 单 seed（s42），不得报统计显著；需补 s43/44 后方可下显著性结论。
- ❌ 不得跨数据集比较绝对 W-F1（4 类 vs 7 类任务难度不可比）。

## 23. 学习型仲裁器探索（2026-10-04，纯 CPU）——回应"oracle 缺口 6 点"

动机：oracle 二选一上界 75.53 与规则门控 69.63 之间有 ~6 点缺口，审稿人会问"为什么不用学习型仲裁器把缺口补上"。本节用逻辑回归（LR）仲裁器做边界探测。

**协议**（`scripts/lr_arbiter.py`，结果 `outputs/arbiter/lr_arbiter.json`，s42 同口径）：
- 特征：p_a、p_b、p_b−p_a、|p_a−p_b|、低置信指示（p_a<0.65）、y_a/y_b 的 dev 先验错误率（7 维，无 gold 泄漏）。
- 训练：dev 全量四元组（1109）中 y_a≠y_b 的分歧子集（223 条，分歧率 20.1%）学 P(y_b 正确 | 特征)。
- 评估：test 只评一次；最终预测 = 分歧且 LR 判 y_b 时取 y_b，否则取 y_a。

### 23.1 主对比（test，n=2610）

| 仲裁策略 | W-F1 | 翻转数 | rescue/harm | 翻转精度 |
|---|---|---|---|---|
| proponent only | 68.60 | 0 | — | — |
| **规则门控（论文方法）** | **69.63** | 154 | 64/38 | 41.6% |
| LR @ thr=0.50（默认） | 69.05 | 37 | 19/8 | 51.4% |
| **LR @ thr=0.35（扫描最优）** | **69.65** | 178 | 78/52 | 43.8% |
| conf-pick（分歧时谁高取谁） | 69.18 | 218 | 83/69 | 38.1% |
| always-flip（分歧全翻） | 67.58 | 533 | 179/208 | 33.6% |
| oracle（分歧完美仲裁） | 75.53 | — | 179/0 | 100% |

### 23.2 三个可写进论文的结论

1. **规则门控已贴近该特征族的可用上限**：LR 全阈值扫描的最优点（69.65@0.35）与手工规则（69.63）持平，说明在"p_a/p_b 置信度 + 类别先验"特征族下，简单规则与学习型仲裁器等效——规则的简洁性成为优点而非缺陷。
2. **oracle 缺口的本质是信噪比约束**：test 分歧池 533 条中 y_b 正确率仅 33.6%（179 潜在 rescue vs 208 潜在 harm vs 146 双错），完美仲裁需要从约 1:1.16 的救/伤先验中逐样本分离；dev 仅 223 个分歧样本可供训练，信号不足以支撑更细粒度分离。LR 高精度区间（thr=0.55 翻转精度 75%）只剩 12 次翻转，覆盖不动缺口主体。
3. **缺口开采的主战场不在仲裁形式**：146 个"双错"分歧样本（27% 的分歧池）两模型都给不出正确答案，任何二选一仲裁都无法获益——这部分缺口需要第三分支（弃权/人工）或更强基座，而非更聪明的仲裁。

### 23.3 允许 / 不允许的主张

- ✅ "学习型仲裁器在相同特征下不超过手工规则（69.65 vs 69.63），规则门控已接近该信号族的仲裁上限"。
- ✅ "oracle 缺口受分歧池信噪比约束（y_b 正确率先验 33.6%）与双错子集（27%）的结构限制"。
- ❌ 单 seed（s42）、探索性分析，正文放 Discussion/附录，不进主结果表。

## 24. Case study（2026-10-04，s42，τ=0.65/margin=0.05 规则下实际翻案）

规则下 test 实际翻转 102 次（rescue 64 / harm 38）。以下案例文本由 `PRC-Emo/data/raw/meld.test.json` 按全局展平 idx 回溯（loader：`src/debate_erc/data/meld_loader.py`）。

### 24.1 Rescue：翻案成功

**案例 R1（idx=1122，gold=disgust，主 anger 0.55 → 审 disgust 0.86）——厌恶与愤怒的细粒度区分**
> Monica: Okay well then, Ill fire him today and you go out with him for another week.
> **Phoebe [TARGET]: Are you kidding?! Another week with that sip, Ill kill**
- 主模型被判刑式语气误导为 anger；复审识别出对"再约会一周"的生理性厌恶（disgust）。呼应 §9  disgust 是最干净的增益类（15:2）。

**案例 R2（idx=934，gold=sadness，主 neutral 0.32 → 审 sadness 0.38）——双低置信下的委婉忧伤**
> Monica: Then why are you smoking?
> **Chandler [TARGET]: Well its very unsettling.**
- 两模型置信度都极低（0.32/0.38），但复审把"unsettling"背后的无奈识别为 sadness 而非字面 neutral。低置信触发（p_a≪τ）使此类样本必被复审。

**案例 R3（idx=161，gold=anger，主 neutral 0.53 → 审 anger 0.62）——压抑语气的恼怒**
> Ross: Four percent. Okay. I tip more than that when theres a bug in my food.
> **Rachel [TARGET]: Ross, tonight was about the two of you getting along. Oh, would you just see my chiropractor, already.**
- 表面是建议句式（主模型判 neutral），复审结合上文 Ross 的挖苦识别出 Rachel 的不耐烦（anger）。

### 24.2 Harm：翻案失败（诚实交代）

**案例 H1（idx=139，gold=neutral，主 neutral 0.59 → 审 anger 0.68，翻错）**
> Monica: Hey, Joey, could you pass the cheese?
> **Joey [TARGET]: Listen uh, Id prefer it if you didnt call me Joey.**
- "Id prefer it if you didnt" 的礼貌克制被复审误读为愤怒前兆。p_a=0.59 处于门控触发带（<0.65），margin 未能挡住——此类"礼貌但疏离"语气是误伤高发模式。

### 24.3 Fear 类正反对照（test 仅 50 条，翻转不可靠的直接证据）

**案例 F1（idx=1725，gold=fear，主 anger 0.64 → 审 fear 0.81，救回）**
> Phoebe: The charity's on fire! / **Phoebe [TARGET]: Help!**
- 单词呼救"Help!"，主模型受上文 "Hey!!" 误导为 anger，复审正确识别为恐惧求助。

**案例 F2（idx=1574，gold=fear，主 fear 0.55 → 审 anger 0.83，翻错）**
> The Stripper: Okay, which one of you guys is Gunther Central-Perk? Hey, Joey?
> **Ross [TARGET]: Wheres my ring? My dead grandmothers wedding ring? Where is it? Where is it?**
- 主模型 0.55 低置信判对 fear（丢失遗物引发的恐慌），复审被重复追问的激烈语气（0.83 高置信）带向 anger。fear 样本仅 50 条，单案例即占 2% 波动——支撑第三分支"极低置信弃权而非翻案"的设计（§9）。

### 24.4 写作要点

- rescue 案例覆盖 disgust/sadness/anger 三大增益类，与 §9 逐类 ΔF1 自洽；harm 案例不回避（rescue/harm=1.28 的代价侧）。
- fear 对照证明"低置信翻案在小样本类上方向不定"，引用为弃权分支动机，不得写成"fear 上方法有效"。

### 22.4 【3-seed 终审更新 2026-10-04】s43/s44 补齐，零调参迁移统计显著

s43/s44 全链完成（队列 15:30 结束），终版 3-seed 数字（n=1623/seed）：

| 口径 | seed42 | seed43 | seed44 | mean±std | 符号 | paired t |
|---|---|---|---|---|---|---|
| 自身 dev 标定 Δ | +0.39 | +0.36 | +0.46 | **+0.40±0.05** | 3/3 正 | p=0.005 |
| **冻结 MELD τ=0.65 迁移 Δ** | **+0.92** | **+1.69** | **+1.02** | **+1.21±0.42** | **3/3 正** | **p=0.037** |

- 迁移口径 base/final W-F1：80.16±0.34 → **81.37±0.45**；触发率 13.6%（12.7–15.3% 稳定）；翻转合计 rescue/harm=124/59。
- 自身标定 s43/s44 的 dev 各自选到低 τ（触发率仅 2.2%/2.7% vs s42 11.2%），增益被调用率压低——进一步印证 §22.2"小 dev 标定噪声大"，冻结大样本 pooled-dev 参数更优（3-seed 迁移 1.21 ≫ 自身标定 0.40）。
- **论文口径升级**：摘要/主文的 IEMOCAP 数字从单点 +0.92 改为 **3-seed +1.21±0.42（p=0.037，3/3 为正）**；跨数据集零调参迁移现在有统计支撑，不再是单 seed 证据。
- 仍禁止跨数据集比较绝对值（MELD 7 类 vs IEMOCAP 4 类）。

### 7.4 【3-seed 终审更新 2026-10-04】hint 边际作用不稳定——s42 单点结论收回

s43/s44 nohint 终审完成（tag `qwen7b_s43_nohint`/`qwen7b_s44_nohint`，固定 τ=0.65/m=0.05），与同 seed 有 hint 主实验配对：

| seed | 有 hint Δ | 无 hint Δ | hint−nohint |
|---|---|---|---|
| 42 | +1.04 | +0.84 | +0.20 |
| 43 | **−0.27** | **+0.59** | **−0.85** |
| 44 | +0.90 | +0.73 | +0.17 |
| **mean±std** | +0.56±0.72 | **+0.72±0.13** | −0.16 |

- nohint 3-seed **全部为正（p=0.010）**，base 68.78±0.19 → 69.49±0.06；触发率 26.7%，翻转合计 215/156。
- **诚实修正**：§7.3 基于 s42 单点的"移除 hint 保留 81%、hint 提高翻案精度"说法**在 3-seed 下不成立**——hint 仅在 2/3 seed 上略正（+0.20/+0.17），s43 上显著为负（−0.85），均值反被 nohint 超出 0.16pp。
- **论文正确主张**（审稿攻防口径）：增益主体来自"异构复审 + 门控"架构本身（nohint 3/3 为正是最硬证据）；CRITIC_REASONING_HINT 是**可有可无的不稳定组件**，不得宣称其有效。Method 中保留 hint 描述但 Results 必须报告该消融的负面/混合结果，反而增强可信度（主动报告一个不 work 的设计组件）。

### 7.5 Self-Consistency 双温度 3-seed 终审（M6，2026-10-04 22:24 完成）

回应审稿质疑"SC 负增益可能是 T=0.7 采样随机性过大"与"单 seed 对照不可信"。协议：同 seed 严格配对（同 m0_sft adapter、同 test 2610、按 idx 对齐同 seed 贪心 y_a、k=5、后验破平），温度 {0.7, 0.3} × seed {42,43,44}，共 6 段。聚合脚本 `scripts/aggregate_sc_3seed.py`，结果 `outputs/self_consistency/sc_k5_s{42,43,44}{,_t03}/`。

**3-seed 主表（mean W-F1 / Δ vs 同 seed 贪心；贪心 3-seed 均值 68.78，辩论 69.33，Δ+0.55）**

| 配置 | W-F1 | Δ vs greedy | mean±std | 配对 t / p | 负 seed |
|---|---|---|---|---|---|
| SC@3 T=0.7 | 67.32 | **−1.45** | −1.45±0.57 | t=−4.38, **p=0.048** | 3/3 |
| SC@5 T=0.7 | 67.80 | **−0.98** | −0.98±0.59 | t=−2.88, p=0.103 | 3/3 |
| SC@3 T=0.3 | 68.45 | **−0.33** | −0.33±0.30 | t=−1.86, p=0.204 | 3/3 |
| SC@5 T=0.3 | 68.70 | **−0.08** | −0.08±0.27 | t=−0.53, p=0.650 | 2/3（s43 +0.17 微正） |
| **异构辩论（我们）** | **69.33** | **+0.55** | — | — | 同 3 seed 2 正 |

逐 seed Δ（SC@3/SC@5）：s42 = −2.01/−1.56（T0.7）、−0.65/−0.36（T0.3）；s43 = −1.49/−1.00、−0.28/**+0.17**；s44 = −0.86/−0.38、−0.05/−0.05。

**结论（双温度 × 3-seed 闭环，质疑解除）**：
1. **12 个 seed×配置格中 11 个为负**；SC 最好的配置（T=0.3/k=5，68.70）花费 5× 预算仍低于贪心 0.08pp；辩论以 1.27× 预算 +0.55pp。高温 SC@3 的负效应在 3-seed 下达到配对显著（p=0.048）。
2. 降温确实收窄损伤（T0.7→T0.3：SC@3 −1.45→−0.33、SC@5 −0.98→−0.08），承认其中含采样噪声成分；但降温让采样退化为近似贪心，SC 只是"不再有害"而非"有益"——唯一微正格（s43 T0.3/k=5, +0.17）对应 5× 预算，不具效率意义。
3. 结构性论点在双温度 × 3-seed 下成立：同参数采样共享相关性盲点，短标签任务投票收益不抵采样扰动；对照从 s42 单点升级为 3-seed 配对证据，论文不再需要 "single-seed control" 脚注。

---

## 25. 置信度校准分析（M3，2026-10-04，纯 CPU）

脚本 `scripts/calibration_analysis.py`，结果 `outputs/calibration/`，图 `outputs/figures/fig_calibration.{pdf,png}`。
口径：top-1 置信度二值校准（records 仅存 top-1 p，非完整 7 类分布）；ECE 为 15 等宽桶，Brier 为 binary Brier。

| 角色（3-seed） | acc | mean conf | ECE↓ | Brier↓ | Spearman(p,correct) | conf−acc |
|---|---|---|---|---|---|---|
| Proponent s42/43/44 | 69.5 | 79.1 | **0.097±0.006** | 0.182±0.002 | **0.440±0.006** | +0.096（过自信） |
| Critic s42/43/44 | 67.7 | 74.3 | **0.070±0.003** | 0.182±0.001 | **0.441±0.010** | +0.067（过自信） |

- 两模型均系统性过自信（ECE≈0.07–0.10），**绝对置信度不能直接当正确率用**；但 Spearman≈0.44 的排序相关性跨 seed 稳定，支持"门控只用排序、不用绝对值"的设计。
- dev 拟合 isotonic → test 评估（s42）：proponent ECE 0.0985→**0.025**，critic 0.0685→**0.019**；Brier 同步改善；Spearman 变化 <0.002（isotonic 平台区 ties 的秩处理，无实质排序反转）——校准误差可在 dev 修复，且单调校准不改变路由排序，门控对校准误差鲁棒。
- 与 §4 AURC=0.156（远优于随机 0.307）一致：信号可用但不完美。

## 26. 外部基线定位（M1，MELD test 官方 split，reported W-F1）

数字均为各论文/官方 leaderboard 报告值（as-reported，**非配对复现**，预处理/上下文窗口/种子数不同，仅供水位定位，不做显著性声称）。来源：InstructERC(arXiv:2309.11911v6)、DialogueLLM(arXiv:2310.11374)、OpenCodePapers MELD leaderboard、EmoTrans(LREC-COLING 2024)、HCAN(arXiv:2309.09799)。

| 类别 | 方法 | 年 | MELD W-F1 | 模态 |
|---|---|---|---|---|
| 判别式 | DialogueRNN | 2019 | 57.0 | 文本 |
| 判别式 | DialogueGCN | 2019 | 58.1 | 文本 |
| 判别式 | DialogueCRN | 2020 | 58.4 | 文本 |
| 判别式 | DialogXL | 2021 | 62.4 | 文本 |
| 判别式 | DAG-ERC | 2021 | 63.1 | 文本 |
| 判别式 | COSMIC | 2022 | 64.2–65.2 | 文本 |
| 判别式 | EmotionFlow-large | 2022 | 66.5 | 文本 |
| 判别式 | SACL-LSTM | 2023 | 66.9（单 seed） | 文本 |
| 判别式 | HiDialog | 2023 | 67.0 | 文本 |
| 判别式 | EmoTrans (RoBERTa-large) | 2024 | 68.0 | 文本 |
| 生成式 LLM | InstructERC (LLaMA2-7B) | 2024 | 69.15 | 文本 |
| 生成式 LLM | CKERC | 2024 | 69.27 | 文本 |
| **本文 base（8-seed 均值）** | LLaMA2-7B SFT | 2026 | **68.88±0.41** | 文本 |
| **本文门控复审（8-seed 均值）** | + Qwen2.5-7B verifier | 2026 | **69.37±0.39**（1.27× 成本） | 文本 |
| 多模态（口径不可直接比） | TelME / M2FNet | 23–24 | 66.7–67.4 | 文+音+视 |
| 多模态 | ELR-GNN | 2024 | 69.9 | 文+音+视 |
| 多模态 | BiosERC | 2024 | 69.83 | 文+视觉知识 |
| 多模态 | DialogueLLM-7B | 2024 | **71.90** | 文+视频知识 |

**定位结论（写作口径）**：
1. 不宣称 SOTA：纯文本前沿 69.2–69.3（InstructERC/CKERC），多模态前沿 69.9–71.9。
2. 本文 base 68.88 即与 InstructERC 69.15 同水位（训练目标刻意保持朴素：纯 SFT、无检索/无多任务/无外部知识），门控以 +27% 期望成本进入 69.3+ 区间。
3. 主张是**正交的部署期增量**：门控复审可叠加于任何更强单模型（包括 InstructERC 类 proponent），本文贡献是"何时调用异构复审"的机制与成本-精度刻画，不是更强的单模型。
4. reported 数字非配对，论文表格显式标注 as-reported。
