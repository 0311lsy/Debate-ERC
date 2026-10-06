# InstructERC 复现协议（统一框架下唯一复现的外部基线）

> **决策状态（2026-10-04）：已采纳方案 B**——69.15 保留 as-reported，
> 66.29 作为"官方已发布管线复现值"列入论文 Table 2 并附代码缺口说明；
> 不启动自实现 unified 训练（方案 A 仅作 rebuttal 储备，原料与补丁已备）。
> 论文落点：`paper_eswa/main.tex` Table 2（§脚注）+ Position 段。

## 为什么只复现这一个

论文 Table 2 的定位主张"朴素 SFT base（68.88）与纯文本前沿 InstructERC
（69.15）同水位"建立在 as-reported 数字上。InstructERC 是其中唯一同基座
（LLaMA2-7B）、同数据集（MELD）、同微调规模（LoRA）的直接锚点。

复现同时承担第二个实验目的：把复现模型作为更强 proponent 接入冻结门控，
检验选择性异构复审的可叠加性。

---

## 历史复现（2026-09-28，meld-only 已发布管线；方案 B 下作为"已发布管线复现值"进论文）

| 项 | 内容 |
|---|---|
| 入口 | 官方 `main_new.py`（单数据集 Plain 路径） |
| 协议 | `--dataset meld`，10 epoch，LoRA dim16/alpha16，lr 2e-4，bs16×ga16，maxlen 1024，seed 42 |
| 产物 | `Agent_Reason/InstructERC/results/meld_lora_10epoch/preds_for_eval_{0..9}.text` |
| 结果 | best **W-F1 66.29（epoch 8）**，末轮 66.21；weighted-F1 口径已核实（`f1_score(average='weighted')`），贪心解码（ModelArgs do_sample=False） |
| 归档证据 | `results/meld_lora10_per_epoch_metrics.json`（10 epoch 指标，best epoch8）、`results/meld_lora10_epoch8_predictions.jsonl`（2610 条 output/target 转储，自算 W-F1 66.26 vs 官方自报 66.294，差 0.03 为标签集处理微差）、`results/epoch8_classification_report.txt` |
| vs reported | 66.29 vs 69.15 差 2.86 点——**协议不同所致，非复现失败**（见下方审计） |

**当时误判**：把该差距归因为复现失败（见
`Agent_Reason/InstructERC/docs/failure_analysis_report.md`）。

**2026-10-04 协议审计结论（更正）**：69.15 与 66.29 根本不是同一个实验——

1. 论文主表 69.15 来自 **unified-label Mixed 协议**
   （`train_and_inference_Mixed.sh`）：MELD+IEMOCAP+EmoryNLP 三数据集统一
   标签空间混合训练，**8 epoch**，全量数据，入口为 **`main_Unilabel.py`**。
   官方 README 数据配比表中 data ratio=1（全量）行 MELD=69.15。
2. 历史复现走的是 **单数据集 Plain 协议**
   （`train_and_inference_Uni.sh`/`main_new.py --dataset meld`），6 epoch
   设计；两者训练数据（单 MELD vs 三集统一混合）与入口代码均不同。
3. **官方 GitHub（LIN-SHANG/InstructERC）从未发布 `main_Unilabel.py`**
   （2026-10-04 经 GitHub API 核对 code/ 目录，仅 6 个文件：
   data_process{,_mixed,_plain}.py、main_new.py、3 个 shell）；
   shell 脚本引用了缺失入口。即 69.15 的精确复现路径**官方未开源**。
4. 本地已具备 mixed 数据处理的全部原料：
   `/home/lsy20252770/InstructERC/original_data/{meld,iemocap,EmoryNLP}/*.pkl`
   均存在；`data_process_mixed.py` 可读，但内含原作者硬编码路径需补丁。

## 可选方案（需作者拍板，成本与可信度不同）

| 方案 | 内容 | GPU 成本 | 风险 |
|---|---|---|---|
| A. 自实现 unified 训练 | 依论文+`data_process_mixed.py` 输出格式自行实现 `main_Unilabel` 等价训练循环（统一标签集 7/6/7 类、混合采样、8 epoch LoRA），3 seed | 约 18–30h（3 seed × 8 epoch × ~24k 样本） | 自造实现偏离官方，审稿人可质疑；且仍可能复现不到 69.15 |
| B. 论文如实处理 | Table 2 标注 InstructERC 69.15 as-reported，并在 footnote/复现说明中写明：官方仓库缺 unified 入口；我们用其**已发布**单数据集管线复现为 66.29（10ep）；本研究自训 m0_sft 68.88 才是同基座实际锚点 | 0 | 少一个配对锚点，但完全诚实；多数审稿人接受"官方代码不完整"的说明 |
| C. 联系作者 | 邮件索取 main_Unilabel.py | 等待数天–数周 | 不可控 |
| D. B 先行 + A 视审稿意见补 | 投稿用 B；若 rebuttal 被要求再投入 A | 0 先行 | 推荐 |

在拍板前**不启动**任何 InstructERC 重训练；`reproduce.sh` 仅保留方案 A
的数据准备与训练骨架（未经验证，不得直接跑）。

## 若执行方案 A：协议规格（骨架，待验证）

- 数据：`data_process_mixed.py --mode mixed --historical_window 12
  --data_percent 1.0` 的补丁副本（替换两处 `/mnt/dolphinfs/...` 硬编码：
  pkl 输入目录与 unified_label 输出目录），输出
  train/test/valid.json 到本目录 `data/unified_label/mixed/`。
  注意其统一标签映射：MELD 标签被改写为
  `neutral, powerful, fear, sad, joyful, disgust, mad`
  （powerful=surprise、mad=anger 的一一映射，W-F1 不受名称影响）。
- 训练：自行实现的 unified 入口，LLaMA2-7B + LoRA（沿用官方 dim16 以对齐），
  lr 2e-4，8 epoch，maxlen 1024，save_steps 设大避免中间 checkpoint
  （历史 10ep 曾产生 260GB DeepSpeed 状态文件）。
- 种子：42/43/44；评估：贪心，MELD test 子集（mixed test.json 含三数据集，
  需按 id 前缀筛 MELD 部分）。
- 诚实规则：复现值与 69.15 并列；偏差不追参数；门控叠加（P3）无论正负
  都进正文。

## P3 门控叠加（方案 A/B 均适用，纯 CPU）

任何一个 InstructERC 协议模型产出 test 逐条 (idx, gold, y_a, p_a) 后，
与现有 `outputs/selective_hetero/cross/critic{42,43,44}_all_test.jsonl`
按 idx 离线组合，冻结 τ=0.65/margin=0.05，复用
`scripts/cross_pair_analysis.py` 规则，无需额外 GPU。
注意：历史 meld-only 模型 66.29 弱于自训 m0 68.88，**不能**充当"更强
proponent"角色，故 P3 必须等方案 A 的 unified 模型（若接近 69）才有论证
价值；对 66.29 模型做叠加只具探索意义。

### P3 执行结果（2026-10-05，v2 修正版）

- 脚本：`scripts/instructerc_gate_stacking_v2.py`（greedy decode 取 y_a，
  生成标签 token 平均 logprob 取 p_a；critic = Qwen2.5-7B s42）
- 基线：test W-F1 66.47 / dev 64.51（与官方 66.29 复现口径一致）
- **R1 冻结 margin 规则（τ=0.65, m=0.05）**：+0.11，触发率仅 2.7%
  （生成置信度中位数 0.9999 严重过自信）→ 零调参不可移植
- **R3 margin 规则仅 τ dev 重校准（τ=0.8）**：+0.05，触发 6.1%，
  rescue/harm 11/10 → 跨尺度比较 p_b>p_a+m 在 p_a≈1 时不可满足
- **R2 双阈值 dev 校准（τ_a=0.995, τ_b=0.60）→ test**：+1.33
  （66.47→67.79），触发 23.8%，rescue/harm 70/35（2:1），成本 1.24×
- 结论：机制可移植，门控参数不可移植——阈值编码了 primary 的置信度
  语义，换管线必须在 dev 上重校准；仲裁须用各自模型的阈值而非跨尺度
  margin。论文已按此口径软化可叠加性主张（§5.3 InstructERC 段 + 摘要 +
  贡献条 + Conclusion）。
- 事故记录：v1（teacher-forced 打分，base 32.78）与 v2 第一次运行
  （greedy 全 neutral）均因 `instructerc_extract_adapter.py` 提取 LoRA 时
  键名处理错误导致 adapter 未加载（第一次 strip 了 "base_model.model."
  前缀；第二次保留了 ".default." 段，而 PEFT 加载时
  `_insert_adapter_name_into_state_dict` 会再插入 adapter 名造成
  "default.default" 双写全 miss）。正确格式：保留前缀、strip
  ".default." 段。验证方法：加载后检查 lora_B 权重非零（96/96）。
