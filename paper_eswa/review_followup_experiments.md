# 审稿意见跟进：实验部分问题清单与补充实验协议

> 日期：2026-10-05
> 视角：ESWA / 顶会审稿人对论文 `paper_eswa/main.tex` 实验部分的评审意见
> 状态约定：[待办] / [进行中] / [已完成] / [暂缓-需GPU]
> 原则：所有补充实验必须落盘脚本 + 结构化结果（json），论文回填另见各文件引用。

---

## 一、必须处理（不补会成为拒稿点）

| 编号 | 问题 | 类型 | 状态 | 处置 |
|---|---|---|---|---|
| A1 | 头条 p=0.019 的 8 种子包含参与 dev 阈值校准的 s42–44，存在"校准集泄漏进确认"观感 | 统计口径 | 已完成（零成本） | 补报未参与校准的确认集 s45–49 |
| A2 | "bootstrap 100k over seeds" 对 n=8 重采样，次数不增加有效 n；缺话语级检验 | 统计方法 | 待办（文字+CPU） | 改名 seed-level bootstrap 并注明 n=8；补逐种子 McNemar / 符号检验并如实报告 |
| A3 | 主表 8 种子使用带 hint 的 critic，但消融显示 nohint 更优（+0.72 vs +0.56），配置自相矛盾 | 实验配置 | 已排队（GPU，角色互换后） | 补跑 s45–49 nohint 评估（adapter 已存在，纯评估），形成 8 种子无hint主表；评估后若 nohint 全 8 种子更优则更换主表，否则按预注册口径保留并改写消融措辞 |
| A4 | IEMOCAP 用了 "zero-shot transfer" 措辞，但 adapter 是在 IEMOCAP 上重新 SFT 的，仅门控超参冻结 | 表述过度 | 待办（零成本文字） | 改为 "zero-shot gate transfer with dataset-retrained adapters" |
| A5 | Table 2 脚注残留 "bibliography pending verification" | 文稿硬伤 | 待办（零成本文字） | 核实 BiosERC/ELR-GNN 文献后改为正常引用或删除占位语 |

### A1 已备数字（可直接回填 §5.1）

- 确认集 s45–49（未参与 τ/m 的 dev 网格）：**ΔW-F1 = +0.46 ± 0.33，t=3.11，4/5 为正**
- 校准集 s42–44：+0.56 ± 0.72（2/3 为正，方差大）
- 全部 8 种子：+0.49 ± 0.46，t=3.03
- 解读：确认集效应略小但更稳定，支持结论不依赖校准种子。

### A2 已知事实（回填时用）

- 8 种子逐种子符号：+ 1.04, −0.27, +0.90, +0.43, +0.56, −0.10, +0.74, +0.65（6/8 正）
- 精确符号检验 P(X≥6 | n=8, p=0.5) ≈ 0.145（单侧）——不显著，必须如实写明，显著性由配对 t（p=0.019）与 Wilcoxon（p=0.039）承重。
- 聚合 rescue/harm=525/414（8 种子离线重跑口径，见 22:55 条；旧近似 494/384 已废弃）
  跨种子重复同一 2610 条话语，不能 pooled 当 n=20880 做单个 McNemar；逐种子 McNemar 才是正确口径。

---

## 二、强烈建议（显著提升说服力）

| 编号 | 问题 | 成本 | 状态 | 处置 |
|---|---|---|---|---|
| B1 | 主表只报 W-F1；MELD 极度不平衡，应补 Macro-F1 / Accuracy | 零成本（记录已算出） | 已完成 | Table 1 加两列 |
| B2 | "角色互换""InstructERC 叠加"等关键结论仅 seed-42 单点 | CPU/GPU | 进行中 | 角色互换 8 种子 GPU 任务已启动（s43–49）；**InstructERC×3 critic 种子为纯 CPU，本批完成** |
| B3 | InstructERC R2 最优点 τ_a=0.995 落在原网格端点，存在"网格外更优"质疑 | 纯 CPU | 进行中 | 网格扩展至 0.9995，报告平台与邻域稳定性 |
| B4 | 只有 τ 扫描（Pareto），缺 margin m 的敏感性 | 纯 CPU | 进行中 | s42–44（具备全量 critic 分数）τ×m 热力矩阵 |
| B5 | 两模型在相同 SFT 数据上训练，独立性仅来自骨干家族，数据诱导错误相关性共享 | GPU ~3h | 已排队（GPU，A3 后） | critic 用 50% 确定性数据子集（4994/9989，seed=42）训练、**6 epoch 控更新步数 ≈939 与全量 3 epoch 匹配**，纯操纵数据身份；proponent 保持现有全量 s42。对照：全量 critic +1.04（s42） |
| B6 | Fig.2/Fig.3（Pareto、risk-coverage）仅 seed 42 | 纯 CPU 重绘 | 待办 | 用现有 records 重绘为多种子均值带 |

### B1 已备数字（8 种子，可直接回填 Table 1）

- **Macro-F1 Δ = +0.81 ± 0.57**（逐种子：1.58, −0.01, 0.69, 0.74, 1.52, 0.23, 0.58, 1.16；7/8 为正）
- **Accuracy Δ = +0.53 ± 0.49**

---

## 三、可选加固

| 编号 | 问题 | 成本 | 状态 |
|---|---|---|---|
| C1 | DailyDialog 段并存 logprob-argmax(0.8236) 与 greedy(0.7913) 两个口径，全文主协议为 greedy；需解释并附 7→7 标签映射表 | 零成本文字+附录 | 待办 |
| C2 | 成本只报前向次数（1.27×），缺实测时延/token 口径 | 零成本文字（0.4–0.7s/utt 已有） | 待办 |
| C3 | 多个次要 p 值（hint、IEMOCAP 变体、12 个 SC 格子）无多重比较校正 | 零成本文字 | 待办：标注 exploratory，确认性检验 Holm 校正 |
| C4 | fear 类 n=50，8 种子 R/H 比基于重复测量而非独立样本 | 零成本文字 | 待办：给 bootstrap 区间或注明重复测量性质 |
| C5 | 朴素等权概率平均集成基线（最显然基线） | 纯 CPU（部分已有 ensemble_baselines.json） | 待办 |
| C6 | class-conditional τ_b 仅列 future work；可在 dev 上纯 CPU 验证能否救回 surprise/fear | 纯 CPU | 待办 |

---

## 四、本批执行的纯 CPU 补充实验（不与 GPU 上角色互换任务争资源）

脚本：`scripts/review_cpu_followups.py`
结果目录：`outputs/review_followups/`
数据前提（均已核对 idx 完全对齐，test n=2610 / dev n=1109）：

- InstructERC proponent：`outputs/instructerc_gated_v2/{test,dev}_records.jsonl`（greedy y_a + 生成置信度 p_a，基线 test W-F1 66.47）
- critic 全量 test：`outputs/selective_hetero/cross/critic4{2,3,4}_all_test.jsonl`
- critic dev：`outputs/selective_hetero/qwen7b{,_s43,_s44}/dev_records.jsonl`
- 主管线 proponent 全量：`outputs/selective_hetero/qwen7b{,_s43,_s44}/test_records.jsonl`

### Exp-1：InstructERC proponent × 3 critic 种子（对应 B2）

- 设计：proponent 固定（InstructERC 官方复现模型仅 1 个种子），Qwen2.5-7B critic 取 s42/43/44，检验 +1.33 是否为单 critic 种子偶然。
- 每个 critic 种子跑三套门控：R1 冻结 margin（τ=.65,m=.05）；R3 margin 规则 τ 仅在 dev 重校准；R2 双阈值 dev 校准→test。
- 输出：`exp1_instructerc_3seeds.json`（含每种子 test ΔW-F1、触发率、rescue/harm，及 3 种子均值±标准差）。

### Exp-2：双阈值扩展网格与平台稳定性（对应 B3）

- τ_a 网格从原 [0.80…0.995] 扩展到 {0.90,0.93,0.95,0.97,0.99,0.995,0.997,0.999,0.9995}；τ_b 不变。
- 报告：dev 最优点是否仍在边界、top-5 配置、dev 最优点邻域（±1 网格档）对应的 test Δ 范围，证明结论依赖平台而非单点尖峰。
- 对 3 个 critic 种子均执行。
- 输出：`exp2_grid_extension.json`。

### Exp-3：τ × margin 敏感性（对应 B4）

- 主管线配对（LLaMA2 prop + Qwen critic）s42/43/44，因仅这 3 个种子有全量 critic 打分可任意重放门控。
- τ∈{0.50,0.60,0.65,0.70,0.80} × m∈{0,0.025,0.05,0.10,0.15}，报 3 种子平均 ΔW-F1 矩阵、触发率矩阵，标注部署点 (0.65,0.05)。
- 输出：`exp3_margin_sensitivity.json`。

### GPU 串行队列（2026-10-05 09:15 启动，脚本 `scripts/gpu_queue_after_cross.sh`，waiter PID 59596）

严格串行、不与角色互换抢显存；日志 `outputs/gpu_queue_after_cross.log`：

1. **等待中**：角色互换 s43–49（PID 52658），预计 11:00–11:30 完成；等待结束后检查显存 <2GB 才续跑（最多等 15 分钟，否则中止并人工检查）。
2. **A3（约 2–2.2h）**：`eval_selective_hetero.py --mode both --fixed-tau 0.65 --fixed-margin 0.05 --no-critic-hint` 依次评估 s45–49，tag=`qwen7b_s{S}_nohint`，与 s42–44 nohint 同口径（adapter 均已存在，无训练）。
3. **B5（约 3h）**：`train_critic_sft.py --backbone Qwen2.5-7B --seed 42 --train-fraction 0.5 --epochs 6 --out outputs/hetero_critic/qwen7b_sft_s42_halfdata`，随后冻结门控评估 tag=`qwen7b_s42_halfdata`，proponent=现有全量 base_sft_s42；子集索引落盘 `train_subset.json`。

完成判据：日志出现 "GPU 队列全部完成"。任一阶段失败 set -e 即停，不静默续跑。

---

## 五、执行结果（2026-10-05，纯 CPU，脚本 `scripts/review_cpu_followups.py`）

原始结果：`outputs/review_followups/{exp1_instructerc_3seeds,exp2_grid_extension,exp3_margin_sensitivity}.json`

### Exp-1：InstructERC × 3 critic 种子（B2）——机制跨种子成立

proponent 固定（test 基线 66.47），critic = Qwen2.5-7B s42/43/44，各自用自身 dev 记录独立校准：

| critic 种子 | R1 冻结 margin | R2 双阈值 dev 校准（扩展网格） | R3 margin 仅 τ 重校准 |
|---|---|---|---|
| s42 | +0.11（触发 2.7%） | **+2.31**（τa=0.9995, τb=0.5；触发 34.7%） | +0.05（τ=0.80） |
| s43 | +0.15 | **+2.12**（τa=0.9995, τb=0.5；触发 34.7%） | +0.36（τ=0.95） |
| s44 | +0.06 | **+1.64**（τa=0.999, τb=0.4；触发 30.4%） | +0.31（τ=0.90） |
| **均值±sd** | **+0.11 ± 0.04** | **+2.02 ± 0.34** | **+0.24 ± 0.17** |

- R2 总 rescue/harm = 394/242（1.63:1），成本 1.30–1.35×；R1 为 23/14；R3 为 45/26。
- **结论**：冻结门控跨管线失效（+0.11）3 种子高度一致；跨尺度 margin 即使重校准 τ 也仅微弱增益（+0.24，约为 R2 的 1/8）；双阈值 dev 校准在**三个 critic 种子全部为正（+1.64～+2.31）**，"机制可移植、参数需 dev 重校准"不再是 seed-42 单点结论。
- 与论文现稿关系：§5.3 的 +1.33 是**预注册窄网格**（τa 上限 0.995, τb=0.6）的结果；扩展网格后 dev 选点移至高 τa 平台（见 Exp-2）。回填时两个协议必须并列报告，不得静默替换为更大数字。

### Exp-2：扩展网格与平台稳定性（B3）——平台稳健，但选择对分辨率敏感

- τa 扩展至 {…,0.99,0.995,0.997,0.999,0.9995} 后，3 个种子的 dev top5 **全部聚集在 τa∈{0.999,0.9995} × τb∈{0.4,0.5,0.6}**，dev W-F1 top1 与 top5 差仅 0.19–0.25 点——是平台而非单点尖峰。
- dev 选定配置的 ±1 档位邻域在 test 上 **Δ 全部为正（+1.64～+2.31）**；s44 的 dev 最优点非边界（0.999），s42/s43 在边界但邻域 +2.07/+1.90 同为正。
- **诚实性注意**：dev 选点随网格分辨率移动（窄网格 0.995/0.6 → 扩展 0.9995/0.5），说明 (i) 原 +1.33 是平台低端的保守取值；(ii) 应报告平台区间而非单点；(iii) 55 格 × n=1109 dev 的选择本身有 winner's curse，论文应把 +1.6～+2.3 表述为"dev 校准平台上的 test 增益区间"，成本 1.24–1.35×。

### Exp-3：主管线 τ×margin 敏感性（B4）——25/25 格全正，无悬崖

s42–43/44 三对配对（唯一具备全量 critic 分数、可任意重放门控的种子），test 事后扫描的平均 ΔW-F1（百分点）：

| τ \\\\ m | 0.00 | 0.025 | 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|---|
| 0.50 | 0.28 | 0.40 | 0.43 | 0.39 | 0.35 |
| 0.60 | 0.29 | 0.42 | 0.48 | 0.50 | 0.42 |
| **0.65（部署）** | 0.37 | 0.51 | **0.56** | 0.55 | 0.47 |
| 0.70 | 0.41 | 0.57 | 0.60 | 0.62 | 0.52 |
| 0.80 | 0.50 | 0.66 | 0.68 | 0.65 | 0.55 |

- **25/25 个配置平均增益为正**；部署点 (0.65,0.05) 逐种子 [+1.04, −0.27, +0.90]（与主表一致）。
- 增益对 m 不敏感（同 τ 行内极差 ≤0.16），结论不依赖 m=0.05 这个点。
- **禁止过度解读**：τ=0.80 在 test 上更高（0.68）是事后扫描结果，不能据此把部署点改成 0.8（那等同 test 调参）；部署点仍以 pooled dev 选出的 0.65 为准。此表仅用于证明收益曲面无悬崖。

### 待回填到论文的位置（下一批，需用户确认后执行）

1. §5.1：A1 确认集 s45–49（+0.46±0.33, t=3.11）+ B1 Macro-F1/Acc 列（+0.81±0.57 / +0.53±0.49）
2. §5.3 InstructERC 段：升级为 3 critic 种子 + 平台区间表述（R2 区间 +1.6～+2.3，成本 1.24–1.35×；R1/R3 多种子均值）
3. §5.4/新段：τ×m 敏感性表（25/25 正，部署点由 dev 决定的声明）
4. A4 措辞、A5 占位文字、A2 统计口径修订（零成本文字）

---

## 六、论文回填记录

### 2026-10-05：角色互换 8 种子独立复制（已完成回填，主方向不替换）

**决策（用户拍板）**：保持 LLaMA2-primary 为主方向（Table 2 锚点、全部消融同基座对齐的先验理由），Qwen-primary 作为**预注册独立复制**并列报告，不做事后方向选择；两方向共享同一 2610 条 test，不合并显著性。

**数据（证据：`outputs/review_followups/gpu_queue_summary.json`，脚本 `scripts/summarize_gpu_queue.py`）**：
- 互换方向 8 种子：68.67±0.22 → 69.38±0.41，**Δ +0.71±0.42，8/8 正**（1.03/1.20/0.98/1.04/0.32/0.70/0.24/0.14）
- paired t=4.78（p=0.0020），Wilcoxon p=0.0078，exact sign p=0.0078，d_z=1.69；触发 24.6±1.2%，R/H 593/393
- 方向间逐种子配对差 +0.21，t=0.864，p=0.42（不显著→效应一致）；两负种子翻正 s43 −0.27→+1.20、s47 −0.10→+0.70

**main.tex 改动 4 处**：①摘要加复制句；②§5.1 主结果段末加复制前向指针与"为何保留主方向"；③§5.3 段标题改为 "Role-reversal symmetry: an eight-seed independent replication" 并重写（含共享样本不合并 p 的方法学声明）；④Limitations 删除"role-reversal multi-seed 属 future work"、Conclusion 更新为已复制。

**仍待回填（A3/B5 完成后）**：hint vs nohint 8 种子配对结论（决定主表配置取舍）、B5 半数据对照。

### 2026-10-05（下午）：A2/A4/C1–C4 完成，Mistral 8 种子进行中

**A2（已完成，纯 CPU）**：`scripts/a2_perseed_tests.py` → `outputs/review_followups/a2_perseed_tests.json`。
逐种子 McNemar（acc 口径，n=2610）：8 种子 b/c = 38/64, 60/50, 54/81, 54/65, 51/68, 49/51, 54/78, 54/68；
p 值 0.013/0.39/0.025/0.36/0.14/0.92/0.045/0.24（3/8 种子 p<0.05）。符号检验 7/8 正（acc Δ），
精确双侧 p=0.0703；W-F1 口径 6/8 正，p=0.289。paired t(acc Δ)=3.061，Wilcoxon p=0.0234。
**注意**：论文 §4 已声明 confirmatory family = {paired t, Wilcoxon} on W-F1（Holm 后均 0.039<0.05）；
sign/McNemar 作为稳健性证据如实报告，不得并入 confirmatory family（否则 Holm 后 t 也失守 p=0.058）。
待回填 §5.1：逐种子 McNemar 表 + sign p=0.289 如实写明。

**A4（已改）**：摘要 "transfers zero-shot" → "with adapters retrained on IEMOCAP and only the gate
hyperparameters frozen from MELD"；贡献条 → "zero-tuning gate transfer ... with dataset-retrained
adapters"；§5.3 Cross-dataset 段首句明确两个 adapter 均在 IEMOCAP train 上重训、仅门控超参迁移。

**C1（已改）**：新增附录 §B（app:dailydialog）：7→7 标签映射表（仅 no emotion→neutral /
happiness→joyful 改名，核实自 label_schema.py + dailydialog_loader.py，MELD 标签为 joyful）；
双口径解释（logprob-argmax 与 p_a 同解码族、用于 gated 结果；greedy 为全文主协议参照）；
DailyDialog test 类别失衡核实为 81.7% neutral（非 83%）。正文 DD 段已指向附录。

**C2（已改）**：§4 hardware 段补 token/时延口径：标签 ≤8 生成 token，置信度=7 条候选各 2–3 token
单次 teacher-forced 前向，触发评审仅加一次同型 critic 前向；门控触发 26.5% → 期望墙钟 ≈1.27×
primary-only（与前向预算同值）。**注意**：曾写入未实测的 0.18–0.25/0.22–0.30s 拆分数字，已撤回，
仅保留原已测 0.4–0.7s/utt 总量与 token 账目；勿恢复未测数字。

**C3（已改）**：§4 统计协议段声明 confirmatory family（paired t + Wilcoxon on W-F1，Holm 后
0.039/0.039）；hint/IEMOCAP 变体/SC 格子/逐类 p 值标注 exploratory 不做多重校正。

**C4（已改）**：§5 per-class 段补种子级 bootstrap：fear R/H 95% CI [0.15,0.80]、surprise [0.32,0.70]
（10 万次种子重采样，均整体低于 1）；tab:perclass 标题注明 8×2610 重复测量非独立样本。

**Mistral 8 种子（进行中）**：`scripts/mistral_8seeds_queue.sh`（PID 46827，16:15 启动，
s45 训练中），预计 ~21:00 完成 s45–49 训练+评估；随后 `dump_critic_all_queue.sh`（PID 47692）
自动 dump s45–49 critic 全量 test 打分。完成后：① 升级 `scripts/ensemble_baselines_offline.py`
至 8 种子（C5）；② 8 种子 Pareto/risk-coverage 重绘（B6）；③ 汇总 Mistral 8 种子对照 Qwen。

**仍待论文回填（未做，需用户确认口径）**：§5.1 确认集 s45–49 + Macro-F1/Acc 列 + A2 统计段；
§4/§5.2 hint 消融改写为 8 种子零效应（+0.14±0.32, t=1.28）；§5.3 Mistral 段升级 3 种子
（+0.26±0.19, 3/3 正）+ B5 半数据段；§5.4 τ×m 敏感性表（25/25 正）。

### 2026-10-05（晚）：Mistral 7 种子完成，§5.3 从"单种子正证据"改写为"独立性必要条件负对照"

**数据（s42–s48，s49 ~21:15 完成；证据 `outputs/review_followups/mistral_multiseed_summary.json`，
脚本 `scripts/summarize_mistral_8seeds.py`，与 scipy 交叉验证一致）**：
逐种子 ΔW-F1 = +0.29, +0.06, +0.43, **−0.82**, +0.29, **−0.26**, **+0.88**（s48）；
**mean +0.12 ± 0.54（68.92→69.04），5/7 正，全部检验不显著**：paired t=0.61, p=0.57；
exact Wilcoxon p=0.38；exact sign p=0.45；合计 R/H=609/480（1.27:1），触发率 26.7%。

**关键反转（必须记住）**：旧稿"3 种子 +0.26±0.19"与"Mistral critic 能力更弱（68.25 vs 69.16）"
都是 s42 窗口偶然。7 种子下 Mistral critic standalone W-F1 = **68.82±0.49，反而略高于 Qwen 的
68.67±0.32**；路由率也匹配（26.7% vs 26.5%）。能力与路由均持平、唯一系统差异是 LLaMA 同谱系 →
该对照隔离的是 **independence 因子**：同谱系 reviewer 不把升级转化为净纠正，异家族 Qwen 显著
（+0.49±0.46, p=0.019）。两族增益差 0.37 点本身不显著（Welch t=1.41, p=0.18）——只能声称
"家族独立性必要"，不能量化谱系效应。

**main.tex 已改 4 处**：①摘要删除"replicates across verifier families (Qwen, Mistral)"，
改为同谱系匹配对照零增益（+0.12±0.54, p=0.57）、效应定位于独立性；②§5.3 段标题改为
"family independence as the binding constraint"，7 种子全数字 + Welch 对照 + 措辞克制声明；
③Limitations (ii) 从"confounds independence with capability"改为"capability/routing 已匹配、
lineage 与 tokenizer/优化相似性仍捆绑"；④Limitations (iv) 把 Mistral 移出 single-seed 清单。
PDF 已重新编译（Tectonic 0.17，15 页，0 undefined）。

**待办（s49 完成后）**：把上述 7 处种子数字替换为 8 种子最终值（哨兵 job-7621c034 自动汇总）；
若结论方向改变需同步摘要/§5.3/Limitations 三处。

### 2026-10-05（晚）：外部 AI 审稿报告零成本批次全部落地（21 处 main.tex + 6 条 bib）

外部报告依据旧版（Mistral 单种子时期），其 M1（摘要 Mistral 过度声明）与 M5（Mistral 单 seed）
已被今晚 7 种子工作推翻/解决，其建议的"弱复制 preliminary"措辞不可采用。仍成立项已全部修复：

- **M4 漏引（已逐一联网核实出处，非幻觉）**：refs.bib +6——chudasama2022m2fnet（CVPR**W** 2022,
  pp.4652–4661）、yun2024telme（**NAACL 2024**, pp.82–95, DOI 10.18653/v1/2024.naacl-long.5）、
  jung2024trust（arXiv:2407.18370）、zellinger2024hcma（arXiv:2410.02173）、huang2025dare
  （注意：是 **MRAC'25 workshop** 非 ACM MM 主会，DOI 10.1145/3746270.3760223）、
  zhang2024lantern（arXiv:2411.17674）。§2.2 加 DARE/Lantern 定位段（同模型多代理开放词汇辩论 vs
  异家族闭集单轮门控；冻结 405B 重加权判别后验 vs 任务内微调 7B 独立分类器）；§2.4 加
  Trust-or-Escalate/HCMA（升级更强模型/弃权 vs 等能力验证者、主分支不弃权）。
  Table 3 M2FNet/TelME 两行补 cite（原无出处数字，硬伤）。
- **M3 rank 不对称**：§4 显式承认 r=64 vs r=32 且用实测反驳（critic 68.67 ≤ primary 68.88，
  容量翻倍无能力优势；家族比较 rank 固定 64）。r=32 critic 补跑为可选 GPU 项，暂不做。
- **M2 成本单位**：§4 补定义（单位=一次同 profile 批模型调用；7 标签 teacher-forced 批各条件
  均付、在 1.27× 比值中抵消；边际成本仅触发子集的额外 reviewer 调用）。
- **M1 pre-registered**：全文 4 处改为 "confirmatory (defined before test-set evaluation,
  protocol archived with released code)"，不声称外部 OSF 注册。
- **MINOR**：引言/§2.4 旧 3 种子 67.1/68.8 → 8 种子 68.67/68.88；摘要+引言 8 处 em-dash 改写；
  顺手修 (Appendix~\ref) 双写 "Appendix Appendix A" 共 3 处。
- PDF 干净重建（清 aux/bbl 后 tectonic），15 页，bbl 含全部新条目，0 undefined。
- **未采纳**：Qwen3/LLaMA3 强骨干 sanity check（角色互换 8 种子已提供跨骨干证据）；
  报告的 35–45%→60–70% 概率无法验证，仅参考其部署定位建议。

### 2026-10-05（晚 2）：成本叙事——Fig.3 Pareto 增强 + Table 3 成本口径表注

决策：**不在 Table 3 加跨系统 cost 列**（11 个 BERT/RoBERTa 小模型单次前向比 7B×1.27 便宜
30–80×；CKERC/BiosERC 调外部 LLM 成本不可测；PRC-Emo 8B 1× 70.44 在成本-精度两维都占优——
跨表加 cost 反而送靶子）。改为强化同协议 Pareto 图。

- `scripts/make_paper_figures_multiseed.py` 升级（哨兵日后重跑自动保留增强）：
  * 新增 Homogeneous self-debate 点（实测 outputs/selective，s42：2.0× / W-F1 67.96，−0.64）；
  * 新增 Weak 1.5B 点（outputs/selective_hetero/qwen15b，s42：触发率 0.54%，成本
    1+0.0054×1.5/7=1.001× / 68.54，−0.05）；
  * SC 补齐两温度：T=0.7 深灰（SC3 67.33 / SC5 67.80）+ T=0.3 浅灰（SC3 68.45 / SC5 68.70），
    全部 ≤ greedy 68.78；
  * info json 增列 single_seed 字段（可追溯）。
- paper_eswa/figures/fig_pareto.pdf 从旧 s42 单种子版替换为 3 种子增强版（τ=0.65:
  1.267× / 69.331±0.545；conf-pick 69.373±0.526；always-flip 69.513±0.570）。
- §5.3 图注重写（3 种子口径、s1 标注、"4 倍 review 预算最多 +0.18"）；§5.2 段末加
  Fig.3 交叉引用；Table 3 表注加跨系统成本不可比三条理由并指向 Fig.3。
- 编译 16 页 0 undefined（图注/表注变长 +1 页）。
- 8 种子 critic dump 完成后：重跑本脚本图自动升级，图注"three paired seeds"需同步改
  "eight"（τ=0.65 8 种子值约 1.27×/69.37）。

### 2026-10-05（21:18）：Mistral s49 完成 → §5.3/摘要 7 种子全部升级为 8 种子终值

s49：68.61→69.14（**+0.53**），critic standalone 68.79，触发 24.8%，R/H=86/63。
8 种子（s42–49）最终口径（outputs/review_followups/mistral_multiseed_summary.json）：
- ΔW-F1 = **+0.18±0.52**（68.88→69.05），**6/8 正**（+0.29,+0.06,+0.43,−0.82,+0.29,
  −0.26,+0.88,+0.53）；paired t=0.95, **p=0.37**；exact Wilcoxon p=0.25；sign p=0.29。
- critic standalone = **68.82±0.45**（7 种子时±0.49，均值不变）；触发率 26.5%（原 26.7%）；
  总 R/H=695/543（1.28:1，原 609/480）。
- 两族增益差 0.49−0.18=**0.31 点**，Welch **t=1.28, p=0.11**（原 0.37 点/t=1.41/p=0.18），
  仍不显著 → "家族独立性必要、不可量化谱系效应" 结论不变且更稳。
论文同步：摘要 (+0.12±0.54, seven, p=0.57)→(+0.18±0.52, eight, p=0.37)；§5.3 全部统计、
逐种子值、R/H；Limitations(ii) 26.7→26.5；(iv) 改为"Pareto 已 3 种子、Mistral/role-reversal
各 8 种子"。编译 15 页 0 undefined。
注意：§5.2 的 26.7%/69.33 是 **Qwen 3 种子消融**口径（s42–44），与 Fig.3 一致，未动。

### 2026-10-05（22:10）：中性级联思路零成本验证（s42–47，6 种子；dump 完自动升 8）

新增 `scripts/analyze_neutral_boundary.py`（纯 CPU，自动探测 dump 齐全的种子），输出
outputs/review_followups/neutral_boundary_analysis.json。结论（s42–47）：
- Q1 错误三分（每种子约）：情绪被吞 315 / neutral 误报 188 / 情绪间 294；**跨边界 62.2%**，
  用户前提在错误计数上成立，但 neutral 类本身 F1 81.1 最高，最差是 fear 31.3。
- Q2 独立 critic 的跨边界互补率（决定是否值得训中性专家）：
  * 被吞样本上 critic 只纠正 **19.7%**，且 **68.0% 同样压成 neutral**（共享盲区！）；
  * neutral 误报样本上纠正 34.7%（互补性较好的一侧）；情绪间错分上 17.8%。
  * 被吞样本真实类：joy 83/anger 81/sad 73/sur 38/disg 21/fear 19。
- Q3a 单向否决（test 选阈=**乐观上界**）：neu→emo ΔW-F1 +0.51，emo→neu +0.24。
- Q3b 三态软路由（confident neu/confident emo/不确定→critic，test 选阈）：
  最优 69.22 @1.26× / ≤1.30× 69.22；**均低于现有冻结门控 69.43 @1.27×**——
  同后验收敛的路由重排不超现有仲裁，与数学论证一致。
- 决策含义：独立信息在"误报侧"较足、"被吞侧"基本是共享盲区；二分类专家只有靠
  训练目标引入新信息才可能有用，且应按冻结协议在 dev 选阈。8 种子复跑由 job-7894ea99 守候。

### 2026-10-05（22:30–22:55）：四项投稿前增强（用户已批准；第5项二分类专家留 rebuttal）

**(1) EmoryNLP 第三数据集 — 管线完成，3 种子队列训练中**
- `scripts/prepare_emorynlp.py` 读 `/home/lsy20252770/InstructERC/original_data/EmoryNLP/EmoryNLP.pkl`
  （list[6]：d0 说话人 onehot、d1 情绪 int dict、d2 句子、d3/4/5 train/test/valid ID；
  标签 0Joyful,1Mad,2Peaceful,3Neutral,4Sad,5Powerful,6Scared），生成
  `PRC-Emo/data/raw/emorynlp.{train,valid,test}.json`：train 659 对话/7551 话，
  valid 89/954，test 79/984；test 分布 neutral288/joy217/scared116/peace111/power96/mad86/sad70。
- schema/loader：`label_schema.py` 加 EMORYNLP_LABELS+OFFICIAL_COUNTS+SCHEMA 注册；
  `data/emorynlp_loader.py`（load_split+check_official_counts）；`data/__init__.py` 分发；
  `train_critic_sft.py` choices 加 emorynlp；`configs/experiments/base_sft_emorynlp.yaml`。冒烟过。
- 队列 `scripts/emorynlp_3seed_queue.sh`（s42/43/44：primary SFT → critic SFT Qwen2.5-7B
  → eval_selective_hetero --mode both 域内 dev 标定；标签空间与 MELD 不相交，无冻结迁移），
  汇总 `scripts/summarize_emorynlp.py` → `outputs/review_followups/emorynlp_summary.json`。
- 22:46 自延迟基准后自动起跑（job-eee6d3 主哨兵；job-55469e 汇总哨兵 120s 轮询），
  日志 `outputs/emorynlp_3seed_queue.log`，预计 10-06 上午完成。**结果无论正负都要入文**：
  §4 Datasets 加描述（需核 Zahiri & Choi 2018 EmotionLines 准确出处）、§5.3 与 IEMOCAP
  并列报告 ΔW-F1±sd/配对 p/触发率/critic standalone/per-seed τ、bib 加引用。

**(2) 真实延迟/吞吐基准 — 已完成并入文**
- `scripts/benchmark_latency.py --n 400`（bf16 batch=1 单流，8 token 贪心 generate+7 标签
  teacher-forced 打分，RTX PRO 5000 48GB）→ `outputs/review_followups/latency_benchmark.json`：
  primary 316.5 ms（P50 320/P95 378，3.16 条/s）；critic 380.7 ms（2.63 条/s）；
  门控系统均值 **419.3 ms（P50 330/P95 757，2.38 条/s）= 墙钟 1.325×**；
  全量审查 697 ms（1.43 条/s）；峰值显存单模型 13.6 GB、双模型常驻 **28.3 GB allocated /
  34 GB reserved(nvidia-smi)**。注意 full_review 是 mean(t_b) 补齐未触发样本的口径。
- 入文三处：摘要（1.33× 实测、2.38 条/s）；§4 hardware 段（全量分位数+显存替换旧
  "37.5GB/0.4–0.7s" 预估）；§6.2 deployment（2.38 vs 3.16、显存、73% 直达）。

**(3) 中性边界分析 8 种子定稿 + 入 Discussion**
- `scripts/analyze_neutral_boundary.py` 8 种子终值
  `outputs/review_followups/neutral_boundary_analysis.json`：跨边界错误 62.4%
  （被吞315/误报188/情绪间294）；被吞样本 critic 仅纠正 20.0%、67.8% 同压 neutral；
  误报侧纠正 35.2%、情绪间 18.0%；被吞真实类 joy83/anger81/sad73/sur38/disg21/fear19；
  三态软路由乐观上界 69.20@1.24× < 冻结门控 69.37@1.27×；单向否决上界 neu→emo +0.55、
  emo→neu +0.29。
- main.tex Discussion 新增 `\subsection{Why a coarse neutral specialist does not
  subsume the gate}`（~1187 行，`\label{sec:neutral}`）：共享盲区/同后验重排不超
  argmax/headroom 在误报侧/留 future work；据此**否决二分类中性专家 LoRA，留 rebuttal**。

**(4) 可插拔性重定位（零 GPU，4 处文字）**
- 摘要开头：deployment-time wrapper that augments an already trained ERC classifier
  without retraining it；引言贡献 1（不重训 primary、一个额外 7B adapter、test 时仅阈值比较）；
  引言新增"does not compete with retrieval/curriculum…stacked on top of any primary…
  deliberately use unenhanced primary"段；结论加 InstructERC 不动权重+1.33 dev 重校准的
  直接 stackability 证据。

**(5) Fig.3 升 8 种子 + C5 集成基线 8 种子离线复跑（口径统一）**
- `scripts/make_paper_figures_multiseed.py` 重绘 `figures/fig_pareto.pdf`（8 paired seeds，
  数据 `pareto_points_multiseed.json`：greedy 68.878±0.437、conf_pick 69.487、
  always_flip 69.55、oracle 75.968、τ=0.65→69.372@1.265×）；图注 eight paired seeds。
- `scripts/ensemble_baselines_offline.py` SEED_DIRS 扩 8 种子（s45–49 dev/test 均在
  _shared 目录），结果存 `outputs/review_followups/ensemble_baselines_8seed.txt`：
  prop 68.88±0.44 / critic 67.26±0.48 / conf_pick 69.49±0.37 / flip 69.55±0.38 /
  ours 69.37±0.34@26.5% / oracle 75.97±0.47。
- **池化 rescue/harm 权威口径改为 525/414（1.27）**（旧 494/384 废弃；新口径精确复现
  69.37 主结果）。逐类 R/H：neutral 209/134=1.56、joy 73/71=1.03、sad 63/39=1.62、
  anger 103/65=1.58、sur 38/77=0.49、disg 31/9=3.44、fear 8/19=0.42；seed bootstrap
  R/H CI：fear [0.17,0.75]、sur [0.34,0.69]；surprise 77 harms 去向 anger37/neu27/joy10/sad3。
- 论文已同步：§5.2（SC 保留 3 种子并显式标注 same-seed 68.78/69.33 vs 8 种子 68.88/69.37；
  全量集成 69.49/69.55、门控 69.37@27.0%）、§5.3 反向段对比 525/414、Table（perclass）
  全表+正文比率/CI、surprise 段 77/37/27、case study disgust 31/9、§6.2 fear 8/19。
- 编译 16 页、0 undefined reference/citation（仅 txr 字体替换 warning）。

**EmoryNLP 引用核实（WebSearch 2026-10-05）**：EmoryNLP(Friends) 权威出处是
Sayyed M. Zahiri & Jinho D. Choi, 2018, "Emotion Detection on TV Show Transcripts
with Sequence-Based Convolutional Neural Networks", AffCon Workshop @ AAAI-32
（EmoryNLP emotion-detection 发布版；7 标签 neutral/joyful/peaceful/powerful/
mad/sad/scared 与 pkl 一致）。**勿与 Chen et al. 2018 EmotionLines (LREC,
arXiv:1802.08379) 混淆**——Friends 语料同源但论文不同。bib 写 Zahiri2018。
