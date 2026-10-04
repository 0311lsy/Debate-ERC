# InstructERC 复现协议（统一框架下唯一复现的外部基线）

## 为什么只复现这一个

论文 Table 2 的定位主张"朴素 SFT base（68.88）与纯文本前沿 InstructERC
（69.15）同水位"目前建立在 as-reported 非配对数字上。InstructERC 是其中
**唯一同基座（LLaMA2-7B）、同数据集（MELD 官方 split）、同微调规模（LoRA）**
的直接锚点，审稿人对口径一致性的全部火力都会集中在这一行。

复现同时承担第二个实验目的：**把复现得到的 InstructERC 模型作为更强 proponent
接入本研究的冻结门控**，检验"选择性异构复审"是否为可叠加的部署期增量
（回应 Discussion 中"门控只在朴素 SFT 上验证"的未来工作）。

## 协议（严格对齐官方，禁止为追数字而调参）

- 代码：`/home/lsy20252770/Agent_Reason/InstructERC/code/main_new.py`
  （官方代码，依赖已升级至 torch 2.7+cu128 / transformers 4.45+）
- 数据：`/home/lsy20252770/Agent_Reason/InstructERC/data/processed/meld_window/`
  （官方 `data_process.py --dataset meld --historical_window 12` 产物，
  检索增强历史 + 多任务口径，不重新处理以避免数据口径漂移）
- 基座：`/home/lsy20252770/InstructERC/LLM_bases/LLaMA2`（NousResearch/Llama-2-7b-hf）
- 训练：LoRA，lr 2e-4，bs16×ga16，max_length 1024，**num_train_epochs=1**
  （官方 `configs/run_meld_lora.sh` 参考配置，对应报告值 69.15）
- 种子：42 / 43 / 44（与本研究主实验三 seed 对齐），共 3 次独立训练
- 评估：贪心解码，MELD test，官方统计模式（Acc_SA / F1_SA）

## 三阶段产出

| 阶段 | 内容 | 产出 | 状态 |
|---|---|---|---|
| P1 | 官方协议训练 3 seed + 官方评估 | `results/s{42,43,44}/` 与 W-F1 | 待 GPU（脚本就绪） |
| P2 | 将 InstructERC test 逐条预测/置信度转储为本研究四元组格式 | `results/s*/test_records_instructerc.jsonl`（idx/gold/y_a/p_a） | 待 P1 后写适配器 |
| P3 | 门控叠加：InstructERC 的 y_a/p_a × 现有 Qwen critic 全量转储 y_b/p_b，冻结 τ=0.65/margin=0.05 离线终审 | `results/gate_stacking.json` | 待 P2，纯 CPU |

P3 离线组合方式与主实验 3×3 因子交叉完全一致
（复用 `scripts/cross_pair_analysis.py` 的离线规则），critic 侧直接使用
`outputs/selective_hetero/cross/critic{42,43,44}_all_test.jsonl`，
无需额外 GPU。

## 诚实报告规则

1. P1 复现值与 reported 69.15 **并列展示**；若有偏差（±0.5 以上），
   两个数字都保留并讨论可能原因（数据处理版本/单 seed 运气/解码细节），
   不得反向调参追 69.15。
2. P3 若门控叠加增益为正：主张升级为"pluggable inference-time gain over
   strong ERC backbones"；若为零或负：作为边界诚实写入 Discussion
   （强模型可补空间小，与 s47 天花板观察一致）。
3. 所有 run_id 显式标注 `instructerc_repro_s{seed}`，防止与本研究
   `base_sft_s*` 产物撞名。

## 运行

GPU 空闲后（或追加到当前队列末尾）：

```bash
bash external_baselines/instructerc_reproduction/reproduce.sh
```

预计：3 次训练（每次约 1.5–2h，单 epoch）+ 评估，共约 6h GPU。
P2/P3 在训练完成后手动/自动接续，P3 为纯 CPU。
