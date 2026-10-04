# PRC-Emo 复现存档（Li et al., AAAI 2026）

> 论文：*Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval,
> and Curriculum Learning*（Li, Liu, Qiao, Xu，大连理工大学，AAAI 2026）
> 官方代码：https://github.com/LiXinran6/PRC-Emo
> 本地原始工程：`/home/lsy20252770/Agent_Reason/PRC-Emo/`（含官方代码、数据、
> 训练日志、5 个 seed 的 LoRA adapter，各约 2.2 GB，不复制入本仓库）

## 1. 论文报告值（as-reported，Table 2/3/4，MELD test，5-seed 均值，Qwen3-8B）

| 配置 | MELD Acc | MELD W-F1 |
|---|---|---|
| **PRC-Emo 完整（P+R+C）** | 71.50 | **70.44** |
| w/o C（去课程） | — | 70.07 |
| w/o R+C（仅 P） | — | 69.62 |
| w/o P+R+C（纯 LoRA 微调） | — | 68.72 |
| w/o S+I+R（curriculum-only，Table 4） | — | 69.34 |

出处：本地存档 PDF `PRC-Emo/docs/PRC_Emo_withAppendix.pdf` 第 6 页。
论文同期报告 InstructERC 69.15、BiosERC 69.83（口径与本文 SOURCES 一致）。

## 2. 本地复现协议（与论文的差异必须显式标注）

| 维度 | 论文 | 本地复现 |
|---|---|---|
| 基座 | Qwen3-8B（MELD） | **LLaMA2-7B 替换**（刻意对齐 InstructERC 锚点） |
| Prompt 模块（P） | 显/隐性情绪描述 + 说话人描述 | **default 简模板，未启用 P** |
| 检索（R） | ERC 专用检索库 top-k | **kshot=0，无检索** |
| 课程（C） | 2 buckets | 相同（curriculum=True, BN=2, CPE=1） |
| 微调 | LoRA r=32, lr 3e-4 linear, 4 epoch, w=5, L=2048 | 相同 |
| seed | 42–46 五次平均 | 42–46 全部训练（2026-09-29） |

**对应论文配置**：本地 ≈ Table 4 "w/o S+I+R"（curriculum-only，论文 69.34），
**不是完整 PRC-Emo（70.44）**。未复现 P/R 两模块：P 依赖 Qwen3-14B 生成情绪
描述的前置管线，R 依赖 HF 检索库构建，均未接入。

## 3. 复现结果（MELD test，n=2610）

| seed | W-F1 | Macro-F1 | Acc | 口径 |
|---|---|---|---|---|
| 42 | **68.56** | 53.08 | 70.04 | 离线忠实重算（dev 68.62） |
| 43 | 67.08 | —* | 68.93 | 修复后解析器在线评估 |
| 44 | 68.22 | —* | 69.35 | 修复后解析器在线评估 |
| 45 | **56.48** | —* | 53.41 | 同上，但推理退化（见 §4） |
| 46 | 68.01 | 52.46 | 69.89 | 修复后解析器在线评估 |
| **5-seed 均值** | **65.67 ± 5.17** | | | 全部保留（含退化 seed） |
| 排除 seed45 | 67.97 ± 0.63 | | | 仅作敏感性参考，不作主张 |

\* seed43/44/45 的官方 macro-F1 因预测串中残留无标签噪声类（如
`angervousubble`，support=0）被 sklearn 计入类别集而严重失真（0.02–0.11）；
**weighted-F1 只按 support 加权，噪声类权重为 0，不受影响，仍为忠实值**。

## 4. 两个必须交代的技术坑（均有诊断证据，见原始工程 docs/）

1. **`<|im_end|>` 冻结病理**（影响全部 seed）：LLaMA2 无原生 chat template，
   `setup_chat_format` 新增的 `<|im_start|>/<|im_end|>` 两个 token embedding 与
   lm_head 行为随机初始化且冻结（LoRA all-linear 不含 tied lm_head），模型
   结构上无法学会输出 `<|im_end|>`。作者的 Qwen 基座自带 template 不触发此问题。
   **修复**：`post_process` 增加按首个换行截断（贪心解码标签即首行），已训模型
   无需重训；seed42 的 0.0 即此坑所致，离线在原始预测串上重算得 68.56
   （与修复后在线评估逐字节等价）。
2. **seed45 退化**（56.48）：训练 loss 正常收敛（≈0.31），但该 seed 对冻结
   特殊 token 的随机初始化更差，推理时大量输出"正确标签前缀 + 无换行乱码"
   （如 `neutralizzazione of Phoebe...`），换行截断无法恢复，计为模型错误。
   这是**替换基座引入的种子敏感病理**，论文 Qwen 配置下不存在；论文引用时
   必须完整报告该 seed（不允许 cherry-pick 删除），并在脚注说明归因。

## 5. 文件清单与原始产物路径

```
prc_emo_reproduction/
├── README.md                         # 本文件
└── results/
    ├── seed42_corrected_metrics.json # seed42 离线忠实重算（dev/test 双集）
    └── five_seed_metrics.json        # 5 seed 全部指标 + 协议元数据 + 统计量
```

原始产物（在 PRC-Emo 工程内，不复制）：

- 训练日志：`PRC-Emo/results/logs/meld_LLaMA2_default_seed{42..46}.log`
  （含每个 seed 的完整 sklearn 分类报告与 `[FINAL TEST]` 行）
- LoRA adapter：`PRC-Emo/results/meld_LLaMA2_*_seed{s}_L2048_default/
  ..._final_full_finetune/`（各约 2.2 GB；如需逐句预测重算可从 adapter 重跑
  评估，约 34 min/seed）
- 诊断报告：`PRC-Emo/docs/failure_analysis_report.md`（§3.4 冻结 token 根因）、
  `PRC-Emo/docs/experiment_log.md`

## 6. 在本研究论文中的使用口径

- **70.44（reported）** 进 Table 2 纯文本生成式区块，标 as-reported，
  使纯文本前沿水位从 69.1–69.3 上移至 70.4；本研究不做跨实现显著性声称。
- 本地复现 65.67（5 seed，含病理 seed）或 67.97（排除）**不与 70.44 直接
  对标**：基座、P/R 模块均不同。其证据价值是：在第三方独立训练代码上再次
  印证"7B 级朴素/弱增强微调 MELD W-F1 落在 66–68"区间，与本文 primary SFT
  68.88 同水位。Table 2 复现行以脚注交代替换基座与 seed45 病理。
