# External Baselines（外部基线存档）

本目录专门保存外部基线的**报告数据（as-reported）**与**复现代码/结果**，
与主实验代码（`scripts/`、`src/`）分离，确保口径可审计。

## 目录结构

```
external_baselines/
├── README.md                          # 本文件：口径约定
├── reported/                          # 直接引用、不复现的外部报告值
│   ├── meld_discriminative.csv        # 10 个纯文本判别式方法（2019–2024）
│   ├── meld_generative_llm.csv        # 生成式 LLM 方法（InstructERC / CKERC / PRC-Emo）
│   ├── meld_multimodal.csv            # 多模态方法（口径不可直接比，物理隔离）
│   └── SOURCES.md                     # 每个数字的论文/表格/链接溯源
├── instructerc_reproduction/          # 复现①：InstructERC（同基座锚点）
│   ├── README.md                      # 复现协议（为何复现、如何复现）
│   ├── reproduce.sh                   # 训练 + 终审 + 门控叠加实验脚本
│   └── results/                       # 10 epoch 指标 + best epoch(8) 逐句预测转储
└── prc_emo_reproduction/              # 复现②：PRC-Emo（AAAI 2026，替换基座简化协议）
    ├── README.md                      # 协议差异、5-seed 结果与两个技术坑的完整审计
    └── results/                       # seed42 忠实重算指标 + 5-seed 汇总清单
```

## 口径约定（审稿合规红线）

1. **`reported/` 下所有数字均为 as-reported**：来自原论文或公开 leaderboard，
   预处理、上下文窗口、backbone 版本、种子数与本研究不同，**仅供领域水位定位**，
   不与本方法做配对显著性检验，论文中不得出现 "significantly outperforms X"
   的措辞（X 为 reported 行）。
2. **10 个判别式方法不复现**：与本方法范式不同（判别式 vs 生成式），
   且为公认公共基准数字，引原论文即可（顶刊审稿惯例）。
3. **本地复现的外部方法共 2 个，协议差异必须随数字一并报告**：
   - **InstructERC**（`instructerc_reproduction/`）：同基座（LLaMA2-7B）、
     同数据集（MELD）、同 LoRA 规模，是"base 与纯文本前沿同水位"主张的直接
     锚点；已发布 meld-only 管线复现 66.29（reported 69.15 属官方未发布代码的
     unified-mixed 协议，详见该目录 README）；best-epoch 逐句预测转储同时供
     **门控可叠加性实验**（更强 proponent）使用。
   - **PRC-Emo (AAAI 2026)**（`prc_emo_reproduction/`）：第三方独立训练代码的
     交叉印证；本地为 LLaMA2-7B **替换基座 + 简化协议**（无 P/R，仅课程学习），
     5-seed 65.67±5.17（含一个替换基座病理 seed，归因已诊断并脚注交代），
     **不得与 reported 完整 PRC-Emo 70.44 直接对标**。
4. **多模态方法物理隔离**：TelME/M2FNet/ELR-GNN/BiosERC/DialogueLLM 使用
   音频/视频/外部知识，与纯文本设置不可比，论文表格独立分块并显式标注模态。
5. 每个 reported 数字在 `SOURCES.md` 中有且仅有一个明确出处
   （论文 + 表号/页码 + arXiv DOI/链接）。
6. **禁止 cherry-picking**：复现的多 seed 结果必须全量保留（如 PRC-Emo
   seed45=56.48），退化 seed 的归因以证据写明，但均值计算不得剔除。

## 与论文的对应关系

- `reported/*.csv` → 论文 Table 2（External positioning on MELD test）
- `instructerc_reproduction/results/` → Table 2 的 InstructERC "reproduced" 行
  + Section 5.3 的叠加实验（门控 × 强 proponent，预测转储已就位）
- `prc_emo_reproduction/results/` → Table 2 的 PRC-Emo reported 70.44 行与
  本地复现行（脚注交代替换基座/简化协议/病理 seed）
