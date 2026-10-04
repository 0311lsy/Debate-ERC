# External Baselines（外部基线存档）

本目录专门保存外部基线的**报告数据（as-reported）**与**复现代码/结果**，
与主实验代码（`scripts/`、`src/`）分离，确保口径可审计。

## 目录结构

```
external_baselines/
├── README.md                          # 本文件：口径约定
├── reported/                          # 直接引用、不复现的外部报告值
│   ├── meld_discriminative.csv        # 10 个纯文本判别式方法（2019–2024）
│   ├── meld_generative_llm.csv        # 生成式 LLM 方法（InstructERC / CKERC）
│   ├── meld_multimodal.csv            # 多模态方法（口径不可直接比，物理隔离）
│   └── SOURCES.md                     # 每个数字的论文/表格/链接溯源
└── instructerc_reproduction/          # 唯一在统一框架下复现的外部方法
    ├── README.md                      # 复现协议（为何复现、如何复现）
    ├── reproduce.sh                   # 训练 + 终审 + 门控叠加实验脚本
    └── results/                       # 复现产出（多 seed W-F1 与 records）
```

## 口径约定（审稿合规红线）

1. **`reported/` 下所有数字均为 as-reported**：来自原论文或公开 leaderboard，
   预处理、上下文窗口、backbone 版本、种子数与本研究不同，**仅供领域水位定位**，
   不与本方法做配对显著性检验，论文中不得出现 "significantly outperforms X"
   的措辞（X 为 reported 行）。
2. **10 个判别式方法不复现**：与本方法范式不同（判别式 vs 生成式），
   且为公认公共基准数字，引原论文即可（顶刊审稿惯例）。
3. **InstructERC 是唯一复现的外部方法**：同基座（LLaMA2-7B）、同数据集（MELD）、
   同 LoRA 规模，是本研究"base 与纯文本前沿同水位"主张的唯一直接锚点；
   复现后同时承担第二个角色——**作为更强 proponent 验证门控机制的可叠加性**。
4. **多模态方法物理隔离**：TelME/M2FNet/ELR-GNN/BiosERC/DialogueLLM 使用
   音频/视频/外部知识，与纯文本设置不可比，论文表格独立分块并显式标注模态。
5. 每个 reported 数字在 `SOURCES.md` 中有且仅有一个明确出处
   （论文 + 表号/页码 + arXiv DOI/链接）。

## 与论文的对应关系

- `reported/*.csv` → 论文 Table 2（External positioning on MELD test）
- `instructerc_reproduction/results/` → 论文 Table 2 的 "reproduced (ours)" 行
  + Section 5.3 的叠加实验（门控 × 强 proponent）
