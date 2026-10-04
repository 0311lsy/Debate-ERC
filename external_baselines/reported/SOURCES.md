# 外部基线数字溯源（MELD test，官方 split，weighted-F1）

> **使用规则**：每个数字对应且仅对应一个出处。论文 Table 2 的每个 as-reported 行
> 必须能追溯到本文件的 source_id。标 ⚠️ 的条目为"转引值"（原论文未在 MELD
> 官方 split 报告或设置不同），审稿前需打开链接二次核对原表。
> 核对日期：2026-10-04。

## 纯文本判别式方法

- **S1. DialogueRNN (2019, AAAI)**
  Majumder, Poria, Hazarika, Mihalcea, Gelbukh, Cambria.
  "DialogueRNN: An Attentive RNN for Emotion Detection in Conversations." AAAI 2019.
  arXiv:1811.00405. 出处：原论文 MELD 结果表，W-F1 ≈ 57.03（文本口径）。
  ⚠️ 后续论文（如 SpikEmo, arXiv:2411.13917 多模态表）转引 58.73，
  DCR(arXiv:2605.04877) 转引 57.95——差异源于上下文窗口/特征口径，
  本表取原论文文本值 57.03，论文中可写 "57.0–58.7 across reported settings"。

- **S2. DialogueGCN (2019, EMNLP)**
  Ghosal, Majumder, Poria, Chhaya, Mihalcea.
  "DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation." EMNLP 2019.
  arXiv:1908.11540.
  ⚠️ 原论文实验在 IEMOCAP/AVEC，未直接在 MELD 官方 split 报告；
  58.10 为后续统一口径论文（EmoTrans/2024 ERC 综述）的转引值。
  投稿前必须核实转引源具体表号。

- **S3. DialogueCRN (2020, IJCAI)**
  Hu, Wei, Huai, Zhang, Chen.
  "Cognitive-Inspired Emotion Reasoning in Conversation." IJCAI 2020.
  出处：原论文 MELD W-F1 58.38（多份 2023–2024 ERC 论文基线表一致转引）。

- **S4. DialogXL (2021, AAAI)**
  Shen, Li, Zhu, Chen.
  "DialogXL: Explaining Long-Term Contexts for Emotion Recognition in Conversation." AAAI 2021.
  出处：原论文 MELD W-F1 62.41。

- **S5. DAG-ERC (2021, COLING)**
  Shen, Zhou, Yang, Yang, Liu.
  "Directed Acyclic Graph Network for Conversational Emotion Recognition." COLING 2021/Findings.
  arXiv:2105.12917（请核对版本）。出处：原论文 MELD W-F1 63.06。

- **S6. COSMIC (2022, Findings of ACL)**
  Ghosal, Majumder, Gelbukh, Mihalcea, Poria.
  "COSMIC: COmmonSense knowledge for eMotion Identification in Conversations." Findings of ACL 2020/2022。
  arXiv:2010.02791（请核对）。出处：RoBERTa-large+ATOMIC 65.24；
  无常识变体 64.23，表内记为区间 64.2–65.2。

- **S7. EmotionFlow (2022, NAACL)**
  Lei, Shen, Feng, Feng, Liu.
  "Emotion Recognition in Dialogue Interactions" / EmotionFlow.
  出处：large 配置 MELD W-F1 66.5（原论文/后续转引一致）。
  ⚠️ 需核对论文准确标题与 NAACL 卷次（疑为 NAACL 2022）。

- **S8. SACL-LSTM (2023)**
  "Symmetric-Affective Conception Learning LSTM"。
  出处：2023 ERC 基线表转引 66.9，原报告单 seed。
  ⚠️ 必须找到原始论文（作者/venue/DOI）后再保留此行，否则投稿前删除。

- **S9. HiDialog (2023, ACL)**
  Zhang et al. "HiDialog: A Compact Hierarchical Model for Emotion Recognition in Conversations."
  出处：原论文 MELD W-F1 67.0。⚠️ 需核对准确标题与作者。

- **S10. EmoTrans (2024, LREC-COLING)**
  "EmoTrans: ... Emotion Transfer ..."，RoBERTa-large。
  出处：LREC-COLING 2024 论文 MELD 表，W-F1 68.0。
  ⚠️ 需补全作者名单与 ACL Anthology 链接。

## 生成式 LLM 方法

- **S11. InstructERC (2024, COLING)**
  Lei, Shen, Sun, Chaturvedi, Liu.
  "InstructERC: Reforming Emotion Recognition in Conversation with a Complementary Retrieval and Multi-task Cognitive Framework."
  arXiv:2309.11911（v6）。代码：https://github.com/bafny/instructERC（请核对仓库）。
  出处：论文 MELD 主表，LLaMA2-7B + LoRA，W-F1 69.15。
  **本研究在本地复现（已发布的 meld-only 管线，66.29）**，见
  `../instructerc_reproduction/`。

- **S12. CKERC (2024)**
  常识知识增强 ERC，7B-class，MELD W-F1 69.27。
  ⚠️ 必须补全：准确论文标题、作者、venue、arXiv DOI、表格编号、开源链接；
  在补全前该行在论文中以 "[ref pending]" 占位，不得直接引用。

- **S18. PRC-Emo (2026, AAAI-26)**
  Xinran Li, Yu Liu, Jiaqi Qiao, Xiujuan Xu（大连理工大学）。
  "Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and
  Curriculum Learning."
  Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40,
  pp. 31778–31786（AAAI OJS article 40446）；arXiv:2511.07061v3（2025-11-24）。
  代码：https://github.com/LiXinran6/PRC-Emo
  出处：本地存档 PDF（含附录版）第 6 页 Table 2：完整 PRC-Emo（Qwen3-8B，
  5-seed 均值）MELD Acc 71.50 / W-F1 70.44；Table 3 消融 w/o C 70.07、
  w/o R+C 69.62、w/o P+R+C 68.72；Table 4 curriculum-only(w/o S+I+R) 69.34。
  **本研究在本地复现（替换基座的简化协议）**，见 `../prc_emo_reproduction/`。

## 多模态 / 外部知识方法（独立分块，不参与纯文本排序）

- **S13. TelMe (2023, ACM MM)** — text+audio+video，MELD 66.7。
  ⚠️ 补作者与 DOI。
- **S14. M2FNet (2023, IJCAI)** — 多模态多粒度融合，MELD 67.4。
  ⚠️ 补作者与 DOI。
- **S15. ELR-GNN (2024)** — 图网络多模态，MELD 69.9。
  ⚠️ 补全 venue 与论文链接后保留。
- **S16. BiosERC (2024)** — 文本 + 视觉常识知识，MELD 69.83。
  ⚠️ 补全准确标题（疑为常识知识/生物信号相关命名）与链接。
- **S17. DialogueLLM-7B (2024)**
  Zhang, Wang, Wu, Tiwari, Li, Wang, Qin.
  "DialogueLLM: Context and Emotion Knowledge-Tuned Large Language Models for ERC."
  arXiv:2310.11374v4（2024-01-17 版本，HTML 全文已核对存在）。
  出处：论文主表 MELD 71.90；使用视频情绪知识构造指令数据（多模态知识增强）。

## 投稿前核查清单（⚠️ 条目清零）

- [ ] S2 DialogueGCN 58.10 的转引源表号
- [ ] S5/S6/S7 arXiv 号与 venue 卷次
- [ ] S8 SACL-LSTM 原始论文
- [ ] S9 HiDialog 准确标题
- [ ] S10 EmoTrans 作者 + ACL Anthology
- [ ] S12 CKERC 完整书目信息
- [ ] S13–S16 多模态四行完整书目信息
- [x] S17 DialogueLLM（arXiv:2310.11374v4 已核对）
- [x] S11 InstructERC（arXiv:2309.11374 已核对）
- [x] S18 PRC-Emo（AAAI OJS 40446 / arXiv:2511.07061v3 已核对）

## 本文自有数字（非 reported，供对照，不属外部基线）

- base plain SFT：8-seed 68.88±0.41（`outputs/selective_hetero/qwen7b*/test_records.jsonl` 聚合）
- gated review：8-seed 69.37±0.39，脚本 `scripts/aggregate_8seeds_uniform.py`
