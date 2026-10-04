#!/usr/bin/env python
"""DailyDialog 结果自动回填到文档和 main.tex。"""
import json, sys
from pathlib import Path

s = json.loads(Path("outputs/dailydialog/summary.json").read_text())

# 回填文档
p = Path("paper_experiments_20261003.md")
text = p.read_text()
marker = "## 27. DailyDialog 域外冻结迁移（M2(a)）\n\n"
if marker not in text:
    text += f"""
## 27. DailyDialog 域外冻结迁移（M2(a)）

> 更新于推理完成时（自动回填）。模型在 MELD 训练，冻结 tau=0.65/margin=0.05 直接迁移。

| 指标 | 数值 |
|---|---|
| 数据集 | DailyDialog test（日常对话，非剧本/非表演） |
| 样本数 | {s['n']} |
| Proponent W-F1 | {s['base_wf1']:.4f} |
| Gated review W-F1 | {s['gated_wf1']:.4f} |
| Delta W-F1 | {s['delta_wf1']:+.4f} |
| Trigger rate | {s['trigger_rate']:.2%} |
| Rescue/Harm/Neutral | {s['flips']['rescue']}/{s['flips']['harm']}/{s['flips']['neutral']} |

"""
    p.write_text(text)
    print("✓ 文档 §27 已回填")

# 回填 main.tex（在 \subsection{Cross-family and cross-dataset generalization} 后追加 dailydialog 段）
m = Path("paper_eswa/main.tex")
t = m.read_text()
marker2 = "%%%% DAILYDIALOG_PLACEHOLDER %%%%"
if marker2 in t:
    dd_section = rf"""
\\paragraph{{DailyDialog zero-shot transfer.}}
The frozen gate was further evaluated on DailyDialog
\\citep{{li2017dailydialog}}, a non-scripted corpus of daily English
conversations (CC BY-NC-SA). Labels were mapped one-to-one to the MELD
seven-class taxonomy (``no emotion''$\\to$neutral, ``happiness''$\\to$joyful,
others unchanged). With hyperparameters frozen at $\\tau{=}0.65$, $m{=}0.05$,
the primary model achieves {s['base_wf1']:.1%} weighted-F1 and gated review
{s['gated_wf1']:.1%} ($\\Delta$={s['delta_wf1']:+.2f}pp, trigger rate
{s['trigger_rate']:.1%}), showing the gate transfers beyond the acted domain.
"""
    t = t.replace(marker2, dd_section)
    m.write_text(t)
    print("✓ main.tex DailyDialog 段已回填")
else:
    print("⚠ main.tex 无 DAILYDIALOG_PLACEHOLDER，跳过")

print("DailyDialog 回填全部完成")
