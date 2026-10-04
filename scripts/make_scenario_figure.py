#!/usr/bin/env python
"""Fig.1 应用场景图（纯 matplotlib，非架构图）：在线客服对话情绪监测。

叙事：对话流逐句进入 -> 主模型（LLaMA2-7B）给出标签+置信度 ->
高置信直答（成本 1x），低置信触发异构复审（Qwen2.5-7B，成本 2x 仅对该句）->
翻案纠错。底部条带量化：27% 触发 -> 期望成本 1.27x，W-F1 68.6 -> 69.6。

输出 outputs/figures/fig_scenario.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs/figures"

C_MAIN = "#1f6f8b"    # 主模型/门控（ours 主色，与其余论文图一致）
C_CRITIC = "#8e7cc3"  # 异构 critic
C_OK = "#2e7d32"      # 正确/直答
C_WARN = "#e69138"    # 低置信
C_BAD = "#cc4125"     # 误判
C_GRAY = "#555555"
C_BG_USER = "#eef3f7"
C_BG_AGENT = "#f7f2ea"


def box(ax, x, y, w, h, text, fc, ec, fontsize=9, tc="black", weight="normal",
        rounding=0.018, lw=1.2, ha="center"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2 if ha == "center" else x + 0.012, y + h / 2, text,
            ha=ha, va="center", fontsize=fontsize, color=tc, weight=weight,
            zorder=4, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.6, style="-|>", ls="-",
          connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                        color=color, lw=lw, linestyle=ls, zorder=2,
                        connectionstyle=connectionstyle)
    ax.add_patch(a)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.4, 5.8), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ===== 顶部场景标题条 =====
    box(ax, 0.015, 0.905, 0.97, 0.072,
        "Deployment scenario: real-time emotion monitoring of contact-center conversations",
        fc="#f5f5f5", ec="#cccccc", fontsize=10.5, weight="bold", tc="#333333")

    # ===== 左：对话流（三条气泡） =====
    ax.text(0.135, 0.862, "Conversation stream", ha="center", fontsize=9.5,
            weight="bold", color=C_GRAY)
    bubbles = [
        (0.775, "Customer:  \"I waited forty minutes.\"", C_BG_USER),
        (0.645, "Agent:  \"I apologize for the delay.\"", C_BG_AGENT),
        (0.515, "Customer:  \"Yeah, great service.\"", C_BG_USER),
    ]
    for yc, txt, fc in bubbles:
        box(ax, 0.03, yc - 0.05, 0.21, 0.10, txt, fc=fc, ec="#bbbbbb",
            fontsize=8.2, ha="left")

    # ===== 中左：主模型逐句打分 =====
    ax.text(0.36, 0.862, "Primary LLM (LLaMA2-7B)", ha="center", fontsize=9.5,
            weight="bold", color=C_MAIN)
    box(ax, 0.26, 0.44, 0.20, 0.30,
        "per-utterance label + confidence\n\n"
        "anger      p = 0.91\n"
        "sadness  p = 0.86\n"
        "joy           p = 0.52",
        fc="#eaf3f6", ec=C_MAIN, fontsize=8.6)
    # 逐句连接线（气泡右缘 -> 主模型左缘）
    for yc in (0.775, 0.645, 0.515):
        arrow(ax, 0.242, yc, 0.258, 0.62, color="#aaaaaa", lw=0.9,
              style="-", ls=":")

    # ===== 门控 =====
    box(ax, 0.505, 0.615, 0.115, 0.11,
        "confidence\ngate\n$p \\geq \\tau$ ?", fc="#fff7e6", ec=C_WARN,
        fontsize=8.6, weight="bold")
    arrow(ax, 0.46, 0.67, 0.503, 0.67, color=C_MAIN, lw=1.8)

    # ===== 上路：高置信直答 =====
    box(ax, 0.68, 0.70, 0.29, 0.13,
        "73% of utterances: answer directly\n"
        "cost = 1$\\times$ forward pass",
        fc="#edf7ed", ec=C_OK, fontsize=9)
    arrow(ax, 0.62, 0.685, 0.68, 0.755, color=C_OK, lw=1.8,
          connectionstyle="arc3,rad=-0.25")
    ax.text(0.640, 0.735, "yes", fontsize=8, color=C_OK, weight="bold")

    # ===== 下路：低置信触发异构复审 =====
    box(ax, 0.68, 0.38, 0.29, 0.15,
        "27% of utterances: heterogeneous review\n"
        "second LLM (Qwen2.5-7B) re-classifies\n"
        "flip only if  $p_b > p_a + m$",
        fc="#f3effa", ec=C_CRITIC, fontsize=9)
    arrow(ax, 0.62, 0.655, 0.68, 0.475, color=C_WARN, lw=1.8,
          connectionstyle="arc3,rad=0.25")
    ax.text(0.640, 0.545, "no", fontsize=8, color=C_WARN, weight="bold")

    # ===== 底部：案例条带（被纠正的一句） =====
    box(ax, 0.26, 0.16, 0.71, 0.13,
        "escalated case:  \"Yeah, great service.\"  (sarcasm)\n"
        "primary: joy (0.52)   $\\rightarrow$   critic: anger (0.78)   $\\rightarrow$   "
        "verdict: anger  (error corrected)",
        fc="white", ec=C_BAD, fontsize=9)
    # 对话流汇入案例条带
    arrow(ax, 0.135, 0.463, 0.135, 0.225, color=C_GRAY, lw=1.4)
    arrow(ax, 0.135, 0.225, 0.258, 0.225, color=C_GRAY, lw=1.4)

    # ===== 底部：成本-收益汇总 =====
    box(ax, 0.03, 0.03, 0.94, 0.085,
        "expected cost = 1.27$\\times$ forward passes      |      "
        "weighted-F1  68.6 $\\rightarrow$ 69.6  (+1.0, seed 42;  8-seed mean +0.49, $p$=0.019)      |      "
        "zero-touch on 73% of traffic",
        fc="#f5f5f5", ec="#cccccc", fontsize=9.2, weight="bold", tc="#333333")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_scenario.{ext}")
    plt.close(fig)
    print(f"Fig.1 已保存至 {OUT}/fig_scenario.{{png,pdf}}")


if __name__ == "__main__":
    main()
