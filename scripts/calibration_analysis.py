"""M3 置信度校准分析（纯 CPU，不碰 GPU）。

口径：top-1 置信度的二值校准（selective-prediction 标准口径）
  - 输入仅含逐条 top-1 p 与 0/1 正确性（records 未存完整 7 类分布）
  - ECE（equal-width 15 桶 / equal-mass 10 桶）、binary Brier、NLL
  - Spearman(p, correctness)：门控只依赖排序的直接证据
  - isotonic 单调校准（仅在 dev 拟合，test 只评一次）：证明 ECE 可修复
    且单调变换不改变门控排序（门控决策不变）
覆盖：proponent 与 critic 各 3 seed（s42/43/44）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/calibration"
FIG = ROOT / "outputs/figures"
LABELS = ["neutral", "joyful", "sadness", "anger", "surprise", "disgust", "fear"]


def load_pairs(path: Path, p_key: str, y_key: str):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-").splitlines()]
    p, c = [], []
    for r in rows:
        if r.get(p_key) is None or r.get(y_key) is None:
            continue
        p.append(float(r[p_key]))
        c.append(1.0 if r[y_key] == r["gold"] else 0.0)
    return np.array(p), np.array(c)


def ece_equal_width(p, c, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    e, gaps, n_b = 0.0, [], []
    for l, r_ in zip(edges[:-1], edges[1:]):
        m = (p >= l) & (p < r_ if r_ < 1 else p <= r_)
        n = int(m.sum())
        n_b.append(n)
        if n:
            gap = abs(c[m].mean() - p[m].mean())
            gaps.append(gap)
            e += n / len(p) * gap
        else:
            gaps.append(0.0)
    return e, edges, gaps, n_b


def ece_equal_mass(p, c, n_bins=10):
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = 0.0, 1.0
    e = 0.0
    for l, r_ in zip(qs[:-1], qs[1:]):
        m = (p >= l) & (p <= r_)
        n = int(m.sum())
        if n:
            e += n / len(p) * abs(c[m].mean() - p[m].mean())
    return e


def all_metrics(p, c):
    rho, _ = spearmanr(p, c)
    return {
        "n": int(len(p)),
        "acc": float(c.mean()),
        "mean_conf": float(p.mean()),
        "conf_gap": float(p.mean() - c.mean()),  # >0 过自信
        "ece_w15": float(ece_equal_width(p, c)[0]),
        "ece_mass10": float(ece_equal_mass(p, c)),
        "brier_binary": float(np.mean((p - c) ** 2)),
        "nll_binary": float(-np.mean(c * np.log(np.clip(p, 1e-12, 1)) +
                                     (1 - c) * np.log(np.clip(1 - p, 1e-12, 1)))),
        "spearman_p_correct": float(rho),
    }


def isotonic_dev_test(p_dev, c_dev, p_test, c_test):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_dev, c_dev)
    p_cal = iso.predict(p_test)
    # 单调变换保序：校准前后 Spearman 应完全一致（门控决策不变的证据）
    rho_before, _ = spearmanr(p_test, c_test)
    rho_after, _ = spearmanr(p_cal, c_test)
    return {
        "ece_w15_raw": float(ece_equal_width(p_test, c_test)[0]),
        "ece_w15_isotonic": float(ece_equal_width(p_cal, c_test)[0]),
        "brier_raw": float(np.mean((p_test - c_test) ** 2)),
        "brier_isotonic": float(np.mean((p_cal - c_test) ** 2)),
        "spearman_raw": float(rho_before),
        "spearman_isotonic": float(rho_after),
        "monotonic_preserves_ranking": bool(np.isclose(rho_before, rho_after)),
    }


def reliability_panel(ax, p, c, title, n_bins=15):
    e, edges, gaps, n_b = ece_equal_width(p, c, n_bins)
    centers, accs, confs = [], [], []
    for l, r_ in zip(edges[:-1], edges[1:]):
        m = (p >= l) & (p < r_ if r_ < 1 else p <= r_)
        if m.sum():
            centers.append((l + r_) / 2)
            accs.append(c[m].mean())
            confs.append(p[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    ax.bar(centers, accs, width=1 / n_bins * 0.9, alpha=0.55,
           edgecolor="C0", label=f"empirical accuracy (ECE={e:.3f})")
    ax.plot(confs, accs, "o-", color="C3", ms=4, lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("top-1 confidence")
    ax.set_ylabel("accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")


def main():
    summary = {"proponent": {}, "critic": {}, "isotonic_dev_fit": {}}

    # ---- proponent 3 seeds（test p_a）----
    prop_paths = {
        42: (ROOT / "outputs/selective_hetero/qwen7b/test_records.jsonl", "p_a", "y_a"),
        43: (ROOT / "outputs/selective_hetero/qwen7b_s43_shared/test_records.jsonl", "p_a", "y_a"),
        44: (ROOT / "outputs/selective_hetero/qwen7b_s44_shared/test_records.jsonl", "p_a", "y_a"),
    }
    for s, (path, pk, yk) in prop_paths.items():
        p, c = load_pairs(path, pk, yk)
        summary["proponent"][f"s{s}"] = all_metrics(p, c)

    # ---- critic 3 seeds（全量 test p_b）----
    crit_paths = {
        42: (ROOT / "outputs/selective_hetero/cross/critic42_all_test.jsonl", "p_b", "y_b"),
        43: (ROOT / "outputs/selective_hetero/cross/critic43_all_test.jsonl", "p_b", "y_b"),
        44: (ROOT / "outputs/selective_hetero/cross/critic44_all_test.jsonl", "p_b", "y_b"),
    }
    for s, (path, pk, yk) in crit_paths.items():
        p, c = load_pairs(path, pk, yk)
        summary["critic"][f"s{s}"] = all_metrics(p, c)

    # ---- 3-seed 汇总 ----
    for role in ("proponent", "critic"):
        eces = [summary[role][f"s{s}"]["ece_w15"] for s in (42, 43, 44)]
        briers = [summary[role][f"s{s}"]["brier_binary"] for s in (42, 43, 44)]
        rhos = [summary[role][f"s{s}"]["spearman_p_correct"] for s in (42, 43, 44)]
        gaps = [summary[role][f"s{s}"]["conf_gap"] for s in (42, 43, 44)]
        summary[role]["mean±std"] = {
            "ece_w15": f"{np.mean(eces):.4f}±{np.std(eces):.4f}",
            "brier_binary": f"{np.mean(briers):.4f}±{np.std(briers):.4f}",
            "spearman": f"{np.mean(rhos):.4f}±{np.std(rhos):.4f}",
            "conf_gap_overconfident": f"{np.mean(gaps):+.4f}±{np.std(gaps):.4f}",
        }

    # ---- isotonic：仅 s42 有 dev 全量双模型 p；dev 拟合、test 评估 ----
    dev_p_a, dev_c_a = load_pairs(ROOT / "outputs/selective_hetero/qwen7b/dev_records.jsonl", "p_a", "y_a")
    dev_p_b, dev_c_b = load_pairs(ROOT / "outputs/selective_hetero/qwen7b/dev_records.jsonl", "p_b", "y_b")
    test_p_a, test_c_a = load_pairs(prop_paths[42][0], "p_a", "y_a")
    test_p_b, test_c_b = load_pairs(crit_paths[42][0], "p_b", "y_b")
    summary["isotonic_dev_fit"]["proponent_s42"] = isotonic_dev_test(dev_p_a, dev_c_a, test_p_a, test_c_a)
    summary["isotonic_dev_fit"]["critic_s42"] = isotonic_dev_test(dev_p_b, dev_c_b, test_p_b, test_c_b)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 可靠性图：s42 两面板 ----
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    reliability_panel(axes[0], test_p_a, test_c_a, "Proponent (LLaMA2-7B, s42)")
    reliability_panel(axes[1], test_p_b, test_c_b, "Critic (Qwen2.5-7B, s42)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_calibration.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 文本摘要 ----
    lines = ["# 置信度校准结果（MELD，top-1 二值口径）", ""]
    for role, name in (("proponent", "Proponent"), ("critic", "Critic")):
        lines.append(f"## {name}（3-seed）")
        lines.append("| seed | acc | mean_conf | ECE(w15) | ECE(mass10) | Brier | NLL | Spearman |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in (42, 43, 44):
            m = summary[role][f"s{s}"]
            lines.append(f"| {s} | {m['acc']:.4f} | {m['mean_conf']:.4f} | {m['ece_w15']:.4f} | "
                         f"{m['ece_mass10']:.4f} | {m['brier_binary']:.4f} | {m['nll_binary']:.4f} | "
                         f"{m['spearman_p_correct']:.4f} |")
        agg = summary[role]["mean±std"]
        lines.append(f"| **均值** | — | — | **{agg['ece_w15']}** | — | **{agg['brier_binary']}** | — | **{agg['spearman']}** |")
        lines.append(f"过自信间隙 conf−acc：{agg['conf_gap_overconfident']}")
        lines.append("")
    lines.append("## Isotonic 单调校准（dev 拟合 → test 评估，s42）")
    lines.append("| 角色 | ECE raw | ECE iso | Brier raw | Brier iso | Spearman raw | Spearman iso | 排序保持 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for role in ("proponent_s42", "critic_s42"):
        m = summary["isotonic_dev_fit"][role]
        lines.append(f"| {role} | {m['ece_w15_raw']:.4f} | {m['ece_w15_isotonic']:.4f} | "
                     f"{m['brier_raw']:.4f} | {m['brier_isotonic']:.4f} | {m['spearman_raw']:.4f} | "
                     f"{m['spearman_isotonic']:.4f} | {m['monotonic_preserves_ranking']} |")
    (OUT / "calibration_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
