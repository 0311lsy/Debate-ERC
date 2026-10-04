"""统计显著性数值边界测试（审查修复 E 任务 6，D 组 significance.py 修复）。

覆盖四组数值契约：
a) paired_ttest 退化情形：全零差 → (0.0, 1.0)；常数非零差 → (inf, 0.0)；
b) bootstrap_ci 已知大差异（A 全对 B 全错）：CI 不含 0 且 Δ≈1；
c) cohens_d 配对口径 d_z：常数差 sd=0 → 0.0（按 D 组实现）；已知差值精确 d_z；
d) 分位插值：构造已知重采样分布，验证 lo < mean < hi。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from debate_erc.evaluation import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    compare_systems,
    paired_ttest,
)

# 已知大差异构造：golds 不含 neutral，B 全预测 neutral → B 每类 F1=0（W-F1=0）
BIG_DIFF_GOLDS = ["joyful", "sadness", "anger", "fear",
                  "disgust", "joyful", "sadness", "anger"]
BIG_DIFF_PREDS_A = list(BIG_DIFF_GOLDS)      # A 全对 → W-F1 = 1.0
BIG_DIFF_PREDS_B = ["neutral"] * len(BIG_DIFF_GOLDS)  # B 全错 → W-F1 = 0.0


class TestPairedTtestDegenerateCases:
    """a) 差值零方差时 t 检验无定义的退化行为（D 组修复）。"""

    def test_all_zero_diffs_returns_identity(self):
        """两系统逐 seed 完全一致 → 全零差 → (0.0, 1.0)（不显著）。"""
        t, p = paired_ttest([0.5, 0.6, 0.7, 0.65], [0.5, 0.6, 0.7, 0.65])
        assert (t, p) == (0.0, 1.0)

    def test_constant_nonzero_diff_returns_inf_zero_p(self):
        """常数非零差（系统间恒定差异）→ (inf, 0.0)。"""
        t, p = paired_ttest([0.7, 0.7, 0.7], [0.5, 0.5, 0.5])
        assert t == float("inf")
        assert p == 0.0

    def test_constant_negative_diff_returns_neg_inf(self):
        """常数负差（B 恒优于 A）→ (-inf, 0.0)（R2-m12：t 值方向随差值符号）。"""
        t, p = paired_ttest([0.5, 0.5, 0.5], [0.7, 0.7, 0.7])
        assert t == float("-inf")
        assert p == 0.0

    def test_normal_pair_detects_improvement(self):
        """常规差值走 scipy.ttest_rel：A 一致优于 B → t 为正、p 显著。"""
        t, p = paired_ttest([0.80, 0.90, 0.70, 0.95], [0.50, 0.60, 0.65, 0.55])
        assert t > 0
        assert p < 0.05

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="长度不一致"):
            paired_ttest([0.5, 0.6], [0.5])

    def test_too_few_pairs_rejected(self):
        with pytest.raises(ValueError, match="< 2"):
            paired_ttest([0.5], [0.4])


class TestBootstrapCiKnownBigDifference:
    """b) 已知大差异：A 全对（W-F1=1）vs B 全错（W-F1=0）→ Δ=1，CI 不含 0。"""

    def test_ci_excludes_zero_with_mean_one(self):
        mean, lo, hi = bootstrap_ci(
            BIG_DIFF_PREDS_A, BIG_DIFF_PREDS_B, BIG_DIFF_GOLDS, n=500
        )
        assert mean == pytest.approx(1.0)
        assert lo > 0.0, f"95% CI 下界须为正（CI 不含 0）: lo={lo}"
        assert hi <= 1.0 + 1e-9
        assert lo <= hi

    def test_identical_systems_zero_delta(self):
        """两系统完全相同 → 每次 Δ=0 → (mean, lo, hi) 全 0（CI 含 0，不显著）。"""
        mean, lo, hi = bootstrap_ci(
            BIG_DIFF_PREDS_A, BIG_DIFF_PREDS_A, BIG_DIFF_GOLDS, n=200
        )
        assert (mean, lo, hi) == (0.0, 0.0, 0.0)

    def test_bootstrap_is_deterministic_given_seed(self):
        """同 seed 复现同结果（多 seed 汇总可复现）。"""
        r1 = bootstrap_ci(BIG_DIFF_PREDS_A, BIG_DIFF_PREDS_B, BIG_DIFF_GOLDS, n=100, seed=7)
        r2 = bootstrap_ci(BIG_DIFF_PREDS_A, BIG_DIFF_PREDS_B, BIG_DIFF_GOLDS, n=100, seed=7)
        assert r1 == r2

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="长度不一致"):
            bootstrap_ci(["a"], ["b", "c"], ["x", "y"])


class TestCohensDPaired:
    """c) 配对口径 d_z = mean(diffs)/sd(diffs)；sd=0 → 0.0（D 组实现）。"""

    def test_constant_diff_zero_sd_returns_zero(self):
        """diffs=[1,1,1]：sd=0 → 返回 0.0（效应量无定义，禁抛异常/禁 inf）。"""
        d = cohens_d([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
        assert d == 0.0

    def test_known_paired_dz_exact(self):
        """a=[1,2,3] vs b=[0.5,1,1.5]：diffs=[.5,1,1.5]，mean=1、sd=.5 → d_z=2.0。"""
        d = cohens_d([1.0, 2.0, 3.0], [0.5, 1.0, 1.5])
        assert d == pytest.approx(2.0)

    def test_paired_not_pooled(self):
        """配对口径：d_z 的分子分母都来自逐对差值序列。

        恒定差（二进制可精确表示的 0.25 步长 → sd 精确为 0）→ 0.0；
        打散配对差后 d_z 有限非零。
        """
        a = [1.0, 2.0, 3.0, 4.0]
        b_const = [0.75, 1.75, 2.75, 3.75]  # 逐对恒差 0.25（精确）→ sd=0
        assert cohens_d(a, b_const) == 0.0
        b_mixed = [1.5, 0.75, 2.75, 3.75]  # 差值序列有波动 → d_z 有限非零
        d = cohens_d(a, b_mixed)
        assert math.isfinite(d) and d != 0.0

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="长度不一致"):
            cohens_d([1.0, 2.0], [1.0])

    def test_too_few_samples_returns_zero(self):
        assert cohens_d([1.0], [0.5]) == 0.0


class TestBootstrapQuantileInterpolation:
    """d) 分位插值：已知重采样分布上验证 lo < mean < hi（线性插值，D 组修复）。"""

    @staticmethod
    def _frac_a(preds, golds) -> float:
        """可控指标：p 中 "a" 的占比（重采样分布可精确推导）。"""
        return sum(1 for x in preds if x == "a") / len(preds)

    def test_lo_lt_mean_lt_hi_on_known_distribution(self):
        """size=2：A=["a","b"]（每位等概率抽中 "a"），B 无 "a"。

        重采样 Δ = idx0_count/2 ∈ {0, 0.5, 1}，概率 {1/4, 1/2, 1/4}
        → 2.5% 分位 = 0、97.5% 分位 = 1、均值 ≈ 0.5（seed 固定，确定性）。
        """
        mean, lo, hi = bootstrap_ci(
            ["a", "b"], ["b", "b"], ["x", "y"],
            metric_fn=self._frac_a, n=1000, seed=42,
        )
        assert lo == 0.0
        assert hi == 1.0
        assert mean == pytest.approx(0.5, abs=0.05)
        assert lo < mean < hi, "CI 分位须夹住均值：lo < mean < hi"

    def test_quantiles_are_interpolated_not_min_max(self):
        """非退化分布上 lo/hi 是内部插值分位而非全局 min/max 端点。"""
        # 与上一用例同构造，但 size=4：Δ ∈ {0,.25,.5,.75,1}，两端概率 1/16
        # n=1000 时 2.5%/97.5% 分位仍夹住均值，且至少有一个内部取值
        mean, lo, hi = bootstrap_ci(
            ["a", "a", "b", "b"], ["b", "b", "b", "b"], ["x"] * 4,
            metric_fn=self._frac_a, n=1000, seed=42,
        )
        assert -1e-9 <= lo < mean < hi <= 1.0 + 1e-9
        assert lo < hi


class TestCompareSystemsIntegration:
    """D 组 compare_systems 修复后的整体行为（bootstrap CI 为主判据）。"""

    def test_big_difference_significant(self):
        report = compare_systems(
            BIG_DIFF_PREDS_A, BIG_DIFF_PREDS_B, BIG_DIFF_GOLDS, n_bootstrap=500
        )
        assert report["delta_wf1"] == pytest.approx(1.0)
        assert report["significant"] is True
        lo, hi = report["ci"]
        assert lo > 0.0, "显著要求 95% CI 不含 0"

    def test_identical_systems_not_significant(self):
        report = compare_systems(
            BIG_DIFF_PREDS_A, BIG_DIFF_PREDS_A, BIG_DIFF_GOLDS, n_bootstrap=200
        )
        assert report["delta_wf1"] == 0.0
        assert report["p_value"] == 1.0
        assert report["significant"] is False
