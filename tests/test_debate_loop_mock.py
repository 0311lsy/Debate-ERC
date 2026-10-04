"""DebateLoop mock 测试：收敛 / 未收敛 / 兜底触发三条路径（不加载真实模型）。

用伪造 BaseAgent 子类（ScriptedAgent，按预设脚本返回固定 AgentOutput）驱动
DebateLoop.run，覆盖 V3 6.1/6.2/6.7.3 的关键分支：
1. 第 1 轮即收敛（无 reask）；
2. 第 2 轮收敛（critique_points / history 注入下一轮）；
3. max_rounds 耗尽未收敛（无 reask → 取 proponent 标签；轮次注入 + 分歧点兜底）；
4. 兜底触发：未收敛 → ReaskLoop 提前终止 / 强制终止（infer 回退与 infer_reask 两路径）；
5. 收敛但低置信 → Reask 条件 2 触发；
6. reasoning_hint 透传（proponent 与 critic）；
7. summarize_debate 新口径（error_consensus / reask_correction 按 reask_pre_label）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from debate_erc.agents.base_agent import BaseAgent
from debate_erc.data.schemas import AgentOutput, DialogueContext
from debate_erc.debate import ConvergenceChecker, DebateLoop, ReaskFallback, summarize_debate
from debate_erc.utils.label_schema import MELD_SCHEMA


def make_context() -> DialogueContext:
    return DialogueContext(
        utterances=("Hi there.", "Did you finish the report?", "No, I was busy."),
        speakers=("A", "B", "A"),
        target_idx=2,
        history_window=2,
        dialogue_id="mock-1",
    )


def out(label: str, conf: float, critique: tuple[str, ...] = ()) -> AgentOutput:
    return AgentOutput(
        label=label,
        confidence=conf,
        critique_points=list(critique),
        causal_chain=f"trigger -> state -> {label}",
        raw_text=f"Trigger: t\nLabel: {label}\nConfidence: {conf}",
    )


class ScriptedAgent(BaseAgent):
    """按预设脚本返回固定 AgentOutput 的伪造 Agent（脚本耗尽后重复最后一项）。

    不实现 infer_reask → reask_loop 走回退 infer 路径。
    """

    def __init__(self, outputs: list[AgentOutput]):
        super().__init__(generator=None, schema=MELD_SCHEMA)
        self.outputs = list(outputs)
        self.calls: list[dict] = []
        self.reask_calls: list[dict] = []

    def infer(self, context, reasoning_hint=None, **kwargs):
        self.calls.append({"reasoning_hint": reasoning_hint, **kwargs})
        idx = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[idx]


class ReaskScriptedAgent(ScriptedAgent):
    """带 infer_reask 的伪造 Agent（reask_loop 走 re-derive 专用路径）。

    脚本序列统一编址：辩论轮次按 infer 调用数、reask 轮次接着
    （infer 调用数 + reask 调用数）往后取。
    """

    def infer_reask(
        self, context, disagreement_points, previous_label,
        previous_confidence, reasoning_hint=None,
    ):
        self.reask_calls.append({
            "disagreement_points": list(disagreement_points),
            "previous_label": previous_label,
            "previous_confidence": previous_confidence,
            "reasoning_hint": reasoning_hint,
        })
        idx = min(len(self.calls) + len(self.reask_calls) - 1, len(self.outputs) - 1)
        return self.outputs[idx]


def test_path1_immediate_convergence():
    ctx = make_context()
    pro = ScriptedAgent([out("anger", 0.8)])
    cri = ScriptedAgent([out("anger", 0.7)])
    loop = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3)
    outcome = loop.run(ctx)

    assert outcome.converged is True
    assert outcome.final_label == "anger"
    assert abs(outcome.confidence - 0.75) < 1e-9  # 双方均值
    assert outcome.rounds == 1
    assert outcome.reask_triggered is False and outcome.reask_rounds == 0
    assert outcome.consensus_but_wrong is False  # 评估阶段才填充
    assert len(pro.calls) == 1 and len(cri.calls) == 1
    assert outcome.latency_seconds >= 0.0
    # 新字段：首轮标签 / 生成调用计数（1 轮 × 2 Agent）/ 未触发兜底
    assert outcome.first_round_label == "anger"
    assert outcome.generation_calls == 2
    assert outcome.reask_pre_label is None
    print("[路径1] 第 1 轮收敛: OK")


def test_path2_second_round_convergence():
    ctx = make_context()
    pro = ScriptedAgent([out("anger", 0.8), out("anger", 0.9)])
    cri = ScriptedAgent(
        [
            out("sadness", 0.9, critique=("trigger mismatch",)),
            out("anger", 0.7),
        ]
    )
    loop = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3)
    outcome = loop.run(ctx)

    assert outcome.converged is True and outcome.rounds == 2
    assert outcome.final_label == "anger" and abs(outcome.confidence - 0.8) < 1e-9
    # 第 2 轮 proponent 收到 critique_points 与前轮摘要 history
    assert pro.calls[1]["critique_points"] == ["trigger mismatch"]
    assert "Round 1" in pro.calls[1]["history"] and "anger" in pro.calls[1]["history"]
    assert outcome.disagreement_points == []  # 收敛轮 critic 无分歧点
    # 新字段：首轮标签是第 1 轮的 anger（非第 2 轮）；2 轮 × 2 Agent = 4 次生成
    assert outcome.first_round_label == "anger"
    assert outcome.generation_calls == 4
    assert outcome.reask_pre_label is None
    print("[路径2] 第 2 轮收敛（critique/history 注入）: OK")


def test_path3_max_rounds_no_convergence():
    ctx = make_context()
    # 3a: critic 有分歧点
    pro = ScriptedAgent([out("anger", 0.8)] * 3)
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 3)
    outcome = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3).run(ctx)

    assert outcome.converged is False and outcome.rounds == 3
    assert outcome.final_label == "anger" and outcome.confidence == 0.8  # 取 proponent
    assert outcome.disagreement_points == ["trigger mismatch"]
    assert outcome.reask_triggered is False  # 无 reask 时保持不触发
    assert outcome.generation_calls == 6  # 3 轮 × 2 Agent

    # 3b: critic 分歧点为空 → 轮次注入 + 分歧点兜底
    pro2 = ScriptedAgent([out("anger", 0.8)] * 3)
    cri2 = ScriptedAgent([out("sadness", 0.9)] * 3)
    outcome2 = DebateLoop(pro2, cri2, ConvergenceChecker(0.6), max_rounds=3).run(ctx)
    assert outcome2.disagreement_points == ["label disagreement: anger vs sadness"]
    # 轮次注入兜底：第 2/3 轮 proponent 收到注入的标签分歧（critic 未给出分歧点）
    expected = ["label disagreement: anger vs sadness"]
    assert pro2.calls[1]["critique_points"] == expected
    assert pro2.calls[2]["critique_points"] == expected
    assert outcome2.first_round_label == "anger"
    print("[路径3] max_rounds 未收敛 + 轮次注入/分歧点兜底: OK")


def test_path4_reask_on_disagreement():
    ctx = make_context()
    # 3 轮辩论（始终分歧）+ 1 次 reask（与辩论结论一致 → 提前终止）
    pro = ScriptedAgent(
        [out("anger", 0.8), out("anger", 0.85), out("anger", 0.9), out("anger", 0.95)]
    )
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 4)
    reask = ReaskFallback(low_confidence=0.6)
    outcome = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3, reask=reask).run(ctx)

    assert outcome.converged is False
    assert outcome.reask_triggered is True
    assert outcome.reask_rounds == 1  # 连续两次 label 一致 → 提前终止
    assert outcome.final_label == "anger" and outcome.confidence == 0.95
    # reask 轮把分歧点反馈给了 Agent1（重提示，非投票；无 infer_reask → 回退 infer）
    assert len(pro.calls) == 4
    assert pro.calls[-1]["critique_points"] == ["trigger mismatch"]
    # 新字段：reask 前的辩论结论 / 3×2 辩论 + 1 次 reask 生成
    assert outcome.reask_pre_label == "anger"
    assert outcome.generation_calls == 7
    assert outcome.first_round_label == "anger"
    print("[路径4a] 兜底触发（分歧重论证，提前终止，infer 回退路径）: OK")

    # 4b: reask 两轮 label 均不一致 → 上限 MAX_REASK 强制终止
    pro2 = ScriptedAgent(
        [
            out("anger", 0.8), out("anger", 0.85), out("anger", 0.9),  # 辩论 3 轮
            out("sadness", 0.7), out("anger", 0.9),                    # reask 两轮
        ]
    )
    cri2 = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 5)
    outcome2 = DebateLoop(
        pro2, cri2, ConvergenceChecker(0.6), max_rounds=3, reask=ReaskFallback(0.6)
    ).run(ctx)
    assert outcome2.reask_rounds == 2  # MAX_REASK 强制终止
    assert outcome2.final_label == "anger" and outcome2.confidence == 0.9  # 取最后一次
    assert outcome2.reask_pre_label == "anger"
    assert outcome2.generation_calls == 8  # 3×2 辩论 + 2 次 reask
    print("[路径4b] 兜底强制终止（上限 MAX_REASK=2）: OK")


def test_path5_reask_on_low_confidence():
    ctx = make_context()
    # 辩论收敛（双方 0.65 ≥ 0.6）但均值 0.65 < low_confidence 0.7 → 条件 2 触发
    pro = ScriptedAgent([out("anger", 0.65), out("anger", 0.9)])
    cri = ScriptedAgent([out("anger", 0.65)])
    reask = ReaskFallback(low_confidence=0.7)
    outcome = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3, reask=reask).run(ctx)

    assert outcome.converged is True
    assert outcome.reask_triggered is True  # 条件 2：一致但低置信
    assert outcome.reask_rounds == 1
    assert outcome.final_label == "anger" and outcome.confidence == 0.9
    assert outcome.reask_pre_label == "anger"  # reask 前的收敛结论
    assert outcome.generation_calls == 3  # 1×2 辩论 + 1 次 reask
    print("[路径5] 兜底触发（收敛但低置信）: OK")


def test_reask_uses_infer_reask():
    """ReaskFallback 优先探测 infer_reask（re-derive 专用路径）。"""
    ctx = make_context()
    # 辩论 3 轮分歧 + reask 两轮：第 1 轮换标签，第 2 轮与第 1 轮一致 → 提前终止
    pro = ReaskScriptedAgent(
        [
            out("anger", 0.8), out("anger", 0.85), out("anger", 0.9),   # 辩论 3 轮
            out("sadness", 0.7), out("sadness", 0.75),                  # reask 两轮
        ]
    )
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 5)
    reask = ReaskFallback(low_confidence=0.6)
    outcome = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3, reask=reask).run(ctx)

    assert outcome.reask_triggered is True and outcome.reask_rounds == 2
    assert len(pro.calls) == 3           # 辩论 3 轮走 infer
    assert len(pro.reask_calls) == 2     # reask 走 infer_reask 专用路径
    first = pro.reask_calls[0]
    assert first["disagreement_points"] == ["trigger mismatch"]
    assert first["previous_label"] == "anger"       # 辩论末轮结论
    assert abs(first["previous_confidence"] - 0.9) < 1e-9
    assert first["reasoning_hint"] is None
    assert outcome.final_label == "sadness" and abs(outcome.confidence - 0.75) < 1e-9
    assert outcome.reask_pre_label == "anger"       # reask 前辩论结论
    assert outcome.generation_calls == 8            # 3×2 辩论 + 2 次 reask
    print("[路径4c] 兜底走 infer_reask 专用路径: OK")


def test_reasoning_hint_passthrough():
    """reasoning_hint 同时透传给 proponent 与 critic（每轮）。"""
    ctx = make_context()
    pro = ScriptedAgent([out("anger", 0.8), out("anger", 0.9)])
    cri = ScriptedAgent(
        [out("sadness", 0.9, critique=("trigger mismatch",)), out("anger", 0.7)]
    )
    loop = DebateLoop(pro, cri, ConvergenceChecker(0.6), max_rounds=3)
    loop.run(ctx, reasoning_hint="focus on the trigger event")

    for calls in (pro.calls, cri.calls):
        assert all(c["reasoning_hint"] == "focus on the trigger event" for c in calls)
    print("[路径6] reasoning_hint 透传（proponent + critic）: OK")


def test_summarize_debate():
    """六样本汇总：新字段填充 + error_consensus / reask_correction 新口径精确分数。

    样本布局（gold 序列 [anger, anger, anger, sadness, anger, sadness]）：
    1. 第 1 轮收敛 anger（正确，无 reask）；
    2. 第 2 轮收敛 anger（正确，无 reask）；
    3. 未收敛 + reask 1 轮 → anger（正确；reask 前结论已正确 → 不算修正）；
    4. 未收敛 + reask 2 轮 → anger（错误；reask 前后都错）；
    5. 收敛 sadness 但 gold=anger（错误共识 → error_consensus 有区分度）；
    6. 未收敛 + reask 2 轮 → sadness（reask 前错 → 正确，reask 修正样本）。
    """
    ctx = make_context()
    outcomes = []

    # 1: 第 1 轮收敛 anger
    pro = ScriptedAgent([out("anger", 0.8)])
    cri = ScriptedAgent([out("anger", 0.7)])
    outcomes.append(DebateLoop(pro, cri, ConvergenceChecker(0.6)).run(ctx))

    # 2: 第 2 轮收敛 anger
    pro = ScriptedAgent([out("anger", 0.8), out("anger", 0.9)])
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",)), out("anger", 0.7)])
    outcomes.append(DebateLoop(pro, cri, ConvergenceChecker(0.6)).run(ctx))

    # 3: 未收敛 + reask 1 轮（anger 提前终止）
    pro = ScriptedAgent([out("anger", 0.8), out("anger", 0.85), out("anger", 0.9), out("anger", 0.95)])
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 4)
    outcomes.append(
        DebateLoop(pro, cri, ConvergenceChecker(0.6), reask=ReaskFallback(0.6)).run(ctx)
    )

    # 4: 未收敛 + reask 两轮强制终止（最终 anger，错误）
    pro = ScriptedAgent([out("anger", 0.8), out("anger", 0.85), out("anger", 0.9),
                         out("sadness", 0.7), out("anger", 0.9)])
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 5)
    outcomes.append(
        DebateLoop(pro, cri, ConvergenceChecker(0.6), reask=ReaskFallback(0.6)).run(ctx)
    )

    # 5: 第 1 轮收敛 sadness（错误共识：gold=anger）
    pro = ScriptedAgent([out("sadness", 0.8)])
    cri = ScriptedAgent([out("sadness", 0.7)])
    outcomes.append(DebateLoop(pro, cri, ConvergenceChecker(0.6)).run(ctx))

    # 6: 未收敛 + reask 两轮（sadness 两次一致 → 修正为 gold）
    pro = ScriptedAgent([out("anger", 0.8), out("anger", 0.85), out("anger", 0.9),
                         out("sadness", 0.7), out("sadness", 0.75)])
    cri = ScriptedAgent([out("sadness", 0.9, critique=("trigger mismatch",))] * 5)
    outcomes.append(
        DebateLoop(pro, cri, ConvergenceChecker(0.6), reask=ReaskFallback(0.6)).run(ctx)
    )

    golds = ["anger", "anger", "anger", "sadness", "anger", "sadness"]
    # 单 Agent 基线：1/6 错（辩论修正）；4/5 对（辩论带错 → introduced error）；6 错（reask 修正）
    single_labels = ["sadness", "anger", "anger", "sadness", "anger", "anger"]

    # 新字段填充核对
    assert [o.first_round_label for o in outcomes] == [
        "anger", "anger", "anger", "anger", "sadness", "anger",
    ]
    assert [o.reask_pre_label for o in outcomes] == [
        None, None, "anger", "anger", None, "anger",
    ]
    assert [o.generation_calls for o in outcomes] == [2, 4, 7, 8, 2, 8]

    summary = summarize_debate(outcomes, golds, single_labels)

    assert summary["n"] == 6
    assert abs(summary["convergence_rate"] - 0.5) < 1e-9          # 样本 1/2/5
    assert abs(summary["error_consensus_rate"] - 1 / 6) < 1e-9    # 样本 5：收敛但错
    assert abs(summary["reask_trigger_rate"] - 0.5) < 1e-9        # 样本 3/4/6
    # 新口径：分子 = reask 前错误且 reask 后正确（仅样本 6；样本 3 reask 前已正确）
    assert abs(summary["reask_correction_rate"] - 1 / 3) < 1e-9
    assert summary["n_reask_corrected"] == 1
    assert abs(summary["debate_accuracy"] - 4 / 6) < 1e-9         # 样本 1/2/3/6
    assert abs(summary["single_accuracy"] - 4 / 6) < 1e-9         # 样本 2/3/4/5
    assert abs(summary["correction_rate"] - 1.0) < 1e-9           # 单Agent 错误样本 1/6 均被修正
    assert abs(summary["introduced_error_rate"] - 0.5) < 1e-9     # 单Agent 正确样本 4/5 被带错
    print("[汇总] summarize_debate 指标（新口径）: OK")


if __name__ == "__main__":
    test_path1_immediate_convergence()
    test_path2_second_round_convergence()
    test_path3_max_rounds_no_convergence()
    test_path4_reask_on_disagreement()
    test_path5_reask_on_low_confidence()
    test_reask_uses_infer_reask()
    test_reasoning_hint_passthrough()
    test_summarize_debate()
    print("\n全部 mock 路径测试通过 ✔")
