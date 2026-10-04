"""Few-ERC 少数类定向合成（V3 7.3）：fear/disgust 1.8x，样本打 synthetic=true 标记。

模板化合成（保持 Friends 风格），不依赖外部 API；后续可替换为 teacher LLM 合成，
接口不变（返回 DialogueSample 列表）。
"""

from __future__ import annotations

import random

from ..utils.label_schema import MELD_SCHEMA
from .schemas import DialogueContext, DialogueSample

# fear 的 5 种语义子类（V3 7.3 / 错误分析 5.1）：惊恐 / 尴尬恐惧 / 担忧 / 威胁 / 焦虑
FEAR_TEMPLATES: dict[str, list[str]] = {
    "panic": [
        "Oh my god, {name}, look out! There's something moving behind you!",
        "No no no, this can't be happening, {name}, tell me this isn't real!",
        "{name}!! Did you hear that noise downstairs? I'm freaking out right now!",
    ],
    "embarrassment": [
        "{name}, please tell me nobody saw that. I want to disappear right now.",
        "Oh no, {name}, I can't believe I just said that in front of everyone...",
        "Please don't bring that up, {name}. Every time I think about it I cringe so hard.",
    ],
    "worry": [
        "{name}, I've been up all night thinking something's really wrong with mom.",
        "I don't know, {name}. What if the test results come back and it's bad news?",
        "{name}, I'm scared to check my bank account. What if the rent check bounced?",
    ],
    "threat": [
        "{name}, back away slowly. That dog does NOT look friendly.",
        "If you go down that alley at night, {name}, something terrible could happen to you.",
        "{name}, I have a really bad feeling about this guy. He's been following us for blocks.",
    ],
    "anxiety": [
        "{name}, my hands won't stop shaking. The interview is in an hour and I can't breathe.",
        "I keep replaying it in my head, {name}. What if I messed everything up?",
        "The deadline's tomorrow and I haven't started, {name}. I'm honestly panicking.",
    ],
}

DISGUST_TEMPLATES: list[str] = [
    "Ugh, {name}, there's a hair in my soup. I think I'm going to be sick.",
    "{name}, that smell coming from the fridge is absolutely revolting.",
    "You actually ate that? {name}, just watching you makes my stomach turn.",
    "Oh gross, {name}, there's mold all over this bread. Get it away from me.",
    "{name}, I can't believe you picked that up off the ground. That's disgusting.",
    "The way he talks with his mouth full, {name}, it makes me want to leave the table.",
    "{name}, you're seriously wearing that shirt again? It's got stains all over it.",
    "I just stepped in something, {name}. This is the most disgusting day of my life.",
]

_SPEAKERS = ("Joey", "Chandler", "Monica", "Rachel", "Ross", "Phoebe")

# 反问/上下文引导句，让合成样本带上下文轮次
_CONTEXT_LEADS = [
    "You have got to see this.",
    "Something happened at work today.",
    "Wait, listen to this.",
    "Okay, don't freak out.",
    "I need to tell you something.",
]


def synthesize_minority(
    samples: list[DialogueSample],
    label: str,
    ratio: float = 1.8,
    seed: int = 42,
    history_window: int = 5,
) -> list[DialogueSample]:
    """把指定类别样本量扩充到原量的 ratio 倍（模板采样，synthetic=true）。

    模板组合无放回去重：已采样的 (spk_a, spk_b, 模板, lead) 组合不再重复，
    重复则重抽；组合空间耗尽（新样本量 ≥ 组合空间大小）时才允许重复。
    """
    originals = [s for s in samples if s.gold_label == label and not s.synthetic]
    target_total = int(round(len(originals) * ratio))
    n_new = max(0, target_total - len(originals))
    if n_new == 0:
        return []
    rng = random.Random(seed)
    if label == "fear":
        n_templates = sum(len(v) for v in FEAR_TEMPLATES.values())
        sub_keys = list(FEAR_TEMPLATES)
    elif label == "disgust":
        n_templates = len(DISGUST_TEMPLATES)
        sub_keys = None
    else:
        raise ValueError(f"synthesize_minority 仅支持 fear/disgust，收到 {label}")
    # 组合空间：说话人对 × 模板 × 上下文引导句
    space_size = len(_SPEAKERS) * (len(_SPEAKERS) - 1) * n_templates * len(_CONTEXT_LEADS)
    seen: set[tuple] = set()
    synthetic: list[DialogueSample] = []
    for i in range(n_new):
        while True:
            spk_a = rng.choice(_SPEAKERS)
            spk_b = rng.choice([s for s in _SPEAKERS if s != spk_a])
            if label == "fear":
                sub = rng.choice(sub_keys)
                template_idx = rng.randrange(len(FEAR_TEMPLATES[sub]))
                template = FEAR_TEMPLATES[sub][template_idx]
            else:
                sub = "disgust"
                template_idx = rng.randrange(len(DISGUST_TEMPLATES))
                template = DISGUST_TEMPLATES[template_idx]
            lead_idx = rng.randrange(len(_CONTEXT_LEADS))
            key = (spk_a, spk_b, sub, template_idx, lead_idx)
            # 组合未采样过，或组合空间已耗尽（才允许重复）
            if key not in seen or len(seen) >= space_size:
                break
        seen.add(key)
        utt = template.format(name=spk_b)
        lead = _CONTEXT_LEADS[lead_idx]
        utterances = (lead, utt)
        speakers = (spk_b, spk_a)
        context = DialogueContext(
            utterances=utterances, speakers=speakers, target_idx=len(utterances) - 1,
            history_window=history_window, dialogue_id=f"syn-{label}-{i}",
        )
        synthetic.append(DialogueSample(
            context=context, gold_label=label, synthetic=True, split="train",
        ))
    return synthetic


def apply_fewshot_synthesis(
    train_samples: list[DialogueSample], ratio: float = 1.8, seed: int = 42
) -> list[DialogueSample]:
    """Few-ERC 入口：fear 与 disgust 分别定向合成后拼接（真实样本在前）。

    合成目标类固定为 MELD 少数类 fear/disgust；训练集中任一目标类不存在
    （如 IEMOCAP 4 类）时直接 raise——静默产出零样本会让 run_id 的 syn 段
    归因错误（表观有合成增强实际没有，R2-M1 教训），配置期显式失败。
    """
    present = {s.gold_label for s in train_samples}
    missing = [lb for lb in ("fear", "disgust") if lb not in present]
    if missing:
        raise ValueError(
            f"定向合成目标类 {missing} 在训练集中不存在（synthesis 仅支持 MELD "
            "7 类的 fear/disgust 少数类；IEMOCAP 4 类不含这两类），"
            "请在配置中关闭 fewshot.synthesis_enabled"
        )
    fear_new = synthesize_minority(train_samples, "fear", ratio, seed)
    disgust_new = synthesize_minority(train_samples, "disgust", ratio, seed)
    return train_samples + fear_new + disgust_new


def synthesis_report(train_samples: list[DialogueSample] , augmented: list[DialogueSample]) -> dict[str, int]:
    """合成前后类目统计（验收标准 2.3-①：fear 268→482、disgust 271→~488）。"""
    from collections import Counter
    before = Counter(s.gold_label for s in train_samples)
    after = Counter(s.gold_label for s in augmented)
    return {label: (before.get(label, 0), after.get(label, 0)) for label in MELD_SCHEMA.labels}
