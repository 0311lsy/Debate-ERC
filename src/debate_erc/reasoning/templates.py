"""推理指引模板（V3 7.1）：按错误模式注入提示的"推理指引"文本。

5 类错误模式 → 5 类模板：
- IRONY          反讽检测：前文冲突 → 字面 vs 语境矛盾 → 标签
- TRIGGER        上下文触发：前文触发事件 → 心理状态变化 → 标签
- MINORITY       少数类语义展开：fear 5 子类逐一比对 → 标签
                 （仅当标签表含 "fear" 时由 TemplateRouter 路由到此模板）
- LONG_UTTERANCE 长话语分段：按标点分段 → 每段情感 → 综合 → 标签
- LABEL_CHECK    标签校验：生成标签 → 合法性 → 映射
"""

from __future__ import annotations

import enum


class TemplateType(enum.Enum):
    """错误模式类型（对应 V3 7.1 模板表）。"""

    IRONY = "irony"
    TRIGGER = "trigger"
    MINORITY = "minority"
    LONG_UTTERANCE = "long_utterance"
    LABEL_CHECK = "label_check"


class ReasoningTemplates:
    """推理指引模板库：get 取模板，register 可扩展新模板。"""

    TEMPLATES: dict[TemplateType, str] = {
        TemplateType.IRONY: (
            "Ironic-utterance detection:\n"
            "Step 1: Scan the previous turns for conflicts (arguments, complaints, criticism).\n"
            "Step 2: Compare the literal wording of the target utterance with its contextual "
            "meaning; flag any contradiction between them.\n"
            "Step 3: If the literal sentiment contradicts the context, trust the contextual "
            "meaning as the true emotion.\n"
            "Step 4: Output the final label."
        ),
        TemplateType.TRIGGER: (
            "Context-trigger analysis:\n"
            "Step 1: Locate the event or prior utterance that triggered the speaker's emotional "
            "state.\n"
            "Step 2: Describe the speaker's psychological state change caused by that trigger.\n"
            "Step 3: Map the state change to one emotion label from the candidate list."
        ),
        TemplateType.MINORITY: (
            "Minority-class semantic expansion for 'fear' - compare the utterance against the "
            "five fear subtypes one by one:\n"
            "1. panic (sudden alarm, physical fright);\n"
            "2. embarrassed fear (awkwardness, social exposure);\n"
            "3. worry (concern about a possible bad outcome);\n"
            "4. threat (feeling endangered or intimidated);\n"
            "5. anxiety (diffuse unease without a clear source).\n"
            "Pick the closest subtype, then output 'fear' if it fits, otherwise the "
            "better-fitting label."
        ),
        TemplateType.LONG_UTTERANCE: (
            "Long-utterance segmentation:\n"
            "Step 1: Split the target utterance into segments at punctuation marks.\n"
            "Step 2: Identify the dominant emotion of each segment.\n"
            "Step 3: Weigh the segments (later segments usually dominate the speaker's current "
            "state) and synthesize one overall emotion.\n"
            "Step 4: Output the final label."
        ),
        TemplateType.LABEL_CHECK: (
            "Label verification:\n"
            "Step 1: Generate your candidate emotion label for the target utterance.\n"
            "Step 2: Check the candidate against the legal label list.\n"
            "Step 3: If it is illegal, map it to the closest legal label (e.g. angry -> anger) "
            "before answering."
        ),
    }

    @classmethod
    def get(cls, template_type: TemplateType) -> str:
        """取指定类型的模板文本；未注册时抛 KeyError。"""
        return cls.TEMPLATES[template_type]

    @classmethod
    def register(cls, template_type: TemplateType, text: str) -> None:
        """注册/覆盖一个模板（可扩展）。"""
        cls.TEMPLATES[template_type] = text
