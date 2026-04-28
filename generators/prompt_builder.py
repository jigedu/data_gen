"""
prompt_builder.py
-----------------
Prompt模板构造模块。

负责为三个生成阶段构建不同风格的系统提示词（System Prompt）与用户提示词（User Prompt），
以驱动LLM生成覆盖不同难度与表达风格的问答对。

生成策略分为两大类：
  1. 简单样本（Simple）：极简口语短句（8~15字），直接、随口，训练后的模型能正确匹配。
  2. 复杂样本（Complex）：包含隐含意图、长文本背景、多轮对话截断、专业术语混用等
     复杂表达，确保模型在生产环境中的鲁棒性。

修改记录：
  - 阶段一 Prompt：字数严格控制在 8~15 字，极简口语风格，参考示例为短句
  - 所有阶段 Prompt：强化 answer 字段格式约束，防止 LLM 将 answer 误填为回复内容
"""

import random
from typing import Dict, Any, List

# ─────────────────────────── 系统提示词 ───────────────────────────

SYSTEM_PROMPT_GENERATOR = """你是一个专业的训练数据生成专家，专门为AI路由模型生成高质量的意图识别训练样本。

你的任务是：根据给定的智能体信息，生成用户可能发出的真实查询问题（question），以及对应的智能体名称（answer）。

严格规则：
1. question 必须是用户的真实自然语言表达，不能直接包含智能体名称
2. question 应该体现用户的实际需求，而非对智能体功能的描述
3. answer 必须严格等于给定的智能体名称字符串，绝对不能填写任何其他内容（不能填回答、建议或解释）
4. 输出格式为 JSON 数组，每个元素只有 question 和 answer 两个字段
5. 不要输出任何额外解释，只输出 JSON 数组
"""

SYSTEM_PROMPT_COMPLEX = """你是一个专业的训练数据生成专家，专门为AI路由模型生成高难度的意图识别训练样本。

你的任务是：根据给定的智能体信息，生成表达复杂、意图隐晦的用户查询问题（question）。

严格规则：
1. question 必须使用复杂的表达方式，包括但不限于：
   - 长文本背景描述后提出需求
   - 口语化、不规范的表达
   - 包含多个子需求，核心需求隐藏在次要信息中
   - 使用行业术语或缩写
   - 类似多轮对话中的截断式提问（如"刚才说的那个问题，帮我处理一下"）
   - 隐含意图，不直接说明需要什么功能
2. question 绝对不能包含智能体名称
3. answer 必须严格等于给定的智能体名称字符串，绝对不能填写任何其他内容（不能填回答、建议或解释）
4. 输出格式为 JSON 数组，每个元素只有 question 和 answer 两个字段
5. 不要输出任何额外解释，只输出 JSON 数组
"""

# ─────────────────────────── 阶段一：冷启动极简短句模板 ───────────────────────────
# 目标：字数 8~15 字，极简口语，像用户随口发出的一句话

STAGE1_SIMPLE_TEMPLATES = [
    """请为智能体「{name}」批量生成 {count} 条极简口语训练样本。

智能体功能：{description}
核心关键词：{keywords}

【核心要求】
- 每条问题字数严格控制在 8~15 字，越短越好
- 模拟用户随口说出的一句话，极度口语化、自然
- 覆盖不同细分场景关键词（{keywords}）
- 问题中不能出现智能体名称「{name}」
- answer 字段只能填写字符串 "{name}"，不得填写任何其他内容

【参考示例，注意这种极简风格】
[
  {{"question": "航班偏航了怎么办", "answer": "{name}"}},
  {{"question": "查下航线偏离", "answer": "{name}"}},
  {{"question": "飞机绕航申请", "answer": "{name}"}},
  {{"question": "备降机场怎么选", "answer": "{name}"}},
  {{"question": "改航路线帮我规划", "answer": "{name}"}}
]

请严格按照上述极简风格，输出包含 {count} 条样本的 JSON 数组，answer 只能是 "{name}"：""",

    """为智能体「{name}」生成 {count} 条极短的用户指令样本。

功能：{description}
关键词：{keywords}

要求：
1. 每条 8~15 字，模拟用户随手输入的简短指令
2. 口语化，像在手机上随手发的消息
3. 不同条目覆盖不同关键词和场景
4. 不含智能体名称「{name}」
5. answer 固定为字符串 "{name}"，不得填写其他任何内容

示例风格（极简短句）：
[
  {{"question": "偏航了怎么处理", "answer": "{name}"}},
  {{"question": "帮我看看绕行方案", "answer": "{name}"}},
  {{"question": "燃油够不够绕路", "answer": "{name}"}}
]

输出 {count} 条 JSON 数组，answer 只能是 "{name}"：""",

    """批量生成针对「{name}」的极简用户查询，共 {count} 条。

智能体描述：{description}

生成规则：
- 字数：8~15 字（硬性要求）
- 风格：口语短句，像日常聊天中的随口一问
- 多样性：覆盖 {keywords} 等不同关键词
- 禁止：问题中不能出现「{name}」
- answer：只能是字符串 "{name}"，不能填回答内容

极简示例：
[
  {{"question": "飞机改航怎么申请", "answer": "{name}"}},
  {{"question": "紧急备降流程", "answer": "{name}"}},
  {{"question": "ATC协调怎么做", "answer": "{name}"}}
]

输出 JSON 数组（{count} 条），answer 只能是 "{name}"：""",
]

# ─────────────────────────── 阶段二：生产泛化模板 ───────────────────────────

# 阶段二-A：口语化样本（20~60字）
STAGE2_COLLOQUIAL_TEMPLATES = [
    """为智能体「{name}」生成 {count} 条口语化、非正式的用户查询样本。

功能：{description}
关键词：{keywords}

要求：
1. 模拟真实用户的口语化表达，可以有语气词、不完整句子
2. 表达不那么正式，像是在聊天或随口一问
3. 意图指向该智能体的核心功能
4. 长度 20~60 字
5. answer 字段只能填写字符串 "{name}"，不得填写任何其他内容

示例格式：
[
  {{"question": "哎，飞机偏航了，最近有什么备降机场推荐吗？", "answer": "{name}"}},
  {{"question": "这偏航路线绕开空域麻烦吗？油够不够用呀？", "answer": "{name}"}}
]

输出 {count} 条 JSON 数组，answer 只能是字符串 "{name}"：""",

    """针对「{name}」生成 {count} 条带语气词的口语化查询。

描述：{description}

要求：
- 像用户在工作群里随手发的消息
- 可以有"啊""呢""吧""哦"等语气词
- 可以有错别字或不完整表达
- 长度 15~50 字
- answer 只能是字符串 "{name}"，绝对不能填写其他内容

输出 JSON 数组（{count} 条），answer 只能是 "{name}"：""",
]

# 阶段二-B：长文本复杂样本（80~200字）
STAGE2_COMPLEX_TEMPLATES = [
    """为智能体「{name}」生成 {count} 条包含丰富背景信息的复杂用户查询。

功能：{description}
领域：{domain}

要求：
1. 先描述具体业务场景，再提出需求（长度 80~200 字）
2. 混合专业术语 + 通俗描述
3. 可以包含多个子问题或多步骤需求
4. 意图最终指向该智能体的核心功能
5. answer 字段只能填写字符串 "{name}"，不得填写任何其他内容

示例格式：
[
  {{"question": "我们在巡航高度过程中突然发生了强侧风导致偏航，现需紧急调整航线绕过受影响的空域，同时评估附近备降机场的适航性和当前剩余燃油是否支持备降。请帮忙分析偏航原因，规划绕行方案并协调ATC。", "answer": "{name}"}},
  ...
]

输出 {count} 条 JSON 数组，answer 只能是字符串 "{name}"：""",

    """生成 {count} 条高难度路由训练样本，针对智能体「{name}」（{domain}领域）。

描述：{description}
关键词：{keywords}

要求：
- 用户描述了一个具体工作场景，需要推断其真实需求
- 使用大量背景信息，核心需求在最后才提出
- 字数 80~150 字
- answer 只能是字符串 "{name}"，绝对不能填写其他内容

输出 JSON 数组（{count} 条），answer 只能是 "{name}"：""",
]

# 阶段二-C：多轮对话截断样本（20~80字）
STAGE2_MULTITURN_TEMPLATES = [
    """为智能体「{name}」生成 {count} 条"多轮对话截断"样本。

功能：{description}

要求：
1. 模拟从一段对话中间截取的片段，用户的问题依赖前文上下文
2. 可以包含"刚才说的那个""继续上面的情况""还有那个问题"等引用前文的表达
3. 问题本身不完整，但结合上下文能判断出意图
4. 长度 20~80 字
5. answer 字段只能填写字符串 "{name}"，不得填写任何其他内容

示例格式：
[
  {{"question": "刚才说的那个偏航导致的燃油紧张问题，备降机场选择上还有什么建议？", "answer": "{name}"}},
  {{"question": "继续上面的情况，绕行路径规划有没有更节油的方案？", "answer": "{name}"}}
]

输出 {count} 条 JSON 数组，answer 只能是字符串 "{name}"：""",
]

# 阶段二-D：隐含意图样本（30~100字）
STAGE2_IMPLICIT_TEMPLATES = [
    """为智能体「{name}」生成 {count} 条"隐含意图"样本。

功能：{description}

要求：
1. 用户描述的是一种现象或状态，而不是直接说明需求
2. 不直接使用该智能体的关键词，但场景明确指向其功能
3. 需要推理才能判断用户真正需要什么
4. 长度 30~100 字
5. answer 字段只能填写字符串 "{name}"，不得填写任何其他内容

示例格式：
[
  {{"question": "飞行途中发现飞机航向突然偏离预定路线，预计将影响正常降落时间。", "answer": "{name}"}},
  {{"question": "由于天气变化，当前航路可能出现禁飞区，需要调整飞行计划以避免风险。", "answer": "{name}"}}
]

输出 {count} 条 JSON 数组，answer 只能是字符串 "{name}"：""",
]

# 阶段二：简单样本补充（确保每个阶段都有简单样本）
STAGE2_SIMPLE_TEMPLATES = [
    """为智能体「{name}」生成 {count} 种不同表达风格的简单用户问题。

功能：{description}
关键词：{keywords}

要求每个问题：
1. 长度 15~50 字
2. 口语化、直接
3. 覆盖不同的使用场景角度
4. 不包含智能体名称「{name}」
5. answer 只能是字符串 "{name}"，不得填写任何其他内容

输出 {count} 条 JSON 数组，answer 只能是 "{name}"：""",

    """针对「{name}」智能体，生成不同角度的简单查询，共 {count} 条。

描述：{description}

请从以下角度各生成问题（每行一个JSON）：
- 初次使用的普通用户视角
- 有一定经验的专业用户视角
- 紧急情况下的简短提问
- 带有具体数据或参数的提问
- 对比或选择类的提问

要求：长度 15~50 字，answer 只能是字符串 "{name}"，不得填写任何其他内容。

输出 JSON 数组（{count} 条），answer 只能是 "{name}"：""",
]


class PromptBuilder:
    """
    Prompt构造器：根据生成阶段、样本类型与智能体元数据，
    构建用于驱动LLM生成训练数据的提示词对。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def build_stage1_prompt(
        self, agent: Dict[str, Any], sample_idx: int = 0, batch_size: int = 5
    ) -> Dict[str, str]:
        """
        构建阶段一（冷启动）的Prompt。
        生成极简口语短句样本，字数控制在 8~15 字。

        Args:
            agent:      智能体元数据字典
            sample_idx: 当前样本序号（用于模板轮换）
            batch_size: 单次批量生成数量

        Returns:
            {"system": str, "user": str}
        """
        template = STAGE1_SIMPLE_TEMPLATES[sample_idx % len(STAGE1_SIMPLE_TEMPLATES)]
        user_prompt = template.format(
            name=agent["name"],
            description=agent["description"],
            keywords="、".join(agent["keywords"]),
            domain=agent["domain"],
            count=batch_size,
        )
        return {"system": SYSTEM_PROMPT_GENERATOR, "user": user_prompt}

    def build_stage2_simple_prompt(
        self, agent: Dict[str, Any], batch_size: int = 5
    ) -> Dict[str, str]:
        """
        构建阶段二（生产泛化）的简单样本Prompt，一次生成多条。

        Args:
            agent:      智能体元数据字典
            batch_size: 单次生成数量

        Returns:
            {"system": str, "user": str}
        """
        template = self.rng.choice(STAGE2_SIMPLE_TEMPLATES)
        user_prompt = template.format(
            name=agent["name"],
            description=agent["description"],
            keywords="、".join(agent["keywords"]),
            domain=agent["domain"],
            count=batch_size,
        )
        return {"system": SYSTEM_PROMPT_GENERATOR, "user": user_prompt}

    def build_stage2_colloquial_prompt(
        self, agent: Dict[str, Any], batch_size: int = 5
    ) -> Dict[str, str]:
        """
        构建阶段二口语化样本Prompt（20~60字）。
        """
        template = self.rng.choice(STAGE2_COLLOQUIAL_TEMPLATES)
        user_prompt = template.format(
            name=agent["name"],
            description=agent["description"],
            keywords="、".join(agent["keywords"]),
            domain=agent["domain"],
            count=batch_size,
        )
        return {"system": SYSTEM_PROMPT_GENERATOR, "user": user_prompt}

    def build_stage2_complex_prompt(
        self, agent: Dict[str, Any], sample_idx: int = 0, batch_size: int = 3
    ) -> Dict[str, str]:
        """
        构建阶段二（生产泛化）的复杂样本Prompt（80~200字）。

        Args:
            agent:      智能体元数据字典
            sample_idx: 当前样本序号（用于模板轮换）
            batch_size: 单次生成数量

        Returns:
            {"system": str, "user": str}
        """
        template = STAGE2_COMPLEX_TEMPLATES[
            sample_idx % len(STAGE2_COMPLEX_TEMPLATES)
        ]
        user_prompt = template.format(
            name=agent["name"],
            description=agent["description"],
            keywords="、".join(agent["keywords"]),
            domain=agent["domain"],
            count=batch_size,
        )
        return {"system": SYSTEM_PROMPT_COMPLEX, "user": user_prompt}

    def build_stage2_multiturn_prompt(
        self, agent: Dict[str, Any], batch_size: int = 3
    ) -> Dict[str, str]:
        """
        构建阶段二多轮对话截断样本Prompt（20~80字）。
        """
        template = self.rng.choice(STAGE2_MULTITURN_TEMPLATES)
        user_prompt = template.format(
            name=agent["name"],
            description=agent["description"],
            keywords="、".join(agent["keywords"]),
            domain=agent["domain"],
            count=batch_size,
        )
        return {"system": SYSTEM_PROMPT_COMPLEX, "user": user_prompt}

    def build_stage2_implicit_prompt(
        self, agent: Dict[str, Any], batch_size: int = 3
    ) -> Dict[str, str]:
        """
        构建阶段二隐含意图样本Prompt（30~100字）。
        """
        template = self.rng.choice(STAGE2_IMPLICIT_TEMPLATES)
        user_prompt = template.format(
            name=agent["name"],
            description=agent["description"],
            keywords="、".join(agent["keywords"]),
            domain=agent["domain"],
            count=batch_size,
        )
        return {"system": SYSTEM_PROMPT_COMPLEX, "user": user_prompt}

    def build_hard_negative_prompt(
        self,
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
        direction: str = "target",
        batch_size: int = 3,
    ) -> Dict[str, str]:
        """
        构建困难负样本Prompt。
        生成一个容易被误判为 confusable_agent 的 target_agent 查询，
        用于强化模型对相似意图的区分能力。

        Args:
            target_agent:     真实目标智能体
            confusable_agent: 易混淆的智能体
            direction:        "target" 表示生成指向target的问题，
                              "negative" 表示生成容易误判为confusable的难样本
            batch_size:       单次生成数量

        Returns:
            {"system": str, "user": str}
        """
        if direction == "target":
            user_prompt = f"""请生成 {batch_size} 条困难负样本，用于训练模型区分两个相似的智能体。

目标智能体（正确答案）：{target_agent["name"]}
  功能：{target_agent["description"]}

易混淆智能体（干扰项）：{confusable_agent["name"]}
  功能：{confusable_agent["description"]}

要求：
1. 每条问题的正确答案是"{target_agent["name"]}"
2. 但问题的表达方式容易让人误以为是"{confusable_agent["name"]}"的功能
3. 问题中要自然地出现"{confusable_agent["name"]}"领域的词汇，但核心需求指向"{target_agent["name"]}"
4. 长度 40~120 字
5. answer 字段只能填写字符串 "{target_agent["name"]}"，不得填写任何其他内容

示例格式：
[
  {{"question": "在飞行计划审核阶段发现航路存在偏航风险，实际飞行中出现绕偏航需求时，该如何开展燃油评估和路径调整？", "answer": "{target_agent["name"]}"}},
  ...
]

输出 {batch_size} 条 JSON 数组，answer 只能是字符串 "{target_agent["name"]}"："""
        else:
            user_prompt = f"""生成 {batch_size} 条边界困难样本，测试模型对以下两个相似智能体的区分能力：

智能体A：{target_agent["name"]}
  功能：{target_agent["description"]}

智能体B（易混淆）：{confusable_agent["name"]}
  功能：{confusable_agent["description"]}

要求：
1. 每条问题的正确答案是"{target_agent["name"]}"
2. 问题描述的场景同时涉及两个智能体的部分功能
3. 但通过仔细分析，核心需求更符合"{target_agent["name"]}"
4. 不要在问题中出现任何智能体名称
5. answer 只能是字符串 "{target_agent["name"]}"，不得填写任何其他内容

输出 {batch_size} 条 JSON 数组，answer 只能是字符串 "{target_agent["name"]}"："""

        return {"system": SYSTEM_PROMPT_COMPLEX, "user": user_prompt}

    def build_batch_prompt(
        self,
        agent: Dict[str, Any],
        count: int,
        style: str = "mixed",
        extra_instructions: str = "",
    ) -> Dict[str, str]:
        """
        构建批量生成Prompt，一次性请求LLM生成多条样本。

        Args:
            agent:              智能体元数据
            count:              本次请求生成的数量
            style:              "simple" | "complex" | "mixed"
            extra_instructions: 额外的生成指令（如针对特定场景）

        Returns:
            {"system": str, "user": str}
        """
        style_desc = {
            "simple": "极简口语短句（每个 8~15 字）",
            "complex": "复杂的、包含背景信息或隐含意图的问题（每个 60~150 字）",
            "mixed": (
                f"其中 {count // 2} 个极简短句（8~15 字）"
                f"和 {count - count // 2} 个复杂问题（60~150 字）"
            ),
        }

        user_prompt = f"""为智能体 "{agent["name"]}" 批量生成 {count} 个训练样本。

智能体功能：{agent["description"]}
核心关键词：{"、".join(agent["keywords"])}
所属领域：{agent["domain"]}

生成要求：
1. 生成 {count} 个不同的用户查询问题
2. 问题类型：{style_desc.get(style, style_desc["mixed"])}
3. 每个问题的表达方式必须不同，覆盖多种使用场景
4. 问题中不能出现智能体名称
5. answer 字段只能填写字符串 "{agent["name"]}"，不得填写任何其他内容
{extra_instructions}

请输出一个JSON数组，格式如下：
[
  {{"question": "问题1", "answer": "{agent["name"]}"}},
  {{"question": "问题2", "answer": "{agent["name"]}"}},
  ...
]

直接输出JSON数组，不要有其他内容："""

        return {"system": SYSTEM_PROMPT_GENERATOR, "user": user_prompt}

    def get_diversity_instructions(self, agent: Dict[str, Any]) -> List[str]:
        """
        为指定智能体生成多样性增强指令列表，
        用于在批量生成时引导LLM覆盖不同的表达维度。

        Returns:
            List[str]: 多样性指令列表
        """
        return [
            f"请特别关注{agent['domain']}行业的专业人员视角，使用行业术语。",
            "请模拟一个不熟悉技术的普通用户，使用日常口语表达。",
            "请模拟一个管理层决策者的视角，关注业务价值和结果。",
            "请模拟一个紧急情况下快速提问的用户，问题简短急迫（8~15字）。",
            "请模拟一个有丰富经验的专家用户，问题包含具体的技术参数。",
            "请模拟一个初次接触该领域的新手用户，问题比较基础。",
            f"请从{agent['domain']}领域的具体业务场景出发，描述一个真实工作中遇到的问题。",
            "请生成一个包含具体数字、时间或地点等具体信息的问题。",
            "请生成一个以'如何'、'怎么'、'能不能'等疑问词开头的问题。",
            "请生成一个描述问题现象而非直接说明需求的问题。",
        ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/agent_router_sft")
    from agents import AGENT_REGISTRY

    builder = PromptBuilder(seed=42)
    agent = AGENT_REGISTRY[0]  # 民航绕偏航处置

    print("=" * 60)
    print("【阶段一 极简短句 Prompt】")
    p = builder.build_stage1_prompt(agent, sample_idx=0, batch_size=5)
    print(f"System: {p['system'][:80]}...")
    print(f"User:\n{p['user']}")

    print("\n" + "=" * 60)
    print("【阶段二-A 口语化 Prompt】")
    p = builder.build_stage2_colloquial_prompt(agent, batch_size=3)
    print(f"User:\n{p['user']}")

    print("\n" + "=" * 60)
    print("【阶段二-B 长文本复杂 Prompt】")
    p = builder.build_stage2_complex_prompt(agent, sample_idx=0, batch_size=3)
    print(f"User:\n{p['user']}")

    print("\n" + "=" * 60)
    print("【阶段二-C 多轮截断 Prompt】")
    p = builder.build_stage2_multiturn_prompt(agent, batch_size=3)
    print(f"User:\n{p['user']}")

    print("\n" + "=" * 60)
    print("【阶段二-D 隐含意图 Prompt】")
    p = builder.build_stage2_implicit_prompt(agent, batch_size=3)
    print(f"User:\n{p['user']}")

    print("\n" + "=" * 60)
    print("【阶段三 困难负样本 Prompt】")
    confusable = next(a for a in AGENT_REGISTRY if a["name"] == "飞行计划审核")
    p = builder.build_hard_negative_prompt(agent, confusable, batch_size=3)
    print(f"User:\n{p['user']}")
