"""
test_single_agent.py
--------------------
对单个智能体"民航绕偏航处置"进行完整的三阶段生成 + 清洗测试。
使用真实 LLM 调用，展示实际生成样本与清洗结果。
"""

import sys
import os
import json
import logging
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("test_single_agent")


# ─────────────────────────── 直接调用 OpenAI ───────────────────────────

def call_llm(prompt: str, n_samples: int = 10, model: str = "gpt-4.1-mini") -> list:
    """直接调用 OpenAI 兼容接口生成样本"""
    from openai import OpenAI

    client = OpenAI()  # 自动读取环境变量中的 API Key 和 Base URL

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个专业的训练数据生成助手。"
                    "请严格按照用户要求的JSON格式输出，不要添加任何额外说明。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.92,
        max_tokens=2048,
        top_p=0.95,
    )

    raw_text = response.choices[0].message.content.strip()
    usage = response.usage

    return raw_text, usage


def extract_json_items(raw_text: str) -> list:
    """从 LLM 响应中提取 JSON 数组"""
    import re

    # 尝试直接解析
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # 提取 markdown 代码块中的 JSON
    code_block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except Exception:
            pass

    # 提取裸 JSON 数组
    array_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    return []


# ─────────────────────────── Prompt 构造 ───────────────────────────

AGENT_META = {
    "name": "民航绕偏航处置",
    "description": "专门处理民用航空器在飞行过程中发生偏航、需要紧急绕行或重新规划航路的场景。包括偏航原因分析、备降机场选择、绕行路径规划、燃油评估、与ATC协调、乘客通报等全流程处置。",
    "keywords": ["偏航", "绕行", "备降", "航路重规划", "ATC协调", "燃油评估", "紧急改航", "飞机改航", "航班绕飞"],
    "domain": "交通运输 / 民航",
}

CONFUSABLE_AGENT = "飞行计划审核"


def build_stage1_prompt(agent: dict, n: int = 10) -> str:
    return f"""请为智能体「{agent['name']}」生成 {n} 条极简口语训练样本。

智能体功能描述：{agent['description']}

要求：
1. 每条样本模拟用户随口发出的极短指令或问题
2. 字数严格控制在 8~15 字之间，越短越好
3. 口语化、自然，像日常对话中随口一问
4. 覆盖不同的细分场景关键词（偏航、绕行、备降、改航、ATC等）
5. 问题中不能直接出现智能体名称「{agent['name']}」
6. "answer" 字段必须固定填写字符串 "{agent['name']}"，不得填写任何其他内容

参考示例（注意这种极简风格）：
[
  {{"question": "航班偏航了怎么办", "answer": "{agent['name']}"}},
  {{"question": "查下航线偏离", "answer": "{agent['name']}"}},
  {{"question": "飞机绕航申请", "answer": "{agent['name']}"}},
  {{"question": "备降机场怎么选", "answer": "{agent['name']}"}},
  ...
]

请严格按照上述格式输出 JSON 数组，answer 只能是 "{agent['name']}"："""


def build_stage2_simple_prompt(agent: dict, n: int = 5) -> str:
    return f"""请为智能体「{agent['name']}」生成 {n} 条口语化、非正式的用户查询样本。

智能体功能：{agent['description']}

要求：
1. 模拟真实用户的口语化表达，可以有语气词、不完整句子
2. 表达不那么正式，像是在聊天或随口一问
3. 意图仍然指向该智能体的核心功能
4. 长度 20~50 字
5. "answer" 字段必须固定填写字符串 "{agent['name']}"，不得填写任何其他内容

示例格式：
[
  {{"question": "哎，飞机偏航了，最近有什么备降机场推荐吗？", "answer": "{agent['name']}"}},
  ...
]

请严格按照上述格式输出 JSON 数组，answer 只能是字符串 "{agent['name']}"："""


def build_stage2_complex_prompt(agent: dict, n: int = 5) -> str:
    return f"""请为智能体「{agent['name']}」生成 {n} 条复杂的用户查询样本。

智能体功能：{agent['description']}

要求：
1. 包含丰富的背景信息，描述具体场景（长度 80~200 字）
2. 混合多种表达方式：专业术语 + 通俗描述
3. 可以包含多个子问题或多步骤需求
4. 意图最终指向该智能体的核心功能
5. 体现真实运营场景的复杂性
6. "answer" 字段必须固定填写字符串 "{agent['name']}"，不得填写任何其他内容

示例格式：
[
  {{"question": "我们在巡航高度过程中突然发生了强侧风导致偏航，现需紧急调整航线绕过受影响的空域，同时评估附近备降机场的适航性和当前剩余燃油是否支持备降。请帮忙分析偏航原因，规划绕行方案并协调ATC。", "answer": "{agent['name']}"}},
  ...
]

请严格按照上述格式输出 JSON 数组，answer 只能是字符串 "{agent['name']}"："""


def build_stage2_multiturn_prompt(agent: dict, n: int = 3) -> str:
    return f"""请为智能体「{agent['name']}」生成 {n} 条"多轮对话截断"样本。

智能体功能：{agent['description']}

要求：
1. 模拟从一段对话中间截取的片段，用户的问题依赖前文上下文
2. 问题本身可能不完整，但结合上下文能判断出意图
3. 可以包含类似"刚才说的那个问题"、"继续上面的情况"等引用前文的表达
4. 长度 20~80 字
5. "answer" 字段必须固定填写字符串 "{agent['name']}"，不得填写任何其他内容

示例格式：
[
  {{"question": "刚才说的那个偏航导致的燃油紧张问题，备降机场选择上还有什么建议？", "answer": "{agent['name']}"}},
  ...
]

请严格按照上述格式输出 JSON 数组，answer 只能是字符串 "{agent['name']}"："""


def build_stage2_implicit_prompt(agent: dict, n: int = 3) -> str:
    return f"""请为智能体「{agent['name']}」生成 {n} 条"隐含意图"样本。

智能体功能：{agent['description']}

要求：
1. 用户描述的是一种现象或状态，而不是直接说明需求
2. 需要推理才能判断用户真正需要什么智能体
3. 不直接使用该智能体的关键词，但场景明确指向其功能
4. 长度 30~100 字

JSON 数组格式，字段同上（question + answer）："""


def build_hard_negative_prompt(agent_a: str, agent_b: str, desc_a: str, n: int = 5) -> str:
    return f"""请生成 {n} 条"困难负样本"，用于区分「{agent_a}」和「{agent_b}」这两个相似的智能体。

「{agent_a}」功能：{desc_a}

要求：
1. 生成的问题表面上与「{agent_b}」相关，但核心需求实际上指向「{agent_a}」
2. 问题中可以同时出现两个智能体的关键词，但主要意图明确属于「{agent_a}」
3. 这类样本专门用于让模型学会区分两者的细微差异
4. 长度 40~150 字

JSON 数组格式，字段同上（question 和 answer 固定为 "{agent_a}"）："""


# ─────────────────────────── 清洗函数 ───────────────────────────

def simple_clean(items: list, agent_name: str) -> tuple:
    """简单清洗：字段检查 + 标签泄露 + 长度过滤"""
    passed, rejected = [], []

    for item in items:
        reason = None

        # 字段检查
        if not isinstance(item, dict):
            reason = "非字典类型"
        elif "question" not in item or "answer" not in item:
            reason = "缺少字段"
        elif not item["question"].strip():
            reason = "问题为空"
        elif item["answer"].strip() != agent_name:
            reason = f"answer错误: {item.get('answer')}"
        # 长度检查
        elif len(item["question"]) < 8:
            reason = "问题过短(<8字)"
        elif len(item["question"]) > 500:
            reason = "问题过长(>500字)"
        # 标签泄露检测
        elif agent_name in item["question"]:
            reason = "标签泄露（问题含智能体名）"
        # 中文比例
        else:
            chinese_chars = sum(1 for c in item["question"] if "\u4e00" <= c <= "\u9fff")
            if len(item["question"]) > 0 and chinese_chars / len(item["question"]) < 0.2:
                reason = "中文比例过低"

        if reason:
            rejected.append({"item": item, "reason": reason})
        else:
            passed.append(item)

    return passed, rejected


# ─────────────────────────── 主测试流程 ───────────────────────────

def run_test():
    agent = AGENT_META
    model = "gpt-4.1-mini"

    print("\n" + "=" * 70)
    print(f"  智能体测试：{agent['name']}")
    print(f"  模型：{model}")
    print("=" * 70)

    all_generated = []
    total_tokens = 0

    # ── 阶段一：简单样本 ──
    print("\n【阶段一】冷启动 - 简单直接样本（目标10条）")
    print("-" * 50)
    prompt = build_stage1_prompt(agent, n=10)
    raw, usage = call_llm(prompt, model=model)
    items = extract_json_items(raw)
    total_tokens += usage.total_tokens
    print(f"  原始响应 Token: {usage.total_tokens}，解析出 {len(items)} 条")
    for i, item in enumerate(items, 1):
        print(f"  [{i:02d}] {item.get('question', '?')}")
    all_generated.extend(items)

    # ── 阶段二：口语化样本 ──
    print("\n【阶段二-A】生产泛化 - 口语化样本（目标5条）")
    print("-" * 50)
    prompt = build_stage2_simple_prompt(agent, n=5)
    raw, usage = call_llm(prompt, model=model)
    items = extract_json_items(raw)
    total_tokens += usage.total_tokens
    print(f"  原始响应 Token: {usage.total_tokens}，解析出 {len(items)} 条")
    for i, item in enumerate(items, 1):
        print(f"  [{i:02d}] {item.get('question', '?')}")
    all_generated.extend(items)

    # ── 阶段二：长文本复杂样本 ──
    print("\n【阶段二-B】生产泛化 - 长文本复杂样本（目标5条）")
    print("-" * 50)
    prompt = build_stage2_complex_prompt(agent, n=5)
    raw, usage = call_llm(prompt, model=model)
    items = extract_json_items(raw)
    total_tokens += usage.total_tokens
    print(f"  原始响应 Token: {usage.total_tokens}，解析出 {len(items)} 条")
    for i, item in enumerate(items, 1):
        q = item.get("question", "?")
        print(f"  [{i:02d}] {q[:80]}{'...' if len(q) > 80 else ''}")
    all_generated.extend(items)

    # ── 阶段二：多轮对话截断 ──
    print("\n【阶段二-C】生产泛化 - 多轮对话截断样本（目标3条）")
    print("-" * 50)
    prompt = build_stage2_multiturn_prompt(agent, n=3)
    raw, usage = call_llm(prompt, model=model)
    items = extract_json_items(raw)
    total_tokens += usage.total_tokens
    print(f"  原始响应 Token: {usage.total_tokens}，解析出 {len(items)} 条")
    for i, item in enumerate(items, 1):
        print(f"  [{i:02d}] {item.get('question', '?')}")
    all_generated.extend(items)

    # ── 阶段二：隐含意图 ──
    print("\n【阶段二-D】生产泛化 - 隐含意图样本（目标3条）")
    print("-" * 50)
    prompt = build_stage2_implicit_prompt(agent, n=3)
    raw, usage = call_llm(prompt, model=model)
    items = extract_json_items(raw)
    total_tokens += usage.total_tokens
    print(f"  原始响应 Token: {usage.total_tokens}，解析出 {len(items)} 条")
    for i, item in enumerate(items, 1):
        print(f"  [{i:02d}] {item.get('question', '?')}")
    all_generated.extend(items)

    # ── 阶段三：困难负样本 ──
    print(f"\n【阶段三】困难负样本 - 区分「{agent['name']}」vs「{CONFUSABLE_AGENT}」（目标5条）")
    print("-" * 50)
    prompt = build_hard_negative_prompt(
        agent_a=agent["name"],
        agent_b=CONFUSABLE_AGENT,
        desc_a=agent["description"],
        n=5,
    )
    raw, usage = call_llm(prompt, model=model)
    items = extract_json_items(raw)
    total_tokens += usage.total_tokens
    print(f"  原始响应 Token: {usage.total_tokens}，解析出 {len(items)} 条")
    for i, item in enumerate(items, 1):
        q = item.get("question", "?")
        print(f"  [{i:02d}] {q[:100]}{'...' if len(q) > 100 else ''}")
    all_generated.extend(items)

    # ── 清洗 ──
    print("\n" + "=" * 70)
    print("  数据清洗结果")
    print("=" * 70)
    passed, rejected = simple_clean(all_generated, agent["name"])
    print(f"  原始生成: {len(all_generated)} 条")
    print(f"  通过清洗: {len(passed)} 条  ({len(passed)/max(len(all_generated),1)*100:.1f}%)")
    print(f"  被过滤:   {len(rejected)} 条")
    if rejected:
        print("\n  被过滤样本原因：")
        for r in rejected:
            print(f"    - [{r['reason']}] {str(r['item'].get('question','?'))[:60]}")

    # ── 统计 ──
    print("\n" + "=" * 70)
    print("  统计摘要")
    print("=" * 70)
    if passed:
        lengths = [len(item["question"]) for item in passed]
        print(f"  总 Token 消耗: {total_tokens:,}")
        print(f"  通过样本数:    {len(passed)} 条")
        print(f"  问题长度:      最短 {min(lengths)} 字 / 最长 {max(lengths)} 字 / 均值 {sum(lengths)//len(lengths)} 字")

    # ── 保存结果 ──
    out_dir = Path("output/test_single_agent")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_generated.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for item in all_generated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    clean_path = out_dir / "cleaned.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for item in passed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  原始数据已保存: {raw_path}")
    print(f"  清洗数据已保存: {clean_path}")
    print("\n" + "=" * 70)

    return passed


if __name__ == "__main__":
    run_test()
