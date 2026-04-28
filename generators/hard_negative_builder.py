"""
hard_negative_builder.py
------------------------
困难负样本（Hard Negatives）专项构造模块。

当智能体数量达到100个时，功能相近的智能体之间必然存在语义边界模糊问题。
本模块专门处理以下场景：

1. 混淆对自动发现（ConfusablePairFinder）
   - 基于agent_registry中的confusable字段构建混淆对图
   - 基于关键词重叠度自动发现潜在混淆对
   - 基于领域内相似度补充发现

2. 困难负样本生成策略（HardNegativeStrategy）
   - 策略A：关键词替换型（用混淆智能体的关键词替换部分词汇）
   - 策略B：场景嫁接型（将目标场景嵌入混淆智能体的上下文）
   - 策略C：反向确认型（先描述混淆智能体的场景，再转向目标需求）
   - 策略D：多需求混合型（同时包含两个智能体的部分需求）

3. 困难负样本质量验证
   - 确保生成的样本确实指向正确的智能体
   - 确保样本具有足够的混淆性（不能太简单）
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────── 混淆对发现 ───────────────────────────

@dataclass
class ConfusablePair:
    """一个混淆对"""
    agent_a: str           # 智能体A名称
    agent_b: str           # 智能体B名称
    source: str            # 发现来源: "registry" | "keyword_overlap" | "domain_similar"
    overlap_score: float   # 相似度分数 (0-1)
    shared_keywords: List[str]  # 共享关键词


class ConfusablePairFinder:
    """
    混淆对自动发现器。
    综合多种策略发现智能体之间的潜在混淆关系。
    """

    def __init__(self, agent_registry: List[Dict[str, Any]]):
        self.agents = agent_registry
        self.agent_map = {a["name"]: a for a in agent_registry}

    def find_all_pairs(
        self,
        keyword_overlap_threshold: float = 0.3,
        max_pairs_per_agent: int = 5,
    ) -> List[ConfusablePair]:
        """
        发现所有混淆对。

        Args:
            keyword_overlap_threshold: 关键词重叠率阈值（超过此值认为存在混淆风险）
            max_pairs_per_agent:       每个智能体最多关联的混淆对数量

        Returns:
            List[ConfusablePair]，已去重
        """
        pairs: Dict[Tuple[str, str], ConfusablePair] = {}

        # 来源1：从registry的confusable字段读取（最高优先级）
        for agent in self.agents:
            for confusable_name in agent.get("confusable", []):
                if confusable_name not in self.agent_map:
                    continue
                key = tuple(sorted([agent["name"], confusable_name]))
                if key not in pairs:
                    pairs[key] = ConfusablePair(
                        agent_a=key[0],
                        agent_b=key[1],
                        source="registry",
                        overlap_score=0.8,
                        shared_keywords=self._find_shared_keywords(
                            agent["name"], confusable_name
                        ),
                    )

        # 来源2：关键词重叠分析
        for i, agent_a in enumerate(self.agents):
            for agent_b in self.agents[i + 1:]:
                key = tuple(sorted([agent_a["name"], agent_b["name"]]))
                if key in pairs:
                    continue

                shared = self._find_shared_keywords(agent_a["name"], agent_b["name"])
                if not shared:
                    continue

                total_keywords = len(
                    set(agent_a["keywords"]) | set(agent_b["keywords"])
                )
                overlap_score = len(shared) / total_keywords if total_keywords > 0 else 0

                if overlap_score >= keyword_overlap_threshold:
                    pairs[key] = ConfusablePair(
                        agent_a=key[0],
                        agent_b=key[1],
                        source="keyword_overlap",
                        overlap_score=overlap_score,
                        shared_keywords=shared,
                    )

        # 来源3：同领域相似度（同领域内的智能体天然容易混淆）
        domain_groups: Dict[str, List[str]] = defaultdict(list)
        for agent in self.agents:
            domain_groups[agent["domain"]].append(agent["name"])

        for domain, names in domain_groups.items():
            for i, name_a in enumerate(names):
                for name_b in names[i + 1:]:
                    key = tuple(sorted([name_a, name_b]))
                    if key in pairs:
                        continue
                    # 同领域内的智能体给予基础混淆分
                    pairs[key] = ConfusablePair(
                        agent_a=key[0],
                        agent_b=key[1],
                        source="domain_similar",
                        overlap_score=0.2,
                        shared_keywords=self._find_shared_keywords(name_a, name_b),
                    )

        # 按overlap_score降序排列，每个智能体限制最大混淆对数量
        all_pairs = sorted(pairs.values(), key=lambda p: -p.overlap_score)
        agent_pair_count: Dict[str, int] = defaultdict(int)
        filtered_pairs = []

        for pair in all_pairs:
            if (
                agent_pair_count[pair.agent_a] < max_pairs_per_agent
                and agent_pair_count[pair.agent_b] < max_pairs_per_agent
            ):
                filtered_pairs.append(pair)
                agent_pair_count[pair.agent_a] += 1
                agent_pair_count[pair.agent_b] += 1

        logger.info(
            f"混淆对发现完成: 共 {len(filtered_pairs)} 对"
            f"（registry={sum(1 for p in filtered_pairs if p.source=='registry')}, "
            f"keyword={sum(1 for p in filtered_pairs if p.source=='keyword_overlap')}, "
            f"domain={sum(1 for p in filtered_pairs if p.source=='domain_similar')}）"
        )
        return filtered_pairs

    def _find_shared_keywords(self, name_a: str, name_b: str) -> List[str]:
        """找出两个智能体的共享关键词"""
        agent_a = self.agent_map.get(name_a)
        agent_b = self.agent_map.get(name_b)
        if not agent_a or not agent_b:
            return []
        set_a = set(agent_a.get("keywords", []))
        set_b = set(agent_b.get("keywords", []))
        return list(set_a & set_b)

    def get_confusion_matrix(self) -> Dict[str, List[str]]:
        """
        生成混淆矩阵：每个智能体 -> 其所有混淆智能体列表。
        用于可视化与分析。
        """
        pairs = self.find_all_pairs()
        matrix: Dict[str, List[str]] = defaultdict(list)
        for pair in pairs:
            matrix[pair.agent_a].append(pair.agent_b)
            matrix[pair.agent_b].append(pair.agent_a)
        return dict(matrix)

    def export_pairs_report(self, output_path: str) -> None:
        """导出混淆对分析报告（JSON格式）"""
        pairs = self.find_all_pairs()
        report = {
            "total_pairs": len(pairs),
            "pairs": [
                {
                    "agent_a": p.agent_a,
                    "agent_b": p.agent_b,
                    "source": p.source,
                    "overlap_score": round(p.overlap_score, 3),
                    "shared_keywords": p.shared_keywords,
                }
                for p in pairs
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"混淆对报告已导出: {output_path}")


# ─────────────────────────── 困难负样本生成策略 ───────────────────────────

class HardNegativeStrategy:
    """
    困难负样本生成策略集合。

    提供多种Prompt构造策略，每种策略从不同角度构造
    容易混淆但答案明确的训练样本。
    """

    @staticmethod
    def strategy_keyword_substitution(
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
    ) -> str:
        """
        策略A：关键词替换型。
        在目标智能体的典型场景中，混入混淆智能体的关键词，
        但核心需求仍然指向目标智能体。
        """
        target_keywords = target_agent.get("keywords", [])[:3]
        confusable_keywords = confusable_agent.get("keywords", [])[:2]

        return f"""请生成一个困难负样本（关键词替换策略）。

目标智能体（正确答案）：{target_agent["name"]}
  功能：{target_agent["description"]}
  核心关键词：{"、".join(target_keywords)}

混淆智能体（干扰项）：{confusable_agent["name"]}
  功能：{confusable_agent["description"]}
  干扰关键词：{"、".join(confusable_keywords)}

生成要求：
1. 问题的核心需求必须指向"{target_agent["name"]}"
2. 但问题中要自然地出现"{confusable_agent["name"]}"的部分关键词（如：{"、".join(confusable_keywords)}）
3. 这些干扰词是问题的背景或次要信息，不是核心需求
4. 问题长度30-80字，表达自然

示例思路：如果目标是"A功能"，混淆是"B功能"，可以写"在B的场景下，我需要用A来处理..."

输出JSON（answer必须是"{target_agent["name"]}"）："""

    @staticmethod
    def strategy_context_grafting(
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
    ) -> str:
        """
        策略B：场景嫁接型。
        将目标智能体的核心需求嵌入混淆智能体的业务上下文中。
        """
        return f"""请生成一个困难负样本（场景嫁接策略）。

目标智能体（正确答案）：{target_agent["name"]}
  功能：{target_agent["description"]}

混淆智能体（背景场景）：{confusable_agent["name"]}
  功能：{confusable_agent["description"]}

生成要求：
1. 以"{confusable_agent["name"]}"的业务场景作为背景（前半部分）
2. 在这个背景下，用户的真实需求是"{target_agent["name"]}"的功能（后半部分）
3. 问题要让人初看以为是"{confusable_agent["name"]}"，但仔细分析后确认是"{target_agent["name"]}"
4. 长度40-100字

输出JSON（answer必须是"{target_agent["name"]}"）："""

    @staticmethod
    def strategy_reverse_confirmation(
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
    ) -> str:
        """
        策略C：反向确认型。
        用户先描述了混淆智能体的场景，然后转向目标智能体的需求。
        """
        return f"""请生成一个困难负样本（反向确认策略）。

目标智能体（正确答案）：{target_agent["name"]}
混淆智能体（误导方向）：{confusable_agent["name"]}

生成要求：
1. 模拟用户先提到了与"{confusable_agent["name"]}"相关的内容
2. 然后通过"但是"、"不过"、"其实"等转折词，表明真实需求是"{target_agent["name"]}"
3. 或者用户在描述完场景后，问的问题实际上需要"{target_agent["name"]}"来回答
4. 长度30-80字，语气自然

输出JSON（answer必须是"{target_agent["name"]}"）："""

    @staticmethod
    def strategy_multi_need_mixed(
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
    ) -> str:
        """
        策略D：多需求混合型。
        用户同时提到了两个智能体的部分需求，但主要需求指向目标智能体。
        """
        return f"""请生成一个困难负样本（多需求混合策略）。

主要目标智能体（正确答案）：{target_agent["name"]}
  功能：{target_agent["description"]}

次要混淆智能体：{confusable_agent["name"]}
  功能：{confusable_agent["description"]}

生成要求：
1. 用户的问题同时涉及两个智能体的功能领域
2. 但主要的、核心的需求是"{target_agent["name"]}"能解决的
3. "{confusable_agent["name"]}"的相关内容只是背景或次要需求
4. 模型需要判断主要需求，选择"{target_agent["name"]}"
5. 长度50-120字

输出JSON（answer必须是"{target_agent["name"]}"）："""

    @staticmethod
    def strategy_implicit_intent(
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
    ) -> str:
        """
        策略E：隐含意图型。
        用户描述的是问题现象，不直接说明需要什么功能，
        但通过推理可以确定是目标智能体。
        """
        return f"""请生成一个困难负样本（隐含意图策略）。

目标智能体（正确答案）：{target_agent["name"]}
  功能：{target_agent["description"]}

混淆智能体（干扰）：{confusable_agent["name"]}
  功能：{confusable_agent["description"]}

生成要求：
1. 用户描述了一个具体的问题现象或工作困境
2. 不直接说明需要什么功能，需要推理才能判断
3. 推理结果应该是"{target_agent["name"]}"，而不是"{confusable_agent["name"]}"
4. 问题中可以包含一些让人误以为是"{confusable_agent["name"]}"的元素
5. 长度40-100字

输出JSON（answer必须是"{target_agent["name"]}"）："""

    STRATEGIES = [
        strategy_keyword_substitution.__func__,
        strategy_context_grafting.__func__,
        strategy_reverse_confirmation.__func__,
        strategy_multi_need_mixed.__func__,
        strategy_implicit_intent.__func__,
    ]

    @classmethod
    def get_prompt(
        cls,
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
        strategy_idx: int = 0,
    ) -> str:
        """
        根据策略索引获取对应的Prompt文本。

        Args:
            target_agent:     目标智能体
            confusable_agent: 混淆智能体
            strategy_idx:     策略索引（0-4循环使用）

        Returns:
            str: Prompt文本
        """
        strategy = cls.STRATEGIES[strategy_idx % len(cls.STRATEGIES)]
        return strategy(target_agent, confusable_agent)


# ─────────────────────────── 困难负样本生成器 ───────────────────────────

class HardNegativeGenerator:
    """
    困难负样本生成器。
    整合混淆对发现与多策略Prompt构造，驱动LLM生成高质量困难负样本。
    """

    SYSTEM_PROMPT = """你是一个专业的AI训练数据生成专家，专门构造困难负样本（Hard Negatives）。

困难负样本是指：表面上容易被误判为其他智能体，但实际上正确答案是指定智能体的训练样本。
这类样本对于提升模型在相似意图之间的区分能力至关重要。

生成规则：
1. 问题必须是真实用户可能发出的自然语言
2. 问题中绝对不能出现任何智能体的名称
3. answer必须严格等于指定的目标智能体名称
4. 问题要有足够的混淆性，但答案必须是明确的
5. 只输出JSON对象，不要有其他内容
"""

    def __init__(
        self,
        llm_client,
        agent_registry: List[Dict[str, Any]],
        output_dir: str,
    ):
        self.client = llm_client
        self.agents = agent_registry
        self.agent_map = {a["name"]: a for a in agent_registry}
        self.output_dir = Path(output_dir)
        self.finder = ConfusablePairFinder(agent_registry)

    def generate_for_pair(
        self,
        pair: ConfusablePair,
        samples_per_direction: int = 25,
        output_file: Optional[Path] = None,
    ) -> List[Dict[str, str]]:
        """
        为一个混淆对生成困难负样本（双向）。

        Args:
            pair:                   混淆对
            samples_per_direction:  每个方向生成的样本数
            output_file:            输出文件路径（None则只返回不写文件）

        Returns:
            List[Dict]: 生成的样本列表
        """
        agent_a = self.agent_map.get(pair.agent_a)
        agent_b = self.agent_map.get(pair.agent_b)
        if not agent_a or not agent_b:
            return []

        all_samples = []

        # 方向1：A -> B的困难样本（正确答案是A，但容易误判为B）
        samples_a = self._generate_one_direction(
            target_agent=agent_a,
            confusable_agent=agent_b,
            count=samples_per_direction,
        )
        all_samples.extend(samples_a)

        # 方向2：B -> A的困难样本（正确答案是B，但容易误判为A）
        samples_b = self._generate_one_direction(
            target_agent=agent_b,
            confusable_agent=agent_a,
            count=samples_per_direction,
        )
        all_samples.extend(samples_b)

        # 写入文件
        if output_file and all_samples:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "a", encoding="utf-8") as f:
                for item in all_samples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return all_samples

    def _generate_one_direction(
        self,
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
        count: int,
    ) -> List[Dict[str, str]]:
        """生成单向困难负样本"""
        from generators.llm_client import JSONExtractor
        samples = []
        strategy_idx = 0

        while len(samples) < count:
            prompt_text = HardNegativeStrategy.get_prompt(
                target_agent=target_agent,
                confusable_agent=confusable_agent,
                strategy_idx=strategy_idx,
            )
            prompt = {
                "system": self.SYSTEM_PROMPT,
                "user": prompt_text,
            }

            response = self.client.call(prompt, expected_answer=target_agent["name"])

            if response.success:
                for item in response.parsed_items:
                    if self._validate_hard_negative(item, target_agent, confusable_agent):
                        samples.append(item)
                        if len(samples) >= count:
                            break
            else:
                logger.debug(f"困难样本生成失败: {response.error}")

            strategy_idx += 1

        return samples[:count]

    def _validate_hard_negative(
        self,
        item: Dict[str, str],
        target_agent: Dict[str, Any],
        confusable_agent: Dict[str, Any],
    ) -> bool:
        """
        验证困难负样本的质量。

        额外验证规则（在基础验证之上）：
        1. 问题不能太简单（长度至少15字）
        2. 问题不能包含目标智能体或混淆智能体的名称
        3. 问题应该与目标智能体的领域相关
        """
        question = item.get("question", "")
        answer = item.get("answer", "")

        if answer != target_agent["name"]:
            return False

        if len(question) < 15:
            return False

        # 不能包含任何智能体名称
        if target_agent["name"] in question or confusable_agent["name"] in question:
            return False

        return True

    def run_full_generation(
        self,
        samples_per_pair: int = 50,
        max_pairs: Optional[int] = None,
        resume: bool = True,
    ) -> int:
        """
        执行全量困难负样本生成。

        Args:
            samples_per_pair: 每个混淆对生成的总样本数（双向合计）
            max_pairs:        最多处理的混淆对数量（None表示全部）
            resume:           是否断点续传

        Returns:
            int: 总生成数量
        """
        pairs = self.finder.find_all_pairs()
        if max_pairs:
            pairs = pairs[:max_pairs]

        logger.info(f"开始困难负样本全量生成，共 {len(pairs)} 个混淆对")

        total_generated = 0
        hard_neg_dir = self.output_dir / "stage3" / "hard_negatives"
        hard_neg_dir.mkdir(parents=True, exist_ok=True)

        for idx, pair in enumerate(pairs):
            pair_filename = f"{self._safe_name(pair.agent_a)}_vs_{self._safe_name(pair.agent_b)}.jsonl"
            output_file = hard_neg_dir / pair_filename

            if resume and output_file.exists():
                existing = self._count_lines(output_file)
                if existing >= samples_per_pair:
                    logger.info(
                        f"[{idx+1}/{len(pairs)}] {pair.agent_a} vs {pair.agent_b}: "
                        f"已有 {existing} 条，跳过"
                    )
                    total_generated += existing
                    continue

            logger.info(
                f"[{idx+1}/{len(pairs)}] {pair.agent_a} vs {pair.agent_b} "
                f"(source={pair.source}, score={pair.overlap_score:.2f})"
            )

            samples = self.generate_for_pair(
                pair=pair,
                samples_per_direction=samples_per_pair // 2,
                output_file=output_file,
            )
            total_generated += len(samples)
            logger.info(f"  生成 {len(samples)} 条困难负样本")

        logger.info(f"\n困难负样本生成完成！总计 {total_generated} 条")
        return total_generated

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace("/", "_").replace("（", "_").replace("）", "_").replace(" ", "_")[:20]

    @staticmethod
    def _count_lines(file_path: Path) -> int:
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/agent_router_sft")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from agents import AGENT_REGISTRY
    from generators.llm_client import LLMClient, LLMClientConfig

    # 测试混淆对发现
    finder = ConfusablePairFinder(AGENT_REGISTRY)
    pairs = finder.find_all_pairs(keyword_overlap_threshold=0.2, max_pairs_per_agent=3)

    print(f"\n发现混淆对总数: {len(pairs)}")
    print("\n前10个混淆对（按相似度排序）:")
    for p in pairs[:10]:
        print(
            f"  [{p.source:15s}] {p.agent_a[:15]:15s} <-> {p.agent_b[:15]:15s} "
            f"score={p.overlap_score:.2f} shared={p.shared_keywords}"
        )

    # 导出报告
    finder.export_pairs_report("/home/ubuntu/agent_router_sft/output/confusable_pairs_report.json")

    # 测试策略生成
    print("\n\n--- 策略A示例Prompt ---")
    agent_a = AGENT_REGISTRY[0]  # 民航绕偏航处置
    agent_b = AGENT_REGISTRY[1]  # 飞行计划审核
    prompt = HardNegativeStrategy.get_prompt(agent_a, agent_b, strategy_idx=0)
    print(prompt[:300] + "...")
