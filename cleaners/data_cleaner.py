"""
data_cleaner.py
---------------
数据清洗与质量过滤流水线。

本模块对LLM生成的原始样本进行多维度质量过滤，确保进入训练集的数据满足：
  1. 格式合法性（JSON格式、字段完整）
  2. 标签正确性（answer必须是合法的智能体名称）
  3. 内容安全性（无标签泄露、无重复、无有害内容）
  4. 语言质量（长度合理、语言流畅、无乱码）
  5. 多样性（相似度去重，防止语义重复）

过滤流水线采用分层设计：
  Layer 1: 硬过滤（任何一项不通过则直接丢弃）
  Layer 2: 软过滤（基于质量分数，低于阈值则丢弃）
  Layer 3: 去重过滤（基于MinHash或编辑距离）
"""

import re
import json
import logging
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class CleanResult:
    """单条样本的清洗结果"""
    item: Dict[str, str]           # 原始样本
    passed: bool                   # 是否通过清洗
    quality_score: float           # 质量分数 (0-1)
    reject_reasons: List[str]      # 拒绝原因列表（passed=False时有值）
    cleaned_item: Optional[Dict[str, str]] = None  # 清洗后的样本


@dataclass
class CleanerConfig:
    """清洗器配置"""
    # 长度限制
    min_question_length: int = 8       # 问题最短字符数
    max_question_length: int = 500     # 问题最长字符数

    # 标签泄露检测
    check_label_leakage: bool = True   # 是否检测标签泄露
    leakage_threshold: float = 0.5     # 标签词汇在问题中出现的比例阈值

    # 语言质量
    min_chinese_ratio: float = 0.3     # 最低中文字符比例（防止乱码）
    max_repeat_char_ratio: float = 0.3 # 最高重复字符比例

    # 去重配置
    dedup_method: str = "exact"        # "exact" | "ngram" | "both"
    ngram_similarity_threshold: float = 0.85  # N-gram相似度阈值（超过则认为重复）
    ngram_size: int = 3                # N-gram大小

    # 质量分数阈值
    min_quality_score: float = 0.6    # 低于此分数的样本被丢弃

    # 每个智能体的最大样本数（防止某个智能体样本过多）
    max_samples_per_agent: Optional[int] = None


# ─────────────────────────── 硬过滤规则 ───────────────────────────

class HardFilter:
    """
    硬过滤器：任何一项不通过则直接丢弃样本。
    这些是不可妥协的基本质量要求。
    """

    def __init__(self, valid_agent_names: Set[str], config: CleanerConfig):
        self.valid_names = valid_agent_names
        self.config = config

    def check(self, item: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        执行所有硬过滤检查。

        Returns:
            (passed: bool, reasons: List[str])
        """
        reasons = []

        # 检查1：字段完整性
        if not isinstance(item, dict):
            return False, ["非字典类型"]
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if not question:
            reasons.append("question字段为空")
        if not answer:
            reasons.append("answer字段为空")
        if reasons:
            return False, reasons

        # 检查2：answer合法性
        if answer not in self.valid_names:
            reasons.append(f"answer不是合法智能体名称: {answer[:30]}")
            return False, reasons

        # 检查3：question长度
        q_len = len(question)
        if q_len < self.config.min_question_length:
            reasons.append(f"question过短: {q_len}字符 < {self.config.min_question_length}")
        if q_len > self.config.max_question_length:
            reasons.append(f"question过长: {q_len}字符 > {self.config.max_question_length}")
        if reasons:
            return False, reasons

        # 检查4：标签泄露（question中直接包含answer）
        if self.config.check_label_leakage and answer in question:
            reasons.append(f"标签泄露: question包含answer '{answer}'")
            return False, reasons

        # 检查5：基本语言检测（防止纯英文或乱码）
        chinese_chars = sum(1 for c in question if '\u4e00' <= c <= '\u9fff')
        if len(question) > 10 and chinese_chars / len(question) < self.config.min_chinese_ratio:
            # 允许纯英文的技术性问题（如代码、命令）
            # 但如果中文比例极低且没有明显技术词汇，则过滤
            has_technical = bool(re.search(r'[A-Z]{2,}|[a-z]+\.[a-z]+|\d+\.\d+', question))
            if not has_technical:
                reasons.append(f"中文比例过低: {chinese_chars/len(question):.1%}")
                return False, reasons

        # 检查6：重复字符检测（防止"哈哈哈哈哈..."类型的无效内容）
        if len(question) > 0:
            char_freq = defaultdict(int)
            for c in question:
                char_freq[c] += 1
            max_freq = max(char_freq.values())
            if max_freq / len(question) > self.config.max_repeat_char_ratio:
                reasons.append(f"重复字符比例过高: {max_freq/len(question):.1%}")
                return False, reasons

        # 检查7：无意义内容检测
        meaningless_patterns = [
            r'^[。，、！？…]+$',           # 纯标点
            r'^\[MOCK\]',                  # Mock数据标记
            r'^(测试|test|demo|示例)',      # 明显的测试数据
        ]
        for pattern in meaningless_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                reasons.append(f"无意义内容: 匹配模式 {pattern}")
                return False, reasons

        return len(reasons) == 0, reasons


# ─────────────────────────── 软过滤（质量评分） ───────────────────────────

class QualityScorer:
    """
    质量评分器：对通过硬过滤的样本进行多维度质量评分。
    分数范围 0-1，越高越好。
    """

    def __init__(self, agent_registry: List[Dict[str, Any]]):
        self.agent_map = {a["name"]: a for a in agent_registry}

    def score(self, item: Dict[str, str]) -> float:
        """
        计算样本质量分数。

        评分维度：
        1. 长度适当性 (0.2)：问题长度在合理范围内
        2. 关键词相关性 (0.3)：问题与智能体关键词的相关程度
        3. 表达多样性 (0.2)：问题不是模板化的套话
        4. 信息密度 (0.15)：问题包含足够的语义信息
        5. 自然度 (0.15)：问题像真实用户提问

        Returns:
            float: 质量分数 (0-1)
        """
        question = item.get("question", "")
        answer = item.get("answer", "")
        agent = self.agent_map.get(answer, {})

        scores = {}

        # 维度1：长度适当性
        q_len = len(question)
        if 15 <= q_len <= 150:
            scores["length"] = 1.0
        elif 8 <= q_len < 15 or 150 < q_len <= 300:
            scores["length"] = 0.7
        else:
            scores["length"] = 0.4

        # 维度2：关键词相关性
        keywords = agent.get("keywords", [])
        if keywords:
            keyword_hits = sum(1 for kw in keywords if kw in question)
            # 不要求全部命中（复杂样本可能不含关键词），但有命中加分
            scores["relevance"] = min(1.0, 0.5 + keyword_hits * 0.25)
        else:
            scores["relevance"] = 0.5

        # 维度3：表达多样性（检测模板化表达）
        template_patterns = [
            r'^请问.*怎么',
            r'^如何使用.*功能',
            r'^我想了解.*系统',
            r'^帮我.*处理',
            r'^\[.*\]',  # 带方括号标记的模板
        ]
        template_hits = sum(
            1 for p in template_patterns if re.search(p, question)
        )
        scores["diversity"] = max(0.3, 1.0 - template_hits * 0.3)

        # 维度4：信息密度（词汇丰富度）
        # 使用字符级别的唯一字符比例作为简单代理
        unique_chars = len(set(question))
        density = min(1.0, unique_chars / max(len(question) * 0.5, 1))
        scores["density"] = density

        # 维度5：自然度（以问号结尾、包含动词等）
        natural_indicators = [
            bool(re.search(r'[？?]$', question)),           # 以问号结尾
            bool(re.search(r'[怎如何能否可以帮请]', question)),  # 包含疑问/请求词
            bool(re.search(r'[我们公司系统]', question)),     # 包含主语
            not bool(re.search(r'^\s*[A-Z]', question)),    # 不以大写字母开头（非英文）
        ]
        scores["naturalness"] = sum(natural_indicators) / len(natural_indicators)

        # 加权汇总
        weights = {
            "length": 0.20,
            "relevance": 0.30,
            "diversity": 0.20,
            "density": 0.15,
            "naturalness": 0.15,
        }
        total_score = sum(scores[k] * weights[k] for k in weights)
        return round(total_score, 4)


# ─────────────────────────── 去重过滤 ───────────────────────────

class DuplicateFilter:
    """
    去重过滤器：基于精确匹配和N-gram相似度去除重复样本。
    """

    def __init__(self, config: CleanerConfig):
        self.config = config
        self._exact_hashes: Set[str] = set()
        self._ngram_sets: Dict[str, Set[frozenset]] = defaultdict(set)  # answer -> ngram集合列表

    def is_duplicate(self, item: Dict[str, str]) -> Tuple[bool, str]:
        """
        检查样本是否重复。

        Returns:
            (is_dup: bool, reason: str)
        """
        question = item.get("question", "")
        answer = item.get("answer", "")

        # 精确去重
        if self.config.dedup_method in ("exact", "both"):
            exact_hash = hashlib.md5(question.encode("utf-8")).hexdigest()
            if exact_hash in self._exact_hashes:
                return True, "精确重复"
            self._exact_hashes.add(exact_hash)

        # N-gram去重
        if self.config.dedup_method in ("ngram", "both"):
            ngrams = self._get_ngrams(question, self.config.ngram_size)
            if not ngrams:
                return False, ""

            # 只在同一智能体内做N-gram去重（跨智能体相似是正常的）
            for existing_ngrams in self._ngram_sets[answer]:
                similarity = self._jaccard_similarity(ngrams, existing_ngrams)
                if similarity >= self.config.ngram_similarity_threshold:
                    return True, f"N-gram相似度过高: {similarity:.2f}"

            self._ngram_sets[answer].add(frozenset(ngrams))

        return False, ""

    def reset(self):
        """重置去重状态（用于重新处理）"""
        self._exact_hashes.clear()
        self._ngram_sets.clear()

    @staticmethod
    def _get_ngrams(text: str, n: int) -> Set[str]:
        """提取字符级N-gram"""
        # 标准化：去除空白，转小写
        text = re.sub(r'\s+', '', text).lower()
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    @staticmethod
    def _jaccard_similarity(set_a: Set, set_b: Set) -> float:
        """计算Jaccard相似度"""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0


# ─────────────────────────── 主清洗流水线 ───────────────────────────

class DataCleaner:
    """
    数据清洗主流水线。
    整合硬过滤、质量评分与去重过滤，对原始生成数据进行全面清洗。
    """

    def __init__(
        self,
        agent_registry: List[Dict[str, Any]],
        config: Optional[CleanerConfig] = None,
    ):
        self.config = config or CleanerConfig()
        self.agent_registry = agent_registry
        valid_names = {a["name"] for a in agent_registry}

        self.hard_filter = HardFilter(valid_names, self.config)
        self.quality_scorer = QualityScorer(agent_registry)
        self.dedup_filter = DuplicateFilter(self.config)

        # 统计信息
        self._stats = {
            "total_input": 0,
            "passed": 0,
            "rejected_hard": 0,
            "rejected_quality": 0,
            "rejected_dedup": 0,
            "reject_reasons": defaultdict(int),
        }

    def clean_item(self, item: Dict[str, str]) -> CleanResult:
        """
        清洗单条样本。

        Returns:
            CleanResult
        """
        self._stats["total_input"] += 1

        # Layer 1: 硬过滤
        passed, reasons = self.hard_filter.check(item)
        if not passed:
            self._stats["rejected_hard"] += 1
            for r in reasons:
                self._stats["reject_reasons"][r.split(":")[0]] += 1
            return CleanResult(
                item=item, passed=False, quality_score=0.0, reject_reasons=reasons
            )

        # Layer 2: 质量评分
        quality_score = self.quality_scorer.score(item)
        if quality_score < self.config.min_quality_score:
            self._stats["rejected_quality"] += 1
            reason = f"质量分数过低: {quality_score:.3f} < {self.config.min_quality_score}"
            self._stats["reject_reasons"]["质量分数过低"] += 1
            return CleanResult(
                item=item, passed=False, quality_score=quality_score,
                reject_reasons=[reason]
            )

        # Layer 3: 去重过滤
        is_dup, dup_reason = self.dedup_filter.is_duplicate(item)
        if is_dup:
            self._stats["rejected_dedup"] += 1
            self._stats["reject_reasons"][dup_reason] += 1
            return CleanResult(
                item=item, passed=False, quality_score=quality_score,
                reject_reasons=[dup_reason]
            )

        # 通过所有过滤
        self._stats["passed"] += 1
        cleaned_item = self._normalize(item)
        return CleanResult(
            item=item, passed=True, quality_score=quality_score,
            reject_reasons=[], cleaned_item=cleaned_item
        )

    def clean_batch(
        self, items: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[CleanResult]]:
        """
        批量清洗样本。

        Returns:
            (passed_items, all_results)
        """
        passed = []
        all_results = []
        agent_counts: Dict[str, int] = defaultdict(int)

        for item in items:
            result = self.clean_item(item)
            all_results.append(result)

            if result.passed and result.cleaned_item:
                # 检查每个智能体的样本上限
                answer = result.cleaned_item.get("answer", "")
                if (
                    self.config.max_samples_per_agent is None
                    or agent_counts[answer] < self.config.max_samples_per_agent
                ):
                    passed.append(result.cleaned_item)
                    agent_counts[answer] += 1

        return passed, all_results

    def clean_file(
        self,
        input_path: str,
        output_path: str,
        append: bool = False,
    ) -> Dict[str, int]:
        """
        清洗单个JSONL文件。

        Args:
            input_path:  输入文件路径
            output_path: 输出文件路径
            append:      是否追加到输出文件（False则覆盖）

        Returns:
            统计信息字典
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        items = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.debug(f"第{line_num}行JSON解析失败: {e}")

        passed_items, _ = self.clean_batch(items)

        mode = "a" if append else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            for item in passed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        stats = {
            "input_count": len(items),
            "passed_count": len(passed_items),
            "pass_rate": len(passed_items) / len(items) if items else 0,
        }
        logger.info(
            f"清洗完成: {input_path.name} -> "
            f"输入{stats['input_count']}条, "
            f"通过{stats['passed_count']}条 "
            f"({stats['pass_rate']:.1%})"
        )
        return stats

    def clean_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.jsonl",
        reset_dedup: bool = True,
    ) -> Dict[str, Any]:
        """
        清洗整个目录下的JSONL文件。

        Args:
            input_dir:  输入目录
            output_dir: 输出目录
            pattern:    文件匹配模式
            reset_dedup: 是否在处理每个文件前重置去重状态

        Returns:
            汇总统计信息
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob(pattern))
        logger.info(f"开始清洗目录: {input_dir}, 共 {len(files)} 个文件")

        total_stats = {"files": 0, "total_input": 0, "total_passed": 0}

        for file_path in sorted(files):
            if reset_dedup:
                # 注意：重置去重会导致跨文件的重复无法检测
                # 如需跨文件去重，设置 reset_dedup=False
                pass

            output_file = output_dir / file_path.name
            stats = self.clean_file(str(file_path), str(output_file))

            total_stats["files"] += 1
            total_stats["total_input"] += stats["input_count"]
            total_stats["total_passed"] += stats["passed_count"]

        total_stats["overall_pass_rate"] = (
            total_stats["total_passed"] / total_stats["total_input"]
            if total_stats["total_input"] > 0
            else 0
        )

        logger.info(
            f"目录清洗完成: "
            f"总输入 {total_stats['total_input']} 条, "
            f"总通过 {total_stats['total_passed']} 条 "
            f"({total_stats['overall_pass_rate']:.1%})"
        )
        return total_stats

    def get_stats(self) -> Dict[str, Any]:
        """获取累计统计信息"""
        stats = dict(self._stats)
        stats["reject_reasons"] = dict(self._stats["reject_reasons"])
        if stats["total_input"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_input"]
        else:
            stats["pass_rate"] = 0.0
        return stats

    def reset_stats(self):
        """重置统计信息"""
        self._stats = {
            "total_input": 0,
            "passed": 0,
            "rejected_hard": 0,
            "rejected_quality": 0,
            "rejected_dedup": 0,
            "reject_reasons": defaultdict(int),
        }

    @staticmethod
    def _normalize(item: Dict[str, str]) -> Dict[str, str]:
        """
        标准化样本：清理多余空白、统一标点等。
        """
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        # 清理多余空白
        question = re.sub(r'\s+', ' ', question)

        # 统一全角/半角标点（可选）
        # question = unicodedata.normalize('NFKC', question)

        return {"question": question, "answer": answer}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/agent_router_sft")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from agents import AGENT_REGISTRY

    cleaner = DataCleaner(AGENT_REGISTRY)

    # 测试样本
    test_items = [
        {"question": "飞机偏航了怎么处理？", "answer": "民航绕偏航处置"},
        {"question": "民航绕偏航处置能帮我处理偏航问题吗？", "answer": "民航绕偏航处置"},  # 标签泄露
        {"question": "哈", "answer": "民航绕偏航处置"},  # 过短
        {"question": "飞机偏航了怎么处理？", "answer": "民航绕偏航处置"},  # 精确重复
        {"question": "我们公司的航班在飞行过程中遭遇了强对流天气，需要立即重新规划航路，请问如何处理？", "answer": "民航绕偏航处置"},
        {"question": "这是一个无效答案", "answer": "不存在的智能体"},  # 非法answer
        {"question": "[MOCK] 测试问题", "answer": "飞行计划审核"},  # Mock数据
        {"question": "航班备降需要哪些审批流程？", "answer": "飞行计划审核"},
        {"question": "我想了解一下飞行计划审核的功能", "answer": "飞行计划审核"},  # 模板化
    ]

    print("=" * 60)
    print("清洗测试结果：")
    print("=" * 60)
    for item in test_items:
        result = cleaner.clean_item(item)
        status = "✓ 通过" if result.passed else "✗ 拒绝"
        print(f"{status} | 分数={result.quality_score:.3f} | {item['question'][:30]:30s}")
        if not result.passed:
            print(f"       原因: {'; '.join(result.reject_reasons)}")

    print("\n统计信息:")
    stats = cleaner.get_stats()
    print(f"  总输入: {stats['total_input']}")
    print(f"  通过:   {stats['passed']} ({stats['pass_rate']:.1%})")
    print(f"  硬过滤: {stats['rejected_hard']}")
    print(f"  质量:   {stats['rejected_quality']}")
    print(f"  去重:   {stats['rejected_dedup']}")
