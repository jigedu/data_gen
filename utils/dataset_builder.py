"""
dataset_builder.py
------------------
数据集合并、统计分析与多格式导出模块。

功能：
1. DatasetMerger：合并三个阶段的清洗后数据，进行全局去重与配额均衡
2. DatasetAnalyzer：对最终数据集进行统计分析，生成质量报告
3. DatasetExporter：将数据集导出为多种格式（JSONL、JSON、CSV、SFT格式）
4. TrainTestSplitter：按比例划分训练集、验证集与测试集
"""

import json
import csv
import random
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────── 数据集合并 ───────────────────────────

@dataclass
class MergeConfig:
    """合并配置"""
    # 各阶段权重（用于配额分配）
    stage1_weight: float = 0.15    # 冷启动样本占比
    stage2_weight: float = 0.70    # 生产泛化样本占比
    stage3_weight: float = 0.15    # 困难负样本占比

    # 每个智能体的目标样本数
    target_per_agent: int = 1000   # 最终每个智能体的目标数量

    # 全局最大样本数
    max_total: Optional[int] = None  # None表示不限制

    # 随机种子
    seed: int = 42

    # 是否进行全局去重
    global_dedup: bool = True


class DatasetMerger:
    """
    数据集合并器。
    将三个阶段的清洗后数据按配置合并为最终训练集。
    """

    def __init__(self, config: MergeConfig, agent_registry: List[Dict[str, Any]]):
        self.config = config
        self.agents = agent_registry
        self.agent_names = {a["name"] for a in agent_registry}
        self.rng = random.Random(config.seed)

    def merge_from_dirs(
        self,
        stage1_dir: str,
        stage2_dir: str,
        stage3_dir: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """
        从三个阶段的目录读取数据并合并。

        Args:
            stage1_dir:  阶段一清洗后数据目录
            stage2_dir:  阶段二清洗后数据目录
            stage3_dir:  阶段三清洗后数据目录
            output_path: 合并后输出文件路径

        Returns:
            合并统计信息
        """
        # 读取各阶段数据
        stage1_data = self._load_from_dir(stage1_dir, tag="stage1")
        stage2_data = self._load_from_dir(stage2_dir, tag="stage2")
        stage3_data = self._load_from_dir(stage3_dir, tag="stage3")

        logger.info(
            f"各阶段数据量: stage1={len(stage1_data)}, "
            f"stage2={len(stage2_data)}, stage3={len(stage3_data)}"
        )

        # 按智能体分组
        stage1_by_agent = self._group_by_agent(stage1_data)
        stage2_by_agent = self._group_by_agent(stage2_data)
        stage3_by_agent = self._group_by_agent(stage3_data)

        # 合并与配额控制
        merged = []
        agent_stats = {}

        for agent_name in sorted(self.agent_names):
            s1 = stage1_by_agent.get(agent_name, [])
            s2 = stage2_by_agent.get(agent_name, [])
            s3 = stage3_by_agent.get(agent_name, [])

            # 计算各阶段配额
            target = self.config.target_per_agent
            s1_quota = int(target * self.config.stage1_weight)
            s2_quota = int(target * self.config.stage2_weight)
            s3_quota = target - s1_quota - s2_quota

            # 随机采样（不足则全取）
            s1_sampled = self._sample(s1, s1_quota)
            s2_sampled = self._sample(s2, s2_quota)
            s3_sampled = self._sample(s3, s3_quota)

            agent_samples = s1_sampled + s2_sampled + s3_sampled
            self.rng.shuffle(agent_samples)

            merged.extend(agent_samples)
            agent_stats[agent_name] = {
                "stage1": len(s1_sampled),
                "stage2": len(s2_sampled),
                "stage3": len(s3_sampled),
                "total": len(agent_samples),
            }

        # 全局去重
        if self.config.global_dedup:
            before_dedup = len(merged)
            merged = self._global_dedup(merged)
            logger.info(f"全局去重: {before_dedup} -> {len(merged)} 条")

        # 全局打乱
        self.rng.shuffle(merged)

        # 限制总量
        if self.config.max_total and len(merged) > self.config.max_total:
            merged = merged[:self.config.max_total]

        # 写出
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for item in merged:
                # 写出时只保留question和answer字段
                clean_item = {"question": item["question"], "answer": item["answer"]}
                f.write(json.dumps(clean_item, ensure_ascii=False) + "\n")

        stats = {
            "total_merged": len(merged),
            "agent_stats": agent_stats,
            "output_path": str(output_path),
        }
        logger.info(f"合并完成: 共 {len(merged)} 条样本 -> {output_path}")
        return stats

    def _load_from_dir(self, dir_path: str, tag: str = "") -> List[Dict]:
        """从目录加载所有JSONL文件"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"目录不存在: {dir_path}")
            return []

        items = []
        for file_path in sorted(dir_path.glob("*.jsonl")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if tag:
                            item["_stage"] = tag
                        items.append(item)
                    except json.JSONDecodeError:
                        pass
        return items

    def _group_by_agent(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """按answer字段分组"""
        groups = defaultdict(list)
        for item in items:
            answer = item.get("answer", "")
            if answer in self.agent_names:
                groups[answer].append(item)
        return dict(groups)

    def _sample(self, items: List[Dict], n: int) -> List[Dict]:
        """随机采样n条，不足则全取"""
        if len(items) <= n:
            return list(items)
        return self.rng.sample(items, n)

    @staticmethod
    def _global_dedup(items: List[Dict]) -> List[Dict]:
        """全局精确去重"""
        seen = set()
        unique = []
        for item in items:
            key = hashlib.md5(item.get("question", "").encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique


# ─────────────────────────── 数据集分析 ───────────────────────────

class DatasetAnalyzer:
    """
    数据集统计分析器。
    生成数据集的质量报告与分布分析。
    """

    def __init__(self, agent_registry: List[Dict[str, Any]]):
        self.agents = agent_registry
        self.agent_map = {a["name"]: a for a in agent_registry}

    def analyze(self, data_path: str) -> Dict[str, Any]:
        """
        分析数据集，返回完整的统计报告。

        Args:
            data_path: JSONL数据文件路径

        Returns:
            统计报告字典
        """
        items = self._load_jsonl(data_path)
        if not items:
            return {"error": "数据集为空"}

        report = {}

        # 基础统计
        report["total_samples"] = len(items)
        report["unique_agents"] = len(set(item.get("answer", "") for item in items))

        # 每个智能体的样本分布
        agent_counts = Counter(item.get("answer", "") for item in items)
        report["agent_distribution"] = {
            "counts": dict(agent_counts),
            "min": min(agent_counts.values()) if agent_counts else 0,
            "max": max(agent_counts.values()) if agent_counts else 0,
            "mean": sum(agent_counts.values()) / len(agent_counts) if agent_counts else 0,
            "std": self._std(list(agent_counts.values())),
        }

        # 缺失智能体（有注册但无样本）
        all_agent_names = {a["name"] for a in self.agents}
        covered = set(agent_counts.keys())
        report["missing_agents"] = list(all_agent_names - covered)
        report["coverage_rate"] = len(covered) / len(all_agent_names) if all_agent_names else 0

        # 问题长度分布
        lengths = [len(item.get("question", "")) for item in items]
        report["question_length"] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "std": round(self._std(lengths), 1),
            "distribution": {
                "short(<=20)": sum(1 for l in lengths if l <= 20),
                "medium(21-80)": sum(1 for l in lengths if 21 <= l <= 80),
                "long(81-200)": sum(1 for l in lengths if 81 <= l <= 200),
                "very_long(>200)": sum(1 for l in lengths if l > 200),
            },
        }

        # 领域分布
        domain_counts = defaultdict(int)
        for item in items:
            answer = item.get("answer", "")
            agent = self.agent_map.get(answer, {})
            domain = agent.get("domain", "未知")
            domain_counts[domain] += 1
        report["domain_distribution"] = dict(domain_counts)

        # 数据质量指标
        report["quality_metrics"] = {
            "avg_question_length": round(sum(lengths) / len(lengths), 1),
            "short_sample_ratio": sum(1 for l in lengths if l <= 15) / len(lengths),
            "long_sample_ratio": sum(1 for l in lengths if l > 100) / len(lengths),
            "balance_score": self._balance_score(list(agent_counts.values())),
        }

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """格式化打印统计报告"""
        print("\n" + "=" * 70)
        print("数据集统计报告")
        print("=" * 70)
        print(f"总样本数:        {report.get('total_samples', 0):,}")
        print(f"覆盖智能体数:    {report.get('unique_agents', 0)}/100")
        print(f"覆盖率:          {report.get('coverage_rate', 0):.1%}")

        dist = report.get("agent_distribution", {})
        print(f"\n每智能体样本分布:")
        print(f"  最少: {dist.get('min', 0)} 条")
        print(f"  最多: {dist.get('max', 0)} 条")
        print(f"  均值: {dist.get('mean', 0):.1f} 条")
        print(f"  标准差: {dist.get('std', 0):.1f}")

        ql = report.get("question_length", {})
        print(f"\n问题长度分布:")
        print(f"  最短: {ql.get('min', 0)} 字符")
        print(f"  最长: {ql.get('max', 0)} 字符")
        print(f"  均值: {ql.get('mean', 0)} 字符")
        for k, v in ql.get("distribution", {}).items():
            total = report.get("total_samples", 1)
            print(f"  {k}: {v} 条 ({v/total:.1%})")

        print(f"\n领域分布:")
        for domain, count in sorted(
            report.get("domain_distribution", {}).items(),
            key=lambda x: -x[1]
        ):
            total = report.get("total_samples", 1)
            print(f"  {domain}: {count} 条 ({count/total:.1%})")

        qm = report.get("quality_metrics", {})
        print(f"\n质量指标:")
        print(f"  均衡度分数: {qm.get('balance_score', 0):.3f} (越接近1越均衡)")
        print(f"  短样本比例: {qm.get('short_sample_ratio', 0):.1%}")
        print(f"  长样本比例: {qm.get('long_sample_ratio', 0):.1%}")

        missing = report.get("missing_agents", [])
        if missing:
            print(f"\n⚠ 缺失样本的智能体 ({len(missing)} 个):")
            for name in missing[:10]:
                print(f"  - {name}")
            if len(missing) > 10:
                print(f"  ... 还有 {len(missing)-10} 个")

        print("=" * 70)

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """保存报告为JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"统计报告已保存: {output_path}")

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return items

    @staticmethod
    def _std(values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _balance_score(counts: List[int]) -> float:
        """
        计算类别均衡度分数（基于Gini系数的反转）。
        1.0 表示完全均衡，0.0 表示极度不均衡。
        """
        if not counts or sum(counts) == 0:
            return 0.0
        total = sum(counts)
        n = len(counts)
        if n == 1:
            return 1.0
        proportions = sorted(c / total for c in counts)
        gini = sum(
            (2 * (i + 1) - n - 1) * p
            for i, p in enumerate(proportions)
        ) / (n * sum(proportions))
        return round(1 - abs(gini), 4)


# ─────────────────────────── 数据集导出 ───────────────────────────

class DatasetExporter:
    """
    数据集多格式导出器。
    支持导出为JSONL、JSON、CSV及各种SFT框架格式。
    """

    @staticmethod
    def to_jsonl(items: List[Dict], output_path: str) -> None:
        """导出为标准JSONL格式（每行一个JSON对象）"""
        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(
                    {"question": item["question"], "answer": item["answer"]},
                    ensure_ascii=False
                ) + "\n")
        logger.info(f"JSONL导出完成: {output_path} ({len(items)} 条)")

    @staticmethod
    def to_json_array(items: List[Dict], output_path: str) -> None:
        """导出为JSON数组格式（原始需求格式）"""
        data = [
            {"question": item["question"], "answer": item["answer"]}
            for item in items
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON数组导出完成: {output_path} ({len(items)} 条)")

    @staticmethod
    def to_csv(items: List[Dict], output_path: str) -> None:
        """导出为CSV格式"""
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            for item in items:
                writer.writerow({
                    "question": item["question"],
                    "answer": item["answer"],
                })
        logger.info(f"CSV导出完成: {output_path} ({len(items)} 条)")

    @staticmethod
    def to_alpaca_format(items: List[Dict], output_path: str) -> None:
        """
        导出为Alpaca指令微调格式。
        适用于LLaMA-Factory、FastChat等框架。
        """
        alpaca_items = []
        for item in items:
            alpaca_items.append({
                "instruction": (
                    "你是一个智能体路由助手。根据用户的问题，"
                    "从以下100个智能体中选择最合适的一个，只输出智能体名称，不要有其他内容。"
                ),
                "input": item["question"],
                "output": item["answer"],
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_items, f, ensure_ascii=False, indent=2)
        logger.info(f"Alpaca格式导出完成: {output_path} ({len(items)} 条)")

    @staticmethod
    def to_sharegpt_format(items: List[Dict], output_path: str) -> None:
        """
        导出为ShareGPT对话格式。
        适用于需要多轮对话格式的SFT框架。
        """
        sharegpt_items = []
        system_msg = (
            "你是一个智能体路由助手。你的任务是根据用户的问题，"
            "从100个预定义的智能体中选择最合适的一个，并只输出该智能体的名称。"
        )
        for item in items:
            sharegpt_items.append({
                "system": system_msg,
                "conversations": [
                    {"from": "human", "value": item["question"]},
                    {"from": "gpt", "value": item["answer"]},
                ],
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sharegpt_items, f, ensure_ascii=False, indent=2)
        logger.info(f"ShareGPT格式导出完成: {output_path} ({len(items)} 条)")

    @staticmethod
    def to_openai_chat_format(items: List[Dict], output_path: str) -> None:
        """
        导出为OpenAI Chat Fine-tuning格式（JSONL）。
        适用于OpenAI Fine-tuning API。
        """
        system_msg = (
            "你是一个智能体路由助手。根据用户的问题选择最合适的智能体，"
            "只输出智能体名称。"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                record = {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"OpenAI Chat格式导出完成: {output_path} ({len(items)} 条)")

    @classmethod
    def export_all_formats(
        cls,
        items: List[Dict],
        output_dir: str,
        prefix: str = "dataset",
    ) -> Dict[str, str]:
        """
        一次性导出所有格式。

        Returns:
            Dict[format_name -> file_path]
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        formats = [
            ("jsonl", cls.to_jsonl, f"{prefix}.jsonl"),
            ("json", cls.to_json_array, f"{prefix}.json"),
            ("csv", cls.to_csv, f"{prefix}.csv"),
            ("alpaca", cls.to_alpaca_format, f"{prefix}_alpaca.json"),
            ("sharegpt", cls.to_sharegpt_format, f"{prefix}_sharegpt.json"),
            ("openai", cls.to_openai_chat_format, f"{prefix}_openai.jsonl"),
        ]

        for fmt_name, export_fn, filename in formats:
            file_path = str(output_dir / filename)
            export_fn(items, file_path)
            paths[fmt_name] = file_path

        return paths


# ─────────────────────────── 训练/验证/测试集划分 ───────────────────────────

class TrainTestSplitter:
    """
    训练集/验证集/测试集划分器。
    支持按智能体分层采样，确保每个类别在各集合中均有代表。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def split(
        self,
        items: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratified: bool = True,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        划分数据集。

        Args:
            items:        全量数据
            train_ratio:  训练集比例
            val_ratio:    验证集比例
            test_ratio:   测试集比例（= 1 - train - val）
            stratified:   是否按类别分层采样

        Returns:
            (train_set, val_set, test_set)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "三个比例之和必须为1"

        if stratified:
            return self._stratified_split(items, train_ratio, val_ratio, test_ratio)
        else:
            return self._random_split(items, train_ratio, val_ratio, test_ratio)

    def _stratified_split(
        self,
        items: List[Dict],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分层采样划分"""
        # 按answer分组
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            groups[item.get("answer", "")].append(item)

        train, val, test = [], [], []

        for answer, group_items in groups.items():
            self.rng.shuffle(group_items)
            n = len(group_items)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val

            if n_test < 0:
                # 数量太少时优先保证训练集
                n_val = max(0, n - n_train)
                n_test = 0

            train.extend(group_items[:n_train])
            val.extend(group_items[n_train:n_train + n_val])
            test.extend(group_items[n_train + n_val:n_train + n_val + n_test])

        self.rng.shuffle(train)
        self.rng.shuffle(val)
        self.rng.shuffle(test)

        return train, val, test

    def _random_split(
        self,
        items: List[Dict],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """随机划分"""
        shuffled = list(items)
        self.rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return (
            shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
        )

    def split_and_save(
        self,
        input_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratified: bool = True,
        export_formats: bool = True,
    ) -> Dict[str, Any]:
        """
        读取数据集，划分后保存到指定目录。

        Returns:
            划分统计信息
        """
        # 读取数据
        items = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        train, val, test = self.split(items, train_ratio, val_ratio, test_ratio, stratified)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exporter = DatasetExporter()

        # 保存各集合
        splits = {"train": train, "val": val, "test": test}
        for split_name, split_data in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)

            # 基础JSONL格式
            DatasetExporter.to_jsonl(
                split_data,
                str(split_dir / f"{split_name}.jsonl")
            )

            if export_formats:
                # 导出额外格式
                DatasetExporter.to_alpaca_format(
                    split_data,
                    str(split_dir / f"{split_name}_alpaca.json")
                )
                DatasetExporter.to_sharegpt_format(
                    split_data,
                    str(split_dir / f"{split_name}_sharegpt.json")
                )
                if split_name == "train":
                    DatasetExporter.to_openai_chat_format(
                        split_data,
                        str(split_dir / f"{split_name}_openai.jsonl")
                    )

        stats = {
            "total": len(items),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_ratio": len(train) / len(items) if items else 0,
            "val_ratio": len(val) / len(items) if items else 0,
            "test_ratio": len(test) / len(items) if items else 0,
        }

        logger.info(
            f"数据集划分完成: "
            f"train={stats['train']}, val={stats['val']}, test={stats['test']}"
        )
        return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/ubuntu/agent_router_sft")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from agents import AGENT_REGISTRY

    # 测试：用test_run的数据进行分析
    test_data_path = "/home/ubuntu/agent_router_sft/output/test_run/stage1/民航绕偏航处置.jsonl"

    if Path(test_data_path).exists():
        analyzer = DatasetAnalyzer(AGENT_REGISTRY)
        # 创建一个简单的测试数据集
        test_items = [
            {"question": f"测试问题{i}", "answer": AGENT_REGISTRY[i % 3]["name"]}
            for i in range(30)
        ]

        # 测试导出
        exporter = DatasetExporter()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = DatasetExporter.export_all_formats(test_items, tmpdir, "test")
            print(f"导出格式: {list(paths.keys())}")

        # 测试划分
        splitter = TrainTestSplitter(seed=42)
        train, val, test = splitter.split(test_items, 0.8, 0.1, 0.1)
        print(f"划分结果: train={len(train)}, val={len(val)}, test={len(test)}")
    else:
        print("测试数据不存在，请先运行生成器")
