"""
stage_generator.py
------------------
三阶段数据生成器核心模块。

实现以下三个生成阶段：
  Stage 1 - 冷启动阶段：每个智能体生成 100~200 条极简短句样本（8~15字）
  Stage 2 - 生产泛化阶段：每个智能体生成 500~1000 条混合复杂度样本
  Stage 3 - 困难负样本阶段：针对易混淆智能体对，定向生成对比样本

并行策略（Stage内并行）：
  - 每个 Stage 内部将所有待生成的 Prompt 任务预先构造为任务列表
  - 通过 LLMClient.batch_call（内部使用 asyncio + Semaphore）并发调用 LLM
  - 并发数由 LLMClientConfig.max_concurrent 控制（默认10）
  - 结果收集完毕后，过滤无效项，按顺序写入输出文件
  - 支持断点续传：已有数据不重复生成

每个阶段均支持：
  - Stage 内并行（同一智能体的多个 Prompt 并发发射）
  - 断点续传（通过检查已有输出文件）
  - 进度日志（含并发数、耗时、成功率）
  - 失败重试（由 LLMClient 内部处理）
  - 样本数量配额控制（超出目标数量自动截断）
"""

import json
import time
import logging
import random
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """各阶段生成配置"""
    # 阶段一配置
    stage1_samples_per_agent: int = 150      # 每个智能体目标样本数（冷启动）
    stage1_batch_size: int = 5               # 单次 LLM 调用生成数量
    stage1_parallel_calls: int = 10          # 阶段一并行 LLM 调用数

    # 阶段二配置
    stage2_samples_per_agent: int = 700      # 每个智能体目标样本数（生产泛化）
    stage2_simple_ratio: float = 0.4         # 简单样本占比（40%简单 + 60%复杂）
    stage2_batch_size: int = 5               # 单次 LLM 调用生成数量
    stage2_parallel_calls: int = 10          # 阶段二并行 LLM 调用数

    # 阶段三配置（困难负样本）
    stage3_pairs_per_agent: int = 50         # 每个混淆对目标样本数
    stage3_batch_size: int = 5               # 单次 LLM 调用生成数量
    stage3_parallel_calls: int = 8           # 阶段三并行 LLM 调用数

    # 输出配置
    output_dir: str = "/home/ubuntu/agent_router_sft/output"

    # 随机种子
    seed: int = 42


class StageGenerator:
    """
    三阶段数据生成器。

    并行模式：
      每个 Stage 内部预先构造全量 Prompt 任务列表，
      通过 LLMClient.batch_call 并发调用，结果汇总后写入文件。
      多个智能体之间仍串行处理（保证断点续传的文件一致性）。
    """

    def __init__(
        self,
        llm_client,
        prompt_builder,
        config: StageConfig,
        agent_registry: List[Dict[str, Any]],
    ):
        self.client = llm_client
        self.builder = prompt_builder
        self.config = config
        self.agents = agent_registry
        self.rng = random.Random(config.seed)
        self._write_lock = threading.Lock()

        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stage in ["stage1", "stage2", "stage3"]:
            (self.output_dir / stage).mkdir(exist_ok=True)

        logger.info(
            f"StageGenerator 初始化完成，输出目录: {self.output_dir}，"
            f"并行数: stage1={config.stage1_parallel_calls}, "
            f"stage2={config.stage2_parallel_calls}, "
            f"stage3={config.stage3_parallel_calls}"
        )

    # ═══════════════════════════════════════════════════════════════
    #  阶段一：冷启动（Stage-内并行）
    # ═══════════════════════════════════════════════════════════════

    def run_stage1(
        self,
        agent_names: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Dict[str, int]:
        """
        执行阶段一：冷启动数据生成。

        并行策略：
          对单个智能体，预先构造 ceil(target/batch_size) 个 Prompt 任务，
          通过 batch_call 并行发射，结果收集后截断至 target_count 写入文件。

        Args:
            agent_names: 指定要处理的智能体名称列表，None 表示全部
            resume:      是否断点续传

        Returns:
            Dict[agent_name -> 实际写入数量]
        """
        logger.info("=" * 60)
        logger.info("开始阶段一：冷启动数据生成（Stage 内并行）")
        logger.info(
            f"目标：每个智能体 {self.config.stage1_samples_per_agent} 条极简短句样本，"
            f"并行调用数={self.config.stage1_parallel_calls}"
        )
        logger.info("=" * 60)

        target_agents = self._filter_agents(agent_names)
        results: Dict[str, int] = {}
        total_generated = 0

        for idx, agent in enumerate(target_agents):
            agent_name = agent["name"]
            output_file = (
                self.output_dir / "stage1" / f"{self._safe_filename(agent_name)}.jsonl"
            )

            existing_count = self._count_existing(output_file)
            target_count = self.config.stage1_samples_per_agent

            if resume and existing_count >= target_count:
                logger.info(
                    f"[{idx+1}/{len(target_agents)}] {agent_name}: "
                    f"已有 {existing_count} 条，跳过"
                )
                results[agent_name] = existing_count
                continue

            remaining = target_count - existing_count
            logger.info(
                f"[{idx+1}/{len(target_agents)}] {agent_name}: "
                f"已有 {existing_count} 条，需生成 {remaining} 条"
            )

            t0 = time.time()
            written = self._parallel_generate_stage1(
                agent=agent,
                target_count=remaining,
                output_file=output_file,
                start_idx=existing_count,
            )
            elapsed = time.time() - t0

            results[agent_name] = existing_count + written
            total_generated += written
            logger.info(
                f"  完成: 新写入 {written} 条，耗时 {elapsed:.1f}s，"
                f"累计 {results[agent_name]} 条"
            )

        logger.info(f"\n阶段一完成！本次新生成 {total_generated} 条样本")
        return results

    def _parallel_generate_stage1(
        self,
        agent: Dict[str, Any],
        target_count: int,
        output_file: Path,
        start_idx: int = 0,
    ) -> int:
        """
        阶段一并行生成核心逻辑。

        步骤：
          1. 计算需要多少次 LLM 调用（call_count = ceil(target / batch_size)）
          2. 构造 call_count 个 Prompt 任务（模板轮换）
          3. 通过 batch_call 并行发射（受 max_concurrent 控制）
          4. 汇总所有解析结果，截断至 target_count
          5. 写入输出文件

        Returns:
            实际写入的条数
        """
        batch_size = self.config.stage1_batch_size
        import math
        call_count = math.ceil(target_count / batch_size)
        # 多生成 20% 以应对解析失败
        call_count = max(call_count, int(call_count * 1.2) + 1)

        # 构造任务列表：(prompt, expected_answer)
        tasks: List[Tuple[Dict[str, str], str]] = []
        for i in range(call_count):
            prompt = self.builder.build_stage1_prompt(
                agent=agent,
                sample_idx=start_idx + i * batch_size,
                batch_size=batch_size,
            )
            tasks.append((prompt, agent["name"]))

        logger.info(
            f"  阶段一并行：构造 {len(tasks)} 个任务，"
            f"并发数={self.config.stage1_parallel_calls}"
        )

        # 并行调用（batch_call 内部使用 asyncio + Semaphore）
        responses = self._batch_call_with_concurrency(
            tasks=tasks,
            max_concurrent=self.config.stage1_parallel_calls,
        )

        # 汇总结果
        all_items: List[Dict[str, str]] = []
        success_calls = 0
        for resp in responses:
            if resp.success:
                all_items.extend(resp.parsed_items)
                success_calls += 1

        logger.info(
            f"  并行调用完成：{success_calls}/{len(tasks)} 次成功，"
            f"共解析 {len(all_items)} 条原始样本"
        )

        # 截断至目标数量
        all_items = all_items[:target_count]

        # 写入文件
        written = self._write_items(output_file, all_items)
        return written

    # ═══════════════════════════════════════════════════════════════
    #  阶段二：生产泛化（Stage-内并行）
    # ═══════════════════════════════════════════════════════════════

    def run_stage2(
        self,
        agent_names: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Dict[str, int]:
        """
        执行阶段二：生产泛化数据生成（Stage 内并行）。

        并行策略：
          将简单样本任务与四种复杂样本任务（口语化/长文本/多轮截断/隐含意图）
          按比例混合构造为任务列表，通过 batch_call 并行发射。
        """
        logger.info("=" * 60)
        logger.info("开始阶段二：生产泛化数据生成（Stage 内并行）")
        logger.info(
            f"目标：每个智能体 {self.config.stage2_samples_per_agent} 条混合样本 "
            f"（简单 {int(self.config.stage2_simple_ratio*100)}% + "
            f"复杂 {int((1-self.config.stage2_simple_ratio)*100)}%），"
            f"并行调用数={self.config.stage2_parallel_calls}"
        )
        logger.info("=" * 60)

        target_agents = self._filter_agents(agent_names)
        results: Dict[str, int] = {}
        total_generated = 0

        for idx, agent in enumerate(target_agents):
            agent_name = agent["name"]
            output_file = (
                self.output_dir / "stage2" / f"{self._safe_filename(agent_name)}.jsonl"
            )

            existing_count = self._count_existing(output_file)
            target_count = self.config.stage2_samples_per_agent

            if resume and existing_count >= target_count:
                logger.info(
                    f"[{idx+1}/{len(target_agents)}] {agent_name}: 已完成，跳过"
                )
                results[agent_name] = existing_count
                continue

            remaining = target_count - existing_count
            logger.info(
                f"[{idx+1}/{len(target_agents)}] {agent_name}: 需生成 {remaining} 条"
            )

            t0 = time.time()
            written = self._parallel_generate_stage2(
                agent=agent,
                target_count=remaining,
                output_file=output_file,
                start_idx=existing_count,
            )
            elapsed = time.time() - t0

            results[agent_name] = existing_count + written
            total_generated += written
            logger.info(
                f"  完成: 新写入 {written} 条，耗时 {elapsed:.1f}s，"
                f"累计 {results[agent_name]} 条"
            )

        logger.info(f"\n阶段二完成！本次新生成 {total_generated} 条样本")
        return results

    def _parallel_generate_stage2(
        self,
        agent: Dict[str, Any],
        target_count: int,
        output_file: Path,
        start_idx: int = 0,
    ) -> int:
        """
        阶段二并行生成核心逻辑。

        任务构造策略：
          - 按 simple_ratio 决定简单/复杂任务数量
          - 复杂任务按 sub_type % 4 轮换四种子类型
          - 多生成 30% 以应对解析失败
        """
        import math
        batch_size = self.config.stage2_batch_size
        simple_ratio = self.config.stage2_simple_ratio

        simple_target = int(target_count * simple_ratio)
        complex_target = target_count - simple_target

        # 计算各类型调用次数（多生成 30%）
        simple_calls = math.ceil(simple_target / batch_size * 1.3) + 1
        complex_calls = math.ceil(complex_target / batch_size * 1.3) + 1

        tasks: List[Tuple[Dict[str, str], str]] = []

        # 构造简单样本任务
        for i in range(simple_calls):
            prompt = self.builder.build_stage2_simple_prompt(
                agent=agent,
                batch_size=batch_size,
            )
            tasks.append((prompt, agent["name"]))

        # 构造复杂样本任务（四种子类型轮换）
        for i in range(complex_calls):
            sub_type = (start_idx + i) % 4
            if sub_type == 0:
                prompt = self.builder.build_stage2_colloquial_prompt(
                    agent=agent, batch_size=batch_size
                )
            elif sub_type == 1:
                prompt = self.builder.build_stage2_complex_prompt(
                    agent=agent, sample_idx=start_idx + i, batch_size=batch_size
                )
            elif sub_type == 2:
                prompt = self.builder.build_stage2_multiturn_prompt(
                    agent=agent, batch_size=batch_size
                )
            else:
                prompt = self.builder.build_stage2_implicit_prompt(
                    agent=agent, batch_size=batch_size
                )
            tasks.append((prompt, agent["name"]))

        # 打乱任务顺序，增加多样性
        self.rng.shuffle(tasks)

        logger.info(
            f"  阶段二并行：构造 {len(tasks)} 个任务 "
            f"（简单={simple_calls}, 复杂={complex_calls}），"
            f"并发数={self.config.stage2_parallel_calls}"
        )

        # 并行调用
        responses = self._batch_call_with_concurrency(
            tasks=tasks,
            max_concurrent=self.config.stage2_parallel_calls,
        )

        # 汇总结果
        all_items: List[Dict[str, str]] = []
        success_calls = 0
        for resp in responses:
            if resp.success:
                all_items.extend(resp.parsed_items)
                success_calls += 1

        logger.info(
            f"  并行调用完成：{success_calls}/{len(tasks)} 次成功，"
            f"共解析 {len(all_items)} 条原始样本"
        )

        # 截断至目标数量
        all_items = all_items[:target_count]

        written = self._write_items(output_file, all_items)
        return written

    # ═══════════════════════════════════════════════════════════════
    #  阶段三：困难负样本（Stage-内并行）
    # ═══════════════════════════════════════════════════════════════

    def run_stage3(
        self,
        agent_names: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Dict[str, int]:
        """
        执行阶段三：困难负样本生成（Stage 内并行）。

        并行策略：
          对单个智能体的所有混淆对，同时构造双向 Prompt 任务，
          通过 batch_call 并行发射，结果汇总后写入文件。
        """
        logger.info("=" * 60)
        logger.info("开始阶段三：困难负样本生成（Stage 内并行）")
        logger.info(
            f"目标：每个混淆对 {self.config.stage3_pairs_per_agent} 条，"
            f"并行调用数={self.config.stage3_parallel_calls}"
        )
        logger.info("=" * 60)

        target_agents = self._filter_agents(agent_names)
        agent_map = {a["name"]: a for a in self.agents}

        results: Dict[str, int] = {}
        total_generated = 0

        for idx, agent in enumerate(target_agents):
            agent_name = agent["name"]
            confusable_names = agent.get("confusable", [])

            if not confusable_names:
                logger.info(
                    f"[{idx+1}/{len(target_agents)}] {agent_name}: 无混淆智能体，跳过"
                )
                continue

            output_file = (
                self.output_dir / "stage3" / f"{self._safe_filename(agent_name)}.jsonl"
            )
            existing_count = self._count_existing(output_file)
            target_count = self.config.stage3_pairs_per_agent * len(confusable_names)

            if resume and existing_count >= target_count:
                logger.info(
                    f"[{idx+1}/{len(target_agents)}] {agent_name}: 已完成，跳过"
                )
                results[agent_name] = existing_count
                continue

            logger.info(
                f"[{idx+1}/{len(target_agents)}] {agent_name}: "
                f"混淆对={confusable_names}，需生成 {target_count - existing_count} 条"
            )

            t0 = time.time()
            written = self._parallel_generate_stage3(
                agent=agent,
                confusable_names=confusable_names,
                agent_map=agent_map,
                target_per_pair=self.config.stage3_pairs_per_agent,
                output_file=output_file,
            )
            elapsed = time.time() - t0

            results[agent_name] = existing_count + written
            total_generated += written
            logger.info(
                f"  完成: 新写入 {written} 条困难负样本，耗时 {elapsed:.1f}s"
            )

        logger.info(f"\n阶段三完成！本次新生成 {total_generated} 条困难负样本")
        return results

    def _parallel_generate_stage3(
        self,
        agent: Dict[str, Any],
        confusable_names: List[str],
        agent_map: Dict[str, Any],
        target_per_pair: int,
        output_file: Path,
    ) -> int:
        """
        阶段三并行生成核心逻辑。

        对每个混淆对，构造 target_per_pair 个任务（双向交替），
        所有混淆对的任务合并后并行发射。
        """
        import math
        batch_size = self.config.stage3_batch_size
        tasks: List[Tuple[Dict[str, str], str]] = []

        for confusable_name in confusable_names:
            confusable_agent = agent_map.get(confusable_name)
            if not confusable_agent:
                logger.warning(f"  找不到混淆智能体: {confusable_name}")
                continue

            call_count = math.ceil(target_per_pair / batch_size * 1.3) + 1
            for i in range(call_count):
                direction = "target" if i % 2 == 0 else "negative"
                prompt = self.builder.build_hard_negative_prompt(
                    target_agent=agent,
                    confusable_agent=confusable_agent,
                    direction=direction,
                    batch_size=batch_size,
                )
                tasks.append((prompt, agent["name"]))

        if not tasks:
            return 0

        logger.info(
            f"  阶段三并行：构造 {len(tasks)} 个任务，"
            f"并发数={self.config.stage3_parallel_calls}"
        )

        responses = self._batch_call_with_concurrency(
            tasks=tasks,
            max_concurrent=self.config.stage3_parallel_calls,
        )

        all_items: List[Dict[str, str]] = []
        success_calls = 0
        for resp in responses:
            if resp.success:
                all_items.extend(resp.parsed_items)
                success_calls += 1

        total_target = target_per_pair * len(confusable_names)
        all_items = all_items[:total_target]

        logger.info(
            f"  并行调用完成：{success_calls}/{len(tasks)} 次成功，"
            f"共解析 {len(all_items)} 条原始样本"
        )

        written = self._write_items(output_file, all_items)
        return written

    # ═══════════════════════════════════════════════════════════════
    #  并行调用核心：ThreadPoolExecutor + batch_call
    # ═══════════════════════════════════════════════════════════════

    def _batch_call_with_concurrency(
        self,
        tasks: List[Tuple[Dict[str, str], str]],
        max_concurrent: int,
    ):
        """
        使用 ThreadPoolExecutor 将任务分批提交给 LLMClient.batch_call，
        实现 Stage 内部的并行 LLM 调用。

        设计说明：
          - LLMClient.batch_call 内部已使用 asyncio + Semaphore 控制并发
          - 此处将全量任务一次性传入 batch_call，由其负责并发调度
          - max_concurrent 参数传入 LLMClient 的 max_concurrent 配置

        Args:
            tasks:          [(prompt_dict, expected_answer), ...]
            max_concurrent: 最大并发调用数

        Returns:
            List[LLMResponse]，与 tasks 一一对应
        """
        if not tasks:
            return []

        # 临时覆盖客户端并发数
        original_concurrent = self.client.config.max_concurrent
        self.client.config.max_concurrent = max_concurrent

        try:
            logger.debug(
                f"  batch_call: {len(tasks)} 个任务，并发数={max_concurrent}"
            )
            responses = self.client.batch_call(tasks)
        finally:
            self.client.config.max_concurrent = original_concurrent

        return responses

    # ═══════════════════════════════════════════════════════════════
    #  工具方法
    # ═══════════════════════════════════════════════════════════════

    def _write_items(self, output_file: Path, items: List[Dict[str, str]]) -> int:
        """将样本列表追加写入 JSONL 文件，返回实际写入条数"""
        if not items:
            return 0
        with self._write_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(items)

    def _filter_agents(self, agent_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """过滤出目标智能体列表"""
        if agent_names is None:
            return self.agents
        name_set = set(agent_names)
        return [a for a in self.agents if a["name"] in name_set]

    @staticmethod
    def _safe_filename(name: str) -> str:
        """将智能体名称转换为安全的文件名"""
        safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe = safe.replace("（", "_").replace("）", "_").replace(" ", "_")
        return safe

    @staticmethod
    def _count_existing(file_path: Path) -> int:
        """统计已有 JSONL 文件中的有效行数"""
        if not file_path.exists():
            return 0
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        count += 1
                    except json.JSONDecodeError:
                        pass
        return count

    def get_progress_report(self) -> Dict[str, Any]:
        """生成当前进度报告"""
        report = {}
        for stage in ["stage1", "stage2", "stage3"]:
            stage_dir = self.output_dir / stage
            stage_report = {}
            total = 0
            if stage_dir.exists():
                for f in stage_dir.glob("*.jsonl"):
                    count = self._count_existing(f)
                    stage_report[f.stem] = count
                    total += count
            report[stage] = {"per_agent": stage_report, "total": total}
        return report


# ═══════════════════════════════════════════════════════════════════
#  单智能体测试入口（真实 LLM 调用）
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, "/home/ubuntu/agent_router_sft")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from agents import AGENT_REGISTRY
    from generators.llm_client import LLMClient, LLMClientConfig
    from generators.prompt_builder import PromptBuilder

    # ── 配置 ──
    USE_REAL_LLM = True   # 改为 False 则使用 dry_run 模式
    TEST_AGENT = "民航绕偏航处置"

    client_config = LLMClientConfig(
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="gpt-4.1-mini",
        temperature=0.85,
        max_concurrent=10,   # 并发上限
        dry_run=not USE_REAL_LLM,
    )

    stage_config = StageConfig(
        stage1_samples_per_agent=20,   # 测试：阶段一生成20条
        stage1_batch_size=5,
        stage1_parallel_calls=4,       # 4路并行

        stage2_samples_per_agent=20,   # 测试：阶段二生成20条
        stage2_simple_ratio=0.4,
        stage2_batch_size=5,
        stage2_parallel_calls=4,

        stage3_pairs_per_agent=10,     # 测试：每混淆对10条
        stage3_batch_size=5,
        stage3_parallel_calls=4,

        output_dir="/home/ubuntu/agent_router_sft/output/parallel_test",
        seed=42,
    )

    client = LLMClient(client_config)
    builder = PromptBuilder(seed=42)

    # 只取目标智能体
    test_agents = [a for a in AGENT_REGISTRY if a["name"] == TEST_AGENT]
    assert test_agents, f"找不到智能体: {TEST_AGENT}"

    generator = StageGenerator(
        llm_client=client,
        prompt_builder=builder,
        config=stage_config,
        agent_registry=AGENT_REGISTRY,   # 全量注册表（用于查找混淆对）
    )

    print(f"\n{'='*60}")
    print(f"测试智能体：{TEST_AGENT}")
    print(f"并行模式：stage1={stage_config.stage1_parallel_calls}路, "
          f"stage2={stage_config.stage2_parallel_calls}路, "
          f"stage3={stage_config.stage3_parallel_calls}路")
    print(f"{'='*60}\n")

    # ── 阶段一 ──
    print("【阶段一：冷启动 - 极简短句并行生成】")
    t0 = time.time()
    r1 = generator.run_stage1(agent_names=[TEST_AGENT], resume=False)
    print(f"阶段一结果: {r1}，耗时 {time.time()-t0:.1f}s\n")

    # ── 阶段二 ──
    print("【阶段二：生产泛化 - 混合样本并行生成】")
    t0 = time.time()
    r2 = generator.run_stage2(agent_names=[TEST_AGENT], resume=False)
    print(f"阶段二结果: {r2}，耗时 {time.time()-t0:.1f}s\n")

    # ── 阶段三 ──
    print("【阶段三：困难负样本 - 并行生成】")
    t0 = time.time()
    r3 = generator.run_stage3(agent_names=[TEST_AGENT], resume=False)
    print(f"阶段三结果: {r3}，耗时 {time.time()-t0:.1f}s\n")

    # ── 进度报告 ──
    print("【进度报告】")
    report = generator.get_progress_report()
    for stage, data in report.items():
        print(f"  {stage}: 总计 {data['total']} 条")
        for agent_name, count in data["per_agent"].items():
            print(f"    {agent_name}: {count} 条")

    # ── LLM 统计 ──
    stats = client.get_stats()
    print(f"\n【LLM 调用统计】")
    print(f"  总请求数: {stats['total_requests']}")
    print(f"  失败请求: {stats['failed_requests']}")
    print(f"  成功率:   {stats['success_rate']:.1%}")
    print(f"  总Token:  {stats['total_tokens']}")

    # ── 展示样本 ──
    print(f"\n【生成样本预览（前5条/每阶段）】")
    for stage_name in ["stage1", "stage2", "stage3"]:
        stage_dir = Path(stage_config.output_dir) / stage_name
        for jsonl_file in stage_dir.glob("*.jsonl"):
            print(f"\n  -- {stage_name} / {jsonl_file.stem} --")
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    try:
                        item = json.loads(line.strip())
                        print(f"    [{i+1}] Q: {item['question']}")
                        print(f"         A: {item['answer']}")
                    except Exception:
                        pass
