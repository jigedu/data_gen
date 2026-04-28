"""
pipeline.py
-----------
全流程 Pipeline 主控模块。

将数据生成、清洗、合并、导出的所有步骤串联为一个可配置的流水线。
支持：
  - 全量运行（run_all）
  - 单阶段运行（run_stage1/2/3）
  - 仅清洗（run_clean）
  - 仅合并导出（run_merge_export）
  - 断点续传（所有阶段均支持）
  - 进度持久化（pipeline_state.json）
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class Pipeline:
    """
    全流程数据生成 Pipeline。
    """

    def __init__(self, config_path: Optional[str] = None):
        # 延迟导入，避免循环依赖
        from config import load_config
        from agents import AGENT_REGISTRY
        from generators import LLMClient, LLMClientConfig, PromptBuilder, StageGenerator, StageConfig
        from generators.hard_negative_builder import HardNegativeGenerator, ConfusablePairFinder
        from cleaners import DataCleaner, CleanerConfig
        from utils import DatasetMerger, MergeConfig, DatasetAnalyzer, DatasetExporter, TrainTestSplitter

        # 加载配置
        self.cfg = load_config(config_path)
        self.agents = AGENT_REGISTRY

        # 初始化 LLM 客户端
        # 构建 api_base（支持 base_url 或 api_base 两种配置名）
        api_base = (
            self.cfg.llm.base_url
            if (hasattr(self.cfg.llm, 'base_url') and self.cfg.llm.base_url)
            else "https://api.openai.com/v1"
        )
        api_key = (
            self.cfg.llm.api_key
            if (hasattr(self.cfg.llm, 'api_key') and self.cfg.llm.api_key)
            else os.environ.get("OPENAI_API_KEY", "")
        )
        llm_cfg = LLMClientConfig(
            model=self.cfg.llm.model,
            api_base=api_base,
            api_key=api_key,
            temperature=self.cfg.llm.temperature,
            max_tokens=self.cfg.llm.max_tokens,
            top_p=self.cfg.llm.top_p,
            max_concurrent=self.cfg.llm.max_workers,
            max_retries=self.cfg.llm.max_retries,
            retry_delay=self.cfg.llm.retry_delay,
            timeout=self.cfg.llm.request_timeout,
            dry_run=self.cfg.llm.dry_run,
        )
        self.llm_client = LLMClient(llm_cfg)

        # 初始化 PromptBuilder
        self.prompt_builder = PromptBuilder(seed=self.cfg.generation.seed)

        # 初始化 StageGenerator
        stage_cfg = StageConfig(
            stage1_samples_per_agent=self.cfg.generation.stage1.samples_per_agent,
            stage1_batch_size=self.cfg.generation.stage1.batch_size,
            stage1_simple_ratio=self.cfg.generation.stage1.simple_ratio,
            stage2_samples_per_agent=self.cfg.generation.stage2.samples_per_agent,
            stage2_batch_size=self.cfg.generation.stage2.batch_size,
            stage2_simple_ratio=self.cfg.generation.stage2.simple_ratio,
            stage3_pairs_per_agent=self.cfg.generation.stage3.samples_per_pair // 2,
            stage3_batch_size=self.cfg.generation.stage3.batch_size,
            output_dir=self.cfg.output.root_dir,
            seed=self.cfg.generation.seed,
        )
        self.stage_generator = StageGenerator(
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
            config=stage_cfg,
            agent_registry=self.agents,
        )

        # 初始化 HardNegativeGenerator
        self.hard_neg_generator = HardNegativeGenerator(
            llm_client=self.llm_client,
            agent_registry=self.agents,
            output_dir=self.cfg.output.root_dir,
        )

        # 初始化 DataCleaner
        clean_cfg = CleanerConfig(
            min_question_length=self.cfg.cleaning.min_question_length,
            max_question_length=self.cfg.cleaning.max_question_length,
            check_label_leakage=self.cfg.cleaning.check_label_leakage,
            min_chinese_ratio=self.cfg.cleaning.min_chinese_ratio,
            max_repeat_char_ratio=self.cfg.cleaning.max_repeat_char_ratio,
            dedup_method=self.cfg.cleaning.dedup_method,
            ngram_similarity_threshold=self.cfg.cleaning.ngram_similarity_threshold,
            ngram_size=self.cfg.cleaning.ngram_size,
            min_quality_score=self.cfg.cleaning.min_quality_score,
            max_samples_per_agent=self.cfg.cleaning.max_samples_per_agent,
        )
        self.cleaner = DataCleaner(self.agents, clean_cfg)

        # 初始化 DatasetMerger
        merge_cfg = MergeConfig(
            stage1_weight=self.cfg.merging.stage1_weight,
            stage2_weight=self.cfg.merging.stage2_weight,
            stage3_weight=self.cfg.merging.stage3_weight,
            target_per_agent=self.cfg.merging.target_per_agent,
            max_total=self.cfg.merging.max_total,
            global_dedup=self.cfg.merging.global_dedup,
            seed=self.cfg.merging.seed,
        )
        self.merger = DatasetMerger(merge_cfg, self.agents)

        # 初始化其他工具
        self.analyzer = DatasetAnalyzer(self.agents)
        self.splitter = TrainTestSplitter(seed=self.cfg.generation.seed)

        # 创建输出目录
        self._ensure_output_dirs()

        # 状态文件
        self.state_file = Path(self.cfg.output.root_dir) / "pipeline_state.json"
        self.state = self._load_state()

        logger.info("Pipeline 初始化完成")

    # ─────────────────────────── 全量运行 ───────────────────────────

    def run_all(
        self,
        agent_names: Optional[List[str]] = None,
        skip_stages: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的数据生成流水线。

        流程：
          1. Stage1 生成（冷启动）
          2. Stage1 清洗
          3. Stage2 生成（生产泛化）
          4. Stage2 清洗
          5. Stage3 生成（困难负样本）
          6. Stage3 清洗
          7. 合并三阶段数据
          8. 统计分析
          9. 划分训练/验证/测试集
          10. 多格式导出

        Args:
            agent_names:  指定要处理的智能体（None 表示全部）
            skip_stages:  跳过的阶段编号列表（如 [2, 3] 表示跳过阶段2和3）

        Returns:
            完整的运行统计信息
        """
        skip_stages = skip_stages or []
        start_time = time.time()
        results = {}

        logger.info("=" * 70)
        logger.info("开始执行完整 Pipeline")
        logger.info(f"  智能体数量: {len(self.agents)}")
        logger.info(f"  LLM 模型:   {self.cfg.llm.model}")
        logger.info(f"  Dry Run:    {self.cfg.llm.dry_run}")
        logger.info("=" * 70)

        # ── Step 1: Stage1 生成 ──
        if 1 not in skip_stages and self.cfg.generation.stage1.enabled:
            logger.info("\n[Step 1/10] Stage1 生成（冷启动）")
            r = self.stage_generator.run_stage1(
                agent_names=agent_names,
                resume=self.cfg.generation.stage1.resume,
            )
            results["stage1_generation"] = r
            self._save_state({"stage1_generation": "done"})
        else:
            logger.info("[Step 1/10] Stage1 生成 - 跳过")

        # ── Step 2: Stage1 清洗 ──
        if 1 not in skip_stages:
            logger.info("\n[Step 2/10] Stage1 清洗")
            r = self.cleaner.clean_directory(
                input_dir=self.cfg.output.stage1_raw_dir,
                output_dir=self.cfg.output.stage1_clean_dir,
                reset_dedup=False,
            )
            results["stage1_cleaning"] = r
        else:
            logger.info("[Step 2/10] Stage1 清洗 - 跳过")

        # ── Step 3: Stage2 生成 ──
        if 2 not in skip_stages and self.cfg.generation.stage2.enabled:
            logger.info("\n[Step 3/10] Stage2 生成（生产泛化）")
            r = self.stage_generator.run_stage2(
                agent_names=agent_names,
                resume=self.cfg.generation.stage2.resume,
            )
            results["stage2_generation"] = r
            self._save_state({"stage2_generation": "done"})
        else:
            logger.info("[Step 3/10] Stage2 生成 - 跳过")

        # ── Step 4: Stage2 清洗 ──
        if 2 not in skip_stages:
            logger.info("\n[Step 4/10] Stage2 清洗")
            self.cleaner.reset_stats()
            r = self.cleaner.clean_directory(
                input_dir=self.cfg.output.stage2_raw_dir,
                output_dir=self.cfg.output.stage2_clean_dir,
                reset_dedup=False,
            )
            results["stage2_cleaning"] = r
        else:
            logger.info("[Step 4/10] Stage2 清洗 - 跳过")

        # ── Step 5: Stage3 生成 ──
        if 3 not in skip_stages and self.cfg.generation.stage3.enabled:
            logger.info("\n[Step 5/10] Stage3 生成（困难负样本）")
            from generators.hard_negative_builder import ConfusablePairFinder
            finder = ConfusablePairFinder(self.agents)
            all_pairs = finder.find_all_pairs(
                keyword_overlap_threshold=self.cfg.generation.stage3.keyword_overlap_threshold,
                max_pairs_per_agent=self.cfg.generation.stage3.max_pairs_per_agent,
            )
            # 如果指定了 agent_names，只处理涉及这些智能体的混淆对
            if agent_names:
                agent_name_set = set(agent_names)
                filtered_pairs = [
                    p for p in all_pairs
                    if p.agent_a in agent_name_set or p.agent_b in agent_name_set
                ]
                logger.info(
                    f"  按 agent_names 过滤后: {len(all_pairs)} -> {len(filtered_pairs)} 个混淆对"
                )
            else:
                filtered_pairs = all_pairs

            r = self.hard_neg_generator.run_full_generation(
                samples_per_pair=self.cfg.generation.stage3.samples_per_pair,
                max_pairs=len(filtered_pairs) if agent_names else None,
                resume=self.cfg.generation.stage3.resume,
            )
            results["stage3_generation"] = {"total": r}
            self._save_state({"stage3_generation": "done"})
        else:
            logger.info("[Step 5/10] Stage3 生成 - 跳过")

        # ── Step 6: Stage3 清洗 ──
        if 3 not in skip_stages:
            logger.info("\n[Step 6/10] Stage3 清洗")
            self.cleaner.reset_stats()
            r = self.cleaner.clean_directory(
                input_dir=str(Path(self.cfg.output.root_dir) / "stage3" / "hard_negatives"),
                output_dir=self.cfg.output.stage3_clean_dir,
                reset_dedup=False,
            )
            results["stage3_cleaning"] = r
        else:
            logger.info("[Step 6/10] Stage3 清洗 - 跳过")

        # ── Step 7: 合并 ──
        logger.info("\n[Step 7/10] 合并三阶段数据")
        merge_stats = self.merger.merge_from_dirs(
            stage1_dir=self.cfg.output.stage1_clean_dir,
            stage2_dir=self.cfg.output.stage2_clean_dir,
            stage3_dir=self.cfg.output.stage3_clean_dir,
            output_path=self.cfg.output.merged_dataset,
        )
        results["merging"] = merge_stats

        # ── Step 8: 统计分析 ──
        logger.info("\n[Step 8/10] 统计分析")
        merged_path = self.cfg.output.merged_dataset
        if Path(merged_path).exists():
            report = self.analyzer.analyze(merged_path)
            self.analyzer.print_report(report)
            report_path = str(Path(self.cfg.output.report_dir) / "dataset_report.json")
            Path(self.cfg.output.report_dir).mkdir(parents=True, exist_ok=True)
            self.analyzer.save_report(report, report_path)
            results["analysis"] = report
        else:
            logger.warning("合并数据集不存在，跳过统计分析")

        # ── Step 9: 划分 ──
        logger.info("\n[Step 9/10] 划分训练/验证/测试集")
        if Path(merged_path).exists():
            split_stats = self.splitter.split_and_save(
                input_path=merged_path,
                output_dir=self.cfg.output.split_dir,
                train_ratio=self.cfg.splitting.train_ratio,
                val_ratio=self.cfg.splitting.val_ratio,
                test_ratio=self.cfg.splitting.test_ratio,
                stratified=self.cfg.splitting.stratified,
                export_formats=self.cfg.splitting.export_formats,
            )
            results["splitting"] = split_stats
            logger.info(
                f"划分完成: train={split_stats['train']}, "
                f"val={split_stats['val']}, test={split_stats['test']}"
            )

        # ── Step 10: 多格式导出 ──
        logger.info("\n[Step 10/10] 多格式导出")
        if Path(merged_path).exists():
            items = self._load_jsonl(merged_path)
            export_paths = DatasetExporter.export_all_formats(
                items=items,
                output_dir=self.cfg.output.export_dir,
                prefix="agent_router_sft",
            )
            results["export"] = export_paths
            logger.info(f"已导出 {len(export_paths)} 种格式到: {self.cfg.output.export_dir}")

        # 完成
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = round(elapsed, 1)
        self._save_state({"pipeline": "completed", "elapsed": elapsed})

        logger.info("\n" + "=" * 70)
        logger.info(f"Pipeline 完成！总耗时: {elapsed:.1f} 秒")
        logger.info("=" * 70)

        return results

    # ─────────────────────────── 单阶段运行 ───────────────────────────

    def run_stage1_only(self, agent_names: Optional[List[str]] = None) -> Dict:
        """仅运行阶段一（生成 + 清洗）"""
        logger.info("运行阶段一：冷启动生成 + 清洗")
        gen_result = self.stage_generator.run_stage1(
            agent_names=agent_names,
            resume=self.cfg.generation.stage1.resume,
        )
        clean_result = self.cleaner.clean_directory(
            input_dir=self.cfg.output.stage1_raw_dir,
            output_dir=self.cfg.output.stage1_clean_dir,
        )
        return {"generation": gen_result, "cleaning": clean_result}

    def run_stage2_only(self, agent_names: Optional[List[str]] = None) -> Dict:
        """仅运行阶段二（生成 + 清洗）"""
        logger.info("运行阶段二：生产泛化生成 + 清洗")
        gen_result = self.stage_generator.run_stage2(
            agent_names=agent_names,
            resume=self.cfg.generation.stage2.resume,
        )
        clean_result = self.cleaner.clean_directory(
            input_dir=self.cfg.output.stage2_raw_dir,
            output_dir=self.cfg.output.stage2_clean_dir,
        )
        return {"generation": gen_result, "cleaning": clean_result}

    def run_stage3_only(self) -> Dict:
        """仅运行阶段三（困难负样本生成 + 清洗）"""
        logger.info("运行阶段三：困难负样本生成 + 清洗")
        total = self.hard_neg_generator.run_full_generation(
            samples_per_pair=self.cfg.generation.stage3.samples_per_pair,
            resume=self.cfg.generation.stage3.resume,
        )
        clean_result = self.cleaner.clean_directory(
            input_dir=str(Path(self.cfg.output.root_dir) / "stage3" / "hard_negatives"),
            output_dir=self.cfg.output.stage3_clean_dir,
        )
        return {"generation": {"total": total}, "cleaning": clean_result}

    def run_merge_export_only(self) -> Dict:
        """仅运行合并与导出（假设各阶段清洗数据已存在）"""
        logger.info("运行合并与导出")
        merge_stats = self.merger.merge_from_dirs(
            stage1_dir=self.cfg.output.stage1_clean_dir,
            stage2_dir=self.cfg.output.stage2_clean_dir,
            stage3_dir=self.cfg.output.stage3_clean_dir,
            output_path=self.cfg.output.merged_dataset,
        )

        merged_path = self.cfg.output.merged_dataset
        if Path(merged_path).exists():
            items = self._load_jsonl(merged_path)
            export_paths = DatasetExporter.export_all_formats(
                items=items,
                output_dir=self.cfg.output.export_dir,
                prefix="agent_router_sft",
            )
            return {"merging": merge_stats, "export": export_paths}

        return {"merging": merge_stats}

    def print_progress(self) -> None:
        """打印当前进度"""
        report = self.stage_generator.get_progress_report()
        print("\n当前生成进度:")
        for stage, data in report.items():
            total = data["total"]
            agent_count = len(data["per_agent"])
            print(f"  {stage}: {total:,} 条（{agent_count} 个智能体）")

    # ─────────────────────────── 工具方法 ───────────────────────────

    def _ensure_output_dirs(self) -> None:
        """创建所有必要的输出目录"""
        dirs = [
            self.cfg.output.root_dir,
            self.cfg.output.stage1_raw_dir,
            self.cfg.output.stage2_raw_dir,
            self.cfg.output.stage3_raw_dir,
            self.cfg.output.stage1_clean_dir,
            self.cfg.output.stage2_clean_dir,
            self.cfg.output.stage3_clean_dir,
            str(Path(self.cfg.output.merged_dataset).parent),
            self.cfg.output.split_dir,
            self.cfg.output.export_dir,
            self.cfg.output.report_dir,
            str(Path(self.cfg.output.logging.file).parent)
            if hasattr(self.cfg.output, "logging") else
            str(Path(self.cfg.output.root_dir) / "logs"),
        ]
        for d in dirs:
            if d:
                Path(d).mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_state(self, update: Dict) -> None:
        self.state.update(update)
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

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


# 避免循环导入
from utils import DatasetExporter
