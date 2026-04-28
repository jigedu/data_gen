#!/usr/bin/env python3
"""
run.py
------
agent_router_sft 命令行入口脚本。

用法示例：
  # 完整流程（使用默认配置）
  python run.py

  # 完整流程（指定配置文件）
  python run.py --config config/config.yaml

  # 仅运行阶段一
  python run.py --stage 1

  # 仅运行阶段二
  python run.py --stage 2

  # 仅运行阶段三（困难负样本）
  python run.py --stage 3

  # 仅合并与导出（假设各阶段数据已存在）
  python run.py --merge-only

  # Dry Run 模式（不调用 LLM，用于测试流程）
  python run.py --dry-run

  # 指定特定智能体
  python run.py --agents "民航绕偏航处置,飞行计划审核"

  # 查看当前进度
  python run.py --progress

  # 生成混淆对分析报告
  python run.py --analyze-pairs

  # 统计已有数据集
  python run.py --analyze-dataset ./output/merged/full_dataset.jsonl
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """配置日志"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="agent_router_sft: 智能体路由模型 SFT 训练数据自动化生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 配置
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径（默认自动发现 config/config.yaml）",
    )

    # 运行模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--stage", "-s",
        type=int,
        choices=[1, 2, 3],
        help="仅运行指定阶段（1=冷启动, 2=生产泛化, 3=困难负样本）",
    )
    mode_group.add_argument(
        "--merge-only",
        action="store_true",
        help="仅执行合并与导出步骤",
    )
    mode_group.add_argument(
        "--progress",
        action="store_true",
        help="查看当前生成进度",
    )
    mode_group.add_argument(
        "--analyze-pairs",
        action="store_true",
        help="生成混淆对分析报告",
    )
    mode_group.add_argument(
        "--analyze-dataset",
        type=str,
        metavar="DATASET_PATH",
        help="统计分析指定数据集文件",
    )

    # 选项
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry Run 模式（不实际调用 LLM API）",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="指定要处理的智能体名称（逗号分隔），默认处理全部",
    )
    parser.add_argument(
        "--skip-stages",
        type=str,
        default=None,
        help="跳过的阶段编号（逗号分隔，如 '2,3'）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="覆盖配置文件中的输出目录",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 设置环境变量（优先级高于配置文件）
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir

    # 初始化日志（临时配置，Pipeline 初始化后会重新配置）
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("agent_router_sft 启动")
    logger.info(f"Python: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")

    # 初始化 Pipeline
    try:
        from pipeline import Pipeline
        pipeline = Pipeline(config_path=args.config)
    except Exception as e:
        logger.error(f"Pipeline 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 重新配置日志（使用 Pipeline 中的配置）
    log_file = pipeline.cfg.logging.file if hasattr(pipeline.cfg, "logging") else None
    setup_logging(args.log_level, log_file)

    # 解析智能体列表
    agent_names = None
    if args.agents:
        agent_names = [name.strip() for name in args.agents.split(",") if name.strip()]
        logger.info(f"指定智能体: {agent_names}")

    # 解析跳过阶段
    skip_stages = []
    if args.skip_stages:
        skip_stages = [int(s.strip()) for s in args.skip_stages.split(",") if s.strip()]
        logger.info(f"跳过阶段: {skip_stages}")

    # 执行对应操作
    try:
        if args.progress:
            pipeline.print_progress()

        elif args.analyze_pairs:
            from generators.hard_negative_builder import ConfusablePairFinder
            finder = ConfusablePairFinder(pipeline.agents)
            pairs = finder.find_all_pairs()
            report_path = str(Path(pipeline.cfg.output.report_dir) / "confusable_pairs.json")
            finder.export_pairs_report(report_path)
            print(f"\n混淆对分析报告已保存: {report_path}")
            print(f"共发现 {len(pairs)} 个混淆对")

        elif args.analyze_dataset:
            report = pipeline.analyzer.analyze(args.analyze_dataset)
            pipeline.analyzer.print_report(report)

        elif args.stage == 1:
            result = pipeline.run_stage1_only(agent_names=agent_names)
            logger.info(f"阶段一完成: {result}")

        elif args.stage == 2:
            result = pipeline.run_stage2_only(agent_names=agent_names)
            logger.info(f"阶段二完成: {result}")

        elif args.stage == 3:
            result = pipeline.run_stage3_only()
            logger.info(f"阶段三完成: {result}")

        elif args.merge_only:
            result = pipeline.run_merge_export_only()
            logger.info(f"合并导出完成: {result}")

        else:
            # 默认：运行全量流程
            result = pipeline.run_all(
                agent_names=agent_names,
                skip_stages=skip_stages,
            )
            logger.info(f"全量流程完成，耗时: {result.get('elapsed_seconds', 0):.1f}s")

        return 0

    except KeyboardInterrupt:
        logger.info("\n用户中断，已保存当前进度")
        return 130
    except Exception as e:
        logger.error(f"运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
