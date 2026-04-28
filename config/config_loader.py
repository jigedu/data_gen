"""
config_loader.py
----------------
配置文件加载与验证模块。

支持：
1. 从 YAML 文件加载配置
2. 环境变量覆盖（优先级：环境变量 > 配置文件 > 默认值）
3. 配置合法性验证
4. 配置对象化访问（通过属性而非字典键）
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigNode:
    """
    配置节点：将字典递归转换为可通过属性访问的对象。

    用法：
        config = ConfigNode({"llm": {"model": "gpt-4o-mini"}})
        config.llm.model  # -> "gpt-4o-mini"
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"ConfigNode({self.to_dict()})"


class ConfigLoader:
    """
    配置加载器。
    负责加载、合并与验证配置。
    """

    # 默认配置（与 config.yaml 保持一致）
    DEFAULTS: Dict[str, Any] = {
        "llm": {
            "model": "gpt-4o-mini",
            "base_url": "",
            "api_key": "",
            "temperature": 0.9,
            "max_tokens": 512,
            "top_p": 0.95,
            "max_workers": 5,
            "max_retries": 3,
            "retry_delay": 2.0,
            "request_timeout": 30,
            "dry_run": False,
        },
        "generation": {
            "seed": 42,
            "stage1": {
                "enabled": True,
                "samples_per_agent": 150,
                "batch_size": 5,
                "simple_ratio": 1.0,
                "resume": True,
            },
            "stage2": {
                "enabled": True,
                "samples_per_agent": 700,
                "batch_size": 10,
                "simple_ratio": 0.40,
                "resume": True,
            },
            "stage3": {
                "enabled": True,
                "samples_per_pair": 50,
                "batch_size": 5,
                "keyword_overlap_threshold": 0.3,
                "max_pairs_per_agent": 5,
                "resume": True,
            },
        },
        "cleaning": {
            "min_question_length": 8,
            "max_question_length": 500,
            "check_label_leakage": True,
            "min_chinese_ratio": 0.3,
            "max_repeat_char_ratio": 0.3,
            "dedup_method": "both",
            "ngram_similarity_threshold": 0.85,
            "ngram_size": 3,
            "min_quality_score": 0.55,
            "max_samples_per_agent": None,
        },
        "merging": {
            "stage1_weight": 0.15,
            "stage2_weight": 0.70,
            "stage3_weight": 0.15,
            "target_per_agent": 1000,
            "max_total": None,
            "global_dedup": True,
            "seed": 42,
        },
        "splitting": {
            "train_ratio": 0.80,
            "val_ratio": 0.10,
            "test_ratio": 0.10,
            "stratified": True,
            "export_formats": True,
        },
        "output": {
            "root_dir": "./output",
            "stage1_raw_dir": "./output/raw/stage1",
            "stage2_raw_dir": "./output/raw/stage2",
            "stage3_raw_dir": "./output/raw/stage3",
            "stage1_clean_dir": "./output/clean/stage1",
            "stage2_clean_dir": "./output/clean/stage2",
            "stage3_clean_dir": "./output/clean/stage3",
            "merged_dataset": "./output/merged/full_dataset.jsonl",
            "split_dir": "./output/split",
            "export_dir": "./output/export",
            "report_dir": "./output/reports",
        },
        "logging": {
            "level": "INFO",
            "file": "./output/logs/pipeline.log",
            "show_progress": True,
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._raw: Dict[str, Any] = {}

    def load(self) -> ConfigNode:
        """
        加载配置并返回 ConfigNode 对象。

        加载优先级（从低到高）：
        1. 内置默认值
        2. YAML 配置文件
        3. 环境变量
        """
        # 从默认值开始
        config = self._deep_copy(self.DEFAULTS)

        # 加载 YAML 文件
        if self.config_path:
            yaml_config = self._load_yaml(self.config_path)
            config = self._deep_merge(config, yaml_config)
        else:
            # 尝试自动发现配置文件
            default_paths = [
                "./config/config.yaml",
                "./config.yaml",
                "/home/ubuntu/agent_router_sft/config/config.yaml",
            ]
            for path in default_paths:
                if Path(path).exists():
                    yaml_config = self._load_yaml(path)
                    config = self._deep_merge(config, yaml_config)
                    logger.info(f"已加载配置文件: {path}")
                    break

        # 环境变量覆盖
        config = self._apply_env_overrides(config)

        # 验证配置
        self._validate(config)

        self._raw = config
        return ConfigNode(config)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载 YAML 文件"""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data or {}
        except ImportError:
            logger.warning("PyYAML 未安装，尝试使用简单解析器")
            return self._simple_yaml_loader(path)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    @staticmethod
    def _simple_yaml_loader(path: str) -> Dict[str, Any]:
        """
        极简 YAML 解析器（仅支持基本键值对，不依赖 PyYAML）。
        当 PyYAML 未安装时作为降级方案。
        """
        result = {}
        current_section = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                # 跳过注释和空行
                if not line or line.lstrip().startswith("#"):
                    continue
                # 顶级键
                if not line.startswith(" ") and ":" in line:
                    key = line.split(":")[0].strip()
                    current_section = key
                    result[key] = {}
                # 二级键值对
                elif current_section and line.startswith("  ") and ":" in line:
                    parts = line.strip().split(":", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip()
                        # 类型转换
                        if v.lower() == "true":
                            v = True
                        elif v.lower() == "false":
                            v = False
                        elif v.lower() == "null":
                            v = None
                        else:
                            try:
                                v = int(v)
                            except ValueError:
                                try:
                                    v = float(v)
                                except ValueError:
                                    v = v.strip('"').strip("'")
                        result[current_section][k] = v

        return result

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """深度合并两个字典，override 中的值覆盖 base"""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _deep_copy(d: Dict) -> Dict:
        """深拷贝字典"""
        import copy
        return copy.deepcopy(d)

    @staticmethod
    def _apply_env_overrides(config: Dict) -> Dict:
        """
        从环境变量覆盖配置。

        支持的环境变量：
          OPENAI_API_KEY     -> config.llm.api_key
          OPENAI_BASE_URL    -> config.llm.base_url
          LLM_MODEL          -> config.llm.model
          DRY_RUN            -> config.llm.dry_run
          OUTPUT_DIR         -> config.output.root_dir
          MAX_WORKERS        -> config.llm.max_workers
        """
        env_map = {
            "OPENAI_API_KEY": ("llm", "api_key"),
            "OPENAI_BASE_URL": ("llm", "base_url"),
            "LLM_MODEL": ("llm", "model"),
            "DRY_RUN": ("llm", "dry_run"),
            "OUTPUT_DIR": ("output", "root_dir"),
            "MAX_WORKERS": ("llm", "max_workers"),
        }

        for env_key, (section, field) in env_map.items():
            value = os.environ.get(env_key)
            if value is not None:
                if section not in config:
                    config[section] = {}
                # 类型转换
                if field in ("dry_run",):
                    value = value.lower() in ("true", "1", "yes")
                elif field in ("max_workers",):
                    value = int(value)
                config[section][field] = value
                logger.debug(f"环境变量覆盖: {env_key} -> config.{section}.{field}")

        return config

    @staticmethod
    def _validate(config: Dict) -> None:
        """验证配置合法性"""
        errors = []

        # 验证比例之和
        splitting = config.get("splitting", {})
        ratio_sum = (
            splitting.get("train_ratio", 0)
            + splitting.get("val_ratio", 0)
            + splitting.get("test_ratio", 0)
        )
        if abs(ratio_sum - 1.0) > 0.01:
            errors.append(
                f"splitting 比例之和必须为1，当前为 {ratio_sum:.3f}"
            )

        # 验证 stage 权重之和
        merging = config.get("merging", {})
        weight_sum = (
            merging.get("stage1_weight", 0)
            + merging.get("stage2_weight", 0)
            + merging.get("stage3_weight", 0)
        )
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                f"merging 权重之和必须为1，当前为 {weight_sum:.3f}"
            )

        # 验证样本数
        gen = config.get("generation", {})
        if gen.get("stage1", {}).get("samples_per_agent", 0) < 10:
            errors.append("stage1.samples_per_agent 不应小于10")

        if errors:
            for err in errors:
                logger.error(f"配置错误: {err}")
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")

        logger.debug("配置验证通过")


def load_config(config_path: Optional[str] = None) -> ConfigNode:
    """
    便捷函数：加载并返回配置对象。

    Args:
        config_path: 配置文件路径，None 则自动发现

    Returns:
        ConfigNode 对象
    """
    loader = ConfigLoader(config_path)
    return loader.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config("/home/ubuntu/agent_router_sft/config/config.yaml")

    print("配置加载成功！")
    print(f"  LLM 模型:         {config.llm.model}")
    print(f"  Dry Run:          {config.llm.dry_run}")
    print(f"  Stage1 样本数:    {config.generation.stage1.samples_per_agent}")
    print(f"  Stage2 样本数:    {config.generation.stage2.samples_per_agent}")
    print(f"  Stage3 样本/对:   {config.generation.stage3.samples_per_pair}")
    print(f"  清洗最低分数:     {config.cleaning.min_quality_score}")
    print(f"  合并目标/智能体:  {config.merging.target_per_agent}")
    print(f"  训练/验证/测试:   {config.splitting.train_ratio}/{config.splitting.val_ratio}/{config.splitting.test_ratio}")
    print(f"  输出目录:         {config.output.root_dir}")
