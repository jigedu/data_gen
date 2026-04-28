"""
llm_client.py
-------------
LLM调用接口封装模块。

提供统一的LLM调用抽象层，支持：
  - OpenAI兼容接口（包括本地部署的vLLM、Ollama等）
  - 自动重试与错误处理
  - 响应解析与JSON提取
  - 并发请求控制（基于asyncio）
  - Token消耗统计

设计原则：
  - 调用方无需关心底层HTTP细节，只需传入Prompt字典即可
  - 所有解析错误均有明确的异常类型，便于上层处理
  - 支持dry_run模式，不实际调用LLM，用于测试流程
"""

import os
import re
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class LLMResponse:
    """LLM单次调用的响应结果"""
    raw_text: str                    # 原始响应文本
    parsed_items: List[Dict]         # 成功解析出的JSON对象列表
    prompt_tokens: int = 0           # 输入Token数
    completion_tokens: int = 0       # 输出Token数
    latency_ms: float = 0.0          # 请求耗时（毫秒）
    model: str = ""                  # 实际使用的模型名称
    error: Optional[str] = None      # 错误信息（None表示成功）

    @property
    def success(self) -> bool:
        return self.error is None and len(self.parsed_items) > 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMClientConfig:
    """LLM客户端配置"""
    # API连接配置
    api_base: str = "https://api.openai.com/v1"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = "gpt-4o-mini"

    # 生成参数
    temperature: float = 0.85        # 较高温度增加多样性
    top_p: float = 0.95
    max_tokens: int = 2048
    frequency_penalty: float = 0.3   # 减少重复

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 2.0         # 重试间隔（秒）
    timeout: float = 60.0            # 单次请求超时（秒）

    # 并发控制
    max_concurrent: int = 10         # 最大并发请求数

    # 调试模式
    dry_run: bool = False            # True时不实际调用LLM，返回mock数据


# ─────────────────────────── JSON解析工具 ───────────────────────────

class JSONExtractor:
    """
    从LLM响应文本中提取JSON对象或数组。
    LLM输出往往包含多余的文本、markdown代码块等，需要鲁棒地提取。
    """

    # 匹配JSON数组
    ARRAY_PATTERN = re.compile(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', re.DOTALL)
    # 匹配单个JSON对象
    OBJECT_PATTERN = re.compile(r'\{[^{}]*"question"\s*:\s*"[^"]*"[^{}]*"answer"\s*:\s*"[^"]*"[^{}]*\}', re.DOTALL)
    # 匹配markdown代码块
    CODE_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)

    @classmethod
    def extract(cls, text: str, expected_answer: str = "") -> List[Dict[str, str]]:
        """
        从文本中提取所有合法的 {"question": ..., "answer": ...} 对象。

        Args:
            text:            LLM原始输出文本
            expected_answer: 期望的answer值（用于验证，空字符串跳过验证）

        Returns:
            List[Dict]: 提取并验证通过的样本列表
        """
        results = []

        # 步骤1：尝试从markdown代码块中提取
        code_blocks = cls.CODE_BLOCK_PATTERN.findall(text)
        if code_blocks:
            for block in code_blocks:
                results.extend(cls._parse_json_text(block, expected_answer))
            if results:
                return results

        # 步骤2：尝试直接解析整个文本为JSON
        stripped = text.strip()
        results.extend(cls._parse_json_text(stripped, expected_answer))
        if results:
            return results

        # 步骤3：正则匹配JSON数组
        for match in cls.ARRAY_PATTERN.finditer(text):
            results.extend(cls._parse_json_text(match.group(), expected_answer))
        if results:
            return results

        # 步骤4：正则匹配单个JSON对象
        for match in cls.OBJECT_PATTERN.finditer(text):
            try:
                obj = json.loads(match.group())
                validated = cls._validate_item(obj, expected_answer)
                if validated:
                    results.append(validated)
            except json.JSONDecodeError:
                pass

        # 步骤5：逐行尝试解析
        if not results:
            for line in text.split('\n'):
                line = line.strip().rstrip(',')
                if line.startswith('{') and line.endswith('}'):
                    try:
                        obj = json.loads(line)
                        validated = cls._validate_item(obj, expected_answer)
                        if validated:
                            results.append(validated)
                    except json.JSONDecodeError:
                        pass

        return results

    @classmethod
    def _parse_json_text(cls, text: str, expected_answer: str) -> List[Dict[str, str]]:
        """尝试将文本解析为JSON并提取合法样本"""
        results = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    validated = cls._validate_item(item, expected_answer)
                    if validated:
                        results.append(validated)
            elif isinstance(data, dict):
                validated = cls._validate_item(data, expected_answer)
                if validated:
                    results.append(validated)
        except json.JSONDecodeError:
            pass
        return results

    @classmethod
    def _validate_item(
        cls, item: Any, expected_answer: str
    ) -> Optional[Dict[str, str]]:
        """
        验证单个样本的合法性。

        验证规则：
        1. 必须包含 "question" 和 "answer" 字段
        2. 两个字段均为非空字符串
        3. question不能包含answer的内容（防止泄露标签）
        4. 若指定expected_answer，answer必须与之完全一致
        """
        if not isinstance(item, dict):
            return None

        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question or not answer:
            return None

        # answer必须与期望一致
        if expected_answer and answer != expected_answer:
            # 尝试修正：如果answer包含expected_answer则截取
            if expected_answer in answer:
                answer = expected_answer
            else:
                logger.debug(f"Answer不匹配: 期望={expected_answer}, 实际={answer}")
                return None

        # question不能直接包含answer（防止数据污染）
        if answer and answer in question:
            logger.debug(f"Question包含Answer，跳过: {question[:50]}")
            return None

        # question长度合理性检查（5-500字）
        if len(question) < 5 or len(question) > 500:
            return None

        return {"question": question, "answer": answer}


# ─────────────────────────── LLM客户端 ───────────────────────────

class LLMClient:
    """
    LLM调用客户端。
    支持同步调用、异步批量调用与dry_run测试模式。
    """

    def __init__(self, config: LLMClientConfig):
        self.config = config
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._total_tokens = 0
        self._total_requests = 0
        self._failed_requests = 0

        if not config.dry_run:
            try:
                from openai import OpenAI, AsyncOpenAI
                self._sync_client = OpenAI(
                    api_key=config.api_key,
                    base_url=config.api_base,
                    timeout=config.timeout,
                )
                self._async_client = AsyncOpenAI(
                    api_key=config.api_key,
                    base_url=config.api_base,
                    timeout=config.timeout,
                )
                logger.info(f"LLM客户端初始化成功: model={config.model}, base={config.api_base}")
            except ImportError:
                raise ImportError("请安装openai库: pip install openai")
        else:
            logger.info("LLM客户端以dry_run模式启动，不会实际调用API")

    # ─────────── 同步调用 ───────────

    def call(
        self,
        prompt: Dict[str, str],
        expected_answer: str = "",
        extra_params: Optional[Dict] = None,
    ) -> LLMResponse:
        """
        同步调用LLM（带自动重试）。

        Args:
            prompt:          {"system": str, "user": str}
            expected_answer: 期望的answer值，用于验证解析结果
            extra_params:    额外的API参数（覆盖默认配置）

        Returns:
            LLMResponse
        """
        if self.config.dry_run:
            return self._mock_response(expected_answer)

        params = extra_params or {}
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = self._sync_client.chat.completions.create(
                    model=params.get("model", self.config.model),
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    temperature=params.get("temperature", self.config.temperature),
                    top_p=params.get("top_p", self.config.top_p),
                    max_tokens=params.get("max_tokens", self.config.max_tokens),
                    frequency_penalty=params.get(
                        "frequency_penalty", self.config.frequency_penalty
                    ),
                )
                latency_ms = (time.time() - start_time) * 1000

                raw_text = response.choices[0].message.content or ""
                parsed = JSONExtractor.extract(raw_text, expected_answer)

                self._total_tokens += response.usage.total_tokens if response.usage else 0
                self._total_requests += 1

                return LLMResponse(
                    raw_text=raw_text,
                    parsed_items=parsed,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    latency_ms=latency_ms,
                    model=response.model,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        self._failed_requests += 1
        return LLMResponse(
            raw_text="",
            parsed_items=[],
            error=last_error,
        )

    # ─────────── 异步批量调用 ───────────

    async def _async_call_single(
        self,
        prompt: Dict[str, str],
        expected_answer: str,
        semaphore: asyncio.Semaphore,
    ) -> LLMResponse:
        """单个异步调用（受semaphore并发控制）"""
        async with semaphore:
            if self.config.dry_run:
                await asyncio.sleep(0.01)  # 模拟网络延迟
                return self._mock_response(expected_answer)

            last_error = None
            for attempt in range(self.config.max_retries):
                try:
                    start_time = time.time()
                    response = await self._async_client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": prompt["user"]},
                        ],
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=self.config.max_tokens,
                        frequency_penalty=self.config.frequency_penalty,
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    raw_text = response.choices[0].message.content or ""
                    parsed = JSONExtractor.extract(raw_text, expected_answer)

                    return LLMResponse(
                        raw_text=raw_text,
                        parsed_items=parsed,
                        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                        completion_tokens=response.usage.completion_tokens if response.usage else 0,
                        latency_ms=latency_ms,
                        model=response.model,
                    )

                except Exception as e:
                    last_error = str(e)
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))

            return LLMResponse(raw_text="", parsed_items=[], error=last_error)

    async def batch_call_async(
        self,
        tasks: List[Tuple[Dict[str, str], str]],
    ) -> List[LLMResponse]:
        """
        异步批量调用LLM。

        Args:
            tasks: List of (prompt_dict, expected_answer)

        Returns:
            List[LLMResponse]，与tasks一一对应
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        coroutines = [
            self._async_call_single(prompt, expected_answer, semaphore)
            for prompt, expected_answer in tasks
        ]
        return await asyncio.gather(*coroutines)

    def batch_call(
        self,
        tasks: List[Tuple[Dict[str, str], str]],
    ) -> List[LLMResponse]:
        """
        同步接口的批量调用（内部使用asyncio）。
        适用于在非async上下文中调用。
        """
        return asyncio.run(self.batch_call_async(tasks))

    # ─────────── Mock模式 ───────────

    def _mock_response(self, expected_answer: str) -> LLMResponse:
        """
        dry_run模式下返回mock响应，用于测试流程正确性。
        不实际调用LLM，生成格式合法的占位数据。
        """
        import random
        mock_questions = [
            f"我需要使用{expected_answer}相关的功能，请帮我处理",
            f"关于{expected_answer}，我有一个问题需要咨询",
            f"请问如何使用{expected_answer}来解决我的问题",
        ]
        question = random.choice(mock_questions)
        # 注意：mock数据中question包含answer，真实清洗时会被过滤
        # 这里仅用于测试流程，实际生成时不会出现此情况
        mock_item = {"question": f"[MOCK] {question}", "answer": expected_answer}
        return LLMResponse(
            raw_text=json.dumps(mock_item, ensure_ascii=False),
            parsed_items=[mock_item],
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=10.0,
            model="mock-model",
        )

    # ─────────── 统计信息 ───────────

    def get_stats(self) -> Dict[str, Any]:
        """返回累计调用统计信息"""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                (self._total_requests - self._failed_requests) / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
            "total_tokens": self._total_tokens,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试dry_run模式
    config = LLMClientConfig(dry_run=True)
    client = LLMClient(config)

    prompt = {
        "system": "你是一个数据生成专家",
        "user": '请生成一个JSON：{"question": "测试问题", "answer": "测试答案"}',
    }

    resp = client.call(prompt, expected_answer="测试答案")
    print(f"Mock响应: success={resp.success}, items={resp.parsed_items}")

    # 测试JSON提取器
    test_text = '''
    好的，这是生成的样本：
    ```json
    [
      {"question": "如何查看航班偏航情况？", "answer": "民航绕偏航处置"},
      {"question": "飞机备降怎么处理", "answer": "民航绕偏航处置"}
    ]
    ```
    '''
    items = JSONExtractor.extract(test_text, "民航绕偏航处置")
    print(f"\nJSON提取测试: 提取到 {len(items)} 条")
    for item in items:
        print(f"  {item}")
