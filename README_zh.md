# GitMem0

> v0.4.0 | Git x Mem0：纯本地、版本控制的 LLM 记忆系统。

GitMem0 为 AI 智能体提供持久化记忆，无需外部 API 或云服务。它将 Git 式版本追踪与 Mem0 的智能记忆层结合——全部运行在你的本机上。

[English](README.md) | 中文

## 特性

- **纯本地** — 无需 API 密钥，无需云端，一切数据留在本地磁盘
- **高性能** — 守护进程架构，模型只加载一次，查询 <0.05s
- **多信号检索** — 语义搜索 + BM25 + 实体图谱 + 时效性 + 重要性评分
- **置信度衰减** — 记忆随时间自然衰减（指数衰减），过期自动归档到 L2
- **记忆整合** — 重复检测、矛盾解决、自动归纳（事件→洞察）、L2 压缩
- **知识图谱** — 自动提取实体（人物、技术、项目……）和关系
- **版本控制** — 每条记忆像 git commit 一样不可变，支持完整 diff/历史
- **LLM 就绪上下文** — Lost-in-the-Middle 排列，token 预算压缩
- **多语言** — 使用 `paraphrase-multilingual-MiniLM-L12-v2`，支持 50+ 种语言
- **LLM Judge 插件** — 可选的 LLM 辅助评分，无 LLM 时自动降级为规则引擎

## 快速开始

```bash
# 从 PyPI 安装
pip install gitmem0

# 一键配置（自动创建配置、数据库、启动守护进程、安装 hooks）
gitmem0 setup

# 或从源码安装（开发）
pip install -e .
gitmem0 setup

# 首次调用自动启动守护进程（加载模型，约 30s），后续调用 <0.1s
python -m gitmem0.client '{"action":"remember","content":"我喜欢深色模式","type":"preference","importance":0.9}'
python -m gitmem0.client '{"action":"query","message":"用户偏好"}'
python -m gitmem0.client '{"action":"search","query":"深色模式"}'
python -m gitmem0.client '{"action":"stats"}'
```

## 架构

```
用户/LLM
    |
    v
client.py（轻量客户端，<0.1s 启动）
    | TCP socket
    v
auto.py 守护进程（端口 19840，模型只加载一次）
    |
    +---> extraction.py  （多信号重要性评分 + 类型推断）
    +---> retrieval.py   （两阶段：召回 → 重排序）
    +---> context.py     （Lost-in-the-Middle + token 预算）
    +---> decay.py       （指数衰减 + 记忆整合）
    +---> entities.py    （知识图谱提取）
    +---> store.py       （SQLite + FTS5 + L0 LRU 缓存）
    +---> embeddings.py  （sentence-transformers，384 维）
```

## CLI 命令

```bash
# Typer CLI（client.py 的替代方案）
gitmem0 setup                                                # 一键配置
python -m gitmem0.cli add "Python 适合快速原型开发" --type fact --importance 0.7
python -m gitmem0.cli search "Python" --top 3
python -m gitmem0.cli context "用户喜欢什么编程语言"
python -m gitmem0.cli extract "我总是使用类型标注。从不跳过测试。"
python -m gitmem0.cli stats
python -m gitmem0.cli decay --dry-run
python -m gitmem0.cli consolidate --threshold 0.85
python -m gitmem0.cli contradictions --dry-run
python -m gitmem0.cli auto-induct --dry-run
python -m gitmem0.cli compress --dry-run
python -m gitmem0.cli metrics
python -m gitmem0.cli export --format jsonl -o backup.jsonl
python -m gitmem0.cli migrate re-embed
```

## Claude Code 集成

GitMem0 通过 hooks 为 Claude Code 提供记忆后端：

```bash
# 安装 hooks
python hooks/setup_claude_code.py
```

安装后，`UserPromptSubmit` 和 `Stop` hooks 会在对话中自动查询和存储记忆。

## 记忆类型

| 类型 | 示例 | 默认重要性 |
|------|------|-----------|
| `preference`（偏好） | "我喜欢深色模式" | 0.9 |
| `instruction`（指令） | "始终使用类型标注" | 0.9 |
| `insight`（洞察） | "React 更适合这个 UI" | 0.8 |
| `fact`（事实） | "Python 是一种编程语言" | 0.7 |
| `event`（事件） | "周一部署了 v2.0" | 0.4 |

## 配置

编辑 `~/.gitmem0/config.toml`：

```toml
[storage]
db_path = "gitmem0.db"
active_threshold = 0.3

[decay]
lambda_rate = 0.01  # 约 70 天半衰期

[retrieval]
weight_semantic = 0.25
weight_bm25 = 0.15
weight_entity = 0.15
weight_recency = 0.10
weight_importance = 0.20
weight_confidence = 0.15

[embedding]
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
dimension = 384

[llm]
backend = "mimo"      # mimo | openai | claude | ollama
api_key = ""          # API 密钥（ollama 不需要）
base_url = ""         # 自定义 base URL（可选）
model = ""            # 模型名称（可选，使用后端默认值）
```

## LLM Judge 插件

### 支持的后端

| 后端 | `backend` 值 | 默认模型 | 需要 API Key |
|------|-------------|---------|-------------|
| 小米 Token Plan | `mimo` | `MiMo` | 需要（`tp-xxxxx`） |
| OpenAI | `openai` | `gpt-4o-mini` | 需要 |
| Claude (Anthropic) | `claude` | `claude-haiku-4-5-20251001` | 需要 |
| Ollama（本地） | `ollama` | `qwen2.5:7b` | 不需要 |

### 快速配置

在 `~/.gitmem0/config.toml` 中添加：

```toml
# 示例：OpenAI
[llm]
backend = "openai"
api_key = "sk-xxxxx"
model = "gpt-4o-mini"

# 示例：Claude
[llm]
backend = "claude"
api_key = "sk-ant-xxxxx"

# 示例：Ollama（本地，无需 API Key）
[llm]
backend = "ollama"
model = "qwen2.5:7b"

# 示例：小米 Token Plan
[llm]
backend = "mimo"
api_key = "tp-xxxxx"
```

或设置环境变量：

```bash
export GITMEM0_LLM_API_KEY="tp-xxxxx"
export GITMEM0_LLM_BASE_URL="https://token-plan-cn.xiaomimimo.com/v1"
```

守护进程自动检测配置并启用 LLM 辅助评分。API 不可用时自动降级为规则引擎。

### 自定义 LLM Judge

实现 `LLMJudge` 协议即可接入任意 LLM：

```python
from gitmem0.extraction import LLMJudge, MemoryType

class MyJudge(LLMJudge):
    def score_importance(self, content, context="") -> float | None:
        # 返回 0.0-1.0，或 None 使用规则默认值
        ...

    def should_remember(self, content) -> bool | None:
        # 返回 True/False，或 None 使用规则默认值
        ...

    def infer_type(self, content) -> MemoryType | None:
        # 返回 MemoryType，或 None 使用模式匹配
        ...

    def summarize(self, memories: list[str]) -> str | None:
        # 返回摘要，或 None 使用拼接方式
        ...

# 传入 AutoMemory
from gitmem0.auto import AutoMemory
auto = AutoMemory(llm_judge=MyJudge())
```

未配置 LLM Judge 时，系统自动降级为规则引擎。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 许可证

MIT
