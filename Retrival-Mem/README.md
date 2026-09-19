# AutoRetri：V4 长期记忆与检索评测框架

AutoRetri 面向长对话智能体的记忆构建、检索、回答与 LoCoMo、
ProactiveMemBench 等基准评测。当前唯一活跃的内置后端是 **V4**；V1、
V2、V3 与 Router 蒸馏工具保存在 `archive/legacy_memory/`，不再参与运行时注册。

## V4 核心约束

- Participant Scope 内构建 `chain_head`、`event`、`semantic_state` 图。
- SQLite schema version 为 4，backend identity 为 `v4`；只支持全新构建，
  不接受历史数据库。
- V4 对外声明的 retrieval plan schema version 为 6；V4 不执行旧静态计划或
  Router artifact。
- 强 Controller 负责 Query 生成、证据充分性、缺失 facet 和动作选择。
- 规则层负责中英文意图识别、固定预算、Query 去重、动作合法性和排序。
- Controller 每轮看到最多 32 条完整累计证据；summary 和 structured facts 不截断。
- 持久图不包含 `SEMANTIC_SIMILAR`、`SHARED_ENTITY`、`TEMPORAL_OVERLAP`。
- 实体补全、跨实体补全和同时性检索分别使用受 Scope/Participant 可见性约束的
  `ENTITY_LOOKUP`、`CROSS_ENTITY_LOOKUP` 和 `TIME_OVERLAP_LOOKUP` 虚拟动作。

固定检索预算为：简单首轮 8，复杂首轮 12，后续全局 6，单 anchor 扩展
3，总候选 32，最终返回 12，最多 4 轮。

## 快速开始

活跃配置在 `configs/`；所有配置都从 `.env` 读取端点与密钥：

| 配置 | 用途 |
| --- | --- |
| `configs/default.yaml` | 默认 V4 评测：Ollama Qwen 控制/窗口规划，DashScope `glm-5.1` 构建与回答，OpenRouter `openai/gpt-4o-mini` 评分 |
| `configs/locomo_v4_openrouter.yaml` | 回答模型换成 OpenRouter `google/gemini-2.5-flash` |
| `configs/locomo_v4_gemini_build_answer.yaml` | 记忆构建与回答都用 OpenRouter `google/gemini-2.5-flash` |
| `configs/locomo_v4_claude_openrouter.yaml` | 回答模型换成 OpenRouter `anthropic/claude-sonnet-4.5` |
| `configs/locomo_v4_sonnet45_openrouter.yaml` | 同上，按 Claude Sonnet 4.5 命名的副本 |
| `configs/ollama_units.yaml` | 可选的多单元 Ollama 推理配置（single/dual 混合），配合 `--ollama-units-config` |
| `configs/ollama_units_3single.yaml` | 三个 single 模式 Ollama 单元的示例 |

已退役的配置保存在 `configs/archive/`。

下载基准并运行评测：

```bash
conda run -n pytorch python scripts/download_benchmarks.py
conda run -n pytorch python scripts/run_eval.py \
  --config configs/default.yaml --benchmarks locomo
```

只复用回答重新评分时追加 `--locomo-rejudge-from runs/<原时间戳>`；结果写入
新的 `runs/<时间戳>/`。

原生 V4 run 快照在通过 benchmark、数据、模型、配置与 schema 校验后可以复用：
`--locomo-reuse-memory-from`、`--locomo-reuse-predictions-from` 分别复用记忆与
预测，`--resume-run-dir` / `--fork-run-from` 在原 run 上续跑或派生，
`--invalidate-stages` 指定需要失效的阶段。非 V4 或异构数据库会被拒绝。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  conda run -n pytorch python -m unittest discover -s tests
```

V4 合同测试集中在 `tests/test_v4_contracts.py`，覆盖版本隔离、意图规则、
Controller 协议、完整证据、固定预算、虚拟查询与旧数据库拒绝。

## 目录

```text
src/memory/v4/          # V4 构建、状态机、SQLite、FAISS、Controller 与检索
src/memory/retrieve/    # 活跃 multi-round 运行时和公共检索 DTO
src/agent/              # 回答执行、分类执行与 Prompt 组装
src/evaluation/         # benchmark runner、复现快照与评分
scripts/                # download_benchmarks.py、run_eval.py
configs/                # 活跃 V4 配置
configs/archive/        # 已退役配置
tests/                  # 单元与合同测试
benchmark/              # 基准数据（只有 LoCoMo 入库，其余本地获取）
archive/legacy_memory/  # V1/V2/V3、静态检索和 Router 蒸馏历史实现
archive/                # 清理归档：old_prompt.py、old_code.py、old_test.py、unused_tests/
```

迁移设计与协议细节见
`docs/memory_v4_migration_and_retrieval_optimization_plan.md`，历史轨迹的
无损证据协议估算见 `docs/memory_v4_offline_comparison_report.md`。
