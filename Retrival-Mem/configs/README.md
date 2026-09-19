# 运行配置

三份配置均使用 Ollama Qwen 检索，DashScope GLM-5.1 构建记忆，OpenRouter `openai/gpt-4o-mini` 普通请求评分。

| 配置 | 回答模型 |
| --- | --- |
| `default.yaml` | DashScope `glm-5.1` |
| `locomo_v4_openrouter.yaml` | OpenRouter `google/gemini-2.5-flash` |
| `locomo_v4_claude_openrouter.yaml` | OpenRouter `anthropic/claude-sonnet-4.5` |

云 API 单次超时 60 秒；Ollama 保留原超时。密钥、端点从 `.env` 读取。`ollama_units.yaml` 是可选的 Ollama 节点配置。

```bash
conda run --no-capture-output -n pytorch python scripts/run_eval.py \
  --config configs/default.yaml --benchmarks locomo
```

仅复用回答重新评分时追加 `--locomo-rejudge-from runs/<原时间戳>`。
结果自动写入新的 `runs/<时间戳>/`。
