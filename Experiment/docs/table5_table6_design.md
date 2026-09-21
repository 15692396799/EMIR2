# MemConflict 表 5 / 表 6（黑白盒）实验设计

对照文档：[`doc_by_human/memconflict_tables_3_5_6.md`](../../doc_by_human/memconflict_tables_3_5_6.md)。
本文回答两件事：**表 5 / 表 6 该怎么跑**，以及**能不能和 conflict-aware（表 3）同步跑**。

---

## 1. 结论

**可以同步跑，而且推荐同步跑：表 3 / 表 5 / 表 6 是同一次 judge 调用的三种投影。**

| 问题 | 答案 |
| --- | --- |
| 需要为表 5 / 表 6 再跑一遍实验（重新 ingest + 重新回答）吗？ | 不需要。三者共用同一份 `results.jsonl` |
| 需要为表 5 / 表 6 再调一次 judge 吗？ | 不需要。judge 一次回答里同时给出 AA、UOCS/CRS、support rank |
| 能不能和 conflict-aware 并行（persona 级并行）一起跑？ | 能。并行只改调度，不改结果；三张表在同一次 `run_scoring.py` 里一起产出 |
| 有没有必须额外调用 judge 的情况？ | 只有一种：想测"判分窗口不同"的对照（E2 的对照组），见 §6 |

落地情况（本次已完成）：`run_scoring.py` 一次写出 `table3.md` / `table5.md` / `table6.md` / `tables.md`；
新增离线重投影工具 `tools/rebuild_tables.py`（0 次 LLM 调用即可重建三表）；离线测试 108 项全绿。

---

## 2. 证据链

### 2.1 上游 schema：表 5 / 表 6 本来就来自同一套字段

`MemConflict/Ablation/Question_Style/eval_scoring.py`（与 `MemConflict/Evaluation/eval_scoring.py` 同源）中：

```python
METRIC_SCHEMAS = {
  "dynamic_conflict":     {"black_box_metrics": ["dynamic_answer_accuracy",
                                                 "update_awareness_and_order_consistency_score"],
                           "white_box_metrics": ["updated_evidence_hit_at_3",
                                                 "updated_evidence_log_rank_score_at_3"]},
  "static_conflict":      {"black_box_metrics": ["static_answer_accuracy",
                                                 "conflict_recognition_score"],
                           "white_box_metrics": ["truth_evidence_hit_at_3",
                                                 "truth_evidence_log_rank_score_at_3"]},
  "conditional_conflict": {"black_box_metrics": ["conditional_answer_accuracy"],
                           "white_box_metrics": ["correct_condition_evidence_hit_at_3",
                                                 "correct_condition_evidence_log_rank_score_at_3"]},
}
```

即：**表 5 = black_box_metrics（AA + UOCS/CRS），表 6 = white_box_metrics（SEH@K + SRS）**，
表 3 = 两者的交叉（AA + SEH@3）。三者不是三次实验，而是同一次判分的三个切片。

### 2.2 上游就是"判一次、派生多窗口"

* `build_llm_judge_prompt()` 对每道题只发一次请求，返回体里同时包含答案分、诊断分和 `*_first_support_rank`。
* `build_white_box_result_by_k()` 用**同一个** `support_rank` 派生出 `WHITE_BOX_TOP_K_VALUES = [2, 3, 5]` 的所有白箱列，
  `log_rank = 1 / log2(rank + 1)`，rank 超出窗口记 0。

所以上游的三张表本身就是在一次 pass 里算出来的，论文数字不存在"表 5 另跑一遍"的可能。

### 2.3 我们这边已经采集了全部字段

`Experiment/memconflict_eval/prompts.py` 的 judge schema 是
`{answer_accuracy, conflict_handling, support_rank, reasoning}`，其中 `conflict_handling` 的提示词明确写为
dynamic 的 UOCS、static 的 CRS；`judging.py` 的 `_coerce_accuracy / _coerce_flag / _coerce_rank`
与上游的 `parse_trinary_score_value / parse_binary_value / parse_support_rank` 逐条对应
（AA 三值 0/0.5/1、conditional 二值、诊断二值、rank 截断到窗口）。

缺的只是"渲染成表 5 / 表 6"这一步，本次已补齐。

### 2.4 截图数字的算术反推（验证论文口径 = 我们的口径）

对截图转录值做分母反推：

| 列 | 与哪个分母自洽 | 含义 |
| --- | --- | --- |
| Dynamic AA（0.5 网格）、Dynamic SEH@3（0/1）、Dynamic UOCS（0/1） | **N = 584** | dynamic 三列共用同一批题 |
| Static AA、Static SEH@3、Static CRS | **N = 72** | 发布版每个 persona 恰好 12 道 static 题 → 72 = 6 个 persona |
| Conditional AA / SEH@3 | 与发布版数据任一合理分母都不自洽 | 论文很可能用的是更早或更小的数据版本 |

两个推论：

1. **UOCS / CRS 在论文里是"每题 0/1 再求均值"**，和我们的二元口径一致，可以对齐；
2. 论文表 3/5/6 的同一冲突类型共用一套分母，佐证"一次判分、多表输出"。

同时这也说明：**发布版 30 personas / 3750 题（2946 dynamic、360 static、444 conditional）与论文分母（584 / 72）不同**，
论文基线数字不能直接和我们的全量结果并列比较，见 §6 的 E4。

---

## 3. 单元格 → 字段 → 计算位置

| 表 | 列 | 我们的字段 | 计算 | 代码 |
| --- | --- | --- | --- | --- |
| 3 | Dynamic/Static/Conditional AA | `Evaluation.Answer_Accuracy` | 类型内均值 | `metrics.aggregate` |
| 3 | Dynamic/Static/Conditional SEH@K | `Evaluation.Support_Rank` | `1[1 ≤ rank ≤ K]` 均值 | 同上（`white_box_k`） |
| 3 | Average AA | 三列 AA | 三类等权平均 | `BenchmarkMetrics.average_aa` |
| 5 | Dynamic UOCS | `Evaluation.Conflict_Handling`（dynamic 桶） | 均值 | `ConflictMetrics.uocs` |
| 5 | Static CRS | `Evaluation.Conflict_Handling`（static 桶） | 均值 | `ConflictMetrics.crs` |
| 5 | Conditional AA | 同表 3 | 均值 | — |
| 6 | Dynamic/Static/Conditional SRS | `Evaluation.Support_Rank` | `1/log2(rank+1)`，窗口外记 0 | `srs_at_k` |
| 6 | Average SEH@K / Average SRS | 三列 | 三类等权平均 | `BenchmarkMetrics.average_*` |
| 6 附 | 任意 Top-K 窗口 | 同上 | 由同一个 rank 派生 | `aggregate_by_k` |

注意：conditional 题没有 UOCS/CRS 定义（表 5 的 conditional 列只有 AA），
所以 `metrics` 现在把 conditional 桶的 `Conflict_Handling` 置为 `None`，避免把噪声当指标。

---

## 4. 本次代码改动

| 文件 | 改动 |
| --- | --- |
| `memconflict_eval/metrics.py` | `white_box_k` 窗口化聚合；`srs_at_k`；`aggregate_by_k`；`ConflictMetrics.uocs/crs`；`render_table5`、`render_table6`、`render_white_box_by_k`、`render_all_tables`；conditional 诊断置空 |
| `run_scoring.py` | 新增 `--white-box-k`（如 `2,3,5`）与 `--judge-top-k`；写出 `table3/5/6.md` + `tables.md`；`metrics.json` 增加 `White_Box_By_K`、`Judge_Top_K`、`Questions_With_Short_Memory_Window`；`--stored-top-k` 不足以覆盖 judge 窗口时告警 |
| `tools/rebuild_tables.py`（新） | 从已有 `scores.jsonl` 离线重建三表，无 LLM、无凭证 |
| `tests/test_experiment.py` | +11 项测试：表 5/6 列与取值、三表重叠单元格严格相等、窗口切分、judge 每题只调一次、离线重建等价、CLI 新参数、conditional 无诊断 |

---

## 5. 怎么跑

### 5.1 同步跑三表（推荐：每题只调 judge 一次）

```powershell
# 1) 作答阶段：把检索到的 5 条记忆落盘，供 Top-2/3/5 三窗口共用
python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml --stored-top-k 5

# 2) 判分阶段：一次 judge，产出表 3 / 5 / 6
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp> --top-k 3 --white-box-k 2,3,5
```

输出：`table3.md`、`table5.md`、`table6.md`、`tables.md`，以及带 `White_Box_By_K` 的 `metrics.json`。
`--white-box-k 2,3,5` 会把 judge 窗口自动抬到 Top-5（与上游一致），这是唯一需要注意的耦合点。

### 5.2 已有 run 只要重出表（0 次 LLM 调用）

```powershell
python Experiment\tools\rebuild_tables.py --run-dir Experiment\runs\<timestamp> --white-box-k 2,3,5
```

### 5.3 与并行调度一起用

```powershell
python Experiment\run_scoring.py --run-dir Experiment\runs\<timestamp> `
  --top-k 3 --white-box-k 2,3,5 --judge-workers 4 --ollama-units http://172.26.94.12:41135
```

`--judge-workers` / `--ollama-units` 是 persona 级并行（point 16/17），三表共享判分结果，
因此并行度与容器数只影响墙钟，不影响任何一个单元格（E3 用来验证这一点）。

---

## 6. 实验设计

| ID | 目的 | 输入 | 命令 | 通过判据 | LLM 成本 |
| --- | --- | --- | --- | --- | --- |
| E1 | 离线重投影等价性 | 已判分的 `scores.jsonl` | `tools/rebuild_tables.py` vs `run_scoring.py` 输出 | 三表所有重叠单元格逐字符相同 | 0 |
| E2 | 判分窗口敏感性（唯一需要"不同步"的实验） | 同 `results.jsonl` 判两遍 | A：`--judge-top-k 5 --white-box-k 2,3,5`；B：`--judge-top-k 3 --top-k 3` | A 的 `SEH@3/SRS@3` 与 B 的题级一致率 ≥ 95%，AA/UOCS/CRS 完全一致 | 2× judge 调用/题 |
| E3 | 并行不改结果 | 同 `results.jsonl` | `--judge-workers 1` vs `--judge-workers 4`（单/多容器） | 三表逐单元格完全相同 | 2× judge 调用/题 |
| E4 | 分母/子集敏感性 | 全量 `scores.jsonl` | 重抽样 6 personas × 20 次离线重算 | 报出 `Average AA / SEH / SRS` 的分布与全量值的 Δ、95% 区间 | 0 |
| E5 | judge 失败题的影响 | 全量 `scores.jsonl` | 含/不含 `Judge_Error` 题各算一次 | 错误率 < 1% 可忽略，否则重跑失败题 | 0（重跑失败题另计） |
| E6 | 与论文基线并表 | 论文表 + 我们的三表 | 人工/脚本拼接 | EMIR² 行的 7 / 6 / 8 列齐全 | 0 |

### E1 —— 先做，它锁死"三表同源"

```powershell
python Experiment\run_scoring.py     --run-dir Experiment\runs\<ts> --top-k 3 --white-box-k 2,3,5
python Experiment\tools\rebuild_tables.py --run-dir Experiment\runs\<ts> --top-k 3 --white-box-k 2,3,5
# 判据：两次写出的 table3.md / table5.md / table6.md 数值列一致
```

已在归档数据上预跑（`Experiment/archive/2026-09-19_smoke_eval_large/scores.jsonl`，2 题）：
Dynamic AA `0.5000`、SEH@3 `0.5000`、UOCS `0.5000`、SRS `0.5000`，
与归档 `metrics.json` 里的 `Answer_Accuracy 0.5 / Conflict_Handling 0.5 / SRS 0.5` 一致 —— 重建通路可信。

### E2 —— 唯一"不能同一次 judge"的对照

其余实验都能在"一次 judge"里完成；E2 想要回答的是
"judge 看到 Top-3 还是 Top-5，会不会改变它对支持记忆的排序判断"。
只有这个对照需要两次判分（成本 = 2× judge 调用/题，建议先缩到 1 个 persona ≈ 122 题）。

### E3 —— 同步并行的安全性

并行是 persona 级的，judge 判定本身与调度无关。判据是"全等"；
如果出现差异，先怀疑 judge 采样温度（应在配置里固定）而不是并行实现。

### E4 —— 为什么必须做

论文三表的 dynamic / static 分母是 584 / 72，对应约 6 个 persona；
发布版是 30 personas / 3750 题（2946 / 360 / 444）。两者不是同一批题，
所以要么自己重跑基线方法，要么在论文中把"全量结果"与"论文子集"分开陈述。
E4 给出抽样的波动范围，用来判断全量值是否落在论文基线的可比区间内。

---

## 7. 必须写进论文/README 的口径说明

1. **窗口**：`--judge-top-k` 必须 ≥ 要报告的最大 K，且 `--stored-top-k` 要落盘同样多的记忆，
   否则白箱列会被"看不见的 miss"污染（`metrics.json` 的 `Questions_With_Short_Memory_Window` 会报这个数）。
2. **子集**：本文三表与论文三表的分母不同（584/72 vs 2946/360/444），数值不可直接对照。
3. **Conditional 无 UOCS/CRS**：表 5 该列只有 AA，我们已把该诊断置空。
4. **UOCS/CRS 口径**：我们用的是 LLM judge 的二元判定；上游另有基于关键词的 rule-based fallback
   （`has_update_order_signal` / `has_conflict_recognition_signal`），论文需声明采用哪一种。
5. **Average 是三类等权平均**，不是题目加权；dynamic 题多不会把 Average 拉向 dynamic。
6. **同一次判分带来的一致性保证**：同一 run 内表 3 的 SEH@3 与表 6 的 SEH@3、表 3 的 AA 与表 5 的 AA
   在构造上必然相等。这既省成本，也堵住了"同一篇论文两张表数字不一致"的审稿风险。
   反过来，如果分两次判分，LLM 抖动会让这两张表出现微小漂移——这正是应该同步跑的理由。

---

## 8. 验收清单

```powershell
python -m unittest discover -s Experiment\tests -t .        # 108 项，离线，不调 LLM
python Experiment\tools\rebuild_tables.py --scores Experiment\archive\2026-09-19_smoke_eval_large\scores.jsonl --white-box-k 2,3,5
```

* [ ] 三张表能在同一次 `run_scoring.py` 里产出，`metrics.json` 含 `White_Box_By_K`
* [ ] 表 3 的 AA / SEH 与表 5 / 表 6 的重叠列完全相等
* [ ] `table6.md` 的 by-K 明细里，@3 列与表 3 一致
* [ ] `Questions_With_Short_Memory_Window == 0`（否则重跑实验阶段并提高 `--stored-top-k`）
* [ ] E3 的两组并行结果逐单元格相等
