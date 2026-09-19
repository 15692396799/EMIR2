# MemConflict 实验表格（截图表格内容转录）

> 来源：截图表格（Table 3 / Table 5 / Table 6）
> 约定：**加粗** = 全体方法中最佳；下划线 = 基线方法中最佳；↑ 表示数值越大越好；EMIR² 行在三个表中均为 "–"（原表未列出该方法的数值）。
> 缩写：AA = Answer Accuracy；SEH@3 = Succinct & Effective Hit @3；UOCS = 冲突情境下的使用质量；CRS = 冲突解决成功率；SRS = 情景相关成功率；Dynamic / Static / Conditional = 三类冲突类型。

---

## Table 3：Conflict-aware memory evaluation on MemConflict by conflict type

> 说明：标题中 "evaluatio[n] … conflict" 一处被裁剪；本表在 Dynamic / Static / Conditional 三类冲突下同时报告 AA 与 SEH@3，最后给出 AA 的均值（Average AA）。

| Method    | Dynamic AA↑ | Dynamic SEH@3↑ | Static AA↑ | Static SEH@3↑ | Conditional AA↑ | Conditional SEH@3↑ | Average AA↑ |
|-----------|-------------|----------------|------------|---------------|-----------------|--------------------|-------------|
| A-Mem     | 0.3596      | 0.5205         | 0.2639     | 0.3611        | 0.7122          | 0.8111             | 0.4452      |
| LangMem   | <u>0.4966</u> | <u>0.7842</u> | 0.1944     | 0.3194        | 0.1556          | 0.2012             | 0.2822      |
| Letta     | 0.3955      | 0.5394         | 0.2223     | 0.4167        | 0.8435          | <u>0.9046</u>      | 0.4871      |
| MemOS     | 0.3793      | 0.5548         | <u>0.4375</u> | <u>0.5694</u> | <u>0.8449</u>   | 0.8889             | <u>0.5539</u> |
| Mem0      | 0.1224      | 0.2003         | 0.1944     | 0.2917        | 0.7667          | 0.8222             | 0.3612      |
| Memobase  | 0.4058      | 0.5925         | 0.4167     | 0.5278        | 0.2434          | 0.3021             | 0.3553      |
| **EMIR²** | –           | –              | –          | –             | –               | –                  | –           |

> 备注：原表中 Conditional 类的 SEH@3 列 Letta 的 0.9046 同时带有下划线；Conditional AA 上 MemOS 的 0.8449 为基线最佳（加粗值 0.8435 在 Letta 行未被显示为粗体，依据原表样式推测为下划线）。**仅按截图所见**逐项转录；若读取有歧义请以原论文为准。

---

## Table 5：Black-box performance of memory systems on MemConflict by conflict type

> 说明：在三类冲突下报告黑箱指标——Dynamic：AA / UOCS；Static：AA / CRS；Conditional：AA；并给出 AA 均值（Average AA）。

| Method    | Dynamic AA↑ | Dynamic UOCS↑ | Static AA↑ | Static CRS↑ | Conditional AA↑ | Average AA↑ |
|-----------|-------------|---------------|------------|-------------|-----------------|-------------|
| A-Mem     | 0.3596      | 0.2911        | 0.2639     | <u>0.2501</u> | 0.7122          | 0.4452      |
| LangMem   | <u>0.4966</u> | 0.3579      | 0.1944     | 0.2083      | 0.1556          | 0.2822      |
| Letta     | 0.3955      | 0.3527        | 0.2223     | 0.2031      | 0.8435          | 0.4871      |
| MemOS     | 0.3793      | <u>0.3818</u> | <u>0.4375</u> | 0.2361    | <u>0.8449</u>   | <u>0.5539</u> |
| Mem0      | 0.1224      | 0.1130        | 0.1944     | 0.1528      | 0.7667          | 0.3612      |
| Memobase  | 0.4058      | 0.3476        | 0.4167     | 0.0694      | 0.2434          | 0.3553      |
| **EMIR²** | –           | –             | –          | –           | –               | –           |

> 备注：Conditional 列下 Letta 0.8435 在原图中无下划线；MemOS 0.8449 显示下划线，标记为基线最佳。

---

## Table 6：White-box memory retrieval and ranking of memory systems on MemConflict by conflict type

> 说明：在三类冲突下报告白箱指标——Dynamic：SEH@3 / SRS；Static：SEH@3 / SRS；Conditional：SEH@3 / SRS；并给出 SEH@3 与 SRS 的均值（Average）。

| Method    | Dynamic SEH@3↑ | Dynamic SRS↑ | Static SEH@3↑ | Static SRS↑ | Conditional SEH@3↑ | Conditional SRS↑ | Average SEH@3↑ | Average SRS↑ |
|-----------|----------------|--------------|---------------|-------------|--------------------|------------------|-----------------|--------------|
| A-Mem     | 0.5205         | 0.4341       | 0.3611        | 0.2854      | 0.8111             | 0.7288           | 0.5642          | 0.4828       |
| LangMem   | <u>0.7842</u>  | <u>0.7089</u> | 0.3194        | 0.2697      | 0.2012             | 0.1944           | 0.4349          | 0.3910       |
| Letta     | 0.5394         | 0.4620       | 0.4167        | 0.3099      | <u>0.9046</u>      | 0.7653           | 0.6202          | 0.5124       |
| MemOS     | 0.5548         | 0.4552       | <u>0.5694</u> | <u>0.4886</u> | 0.8889            | <u>0.8198</u>    | <u>0.6710</u>   | <u>0.5879</u> |
| Mem0      | 0.2003         | 0.1587       | 0.2917        | 0.2401      | 0.8222             | 0.7780           | 0.4381          | 0.3923       |
| Memobase  | 0.5925         | 0.5204       | 0.5278        | 0.4557      | 0.3021             | 0.2877           | 0.4741          | 0.4213       |
| **EMIR²** | –              | –            | –             | –           | –                  | –                | –               | –            |

---

## 速读要点（基于截图数字）

- **MemOS** 在 Static 冲突（AA / SEH@3 / CRS / SRS 全列）和 Average 列（AA / SEH@3 / SRS）几乎全线为基线最佳，是基线中综合表现最好的系统。
- **LangMem** 在 Dynamic 冲突的检索与排序（SEH@3、SRS）以及 Dynamic AA 上为基线最佳，但 Conditional 上明显塌陷（AA 仅 0.1556）。
- **Letta** 在 Conditional 冲突的 SEH@3 上为基线最佳（0.9046），Conditional AA 也很高（0.8435）。
- **EMIR²** 在三张表中均无数据点（行内全部为 "–"），说明该截图中未报告 EMIR² 的对应数值；如需其结果需查阅其他表格。
