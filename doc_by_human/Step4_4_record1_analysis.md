# Step4_4.jsonl 数据结构分析（第一条记录）

分析对象：`MemConflict/Data/Step4_4.jsonl` 的第一条记录（只读该行）。

- 记录 ID：`3c2e5fe5-a0fc-7e3c-b05c-7104ad748705`
- 单行字节数：1,345,767 B（约 1.28 MiB）
- 文件整体：39,671,742 B（37.8 MiB），共 30 条记录

---

## 1. 顶层字段（14 个）

| 字段 | 类型 | 规模 | 说明 |
| --- | --- | --- | --- |
| `ID` | string | uuid | persona 唯一标识 |
| `Fixed_Profile` | object | 6 keys | 不可变事实：姓名、性别、生日、出生地、学历、家庭 |
| `Dynamic_Profile` | object | 7 keys | 可演化状态：居住地、婚姻、子女、职业、工作、健康、社交 |
| `Preference_Profile` | object | 6 keys | 六类偏好，每项为 `偏好项 -> 适用条件` |
| `Personality` | object | 2 keys | MBTI + 标签数组 |
| `Life_Goal` | object | 2 keys | 长期目标类型与描述 |
| `Others_Profile` | object | 8 keys | 8 位相关人物，每人 12–15 个字段 |
| `Full_Session_Chain` | array | 53 | 全部会话链 |
| `metadata` | object | 1 key | 仅 `persona_seed` |
| `token_cost` | object | 2 keys | `current_stage` / `cumulative` 成本 |
| `Total_Dialogue_Token_Length` | int | 206,745 | 全链对话 token 数 |
| `Valid_Session_Dialogue_Count` | int | 53 | 有效会话数 |
| `Total_Session_Question_Count` | int | 122 | 总题数 |
| `Triggered_Session_Count` | int | 42 | 触发提问的会话数 |

### 体积分布

`Full_Session_Chain` 占整条记录 JSON 字符数的 **98.8%**，其余字段合计不足 1.2%。

| 字段 | 字符数 | 占比 |
| --- | ---: | ---: |
| `Full_Session_Chain` | 1,318,993 | 98.8% |
| `Others_Profile` | 10,760 | 0.8% |
| `Life_Goal` | 1,841 | 0.1% |
| `Preference_Profile` | 1,533 | 0.1% |
| `Dynamic_Profile` | 906 | 0.1% |
| 其余 | < 600 | < 0.1% |

---

## 2. 档案字段结构

### Fixed_Profile

```text
Name / Gender / Birthdate / Birthplace
Education_Background: {Highest_Degree, Major, University}
Family_Information:  {Father: {Name, Birth_Date}, Sibling_1: {Type, Name, Birth_Date}}
```

本记录取值：`Jackson Andrews` / Male / 1988-05-14 / Los Angeles, California, USA / Bachelor, Kinesiology, California State University, Long Beach。

### Dynamic_Profile

```text
Residence          : "Darwin, Australia"
Marital_Status     : {Status}
Children_Status    : {Status, Child_1{Name,Birthdate}, Child_2{...}}
Career_Status      : {Employment_Status, Company_Name, Job_Title, Industry, Monthly_Income, Savings_Amount}
Work_Status        : {Current_State}
Health_Status      : {Physical_Health, Mental_Health}
Social_Relationships: {Contacts: {Contacts_1..4}, Social_Status: {Current_State}}
```

### Preference_Profile

六类：`Clothing_ / Beverage_ / Game_ / Reading_ / Food_ / Sport_Preference`，每类 2–4 项。每项的值是**适用条件**而不是打分，例如：

```text
Tight Yoga Pants  -> "Training, rehearsal, or workout"
Neutral Linen Set -> "Client meetings, interviews, or low-key public appearances"
```

这种"偏好项 + 条件"的写法正是 conditional conflict 的构造素材。

### Others_Profile

8 位相关人物：`Father`、`Sibling_1`、`Child_1`、`Child_2`（各 12 字段）、`Contacts_1`–`Contacts_4`（各 15 字段）。字段集合：

```text
Name / Gender / Birthdate / Birthplace / Education_Background / Residence
Career_Status / Work_Status / Health_Status / Preference_Profile
Relationship_To_User / Source_Key            # Contacts_* 额外多 3 个字段
```

每位人物都带完整的职业、健康、偏好画像，是注入"语义相似但属于他人"干扰项的来源。

---

## 3. Full_Session_Chain：53 个会话

时间跨度 **2022-01-03 -> 2026-01-15**，53 个日期互不重复且严格递增。

### 字段出现情况（不固定，需按存在性判断）

| 会话字段 | 出现次数 | 类型 | 备注 |
| --- | ---: | --- | --- |
| `Session_ID` | 53 | int | 取值 0–52 |
| `Date` | 53 | str | 53 个不同日期，严格递增 |
| `Session_Type` | 53 | str | 4 种取值 |
| `Event_Types` | 53 | list | 全部非空，最多 2 个 |
| `Session_Outline` | 53 | str | 53 个不同摘要 |
| `Session_Dialogue` | 53 | dict | 对话主体 |
| `Session_Dialogue_Token_Length` | 53 | int | 2500–6636 |
| `Session_Question_Count` | 53 | int | 0–8 |
| `Session_Questions` | 53 | list | 42 个非空，最多 8 题 |
| `Question_Trigger_Types` | 53 | list | 42 个非空 |
| `Static_Conflict_Information` | 53 | list | 32 个非空，最多 2 条 |
| `Conditional_Conflict_Information` | 53 | list | 25 个非空，最多 3 条 |
| `Others_Dynamic_Information` | 53 | list | 31 个非空，最多 2 条 |
| `Updated_Attributes` | 48 | list | 仅 48 个会话有此键，30 个非空，最多 2 条 |
| `Revealed_Attributes` | 5 | dict | 仅 5 个会话有此键 |

注意 `Revealed_Attributes` 是 **dict**，而 `Updated_Attributes` 是 **list**，两者类型不同。

### 会话类型分布

| Session_Type | 数量 |
| --- | ---: |
| `update` | 30 |
| `chitchat` | 17 |
| `initial_reveal` | 5 |
| `future_plan` | 1 |

### Event_Types 高频值

`Relationship_Status_Change` 8、`Relocation_Update` 7、`Career_Change_Update` 7、`Workload_Change_Update` 7、`Talking_About_Food_and_Mood` 5、`Health_Condition_Update` 4、`Social_Life_Change_Update` 4。

### Question_Trigger_Types

`dynamic_update` 30、`static_conflict` 12、`conditional_conflict` 12。

### Session_Question_Count 分布

| 题数 | 会话数 |
| ---: | ---: |
| 0 | 11 |
| 1 | 6 |
| 2 | 20 |
| 3 | 3 |
| 4 | 7 |
| 5 | 2 |
| 7 | 3 |
| 8 | 1 |

合计 122 题，分布在 42 个会话中。

### Session_Dialogue 格式

结构为 `{dialogue_turn_N: [ {role, content}, ... ]}`，每组固定一对 user/assistant，消息只有 `role` 与 `content` 两个 key。

```text
session 0: 40 组 dialogue_turn_* -> 80 条消息（user 40 / assistant 40）
全部会话:  每会话 80–100 条消息，均值 89.8
```

**顺序陷阱**：turn 的键是字符串，直接按字典序会得到 `turn_1, turn_10, turn_11, ... turn_2`。必须按数字后缀排序。参考实现：

```python
def extract_dialogue_turn_order(key_name: str) -> int:
    try:
        return int(str(key_name).split("_")[-1])
    except Exception:
        return 10**9

ordered_keys = sorted(session_dialogue.keys(), key=extract_dialogue_turn_order)
```

---

## 4. 冲突标注块（benchmark 核心）

同一个 `Conflict_ID` 会以 `Point_A` / `Point_B` 在两个不同会话中成对出现，两点之间的间隔即冲突距离。

### Static_Conflict_Information（34 条，32 个会话有内容）

```json
{"Conflict_ID": "SC_010", "Role": "Point_A", "Target_Field_Path": "Name", "Value": "Jackson Andrews"}
```

字段全集：`Conflict_ID`(34)、`Role`(34)、`Target_Field_Path`(34)、`Value`(34)、`Source_Person_ID`(10)、`Relationship_To_User`(10)。带 `Source_Person_ID` 的 10 条即由他人冒充用户信息制造的假事实。

### Conditional_Conflict_Information（33 条，25 个会话有内容）

```json
{"Conflict_ID": "CC_006", "Rule_ID": "CC_006_R1", "Role": "Point_A",
 "Preference_Type": "Sport_Preference", "Item": "Basketball",
 "Condition": "Casual pickup games in summer to keep cardio fun"}
```

字段全集：`Conflict_ID`(33)、`Role`(33)、`Rule_ID`(21)、`Preference_Type`(21)、`Item`(21)、`Condition`(21)、`Source_Person_ID`(12)、`Relationship_To_User`(12)、`Preference_Key`(12)、`Preference_Description`(12)。存在两种变体写法（`Preference_Type/Item/Condition` 与 `Preference_Key/Preference_Description`）。

### Others_Dynamic_Information（33 条，31 个会话有内容）

```json
{"Attribute": "Career_Status", "Role": "Distractor", "Source_Person_ID": "Contacts_1",
 "Relationship_To_User": "Neighbor",
 "Value": "Owner/operator of a small landscaping and property maintenance business ...",
 "Linked_Left_Session_ID": 1, "Linked_Right_Session_ID": 5}
```

字段全集为 7 个常量字段。`Linked_Left_Session_ID` / `Linked_Right_Session_ID` 明确标注该干扰项插在哪两个会话之间。

### Updated_Attributes（38 条，30 个会话非空）

```json
{"Attribute": "Residence", "Before": "Darwin, Australia", "After": "Melbourne, Australia"}
```

`Before` / `After` 可以是嵌套对象，例如整个 `Career_Status` 的 6 个子字段同时变化（`Future Intelligence/Senior/Media` -> `Northern Logistics/Intern/Legal`）。这是 dynamic conflict 的标准答案来源。

### Revealed_Attributes（5 个会话，dict）

出现在 `initial_reveal` 会话，顶层键覆盖 `Children_Status`、`Career_Status`、`Health_Status`、`Work_Status`、`Marital_Status`、`Social_Relationships`、`Residence`。例（Session 0）：

```json
{"Children_Status": {"Status": "Yes",
  "Child_1": {"Name": "Maya", "Birthdate": "2014-02-04"},
  "Child_2": {"Name": "Jackson", "Birthdate": "2017-03-25"}}}
```

---

## 5. Session_Questions

每题 6 个字段：`question_id`、`question`、`answer`、`conflict_type`、`ability_target`、`difficulty`。

| conflict_type | 数量 | ability_target |
| --- | ---: | --- |
| `dynamic_conflict` | 95 | `track_state_over_time` |
| `conditional_conflict` | 15 | `bind_condition` |
| `static_conflict` | 12 | `recover_truth` |

难度：`easy` 36、`medium` 59、`hard` 27。

### 三种题型的答案形态差异

```text
[dynamic]     Q: "Did the user's residence change recently?"
              A: "Yes."
              A: "They moved from Darwin, Australia to Melbourne, Australia."

[static]      Q: "Which university did the user attend?"
              A: "The information is inconsistent. The correct university is
                  California State University, Long Beach; another source lists MIT."

[conditional] Q: "Under what condition does the user prefer audiobooks?"
              A: "They prefer audiobooks for professional self-improvement and
                  technique study while commuting."
```

static 题明确要求在答案中指出冲突与不确定；conditional 题要求绑定"条件 -> 偏好"。

---

## 6. metadata / token_cost

```json
"metadata": {"persona_seed": "A professional stunt performer who needs help managing their earnings and planning for the future"}

"token_cost": {
  "current_stage": {"input_tokens": 304250, "output_tokens": 35117, "total_tokens": 339367,
                    "total_cost_usd": 0.146298, "model": "gpt-5-mini",
                    "pricing_available": true,
                    "note": "Rule-based session-level question generation"},
  "cumulative":    {"input_tokens": 386767, "output_tokens": 535976, "total_tokens": 922743,
                    "total_cost_usd": 1.168645}
}
```

`token_cost` 是**数据生成阶段**的成本记录，不是评测阶段的成本。

---

## 7. 使用时的注意事项

### 7.1 question_id 不是全局唯一

整条记录只有 8 个不同的 `question_id`，按 `(conflict_type, question_id)` 组合也只有 13 个；dynamic 的 `Q_001`–`Q_007` 在链上重复出现。对齐结果时不能拿 `question_id` 当主键，应使用 `(Session_ID, question_id, conflict_type)` 或数组下标。

### 7.2 编码正常，终端显示可能是假象

- 记录内 U+FFFD 替换字符数量：0
- 原始字节中 `EF BF BD` 序列数量：0
- UTF-8 严格解码与重新编码无损

非 ASCII 字符主要是排版符号：U+2014 em dash 2399 次、U+2019 右单引号 1508 次、U+2013 en dash 629 次、U+2011 不换行连字符 71 次、U+201C/U+201D 各 70 次、U+2018 52 次。

在 Windows PowerShell（GBK 代码页）下直接打印会显示成 `It��s` / `5�C7` 这类乱码，属终端显示问题，不是数据损坏。

### 7.3 末尾会话含多语言与 emoji

| 会话 | 日期 | 内容 | CJK 字符数 |
| --- | --- | --- | ---: |
| Session 46 | 2025-07-02 | 迁居韩国光州 | 28 |
| Session 51 | 2025-12-09 | 迁居中国广州 | 254 |
| Session 50 | 2025-11-03 | 日常闲聊 | 含 3 个 emoji |

这是刻意设计的干扰。检索/embedding 环节若不具备多语言能力，这两段会话会直接失效。

### 7.4 会话字段的存在性必须判空

`Updated_Attributes` 只在 48/53 个会话出现，`Revealed_Attributes` 只在 5 个会话出现；各冲突块大量为空数组。解析时应统一用 `session.get(key, [])` 处理，不要假设字段齐全。
