1. 模型代码的位置：<http://10.245.4.83:3000/kiana/Retrival-Mem/src/branch/mem_v4_end_clean>
2. 需要做的工作，从服务器上拉取代码分支，做冲突测试memoconflict
3. 冲突测试 benchmark的代码位置：<https://github.com/TaoZhen1110/MemConflict>；只需要关注Evaluation和最终的数据Data/Step4_4.jsonl;
4. 总之，先准备好实验环境；在e:\code\github\EMIR2目录下
5. 将代码上传到github git@github.com:15692396799/EMIR2.git上，做版本管理和同步
6. 了解Step4_4.jsonl的数据结构；只读第一条，分析这一条jsonl数据的构成内容
7. 需要修改的代码：MenConflict中的Step4_4.jsonl数据是每一个psersona中的每一个session构建一个QA,与当前Retrieval-Mem适配的locomo不同（一整个example构建完成后进行评测）；即从当前 locomo 格式的数据读取和记忆构建 --> memconflict 的数据读取和记忆构建 + 每构建完一个 session 就要进行相应的 QA。【persona 相互独立，但是同一个 persona 的各个 session 是相关的，后面的 session 是可以看到前面的 session 的，这个默认就是这样，不同的就是 retrieve 和 answer 的时间】
8. **另一个需要修改的点**，是 answer prompt 和 llm judge prompt，现在的 prompt 适配的是 locomo，然后 memconflict 要换一下
9. doc_by_human下的memconflict_tables_3_5_6.md文件中Table3是我们冲突测试实验需要的；我们EMIR2做冲突测试的实验需要的指标如表中所示
10. memory glm5.1 输入$3/M 输出$12/M 缓存$0.66/M；judge gpt-4o-mini 输入 $0.15/M, 输出 $0.6/M; 
11. 先使用小模型跑通流程，再得到允许的情况下再使用大模型进行测试分数
12. 小模型使用百炼或者ollama服务器部署，这里先使用百炼，api key 见 `Experiment/.env` 的 `DASHSCOPE_API_KEY`（模板见 `Experiment/.env.example`，该文件已 git-ignore，不入库）；模型名称qwen3-embedding和qwen3.5:latest；
13. 大模型使用OpenRouter上的api；key 见 `Experiment/.env` 的 `OPENROUTER_API_KEY`（模板见 `Experiment/.env.example`）；memory: glm-5.1; judge: gpt-4o-mini；
14. 172.26.94.12作为GPU服务器IP，想要使用需要登录172.26.4.254:22堡垒机操作开通docker端口服务，需要forgejo作为git服务
15. 查看现在的docker服务 ollama021-1; 宿主机上的docker容器运行你ollama服务似乎出现了问题；宿主机上貌似有好多个ollama容器；
16. 现在在GPU服务器上有四个ollama容器正在运行分别在device4,5,6,7上，做多容器处理
17. 做PERSONA并发处理，节约时间
18. 同一session内的并发答题，与session分片
19. 黑白盒测试
20. 成本估算
21. sessions.jsonl中的Questions为空：MemConflict 只在带 trigger 的 session 上出题；initial reveal：`initial_reveal` 和 `future_plan` 在 MemConflict 里不出题，只有 `update`（触发 `dynamic_update`，以及少量 `static_conflict`/`conditional_conflict`）和 `chitchat`（触发 `static_conflict`/`conditional_conflict`）会挂问题；dynamic 冲突靠 update 的 Before/After 提供标准答案；static 冲突靠早期 `initial_reveal` 说过的真值，去对抗后来闲聊里塞进来的错误说法；现在的scale1h是在1:41启动；
22. `memconflict_eval` 里没有任何模型实现代码，它只做一件事：把 `Retrival-Mem` 这个 checkout 当库导入并使用。入口只有一个函数，真正的实例化只有一处。唯一导入入口：[runtime.py](http://tauri.localhost/E:/code/github/EMIR2/Experiment/memconflict_eval/runtime.py:71 "E:/code/github/EMIR2/Experiment/memconflict_eval/runtime.py") 的 `import_retrival_mem()`
23. 答题用 `runtime.build_chat_client(config.answer_model)`（即上游的 `make_chat_client`）拿到模型，只把 prompt 换成按 `conflict_type` 路由的 MemConflict 版本；检索仍然是上游配置里的 `multi_round` + `teacher` controller，我们只是对每个问题调 `self.system.retrieve(question, namespace)`（[memory.py:204](http://tauri.localhost/E:/code/github/EMIR2/Experiment/memconflict_eval/memory.py:204 "E:/code/github/EMIR2/Experiment/memconflict_eval/memory.py")）。打分侧同理，只换个 judge prompt，把二值判定换成"分级准确率 + 冲突处理 + 支撑记忆排名"，再由 `metrics.py` 聚合成 AA / SEH@K / SRS / UOCS / CRS。MemConflict 自己的评测脚本一行都没 import，[runtime.py](http://tauri.localhost/E:/code/github/EMIR2/Experiment/memconflict_eval/runtime.py:1 "E:/code/github/EMIR2/Experiment/memconflict_eval/runtime.py") 的模块注释里明确写了这个取舍。
24. ingest与MessageCount含义
25. scale1h debug：一个 stale 的中间产物 → 整个 persona 直接挂掉。debug：!! persona 4 failed (90e98aa7-1d5d-0a8f-55d8-0b0886897dd3): SemanticValidationError: reinforce references unknown fact key；来自模型上游的缓存失效缺陷；
26. SRS和UOCS一致，这是巧合吗？还是打分代码错误
27. 分片跑persona，五个一片切成六片；切片的时候persona要随机，不要连续；每次切片跑完都更新table表，让我们看到新的table分数；ollama容器上ollama021-1和ollama021容器，以及ollama021-2-1容器可用
28. 现在的代码是否支持断点续传
29. 现在运行的ollama容器中的qwen模型是否能用百炼上的模型替代？如果替代的话，测出来的切片又是否能合并？
30. debug：C盘临时空间用完
31. debug：模型在adjudication时把方向写反；修改了Retrieval Mem
32. ollama容器多开；我们计划再在device2,3上开新的ollama容器ollama021-3-1和ollama021-3-2
33. 记忆构建和打分模型调换；打分模型更换成gpt-5-mini；记忆构建模型更换成gpt-4o-mini

```

GPU-fa5af6d7-cbd4-ebfc-9475-9a72102d9bad

docker run -d --name ollama021-1 --gpus '"device=6"' -p 0.0.0.0:41135:11434 -v /data/ollama_models:/root/.ollama/models -e OLLAMA_CONTEXT_LENGTH=32768 --restart unless-stopped ollama/ollama:0.21

 docker inspect ollama021 --format '{{json .HostConfig.DeviceRequests}}'


python Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml --start-index 1 --persona-limit 4 --max-sessions 4 --persona-workers 1


# 1 小时规模测试：4 persona 并行 + 每 session 4 线程并行答题
python -u Experiment\run_experiment.py `
  --config Experiment\configs\eval_large.yaml `
  --start-index 11 --persona-limit 4 --max-sessions 11 `
  --persona-workers 4 --answer-workers 4 `
  --extraction-workers 2 --entity-judge-workers 1 `
  --ollama-units http://172.26.94.12:41135 `
  --output-dir Experiment\runs\scale1h

# 一次打分产出 table3 / table5 / table6
python -u Experiment\run_scoring.py --run-dir Experiment\runs\scale1h

python Experiment\tools\shard_plan.py --only 1

#shard_1
python -u Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml --persona-indices 0,4,18,27,28 --persona-workers 6 --answer-workers 4 --extraction-workers 2 --entity-judge-workers 1 --output-dir Experiment\runs\shard_1

#shard_2
python -u Experiment\run_experiment.py --config Experiment\configs\eval_large.yaml --persona-indices 13,16,24,25,29 --persona-workers 6 --answer-workers 4 --extraction-workers 2 --entity-judge-workers 1 --output-dir Experiment\runs\shard_2

#shard_3
python -u Experiment\run_experiment.py --config Experiment\configs\eval_large_bailian.yaml --persona-indices 1,3,7,14,19 --persona-workers 5 --answer-workers 4 --extraction-workers 2 --entity-judge-workers 1 --ollama-units http://172.26.94.12:41136 --output-dir Experiment\runs\shard_3

python Experiment\run_scoring.py --run-dir Experiment\runs\shard_1 --white-box-k 2,3,5

python Experiment\tools\merge_shards.py --allow-partial --out Experiment\runs\merged --shards Experiment\runs\shard_1


python Experiment\run_scoring.py --run-dir Experiment\runs\merged --white-box-k 2,3,5

```