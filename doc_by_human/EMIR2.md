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
18. 
