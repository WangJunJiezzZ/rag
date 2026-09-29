# MCP：把 GraphRAG 做成标准工具服务

> 这一章做的事：把图谱查询和文档检索封装成 **MCP Server**，
> 再写一个 **DeepSeek 驱动的 MCP Client**，把协议的两端都走通。
>
> 本章所有数字都是实测结果。还没测过的内容，会明确标注"未测"。
>
> **只想知道这是什么、怎么讲给别人听：** 读第零节就够了。后面几节是被追问时用的。

---

## 零、白话版：这是什么、干了什么、怎么用

### MCP 是什么

**MCP 就是 AI 的"USB 接口"。**

没有 USB 的年代，每种设备都要配专门的线和驱动。有了 USB，设备只要做一个 USB 口，插到任何电脑上都能用。

AI 领域也一样：
- **没有 MCP 时：** 想让 ChatGPT 查你的数据，要给它写一套插件；换成 DeepSeek，得重写一套；换成 Cursor，还得再写一套。
- **有了 MCP 之后：** 数据只要做成一个 **MCP Server**，也就是装上一个"USB 口"，任何支持 MCP 的 AI 应用都能直接接进来用。

> **一句话：MCP 让 AI 能用上你的数据，而且只接一次，所有 AI 应用都能用。**

### 这个项目用它干了什么

原来的 graphrag-lab 有一个知识图谱，记录《喜羊羊与灰太狼》《铁甲小宝》里角色之间的关系（父母、夫妻、表亲、搭档、封印……），但**只有这个项目自己的程序能用它**。

这一章做了两件事：

| 做了什么 | 相当于 | 文件 |
|---|---|---|
| **MCP Server**：把图谱查询和文档检索包装成 7 个标准工具 | 给数据装上 USB 口 | [`server.py`](../src/graphrag/mcp_app/server.py) |
| **MCP Client**：让 DeepSeek 能通过 MCP 使用这些工具 | 一台插上 USB 的电脑 | [`client.py`](../src/graphrag/mcp_app/client.py) |

装上"USB 口"之后，同一个 Server 不改一行代码，就能被命令行（DeepSeek）、Cherry Studio、Cursor、Claude Desktop 等任何 MCP 客户端接入。

**注意：MCP Server 本身不调用任何 AI，也不需要 API key。** 它只是在本地查图谱和文档。调用 DeepSeek、需要 key 的是 Client 这一侧。

### 怎么用

```bash
source .venv/bin/activate          # 每次新开终端都要执行一次

# 问 DeepSeek 一个问题，它会自己决定调用哪些工具
python run.py mcp-chat -- --graph data/synthetic/graph.json -q "小灰灰和小香香是什么关系？"
```

屏幕上会逐行打印 AI 调用了哪些工具，最后给出答案：
```
  -> find_entity({"name": "小灰灰"})                      先确认问的是哪个角色
  -> find_entity({"name": "小香香"})
  -> explain_relation({"a": "C10", "b": "C37"})          找两者之间的关系路径
  -> read_document({"doc_id": "doc-0014"})               读原始档案核实证据
  ...
```

**这些步骤是 AI 自己决定的，不是写死的。** 这就是演示时最想让对方看到的东西。

另外两种用法：
- **MCP Inspector**：`npx @modelcontextprotocol/inspector .venv/bin/python src/graphrag/mcp_app/server.py`，在网页上手动调用工具。这是开发者的调试工具，相当于电工用的万用表，**不用于演示**。工具参数要填角色名，比如 `灰太狼`，不要填整句问题；整句问题要交给 `search_text` 或 `ask`
- **接入其他 AI 应用**：见第二节的配置

### 演示时要让对方看到的三件事

1. **AI 会自己决定怎么查**：屏幕上逐行出现的工具调用
2. **同一个 Server，谁都能接**：不改代码，接入不同的客户端
3. **你知道它为什么对、什么时候会错**：见下面这个真实案例，这是最能体现水平的一点

### 真实案例：同一道题，两份图谱各跑一次

问题是"小灰灰和小香香是什么关系？"，也就是评测集里的 q-0084。标准答案是**表亲**：小灰灰的妈妈红太狼，是小香香的妈妈香太狼的表姐。三条关系分别写在小灰灰和香太狼的档案里，**没有任何一份文档同时提到小灰灰和小香香**。

| | 用自动抽取的图谱 | 用标准图谱 |
|---|---|---|
| 结论 | ✅ 表亲，路径与标准答案一致 | ✅ 表亲，路径与标准答案一致 |
| 工具调用 | 4 次 | 6 次（多了两次 `read_document`） |
| 模型调用 / token | 3 次 / 约 10200 | 4 次 / 约 14800 |
| 花费 | 约 $0.003 | 约 $0.005 |
| 引用的证据文档 | 2 份标准证据全部引用 | 2 份标准证据全部引用 |

**两次都答对，但做法不同：** 用标准图谱的那次，拿到 `explain_relation` 的路径后又回去读了两份原始档案，逐跳核实；用抽取图谱的那次直接作答，少一轮调用，但结论完全押在抽取质量上。

**从这个案例能说明三点：**
1. **Agent 的上限取决于底下的数据**。这次抽取图没有漏边（v4 抽取 prompt 的 recall 是 100%），所以两次结论一样；抽取一旦漏掉「香太狼是红太狼的表妹」这一条，工具就会如实返回"没有路径"，模型也会如实回答"找不到关系"——那种自信的漏报最危险
2. **模型会自己决定要不要核实**。同样的工具、同样的问题，一次读原文、一次不读，成本差 1.6 倍。值不值得读，要靠评测集回答，不能靠一道题
3. **只看一道题会得出错误结论**。必须对照标准答案，才知道"看起来成功"的那次是不是真的对

> 早期版本用的是合成基金语料，那时同样的流程出现过一次真实的答错：两家同名公司靠「（2）」区分，而董事公告里没写「（2）」，抽取时挂错了公司，关联链从第一跳就断了。那个教训仍然成立，只是新语料上没有复现。

两次运行都已缓存，加上 `GRAPHRAG_OFFLINE=1` 就能离线原样回放，演示时不花钱，结果也不会变。

### 线上怎么展示

静态站（GitHub Pages / HF Static Space）没有后端，跑不了 MCP Server，也**绝不能**把 API key 放进网页。所以线上用的是**回放页面** `agent.html`，首页右上角有入口。它把真实运行的每一轮工具调用、参数、原始返回和最终答案逐步展示出来，也包含两份图谱的对比表。页面是纯静态的，不调用任何模型。

`python run.py build-static` 构建时，会从缓存离线重放这些运行，生成 `data/agent_traces.json`，然后照常发布。

### 换题或加题

回放页面只能展示**已经在线跑过**的问题，因为构建时要从缓存逐轮重放。步骤如下：

1. **从评测集里选题**，这样才有标准答案可以判断对错。适合 Agent 演示的题型有 `relation_path`（A 和 B 是什么关系）、`hop3`（三跳关系）、`hop2`（两跳关系）和 `negative`（语料里没有答案，看 AI 会不会乱编）。可以用下面的命令列出题目：
   ```bash
   python -c "import json;[print(q['id'],q['type'],q['question']) for q in map(json.loads,open('data/eval/eval_set.jsonl')) if q['type'] in ('relation_path','hop3','hop2','negative')]"
   ```
2. **在两份图谱上各在线跑一次**，问题要写成你想在页面上展示的样子。每题约 $0.01：
   ```bash
   python run.py mcp-chat -- --graph data/synthetic/graph.json -q "你的问题"
   python run.py mcp-chat -- -q "你的问题"
   ```
3. **登记到 [`scripts/export_static.py`](../scripts/export_static.py) 的 `AGENT_DEMOS` 列表里**。`question` 必须和第 2 步**一字不差**；`verdict`（`correct` 或 `wrong`）和 `note` 是你对照标准答案后填写的判断，不填时页面显示"待核对"：
   ```python
   {"qid": "q-0087", "question": "你的问题",
    "runs": {"oracle":    {"label": "标准图谱"},
             "extracted": {"label": "自动抽取的图谱"}}},
   ```
   列表里有多道题时，页面顶部会自动出现选题栏。
4. **重新构建并发布**：`python run.py build-static`。如果某道题没在线跑过，会打印 `[warn] Agent 回放缓存未命中`，并提示需要补跑的命令。

---

## 一、一句话讲清 MCP

**MCP 管的是"工具怎么接进来"，function calling 管的是"模型怎么表达想调工具"。**

```
┌──────────── Host（client.py）────────────┐          ┌──── MCP Server（server.py）────┐
│                                           │          │                               │
│  DeepSeek ◄── function calling ──► 翻译层 ◄── MCP ──► 7 个 tools                      │
│  (模型层)      tools / tool_calls         │ JSON-RPC │ 1 个 resource + 1 个模板       │
│                                           │ stdio 或 │ 1 个 prompt                    │
│                                           │ HTTP     │        │                      │
└───────────────────────────────────────────┘          │        ▼                      │
                                                       │  tools.py → RAGService / 图谱 │
                                                       └───────────────────────────────┘
```

- **function calling** 是模型的能力。模型会输出"我想调 `explain_relation(a=..., b=...)`"，但它不知道这个函数在哪、怎么调用。
- **MCP** 是一套协议，规定工具怎么被发现（`list_tools`）、怎么描述（JSON Schema）、怎么调用（`call_tool`）、结果怎么返回（content 块）。它和具体用哪个模型无关。
- **Host** 负责两边的翻译：把 MCP 的 Tool 转成 OpenAI 的 tools 格式，把模型输出的 `tool_calls` 转成 `call_tool` 请求，再把工具结果包装成 `role=tool` 的消息交回模型。

**为什么需要 MCP：** 没有 MCP 时，M 个应用要接 N 个工具，每个应用都得把每个工具的 schema 和调用代码写一遍，一共 M×N 份。有了 MCP，工具方写一次 Server，应用方写一次 Client，变成 M+N 份。
本项目的 Server 可以被任何 MCP 客户端直接使用，不局限于 DeepSeek。

---

## 二、怎么跑

```bash
pip install "mcp>=2.2"

# 1. 离线自检：不需要 API key，约 15 秒
python run.py verify-mcp

# 2. 用 DeepSeek 对话（需要先 export DEEPSEEK_API_KEY=...，不要写进仓库）
python run.py mcp-chat
python run.py mcp-chat -- -q "小灰灰和小香香是什么关系？"

# 3. 以 HTTP 方式启动 Server，供远程客户端连接
python run.py mcp-server -- --http --port 8765
python run.py mcp-chat -- --url http://127.0.0.1:8765/mcp
```

`mcp-chat` 会把每一步工具调用打印到 stderr：

```
[mcp] 已连接, 发现 7 个工具: find_entity, get_neighbors, ...
  -> find_entity({"name": "小灰灰"})  74 字符
  -> explain_relation({"a": "E0022", "b": "E0027"})  784 字符
```

**接入其他 MCP 客户端**（Cherry Studio、Cursor、Claude Desktop 等，它们都支持 stdio Server）。在客户端的 MCP 配置里加上下面这段，路径换成你自己的：

```json
{
  "mcpServers": {
    "graphrag-lab": {
      "command": "/绝对路径/graphrag-lab/.venv/bin/python",
      "args": ["/绝对路径/graphrag-lab/src/graphrag/mcp_app/server.py"]
    }
  }
}
```

---

## 三、代码导读（建议按这个顺序读）

| 文件 | 作用 | 读的时候重点看 |
|---|---|---|
| [`mcp_app/tools.py`](../src/graphrag/mcp_app/tools.py) | 工具的实现，纯 Python，不依赖 MCP SDK | 为什么拆成原子工具；"找不到"时怎么返回 |
| [`mcp_app/server.py`](../src/graphrag/mcp_app/server.py) | 用 `MCPServer` 注册 tools / resources / prompt | 工具描述怎么写；三种原语怎么选 |
| [`mcp_app/client.py`](../src/graphrag/mcp_app/client.py) | Host：连接 Server，把工具翻译给 DeepSeek，运行工具调用循环 | `mcp_tool_to_openai`；`ask()` 里的循环 |
| [`llm.py`](../src/graphrag/llm.py) 中的 `chat()` | DeepSeek 工具调用 + 磁盘缓存 | 缓存键为什么是完整的消息历史 |
| [`scripts/verify_mcp.py`](../scripts/verify_mcp.py) | 用剧本式的假 LLM 跑真实协议 | 覆盖了哪些失败路径 |

### 7 个工具

| 工具 | 什么时候用 | 对应的已有能力 |
|---|---|---|
| `find_entity` | 任何图查询之前，先把名称解析成 id | `KnowledgeGraph.resolve` + `EntityLinker` + 子串匹配 |
| `get_neighbors` | 一跳关系，例如父母、搭档、原型、口头禅 | `kg.out` / `kg.inn` |
| `find_paths` | A 和 B 之间的全部路径（包括经由节目节点的） | `kg.find_paths` |
| `explain_relation` | A 和 B 是什么关系 | `kg.find_paths` + 枢纽阻断，与图检索器的口径一致 |
| `search_text` | 描述性问题，或者图里查不到的对象 | BM25 + Dense 的 RRF 融合 |
| `read_document` | 核实证据原文 | 文档库 |
| `ask` | 一站式问答（固定流水线） | `RAGService.ask` |

---

## 四、设计决策与踩过的坑

### 1. 为什么拆成原子工具，而不是只暴露一个 `ask`

`ask` 是写死的流水线：规则路由 → 检索 → 生成。调用方只能问一句、拿一个答案，没法在中途判断"这一步没找到，换条路"。
拆成原子工具后，**编排的决策权交给了调用方的模型**。至于模型编排到底比手写规则好还是差，这正是下一章（项目 B：Agentic RAG）要用评测集回答的问题。`ask` 保留下来作为基线。

### 2. 工具描述要写"什么时候该用我"

对模型来说，工具描述是它选工具的**唯一依据**。所以描述里不能只写"列出一跳关系"，而要写"适合：角色的父母、配偶、搭档……"。关系的方向也写进了 schema resource（比如 `PARENT_OF` 是"父母 → 子女"），否则模型容易把方向查反，把爷爷答成孙子。
> 这些描述写得好不好，本章**还没有用真实模型评测过**，放到项目 B 里测。

### 3. "找不到"时返回提示，不抛异常

```json
{"error": "图中找不到实体「查无此人」",
 "hint": "先用 find_entity 做模糊查找拿到实体 id; 若仍找不到, 该对象可能只在文档里出现, 改用 search_text。"}
```
模型能读懂 hint 然后换个工具，但读不懂 Python 的 traceback。参数越界（比如 `limit=999`）由 SDK 按 JSON Schema 校验，把错误信息作为 `is_error` 结果返回；模型输出坏 JSON 时，Client 同样把错误回给模型，而不是让程序崩溃。这三种情况在 `verify-mcp` 里都有覆盖。

### 4. 工具结果返回紧凑 JSON（实测）

工具函数如果直接返回 dict，SDK 会按 `indent=2` 格式化。实测 5 类工具结果，带缩进的版本**字符数多 5%～77%，平均多 46%**（嵌套越深的结果越吃亏，`get_neighbors` 最多，`read_document` 基本是一整段正文，几乎不受影响）。
注意这是字符数。token 的增幅会小一些，因为分词器常把连续空格合并，这一点未单独测量。工具输出每一轮都会原样进入模型上下文，所以改成了 `separators=(",", ":")` 的紧凑格式。

### 5. 懒加载（实测）

建向量索引和加载图谱，在早期 326 份文档的基金语料上实测需要 **约 11 秒**；换成现在 42 份文档的动画语料后约 1.5 秒。语料一大，Server 启动时就初始化会让客户端的 `initialize` 握手等很久，有的客户端会判定超时。所以改成第一次调用工具时才初始化。

### 6. stdio 下 stdout 是协议通道（实测）

用 stdio 传输时，Server 的 stdout 就是 JSON-RPC 消息流，**随手 print 一行就会混进协议里**。
- mcp 2.x 的 `stdio_server` 会把文件描述符 1 转到 stderr 兜底，1.x 没有这个机制。`tools.py` 在初始化期间也会显式把 stdout 重定向到 stderr。
- `run.py` 启动子脚本前会往 stdout 回显一行 `$ python ...`。实测官方 Python SDK 客户端遇到这行时，会报 `Failed to parse JSONRPC message` 然后跳过，调用仍然成功。但不能保证其他客户端也这么宽容，所以这行回显改成了走 stderr。

### 7. SDK 2.x 的 API 与多数教程不兼容

网上大部分教程用的是 1.x 的 `from mcp.server.fastmcp import FastMCP`，在 2.x 里会直接报错，这个类已经改名为 `MCPServer`。
2.x 还新增了一个高层的 `mcp.Client`，可以直接接收 `MCPServer` 实例在进程内连接，写测试很方便。

### 8. 工具调用循环里的消息配对

```
assistant  {content: "", tool_calls: [{id: "c1", function: {...}}]}   ← 必须原样保留 tool_calls
tool       {tool_call_id: "c1", content: "..."}                        ← 靠 id 和上面配对
```
如果漏了 assistant 消息里的 `tool_calls`，或者 id 对不上，下一轮 API 会直接报错。另外还设了一个 `max_steps` 上限（默认 8），防止模型一直调工具停不下来。

### 9. 可复现：缓存键是完整的消息历史

`LLM.chat()` 的缓存键包含全部 messages 和 tools。Agent 每一步的输入都包括之前所有的工具结果，只要工具本身是确定性的，**整条调用轨迹都能从缓存逐步回放**。这样评测可以复现，重跑也不花钱。
改动工具描述的任何一个字，缓存键都会变，从而触发重新调用。这是有意的设计，因为工具描述本身就是被评测的变量。

---

## 五、面试问答

> 前三题是最基础的，一定会被问到；后面几题是被追问时用的。

### Q0-1：用一句话说，MCP 是什么？

> MCP 是一个开放协议，让 AI 应用用统一的方式接入外部的工具和数据，可以理解成 AI 的 USB 接口。
> 工具方写一次 MCP Server，所有支持 MCP 的 AI 应用都能直接使用，不用给每个应用各写一套插件。

### Q0-2：你这个项目里 MCP 用来做什么？

> 我把知识图谱的查询能力做成了一个 MCP Server，一共 7 个工具，比如查角色、查关系、解释两个角色是什么关系、全文检索、读原文。
> 然后自己写了一个 MCP Client，让 DeepSeek 通过 MCP 调用这些工具来回答问题。
> 这样图谱就不再只能被我自己的程序使用，任何支持 MCP 的 AI 应用都能接进来。

### Q0-3：你的演示想说明什么？

> 三件事。第一，AI 会自己决定调用哪些工具、按什么顺序调用，这些不是我写死的。第二，同一个 Server 不改代码，就能接到不同的 AI 应用里。
> 第三点最重要：我知道它为什么对。同一道"小灰灰和小香香是什么关系"，在抽取图和标准图上各跑一次都答对了，但一次回去读了原文核实、一次没读，成本差 1.6 倍。抽取图能答对，是因为抽取环节在这几份档案上没有漏边——漏一条，工具就会如实返回"没有路径"，模型也会如实说"找不到关系"。
> 这说明 Agent 的上限取决于底下的数据，而这种自信的漏报是最难发现的错误。

### Q1：MCP 和 function calling 是什么关系？是替代关系吗？

> 不是替代关系，它们在不同的层。function calling 是模型层的能力，模型输出结构化的"我想调某个函数"。MCP 是工具接入层的协议，规定工具怎么被发现、怎么描述、怎么调用。
> 我自己写过中间那层 Host：从 MCP 的 `list_tools` 拿到 JSON Schema，改几个字段名就变成 DeepSeek 的 tools 参数；模型返回 `tool_calls` 后，我转成 MCP 的 `call_tool`，再把结果包成 `role=tool` 消息交回去。两边都用 JSON Schema 描述参数，所以翻译几乎是零成本的，这也是 MCP 能做到和模型无关的原因。

### Q2：Tools、Resources、Prompts 怎么选？

> 看谁来决定什么时候用。模型决定的是 tool；应用或客户端决定的是 resource；用户主动选择的是 prompt。
> 比如图谱的 schema，我放在 resource 里，由客户端在对话开始前读一次，放进 system prompt，这样模型就不用浪费一轮工具调用去问"有哪些关系"。"角色关系调查"是一个固定的调查流程，做成 prompt 模板，由用户来选。

### Q3：工具粒度怎么定？为什么不只给一个 ask？

> 粒度决定了谁来做编排。只给一个 `ask`，编排逻辑写死在服务端；拆成原子工具，编排交给模型。
> 我两种都保留了，因为这两者谁更好是一个需要测的问题，不能靠拍脑袋。我在下一阶段用同一套评测集做了对比（见项目 B）。
> 工具也不能拆得太碎，每多一个工具，模型选错的概率就高一点，上下文里的工具描述也更长。

### Q4：工具出错时怎么处理？

> 分三种情况。参数不符合 schema 的，由 SDK 校验后返回 `is_error`；业务上找不到的，返回带 hint 的结构化结果，告诉模型下一步该用哪个工具；模型输出坏 JSON 的，Client 把解析错误作为工具结果回给模型。
> 原则是：**所有错误都变成模型能读懂的文本，让它有机会自我纠正**，而不是让整条链路崩掉。这三种情况我都写了自检。

### Q5：stdio 和 Streamable HTTP 怎么选？

> stdio 适合本地场景：客户端把 Server 当子进程拉起，不需要网络和鉴权，隔离也好。但有个坑，stdout 就是协议通道，不能随便 print。
> HTTP 适合远程或多客户端共享一个 Server。代价是要自己处理鉴权。我的 Server 在 HTTP 模式下默认只绑定 127.0.0.1，**没有做鉴权**，要上生产得先加上。

### Q6：怎么测试 MCP Server？没有 API key 怎么办？

> 写了一个剧本式的假 LLM，按预设顺序返回 `tool_calls`，驱动真实的 Client 循环，通过真实的 stdio 协议和 Server 通信。这样不花一分钱，就能验证协议发现、消息配对和失败路径。
> 真实模型的工具选择质量是另一回事，需要评测集来测。那部分放在项目 B 里做。

### Q7：MCP 有什么安全风险？

> 最主要的是：**工具结果会原样进入模型上下文**。如果工具返回的内容来自不可信的来源，比如用户上传的文档，里面就可能藏着 prompt 注入。另外，有副作用的工具需要设计权限和确认机制。
> 我这个 Server 里的工具全部是只读的，通过 `read_only_hint` 注解声明出来，客户端可以据此决定是否需要用户确认。文档是我自己根据公开资料整理、由程序渲染的，所以注入风险可控。但如果换成用户上传的文档，这是第一个要处理的问题。

### Q8：做这个遇到的最大坑是什么？

> 两个。第一是 SDK 版本：网上大多数教程是 1.x 的写法，2.x 里 `FastMCP` 改名成了 `MCPServer`，照着教程写会直接报错。
> 第二是 stdout 污染。我的任务入口 `run.py` 会往 stdout 回显一行命令。我实测过，官方 SDK 客户端会报解析错误然后跳过这一行，所以没有真的挂掉，但不能指望所有客户端都这么宽容，于是把回显改到了 stderr。
> 这两个坑的共同教训是：**先实测，再下结论**。我一开始以为那行回显会导致握手失败，测了才发现并没有。

---

## 六、还没做的

- **真实模型下的工具选择准确率**：目前只手动跑过一道题（见第零节），还没在整套评测集上统计，放在项目 B 里做
- **Agent 的成本控制**：用标准图谱那次，第 2 轮的 `explain_relation` 已经给出了完整路径，模型又读了两份原文核实，成本是另一次的 1.6 倍。这样做值不值得，放在项目 B 里用评测集测
- **枢纽阈值的弱关联**："喜羊羊和灰太狼是什么关系"会返回一条 4 跳路径（喜羊羊住羊村 → 暖羊羊也住羊村 → 蕉太狼是暖羊羊的好朋友 → 灰太狼是蕉太狼的二叔）。羊村只连着 8 条边，没到枢纽阈值 12，所以允许中转。这不算错，但也不是一般人说的"关系"——阈值该按节点类型设，还是按边的类型设，还没测
- **结构化输出**：工具返回的是紧凑 JSON 字符串，在 Inspector 里会被包成 `{"result": "..."}`。对 DeepSeek 没有影响，但对使用结构化输出的客户端不够友好
- **HTTP 模式的鉴权**：没做，默认只绑定 127.0.0.1
- **流式返回、进度通知**：没做。`ask` 需要几秒钟，可以用 MCP 的 progress notification 改善体验
