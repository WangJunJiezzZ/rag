---
id: extract_triples/v1_naive
base: -
changes: 起点版本。只说"抽三元组", 不给 schema、不给约束、不给示例。
hypothesis: 能抽出东西, 但关系名会五花八门(父亲/爸爸/是……的父亲), 实体边界不稳, 且会顺手把性格描写、资料说明之类的套话也抽出来。precision 低。
---
# system
你是信息抽取助手。从给定的中文动画角色资料中抽取实体和它们之间的关系。

# user
文档标题：{title}

正文：
{text}

请抽取上述文档中的实体关系三元组，以 JSON 输出。
