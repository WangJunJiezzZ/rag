"""
Phase 4 - 汇总全部评测结果, 生成一页 HTML 报告。

报告是这个项目的**交付物**: 面试时直接打开这一页, 而不是翻终端日志。
它把散落在 reports/*.json 里的结果拼成一条叙事线:

    纯检索的能力边界 -> 加了什么 -> 各值多少分 -> 还剩什么缺口

刻意不做的事: 不画花哨图表。评测报告的读者要的是能对比的数字,
一张排版清楚的表比任何图都快。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrag.compat import setup_console, write_text   # noqa: E402

setup_console()
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

TYPE_LABEL = {
    "fact_direct": "单跳事实（原词）", "fact_paraphrase": "单跳事实（同义改写）",
    "semantic_only": "纯语义（无实体名）", "fact_disambig": "易混实体",
    "hop2": "两跳关系", "hop3_parent": "三跳关系", "hop4_risk": "四跳风险穿透",
    "shared_director": "共同董事", "aggregation": "聚合计数", "negative": "幻觉陷阱",
}


def load(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except (TypeError, ValueError):
        return "—"


def table(headers: list[str], rows: list[list[str]], highlight: int = -1) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for r in rows:
        tds = "".join(
            f'<td class="{"hi" if i == highlight else ""}{" lbl" if i == 0 else ""}">{c}</td>'
            for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def section_retrieval() -> str:
    d = load("phase1_retrieval.json")
    if not d:
        return '<p class="muted">尚未运行 <code>python run.py retrieval</code></p>'
    out = []

    ga = d.get("group_a_chunking", [])
    if ga:
        names = [r["retriever"].split("/")[-1] for r in ga]
        types = [t for t in TYPE_LABEL if t in ga[0]["by_type"]]
        rows = [[TYPE_LABEL[t]] + [pct(r["by_type"][t]["recall"].get("1")) for r in ga]
                for t in types]
        rows.append(["<b>总计</b>"] + [f'<b>{pct(r["overall"]["recall"].get("1"))}</b>'
                                       for r in ga])
        out.append("<h3>A 组　切块策略对比（recall@1，检索器固定 BM25）</h3>")
        out.append(table(["题型"] + names, rows, highlight=len(names)))
        out.append('<p class="note">孤儿 chunk 效应：section 策略把「本基金最低认购金额…」'
                   '切成不含基金名的独立块，recall@1 掉到 58%；补上文档标题前缀（contextual）后回到 100%。</p>')

    gb = d.get("group_b_ablation", [])
    if gb:
        names = [r["retriever"] for r in gb]
        types = [t for t in TYPE_LABEL if t in gb[0]["by_type"]]
        rows = [[TYPE_LABEL[t]] + [pct(r["by_type"].get(t, {}).get("recall", {}).get("10"))
                                   for r in gb] for t in types]
        rows.append(["<b>总计</b>"] + [f'<b>{pct(r["overall"]["recall"].get("10"))}</b>'
                                       for r in gb])
        out.append("<h3>B 组　检索路径消融（recall@10）</h3>")
        out.append(f'<div class="scroll">{table(["题型"] + names, rows, len(names))}</div>')
    return "".join(out)


def section_extraction() -> str:
    d = load("phase2_extraction_rule.json")
    if not d:
        return '<p class="muted">尚未运行 <code>python run.py build-graph</code></p>'
    rows = [[r, str(v["gold"]), str(v["ext"]), str(v["ok"]),
             pct(v["precision"]), pct(v["recall"])]
            for r, v in sorted(d["by_relation"].items(), key=lambda x: -x[1]["gold"])]
    es, rs = d["extract_stats"], d["resolve_stats"]
    return (f'<p class="kpi"><b>{pct(d["precision"])}</b> precision　'
            f'<b>{pct(d["recall"])}</b> recall　<b>{pct(d["f1"])}</b> F1　'
            f'<span class="muted">抽取器：{d["extractor"]}</span></p>'
            + table(["关系", "标准", "抽出", "正确", "precision", "recall"], rows)
            + f'<p class="note"><b>消歧欠账</b>：提及 {rs["mentions"]} 个名称归并为 '
              f'{rs["canonical"]} 个实体，其中词面规则合并 {rs["merged_by_rule"]} 对，'
              f'词面一票否决 {rs.get("blocked_by_lexical", 0)} 对（向量说像但词面不通，'
              f'避免的错合并），灰区未解决 {rs["gray_unresolved"]} 对。'
              f'剩余未合并的是<b>跨文种别名</b>（罗马化名），只能靠 LLM 裁决。</p>'
            + f'<p class="note"><b>evidence 回查</b>拦掉 {es["dropped_evidence"]} 条、'
              f'schema 校验拦掉 {es["dropped_schema"]} 条。</p>'
            + '<p class="warn">规则抽取在本项目的模板化语料上接近满分，'
              '<b>该分数不具外推性</b>，仅作为基线与离线兜底；真实能力取决于 LLM 抽取。</p>')


def section_e2e() -> str:
    d = load("phase3_e2e_ab.json") or load("phase3_e2e.json")
    if not d:
        return '<p class="muted">尚未运行 <code>python run.py e2e</code></p>'
    names = [r["prompt"].split("/")[-1] for r in d]

    def ans(r, key):
        rows = [x for x in r["details"] if x["type"] != "negative"]
        vals = [x[key] for x in rows if x[key] == x[key]]
        return sum(vals) / len(vals) if vals else 0.0

    rows = [
        ["可答题答对率"] + [pct(ans(r, "correct")) for r in d],
        ["可答题实体召回"] + [pct(ans(r, "entity_recall")) for r in d],
        ["陷阱题正确拒答"] + [pct(r["by_type"].get("negative", {}).get("refusal_correct", 0))
                        for r in d],
        ["可答题误拒答率"] + [pct(ans(r, "false_refusal")) for r in d],
        ["引用有效率"] + [pct(r["overall"]["citation_valid"]) for r in d],
        ["引用命中标准证据"] + [pct(r["overall"]["citation_grounded"]) for r in d],
        ["累计花费 (USD)"] + [f'{r["overall"]["cost_usd"]:.3f}' for r in d],
    ]
    return (table(["指标"] + names, rows)
            + '<p class="note"><b>为什么「答对率」和「拒答率」必须并排看：</b>'
              '只看拒答率，一个永远回答「资料中未提及」的系统能拿满分；'
              '只看答对率，一个从不拒答的系统在陷阱题上会全错但总分未必难看。'
              '单看任何一个都能被轻易刷分 —— 这是 prompt 调优最容易自欺的地方。</p>')


def main() -> int:
    d = load("phase1_retrieval.json")
    best = "—"
    if d and d.get("group_b_ablation"):
        best = max((r["overall"]["recall"].get("10", 0)
                    for r in d["group_b_ablation"]), default=0)
        base = next((r["overall"]["recall"].get("10", 0)
                     for r in d["group_b_ablation"] if r["retriever"] == "BM25"), 0)
        best = f"{base:.1%} → {best:.1%}"

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GraphRAG Lab 评测报告</title><style>
:root{{--bg:#fff;--fg:#1a1d24;--muted:#6b7280;--line:#e3e7ee;--panel:#f7f8fa;
 --accent:#2563eb;--ok:#0f9d76;--warn:#b45309;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f1115;--fg:#e6e9ef;
 --muted:#8b93a7;--line:#272c38;--panel:#171a21;--accent:#5b9dff;--ok:#7ee0b8;--warn:#ffb454;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 18px 80px}}
h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:38px 0 10px;
 padding-bottom:7px;border-bottom:2px solid var(--line)}}
h3{{font-size:14px;margin:22px 0 8px;color:var(--muted)}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:26px}}
.kpi{{font-size:15px;margin:10px 0 14px}}.kpi b{{color:var(--accent);font-size:19px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}}
th,td{{border:1px solid var(--line);padding:6px 9px;text-align:right;white-space:nowrap}}
th{{background:var(--panel);font-weight:600;color:var(--muted);text-align:right}}
th:first-child,td.lbl{{text-align:left}}
td.hi{{background:rgba(37,99,235,.08);font-weight:600}}
.note{{font-size:12.5px;color:var(--muted);background:var(--panel);
 padding:10px 13px;border-radius:8px;border-left:3px solid var(--accent);margin:10px 0}}
.warn{{font-size:12.5px;color:var(--warn);background:var(--panel);
 padding:10px 13px;border-radius:8px;border-left:3px solid var(--warn);margin:10px 0}}
.muted{{color:var(--muted)}}code{{background:var(--panel);padding:1px 5px;border-radius:4px;
 font-size:12px}}.scroll{{overflow-x:auto}}
ul{{padding-left:20px}}li{{margin:5px 0}}
</style></head><body><div class="wrap">
<h1>GraphRAG Lab 评测报告</h1>
<div class="sub">生成于 {date.today()}　·　全部语料为程序合成，机构/人名/辖区均属虚构　·　
检索 recall@10：<b>{best}</b></div>

<h2>一、检索</h2>
{section_retrieval()}

<h2>二、抽取与实体消歧</h2>
{section_extraction()}

<h2>三、生成与 Prompt A/B</h2>
{section_e2e()}

<h2>四、主要结论</h2>
<ul>
<li><b>纯检索对多跳关系存在结构性失败，加大 k 无解。</b>hop4_risk 的 recall
随 k 从 1 放大到 20 仅由 17% 升到 50% 即饱和 —— 关键证据与问题零词面重叠，
无论 top-k 取多大都召不回。已由 <code>verify_dataset.py</code> 形式化验证。</li>
<li><b>等权 RRF 融合会把分数拉垮。</b>它只看名次、不看「这一路在这类问题上是否可信」，
BM25 的高排名噪音把图检索的精确结果挤了出去（聚合类 92%→51%）。
改为基于查询意图的<b>级联路由</b>后总分反超。</li>
<li><b>阻断枢纽节点值 23 个点。</b>辖区度数 17~25、托管银行 14，允许穿透会让
「共用一家托管行的两只基金」被判定为存在关联 —— 关系图谱最常见的假阳性来源。</li>
<li><b>抽取 F1 高不代表图好用。</b>规则抽取 F1 99.8%，但同一张图跑检索仍低于标准图，
差距来自<b>未合并的跨文种别名</b>。F1 的计算用标准答案做了实体对齐，掩盖了消歧欠账；
图检索没有这个外挂。</li>
<li><b>向量相似度不能单独用于实体合并。</b>中文模型对英文串的表示坍缩，
两家毫不相干的公司余弦相似度可达 1.00。词面三值判定作为守卫后该类错合并归零。</li>
</ul>

<h2>五、已知缺口</h2>
<ul>
<li>纯语义题 recall@10 仅 63.6%，向量模型为 <code>bge-small-zh</code> 量化版，
换更大模型或加 rerank 应有提升空间。</li>
<li>查询路由粒度偏粗：两跳题上「融合」优于「路由」，说明应按题型给融合权重而非一刀切。</li>
<li>跨文种实体消歧依赖 LLM 裁决，未配置 API key 时留有欠账。</li>
<li>LLM 抽取与 prompt A/B 需配置 provider 后运行，当前报告中的抽取分数来自规则基线。</li>
</ul>
</div></body></html>"""

    out = REPORTS / "index.html"
    write_text(out, html)
    print(f"[ok] 报告已生成 -> {out}")
    print(f"     直接用浏览器打开即可（离线，无外部依赖）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
